import os
from datetime import datetime
import argparse
import copy
from pathlib import Path
import pickle
from pcdet.datasets import build_dataloader
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils, box_utils
from pcdet.models.model_utils.model_nms_utils import class_agnostic_nms
from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu

import torch
import numpy as np
from easydict import EasyDict
from tqdm import trange
import torch.nn.functional as F

import sys
sys.path.insert(0, "../..")
from data_utils import forward_transform


def load_pickle(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)


def get_points_update(
        x_cur, net, t_cur, t_next=0, pp_score=None, mask=None,
        num_steps=64, sigma_min=0.002, sigma_max=80, rho=7,
        S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    if mask is None:
        mask = torch.zeros(x_cur.shape[:-1], dtype=torch.bool, device=x_cur.device)

    gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    t_hat = net.round_sigma(t_cur + gamma * t_cur)
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

    denoised = net(x_hat, t_hat, pp_score=pp_score, mask=mask).to(torch.float64)
    d_cur = (x_hat - denoised) / t_hat
    points_update = (t_next - t_hat) * d_cur

    if t_next > 0:
        x_next = x_hat + points_update
        denoised = net(x_next, t_next, pp_score=pp_score, mask=mask).to(torch.float64)
        d_prime = (x_next - denoised) / t_next
        points_update = (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return points_update


@torch.enable_grad()
def get_box_update(ptc, box, points_update, pp_score=None, points_mask=None, limit=2.0):
    box1 = box.clone().detach().contiguous()
    target = forward_transform(ptc, box, pp_score, points_mask, limit)[0] + points_update
    target = target.detach().contiguous()
    for _ in range(1):
        box1.requires_grad = True
        optimizer = torch.optim.LBFGS([box1], lr=0.5, max_iter=20)
        def closure():
            outputs = forward_transform(ptc, box1, pp_score, points_mask, limit)[0]
            loss = F.mse_loss(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            return loss
        optimizer.step(closure)
        box1 = box1.detach().contiguous()
    return box1 - box

def subsample_indices(n_points, subsample_size=256, seed=-1):
    if seed >= 0:
        rng = np.random.default_rng(seed=seed)
        indices = np.sort(rng.choice(n_points, size=subsample_size, replace=False))
    else:
        indices = np.sort(np.random.choice(n_points, size=subsample_size, replace=False))
    return indices


def to_flattened_numpy(x):
    return x.detach().cpu().numpy().reshape((-1,))

def from_flattened_numpy(x, shape):
    return torch.from_numpy(x.reshape(shape))

def parse_config():
    parser = argparse.ArgumentParser(description='Evaluate denoiser')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='lyft', choices=["lyft", "ithaca365", "nuscenes"], help='The dataset that the detections were obtained from')
    parser.add_argument('--trained-on-dataset', type=str, default="kitti", choices=['kitti'], help='The dataset that the diffusion model is trained on')
    parser.add_argument('--category', type=str, default='car', choices=["car", "cyclist", "pedestrian"], help='The class of traffic participants')
    parser.add_argument('--ckpt', type=str, default=None, help='Checkpoint path for the diffusion model')
    parser.add_argument('--save-dir', required=True, type=str, help='Folder to save the outputs')

    # denoising configs
    parser.add_argument('--num-steps', type=int, default=16)
    parser.add_argument('--early-stop', type=int, default=2)
    parser.add_argument('--sigma_min', type=float, default=0.02)
    parser.add_argument('--sigma_max', type=float, default=80)
    parser.add_argument('--sigma_max_lb', type=float, default=10.)
    parser.add_argument('--shape-weight', type=float, default=0.1)
    parser.add_argument('--min-score', type=float, default=0.0)
    parser.add_argument('--max-score', type=float, default=1.0)
    parser.add_argument('--rho', type=float, default=1)

    parser.add_argument('--context-limit', type=float, default=4.0)
    
    parser.add_argument('--use-ot', action="store_true", help='Perform output transformation adaptation')
    parser.add_argument('--eval-only', action='store_true', help='Evaluate detection performance without denoising')
    parser.add_argument('--apply-nms', action='store_true', default=False, help='Perform NMS')
    

    parser.add_argument('--det-path', default=None, type=str, required=True, help='Pickle file of detected boxes')
    parser.add_argument('--custom-cfg-file', default=None, type=str, help='Custom config file for loading the dataset')
    

    args = parser.parse_args()

    if args.trained_on_dataset is None:
        args.trained_on_dataset = args.dataset
    if args.dataset == args.trained_on_dataset:
        assert not args.use_ot

    # cfg is only used to load test set, model doesn't matter here
    if args.custom_cfg_file is not None:
        cfg_file = args.custom_cfg_file
    elif args.dataset == "kitti":
        cfg_file = "cfgs/kitti_models/pv_rcnn_xyz.yaml"
    elif args.dataset == "lyft":
        cfg_file = "cfgs/lyft_models/pointrcnn_xyz_dense.yaml"
    elif args.dataset == "ithaca365":
        cfg_file = "cfgs/ithaca365_models_kitti_format/pointrcnn_xyz_dense.yaml"
    elif args.dataset == 'nuscenes':
        cfg_file = "cfgs/nuscenes_models_kitti_format/pointrcnn_xyz.yaml"
    else:
        raise NotImplementedError

    cfg_from_yaml_file(cfg_file, cfg)
    cfg.TAG = Path(cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(cfg_file.split('/')[1:-1])

    torch.manual_seed(args.seed)

    return args, cfg


def main():
    device = torch.device("cuda")
    args, cfg = parse_config()

    S_churn = 0
    S_min = 0
    S_max = float('inf')
    S_noise = 1

    logger = common_utils.create_logger()
    test_set, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False, workers=4, logger=logger, training=False
    )
    eval_gt_annos = [copy.deepcopy(info['annos']) for info in test_set.kitti_infos]
    
    eval_gt_annos = eval_gt_annos

    shape_templates = {
        "kitti": {
            "car": torch.tensor([3.89, 1.62, 1.53]).to(device),
            "cyclist": torch.tensor([1.77, 0.57, 1.72]).to(device),
            "pedestrian": torch.tensor([0.82, 0.63, 1.77]).to(device),
        },
        "lyft": {
            "car": torch.tensor([4.74, 1.91, 1.71]).to(device),
            "cyclist": torch.tensor([1.75, 0.61, 1.36]).to(device),
            "pedestrian": torch.tensor([0.80, 0.78, 1.74]).to(device),
        },
        "ithaca365": {
            "car": torch.tensor([4.41, 1.75, 1.55]).to(device),
            "cyclist": torch.tensor([1.70, 0.70, 1.53]).to(device),
            "pedestrian": torch.tensor([0.60, 0.61, 1.70]).to(device),
        },
        "nuscenes": {
            "car": torch.tensor([4.63, 1.95, 1.73]).to(device),
            "cyclist": torch.tensor([1.93, 0.71,  1.42]).to(device),
            "pedestrian": torch.tensor([0.73, 0.67, 1.78]).to(device),
        }
    }
    shape_template = shape_templates[args.dataset][args.category]

    det_path = args.det_path
    
    if args.ckpt is None:
        args.ckpt = f"{args.trained_on_dataset}-{args.category}-contextlimit{args.context_limit:.0f}.pt"
    assert os.path.isfile(args.ckpt), 'Checkpoint does not exist'

    print(f"Adapting {args.trained_on_dataset} -> {args.dataset}, {args.category}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Detection path: {det_path}")
    print("Apply NMS:", args.apply_nms)

    net = torch.load(args.ckpt)["net"].eval()

    det_annos = load_pickle(det_path)
    assert len(det_annos) == len(test_set)

    _, ap_dict_b4 = kitti_eval.get_official_eval_result(
        eval_gt_annos, det_annos, cfg.CLASS_NAMES)
    
    with torch.no_grad():
        for frame_id in trange(len(eval_gt_annos)):
            det_anno = det_annos[frame_id]
            det_mask = det_anno['name'] == args.category.title()
            if np.sum(det_mask) == 0:
                continue

            gt_anno = test_set[frame_id]
            gt_mask = gt_anno['gt_boxes'][:, 7] == cfg.CLASS_NAMES.index(args.category.title()) + 1
            if np.sum(gt_mask) == 0:
                continue
            gt_boxes = torch.from_numpy(gt_anno['gt_boxes'][gt_mask, :7]).float().cuda()

            ptc = torch.from_numpy(gt_anno['points'][:, :3]).float().cuda()
            pp_score = None

            for obj_cnt, obj_id in enumerate(np.where(det_mask)[0]):
                if det_anno['score'][obj_id] < args.min_score or det_anno['score'][obj_id] > args.max_score:
                    continue
                x_box = torch.from_numpy(det_anno['boxes_lidar'][obj_id]).float().cuda().clone().detach().contiguous()
                if args.use_ot:
                    x_box[3:6] *= shape_templates[args.dataset][args.category] / shape_templates[args.trained_on_dataset][args.category]
                iou = boxes_iou3d_gpu(x_box.unsqueeze(dim=0), gt_boxes)[0]
                gt_idx = torch.argmax(iou).item()
                if iou[gt_idx] <= 1e-6:
                    continue

                sigma_min = max(args.sigma_min, net.sigma_min)

                if not args.eval_only:
                    sigma_max = (args.sigma_max ** (1 / args.rho) + det_anno['score'][obj_id] * (
                                args.sigma_max_lb ** (1 / args.rho) - args.sigma_max ** (1 / args.rho))) ** args.rho

                    step_indices = torch.arange(args.num_steps, device=device)
                    t_steps = (sigma_max ** (1 / args.rho) + step_indices / (args.num_steps - 1) * (
                                sigma_min ** (1 / args.rho) - sigma_max ** (1 / args.rho))) ** args.rho
                    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

                    for i, (t_cur, t_next) in list(enumerate(zip(t_steps[:-1], t_steps[1:])))[:args.num_steps-args.early_stop]:
                        x_points, x_points_pp, x_points_mask = forward_transform(ptc, x_box, pp_score, limit=args.context_limit)

                        n_points_orig = x_points.shape[0]
                        if x_points.shape[0] > 1024:
                            indices = subsample_indices(n_points_orig, subsample_size=1024, seed=-1)
                            x_points = x_points[indices, :]
                            x_points_pp = x_points_pp[indices] if x_points_pp is not None else None
                            indices_not_selected = np.delete(np.arange(n_points_orig), indices)
                            x_points_mask_indices_not_selected = np.where(x_points_mask.detach().cpu().numpy())[0][
                                indices_not_selected]
                            x_points_mask[x_points_mask_indices_not_selected] = False

                        x_points = x_points.unsqueeze(dim=0)
                        x_points_pp = x_points_pp.unsqueeze(dim=-1).unsqueeze(dim=0) if x_points_pp is not None else None

                        points_update = get_points_update(
                            x_points, net, t_cur, t_next, pp_score=x_points_pp,
                            num_steps=args.num_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=args.rho,
                            S_churn=S_churn, S_min=S_min, S_max=S_max, S_noise=S_noise,
                        )
                        box_update = get_box_update(ptc, x_box, points_update.float()[0], pp_score, x_points_mask, limit=args.context_limit).float()

                        x_box = x_box + box_update
                        x_box[3:6] = x_box[3:6] * (1 - args.shape_weight) + args.shape_weight * shape_template

                det_anno['boxes_lidar'][obj_id] = x_box.detach().cpu().numpy()

            if args.apply_nms and np.sum(det_mask) > 0:
                selected, _ = class_agnostic_nms(
                    box_scores=torch.from_numpy(det_anno['score']).cuda().float(),
                    box_preds=torch.from_numpy(det_anno['boxes_lidar']).cuda().float(),
                    nms_config=EasyDict(
                        NMS_PRE_MAXSIZE=det_anno['score'].shape[0],
                        NMS_TYPE='nms_gpu',
                        NMS_THRESH=0.1,
                        NMS_POST_MAXSIZE=det_anno['score'].shape[0])
                )
                selected = selected.detach().cpu().numpy()
                det_anno['boxes_lidar'] = det_anno['boxes_lidar'][selected]
                det_anno['score'] = det_anno['score'][selected]
                det_anno['bbox'] = det_anno['bbox'][selected]
                det_anno['name'] = det_anno['name'][selected]
                det_anno['alpha'] = det_anno['alpha'][selected]
                det_anno['rotation_y'] = det_anno['rotation_y'][selected]
                det_mask = det_mask[selected]

            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(det_anno['boxes_lidar'], gt_anno['calib'])
            det_anno['dimensions'] = pred_boxes_camera[:, 3:6]
            det_anno['location'] = pred_boxes_camera[:, 0:3]
            det_anno['alpha'] = det_anno['alpha'] + pred_boxes_camera[:, 6] - det_anno['rotation_y']
            det_anno['rotation_y'] = pred_boxes_camera[:, 6]

    _, ap_dict_after = kitti_eval.get_official_eval_result(
        eval_gt_annos, det_annos, cfg.CLASS_NAMES)

    for k, v in vars(args).items():
        print(f"{k}: {v}")

    for k in ap_dict_after:
        if args.category.title() in k and ("3d" in k or "bev" in k) and "R40" in k:
            print(f"{k}: {ap_dict_b4[k]:0.2f} -> {ap_dict_after[k]:0.2f}")

    info_str = []
    for suffix, iou_threshold in zip(["_R40", "_R40_0.5", "_R40_0.25"], ["0.7", "0.5", "0.25"]):
        s0 = [f"mAP@{iou_threshold}: "]
        for dist in ["0-30m", "30-50m", "50-80m", "0-80m"]:
            s = []
            for metric in ["bev", "3d"]:
                k = f"{args.category.title()}_{metric}/{dist}{suffix}"
                s.append(f"{ap_dict_after[k]:0.2f}")
            s0.append(" / ".join(s))
        info_str.append("\t".join(s0))
    info_str = "\n".join(info_str)

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    fname = f"{args.dataset}_{args.category}_{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}"
    save_path = os.path.join(save_dir, fname + ".pkl")
    with open(save_path, "wb") as f:
        pickle.dump(det_annos, f)

    save_path = os.path.join(save_dir, fname + ".txt")
    with open(save_path, "w") as f:
        f.write(f"det_path: {det_path}\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write(info_str)
    print(info_str)
    
    print(f"Results saved to: {save_path}")

if __name__ == "__main__":
    main()