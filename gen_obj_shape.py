import os
from typing import Tuple, Union
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from data_utils import get_fov_flag, get_transform_multi, load_velo_scan

from pcdet.utils import calibration_kitti
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("--context-limit", type=int, default=4)
parser.add_argument("--out-dir", type=str, required=True)
parser.add_argument("--dataset", type=str, default='kitti', choices=['kitti'])
parser.add_argument("--dataset-path", type=str, default="OpenPCDet/data/kitti")
args = parser.parse_args()

def check_type_and_convert(
    ptc: Union[np.ndarray, torch.Tensor],  # (N_points, 4)
    pp_score: Union[np.ndarray, torch.Tensor],  # (N_points,)
    boxes: Union[np.ndarray, torch.Tensor],
    extreme: float = 0.2
):
    assert type(ptc) == type(pp_score) == type(boxes)
    np_input = isinstance(ptc, np.ndarray)
    assert ptc.ndim == 2 and pp_score.ndim == 1 and ptc.shape[1] == 4 and ptc.shape[0] == pp_score.shape[0]
    assert boxes.ndim >= 2 and boxes.shape[-1] == 7
    _shape = boxes.shape[:-1]
    if np_input:
        ptc = torch.from_numpy(ptc).float()
        pp_score = torch.from_numpy(pp_score).float()
        boxes = torch.from_numpy(boxes).view(-1, 7).float()  # [M, 7]
    if torch.cuda.is_available():
        ptc = ptc.cuda()
        pp_score = pp_score.cuda()
        boxes = boxes.cuda()
    pp_score = torch.clamp(pp_score, min=extreme, max=1-extreme)
    boxes = torch.cat([boxes[:, :3], torch.clamp(boxes[:, 3:6], min=1e-6), boxes[:, 6:]], dim=1)  # removes negative box sizes
    return ptc, pp_score, boxes, np_input, _shape

def get_image_size(path):
    img = Image.open(path)
    img_size = img.size
    img.close()
    assert img_size[1] <= img_size[0]
    return (img_size[1], img_size[0])

class KITTITrain(Dataset):
    def __init__(self, dataset_path):
        print("Loading KITTI")
        self.dataset_path = dataset_path
        idx_path = "gen_obj_shape_info/kitti_train_idx_hasobj.txt"  # filtered out scenes with no objects
        self.train_set = [int(x) for x in open(idx_path).readlines()]

        gt_info = pickle.load(open("gen_obj_shape_info/kitti_labels.pkl", "rb"))
        self.gt_dict = {}
        for gt in gt_info:
            self.gt_dict[int(gt['frame_id'])] = gt

        self.ptc_path = os.path.join(self.dataset_path, 'training', 'velodyne')
        self.calib_path = os.path.join(self.dataset_path, 'training', 'calib')
        self.image_path = os.path.join(self.dataset_path, 'training', 'image_2')

    def __len__(self):
        return len(self.train_set)

    def __getitem__(self, item):
        idx = self.train_set[item]
        ptc = load_velo_scan(os.path.join(self.ptc_path, f"{idx:06d}.bin"))
        curr_boxes = self.gt_dict[idx]['boxes_lidar']
        categories = self.gt_dict[idx]['name']

        curr_calib = calibration_kitti.Calibration((os.path.join(self.calib_path, f"{idx:06d}.txt")))
        img_size = get_image_size(os.path.join(self.image_path, f"{idx:06d}.png"))
        curr_fov_flag = get_fov_flag(ptc, img_size, curr_calib)
        ptc = ptc[curr_fov_flag]
        pp_score = None

        return ptc, pp_score, curr_boxes, categories

def get_scale(
    ptc: torch.Tensor,  # (N_points, 4)
    boxes: torch.Tensor,  # (M, 7)
) -> Tuple[torch.Tensor, torch.Tensor]:
    ptc = torch.cat([ptc[:, :3], torch.ones_like(ptc[:, :1])], dim=1)  # [N_points, 4]
    ptc = ptc.unsqueeze(dim=0).expand(boxes.shape[0], ptc.shape[0], 4)  # [M, N_points, 4]

    trs = get_transform_multi(boxes)  # [M, 4, 4]
    ptc = torch.bmm(ptc, trs)[:, :, :3]  # [M, N_points, 3]
    scale = ptc / (boxes[:, 3:6].unsqueeze(dim=1) * 0.5)

    scale = torch.max(torch.abs(scale), dim=2).values
    return ptc, scale


if __name__ == "__main__":
    args.out_dir = f"{args.out_dir}/{args.dataset}_train_{args.context_limit}"
    print(f"save to {args.out_dir}")
    os.makedirs(args.out_dir, exist_ok=True)
    if args.dataset == 'kitti':
        dataset = KITTITrain(args.dataset_path)
    else:
        raise NotImplementedError
        
    category_count = {}
    for ptc, pp_score, boxes, categories in tqdm(dataset):
        if pp_score is None:
            pp_score = np.zeros(ptc.shape[0])
            pp_score.fill(np.nan)
            if not isinstance(ptc, np.ndarray):
                pp_score = torch.tensor(pp_score)
        if not type(ptc) == type(pp_score) == type(boxes):
            import ipdb; ipdb.set_trace()
        ptc, pp_score, boxes, np_input, _shape = check_type_and_convert(ptc, pp_score, boxes, extreme=0.0)
        ptc, scale = get_scale(ptc, boxes)
        for box_id in range(boxes.shape[0]):
            category = str(categories[box_id])
            if category not in category_count:
                c_idx = category_count[category] = 0
                os.makedirs(os.path.join(args.out_dir, category), exist_ok=True)
            else:
                c_idx = category_count[category] = category_count[category] + 1
            save_path = os.path.join(args.out_dir, category, f"{c_idx:06d}.pkl")

            mask = scale[box_id] < args.context_limit
            box_ptc = ptc[box_id][mask]
            box_pp = pp_score[mask].unsqueeze(dim=1)

            data_dict = {
                "ptc": box_ptc.cpu().numpy(),
                "pp_score": box_pp.cpu().numpy(),
                "size": boxes[box_id, 3:6].cpu().numpy(),
                "translation": boxes[box_id, 0:3].cpu().numpy(),
            }

            with open(save_path, "wb") as f:
                pickle.dump(data_dict, f)
