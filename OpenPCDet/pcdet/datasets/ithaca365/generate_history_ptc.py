import hydra
import numpy as np
import os
import os.path as osp
from omegaconf import DictConfig, OmegaConf
from ithaca365.ithaca365 import Ithaca365
from ithaca365.utils import splits
from tqdm.auto import tqdm
import sys
import pickle
import ithaca365_utils

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def display_args(args):
    eprint("========== history info ==========")
    eprint("host: {}".format(os.getenv('HOSTNAME')))
    eprint(OmegaConf.to_yaml(args))
    eprint("==================================")


@hydra.main(config_path="configs/", config_name="compute_history.yaml")
def main(args: DictConfig):
    display_args(args)
    dataset = Ithaca365(version=args.version,
                     dataroot=args.data_path, verbose=True)

    if args.only_accurate_localization:
        sample_list = dataset.sample_with_accurate_localization
    else:
        sample_list = dataset.sample

    if args.val_only:
        val_scenes = splits.val
        available_scenes = ithaca365_utils.get_available_scenes(dataset)
        available_scene_names = [s['name'] for s in available_scenes]
        val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
        val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes])
        sample_list = [sample for sample in sample_list if sample['scene_token'] in val_scenes]

    idx_list = list(range(len(sample_list)))
    if args.total_part > 1:
        idx_list = np.array_split(
            idx_list, args.total_part)[args.part]
    os.makedirs(args.target_save_path, exist_ok=True)
    for idx in tqdm(idx_list):
        sample = sample_list[idx]
        ref_sd_token = sample['data']['LIDAR_TOP']
        ref_sd_rec = dataset.get('sample_data', ref_sd_token)
        target_save_path = osp.join(
            args.target_save_path,
            osp.basename(ref_sd_rec['filename']).split(".")[0]+".pkl")
        if osp.exists(target_save_path):
            continue
        if args.history_type == 'any':
            history_scans = dataset.get_other_traversals(
                ref_sd_token, sorted_by='pos_type',
                increasing_order=False, ranges=args.ranges,
                num_history=args.num_history, every_x_meter=args.every_x_meter
            )
        elif args.history_type == 'time_valid_least_recent':
            history_scans = dataset.get_other_traversals(
                ref_sd_token, sorted_by='time',
                increasing_order=True, ranges=args.ranges,
                num_history=args.num_history, every_x_meter=args.every_x_meter,
                time_valid=True, accurate_history_only=True
            )
        elif args.history_type == 'time_valid_most_recent':
            history_scans = dataset.get_other_traversals(
                ref_sd_token, sorted_by='time',
                increasing_order=False, ranges=args.ranges,
                num_history=args.num_history, every_x_meter=args.every_x_meter,
                time_valid=True, accurate_history_only=True
            )
        elif args.history_type == 'best_hgt_std':
            history_scans = dataset.get_other_traversals(
                ref_sd_token, sorted_by='hgt_stdev',
                increasing_order=True, ranges=args.ranges,
                num_history=args.num_history, every_x_meter=args.every_x_meter,
                time_valid=False, accurate_history_only=True
            )
        else:
            raise NotImplementedError()

        for k in history_scans:
            history_scans[k] = history_scans[k][:, :3]
        pickle.dump(
            history_scans,
            open(target_save_path, "wb")
        )

if __name__ == "__main__":
    main()
