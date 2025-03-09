import os
import numpy as np
import torch
import torch.nn.functional as F
import pickle


def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan


def get_fov_flag(points, img_shape, calib):
    """
    Args:
        points (numpy.ndarray): A numpy array of shape `(N, 4)` containing lidar points. Each lidar point is represented by its x, y, z, and intensity values.
        img_shape (tuple): A tuple of length 2 representing the height and width of the image in pixels.
        calib (calibration_kitti.Calibration): An instance of the `Calibration` class from the `calibration_kitti` library that contains calibration information for the lidar and camera sensors.

    Returns:
        pts_valid_flag (numpy.ndarray): A boolean numpy array of shape `(N,)` where `N` is the number of lidar points. The array contains True for points that are within the field of view (FOV) of the camera and False for points that are outside the FOV.
    """
    pts_rect = calib.lidar_to_rect(points[:, 0:3])
    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

    return pts_valid_flag


class LyftTrain(torch.utils.data.Dataset):
    def __init__(
            self,
            fov: bool=True,
            mode: str="all",
    ):
        assert mode in ("all", "car", "car_truck")
        self.fov = fov
        if mode == "car":
            idx_path = os.path.expanduser("~/datasets/lyft_fw70_2m_train_idx_hascar.txt")
        else:
            idx_path = os.path.expanduser("~/datasets/lyft_fw70_2m_train_idx_hasobj.txt")  # filtered out scenes with no objects
        self.train_set = [int(x) for x in open(idx_path).readlines()]

        if mode == "car_truck":
            gt_dict_path = os.path.expanduser("~/datasets/lyft_infos_train_filtered_car_truck_iou_lt_0.5.pkl")
        elif mode == "all":
            gt_dict_path = os.path.expanduser("~/datasets/lyft_infos_train_filtered_iou_lt_0.5.pkl")
        elif mode == "car":
            gt_dict_path = os.path.expanduser("~/datasets/lyft_infos_train_filtered_car_iou_lt_0.5.pkl")
        self.gt_dict = pickle.load(open(gt_dict_path, "rb"))
        self.ptc_path = "~/lyft/training/velodyne/"
        if not os.path.isdir(self.ptc_path):
            self.ptc_path = "~/datasets/lyft_release_test/training/velodyne/"
        self.calib_path = "~/datasets/lyft_release_test/training/calib/"
        self.p2score_path = "~/lyft/training/pp_score_fw70_2m_r0.3/"
        if not os.path.isdir(self.p2score_path):
            self.p2score_path = "~/datasets/lyft_release_test/training/pp_score_fw70_2m_r0.3/"

    def __len__(self):
        return len(self.train_set)

    def __getitem__(self, item):
        idx = self.train_set[item]
        ptc = load_velo_scan(os.path.join(self.ptc_path, f"{idx:06d}.bin"))
        pp_score = np.load(os.path.join(self.p2score_path, f"{idx:06d}.npy"))
        curr_boxes = self.gt_dict[idx]
        return ptc, pp_score, curr_boxes


def convert_to_tensor(
        ptc: np.ndarray,  # (N_points, 4)
        pp_score: np.ndarray,  # (N_points,)
        boxs: np.ndarray,
        extreme: float = 0.0
):
    assert type(ptc) == type(pp_score) == type(boxs)
    assert ptc.ndim == 2 and pp_score.ndim == 1 and ptc.shape[1] == 4 and ptc.shape[0] == pp_score.shape[0]
    assert boxs.ndim >= 2 and boxs.shape[-1] == 7

    ptc = torch.from_numpy(ptc).float()
    pp_score = torch.from_numpy(pp_score).float()
    boxs = torch.from_numpy(boxs).view(-1, 7).float()  # [M, 7]

    if torch.cuda.is_available():
        ptc = ptc.cuda()
        pp_score = pp_score.cuda()
        boxs = boxs.cuda()
    pp_score = torch.clamp(pp_score, min=extreme, max=1 - extreme)
    boxs = torch.cat([boxs[:, :3], torch.clamp(boxs[:, 3:6], min=1e-6), boxs[:, 6:]],
                     dim=1)  # removes negative box sizes
    return ptc, pp_score, boxs


def get_transform(box: torch.Tensor) -> torch.Tensor:
    # Extract box center coordinates, dimensions and rotation angle
    center_xyz = box[:3]
    dimensions = box[3:6]
    rotation_xy = box[6]

    # Compute rotation matrix around the z-axis
    cos, sin = torch.cos(rotation_xy), torch.sin(rotation_xy)
    zero, one = torch.zeros_like(cos), torch.ones_like(cos)
    rotation_z = torch.stack([cos, -sin, zero, zero,
                              sin, cos, zero, zero,
                              zero, zero, one, zero,
                              zero, zero, zero, one], dim=-1).view(4, 4)

    # Compute translation matrix to move the center of the box to the origin
    translation = torch.eye(4).to(box.device)
    translation[3, :3] = -center_xyz

    scale = torch.diag(F.pad(2 / box[3:6], (0, 1), "constant", 1))
    return torch.matmul(torch.matmul(translation.to(box.dtype), rotation_z.to(box.dtype)), scale)

def get_transform_multi(boxes: torch.Tensor) -> torch.Tensor:
    """
    Computes the homogeneous transforms on pointclouds to center at each box.

    Parameters
    ----------
    boxs : torch.Tensor, shape=(M, 7), the last dimension represents [center_x, center_y, center_z, dx, dy, dz, rotation_xy]

    Returns
    -------
    numpy.ndarray, shape=(M, 4, 4)
        Homogeneous transforms that map pointclouds to center at each box.
    """

    # Extract box center coordinates, dimensions and rotation angle
    center_xyz = boxes[:, :3]
    dimensions = boxes[:, 3:6]
    rotation_xy = boxes[:, 6]

    # Compute rotation matrix around the z-axis
    cos, sin = torch.cos(rotation_xy), torch.sin(rotation_xy)
    zero, one = torch.zeros_like(cos), torch.ones_like(cos)
    rotation_z = torch.stack([cos, -sin, zero, zero,
                              sin, cos, zero, zero,
                              zero, zero, one, zero,
                              zero, zero, zero, one], dim=-1).view(-1, 4, 4)

    # Compute translation matrix to move the center of the box to the origin
    translation = torch.eye(4).to(boxes.device).unsqueeze(0).repeat(center_xyz.shape[0], 1, 1)
    translation[:, 3, :3] = -center_xyz
    return torch.matmul(translation.to(boxes.dtype), rotation_z.to(boxes.dtype))

def forward_transform(ptc, box, pp_score=None, mask=None, limit=2.0):
    ptc = torch.cat([ptc[:, :3], torch.ones_like(ptc[:, :1])], dim=1)
    trs = get_transform(box)
    ptc = torch.matmul(ptc, trs)[:, :3]
    if mask is None:
        mask = torch.all(torch.abs(ptc) < limit, dim=-1)

    ptc = ptc[mask]
    if pp_score is not None:
        pp_score = pp_score[mask]
    return ptc, pp_score, mask