import os
import pickle
import torch
import torch.nn.functional as F

from data_utils import get_transform


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        name='kitti',
        path='diffusion_train_data',
        classes=("Car",),
        event_shape=(256, 4),
        noise_level=(0.7, 0.35, 0.15, 0.4, 0.2, 0.2, 0.3),
        limit=4.0,
        augment_scale=False,
        use_condition_xyz=False,
    ):
        super().__init__()
        assert event_shape[1] in [3, 4]

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"Loading {name} from {path}")
        self.name = name
        self.path = path = os.path.join(path, f"{name}_train_{limit:.0f}")
        self.classes = classes
        self.event_shape = event_shape
        assert len(noise_level) == 7
        self.noise_level = torch.tensor(noise_level)
        self.limit = limit
        self.augment_scale = augment_scale
        self.use_condition_xyz = use_condition_xyz

        self.data_list = []
        for c in classes:
            self.data_list += [os.path.join(path, c, x) for x in os.listdir(os.path.join(path, c))]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        data = load_pickle(self.data_list[item])
        points = torch.from_numpy(data["ptc"]).float()
        size = torch.from_numpy(data["size"]).float()
        if self.use_condition_xyz:
            condition_xyz = torch.from_numpy(data["translation"]).float()
        else:
            condition_xyz = None

        # random flip
        if torch.randint(2, size=(), dtype=torch.bool):
            points = points * torch.tensor([[1, -1, 1]], dtype=points.dtype)
        if self.augment_scale:
            points = points * torch.empty(1, 3, dtype=points.dtype).uniform_(0.9, 1.1)

        sigma = torch.rand(1) * 79.9875 + 0.0125
        noise = torch.randn(7) * sigma / 80 * self.noise_level * 2
        new_box = torch.cat([
            noise[:3],
            size * noise[3:6].exp(),
            noise[6:7]
        ])
        ptc = torch.cat([points[:, :3], torch.ones_like(points[:, :1])], dim=1)
        trs = get_transform(new_box)
        ptc = torch.matmul(ptc, trs)[:, :3]
        # scale = torch.diag(F.pad(2 / new_box[3:6], (0, 1), "constant", 1))
        mask = torch.all(torch.abs(ptc) < self.limit, dim=-1)

        points = points[mask] * 2 / size.unsqueeze(dim=0)
        ptc = ptc[mask]

        if self.event_shape[1] == 4:
            pp_score = torch.from_numpy(data["pp_score"]).float()[mask]
        else:
            pp_score = None

        num_points = points.shape[0]
        if num_points > self.event_shape[0]:
            selection = torch.randperm(num_points)[:self.event_shape[0]]
            points = points[selection]
            ptc = ptc[selection]
            if pp_score is not None:
                pp_score = pp_score[selection]
            mask = torch.zeros(self.event_shape[0], dtype=torch.bool)
        else:
            points = F.pad(points, (0, 0, 0, self.event_shape[0] - num_points), "constant", 0)
            ptc = F.pad(ptc, (0, 0, 0, self.event_shape[0] - num_points), "constant", 0)
            if pp_score is not None:
                pp_score = F.pad(pp_score, (0, 0, 0, self.event_shape[0] - num_points), "constant", -1)
            mask = F.pad(
                torch.zeros(num_points, dtype=torch.bool),
                (0, self.event_shape[0] - num_points), "constant", 1)

        return points, ptc, sigma, pp_score, mask, condition_xyz

if __name__ == "__main__":
    dataset = PointCloudDataset()
    print(len(dataset))
    for i in range(20):
        data, pp_score, mask = dataset[i]
        print(data.shape, pp_score.shape, mask.shape)