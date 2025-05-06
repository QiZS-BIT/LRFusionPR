import cv2
import torch
import numpy as np
from torchvision.transforms import transforms
from PIL import Image


def find_nearest_ndx(ts, timestamps):
    ndx = np.searchsorted(timestamps, ts)
    if ndx == 0:
        return ndx
    elif ndx == len(timestamps):
        return ndx - 1
    else:
        assert timestamps[ndx-1] <= ts <= timestamps[ndx]
        if ts - timestamps[ndx-1] < timestamps[ndx] - ts:
            return ndx - 1
        else:
            return ndx


def load_radar_polar_pool(file_pathname, w=225, h=50):
    radar_img = cv2.imread(file_pathname, 0).transpose(1, 0)[:1826, :]
    radar_img = radar_img.astype(np.float32)
    radar_img = radar_img[np.newaxis, :, :]
    radar_img = torch.from_numpy(radar_img)
    radar_img = radar_img.unsqueeze(0)

    img_transforms = transforms.Compose([transforms.Resize((h, w))])

    downsampled_radar_img = img_transforms(radar_img)

    downsampled_radar_img = downsampled_radar_img.squeeze(0).squeeze(0).numpy().astype(np.uint8)

    # downsampled_radar_img[downsampled_radar_img < 60] = 0
    return downsampled_radar_img


def load_pc(file_pathname):
    pc = np.fromfile(file_pathname, dtype=np.float32)
    pc = np.reshape(pc, (4, -1)).transpose(1, 0)[:, :3]

    mask = np.all(np.isclose(pc, 0.), axis=1)
    pc = pc[~mask]

    dist_mask = np.linalg.norm(pc[:, :3], 2, axis=1) < 80.0
    pc = pc[dist_mask]

    mask = pc[:, 2] < 1.0
    pc = pc[mask]

    # coord alignment
    transformation_matrix = np.array([[-1., 0., 0., 0.1],
                                      [0, -1, 0, -0.47],
                                      [0., 0., 1., 0.28],
                                      [0., 0., 0., 1.]])
    pc = transformation_matrix.dot(np.concatenate([pc, np.ones_like(pc[:, 0:1])], axis=-1).T).T[:, :3]
    return pc
