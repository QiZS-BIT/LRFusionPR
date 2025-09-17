import cv2
from PIL import Image
import numpy as np
from torchvision.transforms import transforms
import torch
from scipy.spatial import distance_matrix


# Coordinates of test region centres (in Sejong sequence)
TEST_REGION_CENTRES = np.array([[345090.0743, 4037591.323], [345090.483, 4044700.04],
                                [350552.0308, 4041000.71], [349252.0308, 4044800.71]])

# Radius of the test region
TEST_REGION_RADIUS = 500

# Boundary between training and test region - to ensure there's no overlap between training and test clouds
TEST_TRAIN_BOUNDARY = 50


def in_train_split(pos):
    # returns true if pos is in train split
    assert pos.ndim == 2
    assert pos.shape[1] == 2
    dist = distance_matrix(pos, TEST_REGION_CENTRES)
    mask = (dist > TEST_REGION_RADIUS + TEST_TRAIN_BOUNDARY).all(axis=1)
    return mask


def in_test_split(pos):
    # returns true if position is in evaluation split
    assert pos.ndim == 2
    assert pos.shape[1] == 2
    dist = distance_matrix(pos, TEST_REGION_CENTRES)
    mask = (dist < TEST_REGION_RADIUS).any(axis=1)
    return mask


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


def load_pc(file_pathname):
    # Load point cloud, clip x, y and z coords (points far away and the ground plane)
    # Returns Nx3 matrix
    pc = np.fromfile(file_pathname, dtype=np.float32)
    # PC in Mulran is of size [num_points, 4] -> x,y,z,reflectance
    pc = np.reshape(pc, (-1, 4))[:, :3]

    mask = np.all(np.isclose(pc, 0.), axis=1)
    pc = pc[~mask]

    dist_mask = np.linalg.norm(pc[:, :3], 2, axis=1) < 80.0
    pc = pc[dist_mask]

    mask = pc[:, 2] > -0.9
    pc = pc[mask]
    return pc


def load_radar_polar(file_pathname, w=225, h=50):
    # 3360-200m 1344-80m
    radar_img = cv2.imread(file_pathname, 0)[:1344, :]
    radar_img = Image.fromarray(radar_img)

    img_transforms = transforms.Compose([transforms.Resize((h, w)),
                                         transforms.ToTensor()])

    # 3360-200m 1344-80m
    radar_img_tensor = img_transforms(radar_img)
    radar_img = radar_img_tensor.squeeze(0).numpy() * 255
    # radar_img[radar_img < 50] = 0
    return radar_img


def load_radar_polar_pool(file_pathname, w=225, h=50):
    radar_img = cv2.imread(file_pathname, 0)[:1344, :]  # 80m-1344
    radar_img = radar_img.astype(np.float32)
    radar_img = radar_img[np.newaxis, :, :]
    radar_img = torch.from_numpy(radar_img)
    radar_img = radar_img.unsqueeze(0)

    img_transforms = transforms.Compose([transforms.Resize((h, w))])

    downsampled_radar_img = img_transforms(radar_img)

    downsampled_radar_img = downsampled_radar_img.squeeze(0).squeeze(0).numpy().astype(np.uint8)

    # downsampled_radar_img[downsampled_radar_img < 60] = 0
    return downsampled_radar_img


def polar_to_cartesian(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def cartesian_to_polar(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return r, theta
