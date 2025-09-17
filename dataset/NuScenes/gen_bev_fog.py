import os
import pickle
import cv2
import open3d as o3d
import numpy as np
from dataset.NuScenes.utils import load_lidar_data_fog
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm


def radar_polar_bev_projection(pcl_input, measure_range, w=225, h=50):
    mat_global_image = np.zeros((h, w), dtype=np.uint8)

    yaw = -np.arctan2(pcl_input[:, 1], pcl_input[:, 0])
    scan_r = np.linalg.norm(pcl_input[:, :2], axis=1)

    w_ind = np.floor((0.5 * (yaw / np.pi + 1.0)) * w).astype(int)
    h_ind = np.floor(scan_r / measure_range * h).astype(int)

    valid_mask = (w_ind >= 0) & (w_ind < w) & (h_ind >= 0) & (h_ind < h)

    valid_w_ind = w_ind[valid_mask]
    valid_h_ind = h_ind[valid_mask]

    np.add.at(mat_global_image, (valid_h_ind, valid_w_ind), 1)
    mat_global_image = np.clip(mat_global_image, 0, 10)

    mat_global_image = mat_global_image * 10

    mat_global_image[np.where(mat_global_image > 255)] = 255
    mat_global_image = mat_global_image / np.max(mat_global_image) * 255

    return mat_global_image


def lidar_polar_bev_projection(pcl_input, measure_range, resolution, w=900, h=200):
    pcl_input_pc = o3d.geometry.PointCloud()
    pcl_input_pc.points = o3d.utility.Vector3dVector(pcl_input)
    pcl_input_pc = pcl_input_pc.voxel_down_sample(voxel_size=resolution)
    pcl_input_np = np.asarray(pcl_input_pc.points)

    mat_global_image = np.zeros((h, w), dtype=np.uint8)

    yaw = -np.arctan2(pcl_input_np[:, 1], pcl_input_np[:, 0])
    scan_r = np.linalg.norm(pcl_input_np[:, :2], axis=1)

    w_ind = np.floor((0.5 * (yaw / np.pi + 1.0)) * w).astype(int)
    h_ind = np.floor(scan_r / measure_range * h).astype(int)

    valid_mask = (w_ind >= 0) & (w_ind < w) & (h_ind >= 0) & (h_ind < h)

    valid_w_ind = w_ind[valid_mask]
    valid_h_ind = h_ind[valid_mask]

    np.add.at(mat_global_image, (valid_h_ind, valid_w_ind), 1)
    mat_global_image = np.clip(mat_global_image, 0, 10)

    mat_global_image = mat_global_image * 10

    mat_global_image[np.where(mat_global_image > 255)] = 255
    mat_global_image = mat_global_image / np.max(mat_global_image) * 255

    return mat_global_image


def gen_bev(bev_fileroot, info_filepath, bs_whole, nusc_trainval, nusc_test, nusc_root):
    if not os.path.exists(bev_fileroot):
        os.makedirs(bev_fileroot)
    lidar_bev_root = os.path.join(bev_fileroot, 'lidar_fog_0.0128')
    if not os.path.exists(lidar_bev_root):
        os.makedirs(lidar_bev_root)
    radar_bev_root = os.path.join(bev_fileroot, 'radar')
    if not os.path.exists(radar_bev_root):
        os.makedirs(radar_bev_root)

    with open(info_filepath, 'rb') as f:
        infos = pickle.load(f)

    for i in tqdm(range(0, bs_whole.shape[0])):
        cur_ind = bs_whole[i][0]
        cur_info = infos[int(cur_ind)]
        token = cur_info['sample_token']
        is_trainval = cur_info['is_trainval']
        print(f'---------------> Generating dataset of {i} th sample: {token}')

        if is_trainval:
            current_sample = nusc_trainval.get('sample', token)
            ref_sd_token = current_sample['data']['LIDAR_TOP']
            ref_sd_rec = nusc_trainval.get('sample_data', ref_sd_token)
            filename = os.path.join(nusc_root, ref_sd_rec['filename'])
            filename = filename.rsplit('/', 1)[0] + '_fog_0.0128/' + filename.rsplit('/', 1)[1]
            lidar_pcd = load_lidar_data_fog(filename)
        else:
            current_sample = nusc_test.get('sample', token)
            ref_sd_token = current_sample['data']['LIDAR_TOP']
            ref_sd_rec = nusc_test.get('sample_data', ref_sd_token)
            filename = os.path.join(nusc_root, ref_sd_rec['filename'])
            filename = filename.rsplit('/', 1)[0] + '_fog_0.0128/' + filename.rsplit('/', 1)[1]
            lidar_pcd = load_lidar_data_fog(filename)

        ground_mask = lidar_pcd[:, 2] > -1.0
        lidar_pcd = lidar_pcd[ground_mask]
        dist_mask = np.linalg.norm(lidar_pcd[:, :3], 2, axis=1) < 80.0
        lidar_pcd = lidar_pcd[dist_mask]

        l_bev_img = lidar_polar_bev_projection(lidar_pcd, 80.0, 0.4, 900, 200)

        filename_l = os.path.basename(cur_info['lidar_infos']['LIDAR_TOP']['filename']).split('.')[0] + '.png'
        cv2.imwrite(os.path.join(lidar_bev_root, filename_l), l_bev_img)


if __name__ == '__main__':
    nuscenes_root = '/media/octane17/T7ShieldNus/NuScenes'
    dataset_root = '/home/octane17/LRFusionPR/data'
    nuscenes_trainval = NuScenes(version='v1.0-trainval', dataroot=nuscenes_root, verbose=True)
    nuscenes_test = NuScenes(version='v1.0-test', dataroot=nuscenes_root, verbose=True)

    bev_root = os.path.join(dataset_root, 'sq_bev')
    infos_path = os.path.join(dataset_root, 'nuscenes_infos-sq.pkl')
    test_query_ind_root = os.path.join(dataset_root, 'sq_test_query.npy')
    ind_test_query = np.load(test_query_ind_root)
    ind_whole = np.concatenate([ind_test_query], axis=0)
    gen_bev(bev_root, infos_path, ind_whole, nuscenes_trainval, nuscenes_test, nuscenes_root)
