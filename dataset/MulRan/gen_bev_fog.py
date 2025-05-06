import copy
import os
import pickle
import cv2
import open3d as o3d
import numpy as np
from dataset.MulRan.utils import load_pc, load_radar_polar_pool
from tqdm import tqdm


def lidar_polar_bev_projection(pcl_input, measure_range, resolution, w=900, h=200):
    pcl_input_pc = o3d.geometry.PointCloud()
    pcl_input_pc.points = o3d.utility.Vector3dVector(pcl_input)
    pcl_input_pc = pcl_input_pc.voxel_down_sample(voxel_size=resolution)
    pcl_input_np = np.asarray(pcl_input_pc.points)

    mat_global_image = np.zeros((h, w), dtype=np.uint8)

    for i in range(pcl_input_np.shape[0]):
        depth = np.linalg.norm(pcl_input_np[i, :3], 2)
        yaw = -np.arctan2(pcl_input_np[i, 1], pcl_input_np[i, 0])
        pitch = np.arcsin(pcl_input_np[i, 2] / depth)
        scan_r = depth * np.cos(pitch)
        w_ind = np.floor((0.5 * (yaw / np.pi + 1.0)) * w).astype(int)
        h_ind = np.floor(scan_r / measure_range * h).astype(int)
        if w_ind < 0 or w_ind >= w or h_ind < 0 or h_ind >= h:
            continue
        if mat_global_image[h_ind, w_ind] < 10:
            mat_global_image[h_ind, w_ind] += 1

    mat_global_image = mat_global_image * 10

    mat_global_image[np.where(mat_global_image > 255)] = 255
    mat_global_image = mat_global_image / np.max(mat_global_image) * 255

    return mat_global_image


def gen_bev(bev_fileroot, info_filepath, bs_whole):
    if not os.path.exists(bev_fileroot):
        os.makedirs(bev_fileroot)
    lidar_bev_root = os.path.join(bev_fileroot, 'lidar_fog_0.0163')
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

        lidar_filename = cur_info['lidar_infos']['LIDAR_TOP']['filename']
        lidar_filename = lidar_filename.rsplit('/', 1)[0] + '_fog_0.0163/' + lidar_filename.rsplit('/', 1)[1]
        lidar_pcd = load_pc(lidar_filename)

        l_bev_img = lidar_polar_bev_projection(lidar_pcd, 80.0, 0.4, 900, 200)

        filename_l = os.path.basename(cur_info['lidar_infos']['LIDAR_TOP']['filename']).split('.')[0] + '.png'
        cv2.imwrite(os.path.join(lidar_bev_root, filename_l), l_bev_img)


if __name__ == '__main__':
    dataset_root = '/home/octane17/LRFusionPR/data'

    bev_root = os.path.join(dataset_root, 'sejong_multi_bev')
    infos_path = os.path.join(dataset_root, 'mulran_infos-sejong.pkl')
    test_query_ind_root = os.path.join(dataset_root, 'sejong_test_query.npy')
    ind_test_query = np.load(test_query_ind_root)
    ind_whole = np.concatenate([ind_test_query], axis=0)

    gen_bev(bev_root, infos_path, ind_whole)
