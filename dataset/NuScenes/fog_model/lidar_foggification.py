import argparse
import os
import time
from nuscenes.nuscenes import NuScenes
import pickle
from dataset.NuScenes.utils import get_location_indices
import numpy as np
from tqdm import tqdm
from beta_modification import BetaRadomization


dmax_list = []
point_max_list= []
beta_usefull_list = []


def load_lidar_data(base_dataset_path, f_name):
    fname = os.path.join(base_dataset_path, f_name)
    pc = np.frombuffer(open(fname, "rb").read(), dtype=np.float32)
    pc = np.array(pc.reshape(-1, 5))[:, 0:4]
    return pc


def haze_point_cloud(pts_3D, Radomized_beta, fraction_random):
    n = 0.015
    g = 0.45
    dmin = 2

    d = np.sqrt(pts_3D[:,0] * pts_3D[:,0] + pts_3D[:,1] * pts_3D[:,1] + pts_3D[:,2] * pts_3D[:,2])
    detectable_points = np.where(d>dmin)
    d = d[detectable_points]
    pts_3D = pts_3D[detectable_points]
    point_max_list.append(np.max(np.linalg.norm(pts_3D[:, :3], axis=-1)))

    beta_usefull = Radomized_beta.get_beta(pts_3D[:,0], pts_3D[:, 1], pts_3D[:, 2])
    dmax = -np.divide(np.log(np.divide(n,(pts_3D[:,3] + g))),(2 * beta_usefull))
    beta_usefull_list.append(np.mean(beta_usefull))
    dmax_list.append(np.mean(dmax))
    dnew = -np.log(1 - 0.5) / (beta_usefull)

    probability_lost = 1 - np.exp(-beta_usefull*dmax)
    lost = np.random.uniform(0, 1, size=probability_lost.shape) < probability_lost

    if Radomized_beta.beta == 0.0:
        dist_pts_3d = np.zeros((pts_3D.shape[0], 5))
        dist_pts_3d[:, 0:4] = pts_3D
        dist_pts_3d[:, 4] = np.zeros(np.shape(pts_3D[:, 3]))
        return dist_pts_3d,  []

    cloud_scatter = np.logical_and(dnew < d, np.logical_not(lost))
    random_scatter = np.logical_and(np.logical_not(cloud_scatter), np.logical_not(lost))
    idx_stable = np.where(d<dmax)[0]
    old_points = np.zeros((len(idx_stable), 5))
    old_points[:,0:4] = pts_3D[idx_stable,:]
    old_points[:,3] = old_points[:,3]*np.exp(-beta_usefull[idx_stable]*d[idx_stable])
    old_points[:, 4] = np.zeros(np.shape(old_points[:,3]))

    cloud_scatter_idx = np.where(np.logical_and(dmax<d, cloud_scatter))[0]
    cloud_scatter = np.zeros((len(cloud_scatter_idx), 5))
    cloud_scatter[:,0:4] =  pts_3D[cloud_scatter_idx,:]
    cloud_scatter[:,0:3] = np.transpose(np.multiply(np.transpose(cloud_scatter[:,0:3]), np.transpose(np.divide(dnew[cloud_scatter_idx],d[cloud_scatter_idx]))))
    cloud_scatter[:,3] = cloud_scatter[:,3]*np.exp(-beta_usefull[cloud_scatter_idx]*dnew[cloud_scatter_idx])
    cloud_scatter[:, 4] = np.ones(np.shape(cloud_scatter[:, 3]))

    # Subsample random scatter abhaengig vom noise im Lidar
    random_scatter_idx = np.where(random_scatter)[0]
    scatter_max = np.min(np.vstack((dmax, d)).transpose(), axis=1)
    drand = np.random.uniform(high=scatter_max[random_scatter_idx])
    # scatter outside min detection range and do some subsampling. Not all points are randomly scattered.
    # Fraction of 0.05 is found empirically.
    drand_idx = np.where(drand>dmin)
    drand = drand[drand_idx]
    random_scatter_idx = random_scatter_idx[drand_idx]
    # Subsample random scattered points to 0.05%
    print(len(random_scatter_idx), fraction_random)
    subsampled_idx = np.random.choice(len(random_scatter_idx), int(fraction_random*len(random_scatter_idx)), replace=False)
    drand = drand[subsampled_idx]
    random_scatter_idx = random_scatter_idx[subsampled_idx]

    random_scatter = np.zeros((len(random_scatter_idx), 5))
    random_scatter[:,0:4] = pts_3D[random_scatter_idx,:]
    random_scatter[:,0:3] = np.transpose(np.multiply(np.transpose(random_scatter[:,0:3]), np.transpose(drand/d[random_scatter_idx])))
    random_scatter[:,3] = random_scatter[:,3]*np.exp(-beta_usefull[random_scatter_idx]*drand)
    random_scatter[:, 4] = 2*np.ones(np.shape(random_scatter[:, 3]))

    dist_pts_3d = np.concatenate((old_points, cloud_scatter,random_scatter), axis=0)

    color = []
    return dist_pts_3d, color


def initialize_window():
    w= None
    return w


def add_random_noise(velodyne_scan):
    random_noise = np.random.normal(0.0, 5, np.shape(velodyne_scan))
    velodyne_scan = velodyne_scan + random_noise
    return  velodyne_scan


def set_color(dist_pts_3d):
    color = []
    for pts in dist_pts_3d:
        if pts[4] == 0:
            color.append([0, 255, 255, 1])
        else:
            color.append([pts[3],0, 0, 1])

    return np.asarray(color)


def main(nusc_tv, nusc_test, selected_tv_ind, selected_test_ind, all_query_filename,
         scene_ind_tv, scene_ind_test, nusc_root, target_root, beta):
    for ind in tqdm(range(len(scene_ind_tv))):
        if ind not in selected_tv_ind:
            continue
        cur_scene = nusc_tv.scene[scene_ind_tv[ind]]
        cur_sample_token = cur_scene['first_sample_token']
        B = BetaRadomization(beta)
        B.propagate_in_time(10)
        while not cur_sample_token == '':
            cur_sample = nusc_tv.get('sample', cur_sample_token)
            cur_lidar_data = nusc_tv.get('sample_data', cur_sample['data']['LIDAR_TOP'])
            cur_lidar_filename = cur_lidar_data['filename']
            # ---------------foggification start----------------
            velodyne_scan = load_lidar_data(nusc_root, cur_lidar_filename)
            velodyne_scan[:, 3] = velodyne_scan[:, 3] / 255
            dist_pts_3d, color = haze_point_cloud(velodyne_scan, B, 0.05)
            B.propagate_in_time(5)
            if cur_lidar_filename in all_query_filename:
                save_path_velo = os.path.join(target_root, os.path.basename(cur_lidar_filename))
                dist_pts_3d.astype(np.float32).tofile(save_path_velo)
            # ---------------foggification end----------------
            cur_sample_token = cur_sample['next']
    for ind in tqdm(range(len(scene_ind_test))):
        if ind not in selected_test_ind:
            continue
        cur_scene = nusc_test.scene[scene_ind_test[ind]]
        cur_sample_token = cur_scene['first_sample_token']
        B = BetaRadomization(beta)
        B.propagate_in_time(10)
        while not cur_sample_token == '':
            cur_sample = nusc_test.get('sample', cur_sample_token)
            cur_lidar_data = nusc_test.get('sample_data', cur_sample['data']['LIDAR_TOP'])
            cur_lidar_filename = cur_lidar_data['filename']
            # ---------------foggification start----------------
            velodyne_scan = load_lidar_data(nusc_root, cur_lidar_filename)
            velodyne_scan[:, 3] = velodyne_scan[:, 3] / 255
            dist_pts_3d, color = haze_point_cloud(velodyne_scan, B, 0.05)
            B.propagate_in_time(5)
            if cur_lidar_filename in all_query_filename:
                save_path_velo = os.path.join(target_root, os.path.basename(cur_lidar_filename))
                dist_pts_3d.astype(np.float32).tofile(save_path_velo)
            # ---------------foggification end----------------
            cur_sample_token = cur_sample['next']


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)

    nuscenes_root = '/media/octane17/T7ShieldNus/NuScenes'
    target_root = '/media/octane17/T7ShieldNus/NuScenes/samples/LIDAR_TOP_fog_0.0128'
    info_filename = "/home/octane17/LR-Net/data/nuscenes_infos-sq.pkl"
    query_index_filename = "/home/octane17/LR-Net/data/sq_test_query.npy"
    beta = 0.0128

    nusc_trainval = NuScenes(version='v1.0-trainval', dataroot=nuscenes_root, verbose=True)
    nusc_test = NuScenes(version='v1.0-test', dataroot=nuscenes_root, verbose=True)
    query_index_n_pos = np.load(query_index_filename)
    with open(info_filename, 'rb') as f:
        infos = pickle.load(f)

    all_query_lidar_filename = []
    for i in range(len(query_index_n_pos)):
        cur_inf = infos[int(query_index_n_pos[i, 0])]
        all_query_lidar_filename.append(cur_inf['lidar_infos']['LIDAR_TOP']['filename'])

    scene_indices_trainval = get_location_indices(nusc_trainval, location='singapore-queenstown')
    scene_indices_test = get_location_indices(nusc_test, location='singapore-queenstown')

    selected_trainval_scene_ind = []
    for i in range(len(scene_indices_trainval)):
        scene = nusc_trainval.scene[scene_indices_trainval[i]]
        sample_token = scene['first_sample_token']
        while not sample_token == '':
            sample = nusc_trainval.get('sample', sample_token)
            lidar_data = nusc_trainval.get('sample_data', sample['data']['LIDAR_TOP'])
            lidar_filename = lidar_data['filename']
            sample_token = sample['next']
            if lidar_filename in all_query_lidar_filename:
                selected_trainval_scene_ind.append(i)
                break
    selected_test_scene_ind = []
    for i in range(len(scene_indices_test)):
        scene = nusc_test.scene[scene_indices_test[i]]
        sample_token = scene['first_sample_token']
        while not sample_token == '':
            sample = nusc_test.get('sample', sample_token)
            lidar_data = nusc_test.get('sample_data', sample['data']['LIDAR_TOP'])
            lidar_filename = lidar_data['filename']
            sample_token = sample['next']
            if lidar_filename in all_query_lidar_filename:
                selected_test_scene_ind.append(i)
                break

    main(nusc_trainval, nusc_test, selected_trainval_scene_ind, selected_test_scene_ind, all_query_lidar_filename,
         scene_indices_trainval, scene_indices_test, nuscenes_root, target_root, beta)
    print(np.mean(np.array(dmax_list)), np.mean(np.array(beta_usefull_list)), np.mean(np.array(point_max_list)))
