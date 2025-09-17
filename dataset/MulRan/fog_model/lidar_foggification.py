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
point_max_list = []
beta_usefull_list = []


def load_lidar_data(file_pathname):
    # Load point cloud, clip x, y and z coords (points far away and the ground plane)
    # Returns Nx3 matrix
    pc = np.fromfile(file_pathname, dtype=np.float32)
    # PC in Mulran is of size [num_points, 4] -> x,y,z,reflectance
    pc = np.reshape(pc, (-1, 4))
    return pc


def haze_point_cloud(pts_3D, Radomized_beta, fraction_random):
    n = 0.02
    g = 0.45
    dmin = 2 # Minimal detectable distance

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

    dist_pts_3d = np.concatenate((old_points, cloud_scatter, random_scatter), axis=0)

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


def main(dataset_fileroot, target_fileroot, query_filename, all_filename, beta):
    B = BetaRadomization(beta)
    B.propagate_in_time(10)
    for lidar_filename in query_filename:
        cur_lidar_filename = os.path.join(dataset_fileroot, lidar_filename)
        print(cur_lidar_filename)
        # ---------------foggification start----------------
        velodyne_scan = load_lidar_data(cur_lidar_filename)
        velodyne_scan[:, 3] = velodyne_scan[:, 3] / 255
        dist_pts_3d, color = haze_point_cloud(velodyne_scan, B, 0.05)

        # import open3d
        # point_cloud0 = open3d.geometry.PointCloud()
        # point_cloud0.points = open3d.utility.Vector3dVector(velodyne_scan[:, :3])
        # point_cloud0.paint_uniform_color([1, 0, 0])
        # point_cloud1 = open3d.geometry.PointCloud()
        # point_cloud1.points = open3d.utility.Vector3dVector(dist_pts_3d[:, :3])
        # point_cloud1.paint_uniform_color([0, 1, 0])
        # vis = open3d.visualization.Visualizer()
        # vis.create_window()
        # opt = vis.get_render_option()
        # opt.point_size = 3
        # vis.add_geometry(point_cloud0)
        # vis.add_geometry(point_cloud1)
        # vis.run()
        # vis.destroy_window()

        B.propagate_in_time(5)
        if lidar_filename in query_filename:
            save_path_velo = os.path.join(target_fileroot, os.path.basename(lidar_filename))
            dist_pts_3d[:, :4].astype(np.float32).tofile(save_path_velo)


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)

    mulran_root = '/home/octane17/LRFusionPR/data'
    dataset_root = '/media/octane17/T7 Shield/MulRan/Sejong02/Ouster'
    target_root = '/media/octane17/T7 Shield/MulRan/Sejong02/Ouster_fog_0.0163'
    test_query_ind_root = os.path.join(mulran_root, 'sejong_test_query.npy')
    infos_path = os.path.join(mulran_root, 'mulran_infos-sejong.pkl')
    beta = 0.0163

    with open(infos_path, 'rb') as f:
        infos = pickle.load(f)
    query_index_n_pos = np.load(test_query_ind_root)

    query_lidar_filename = []
    for i in range(len(query_index_n_pos)):
        cur_inf = infos[int(query_index_n_pos[i, 0])]
        query_lidar_filename.append(os.path.basename(cur_inf['lidar_infos']['LIDAR_TOP']['filename']))

    all_lidar_filenames = sorted(os.listdir(dataset_root))

    main(dataset_root, target_root, query_lidar_filename, all_lidar_filenames, beta)
