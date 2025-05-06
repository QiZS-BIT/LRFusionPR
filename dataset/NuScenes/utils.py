import os
import numpy as np
from nuscenes.nuscenes import RadarPointCloud, LidarPointCloud


def get_location_sample_tokens(nusc, location):
    # select scenes sampled in specific locations
    location_indices = get_location_indices(nusc, location)
    sample_token_list = []
    # get sequential sample tokens
    for scene_index in location_indices:
        scene = nusc.scene[scene_index]
        sample_token = scene['first_sample_token']
        while not sample_token == '':
            sample = nusc.get('sample', sample_token)
            sample_token_list.append(sample_token)
            sample_token = sample['next']
    return sample_token_list


def get_location_indices(nusc, location):
    location_indices = []
    for scene_index in range(len(nusc.scene)):
        scene = nusc.scene[scene_index]
        sample = nusc.get('sample', scene['last_sample_token'])
        if nusc.get('log', scene['log_token'])['location'] != location:
            continue
        location_indices.append(scene_index)
    return np.array(location_indices)


def process_radar_pcd(nusc, sample, nsweep=7):
    ref_chan = 'LIDAR_TOP'
    chans = ['RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT', 'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT']
    pcl_all_ = np.zeros((0, 18))
    for chan in chans:
        pc, times = RadarPointCloud.from_file_multisweep(nusc=nusc, sample_rec=sample, nsweeps=nsweep, chan=chan, ref_chan=ref_chan)
        pt = pc.points[:17, :].transpose()  # (71, 17)
        pt = np.hstack((pt, times.transpose()))  # (71, 18)
        pcl_all_ = np.concatenate((pcl_all_, pt), axis=0)
    return pcl_all_


def process_lidar_pcd(nusc, sample, nsweep=1):
    chan = 'LIDAR_TOP'
    pc, times = LidarPointCloud.from_file_multisweep(nusc=nusc, sample_rec=sample, nsweeps=nsweep, chan=chan, ref_chan=chan)
    pcl_all_ = pc.points[:3, :].transpose()
    return pcl_all_


def load_lidar_data_fog(f_name):
    pc = np.fromfile(f_name, dtype=np.float32)
    pc = np.array(pc.reshape(-1, 5))[:, :3]
    x_filt = np.abs(pc[:, 0]) < 1.0
    y_filt = np.abs(pc[:, 1]) < 1.0
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    pc = pc[not_close, :]
    return pc


def polar_to_cartesian(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def cartesian_to_polar(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return r, theta
