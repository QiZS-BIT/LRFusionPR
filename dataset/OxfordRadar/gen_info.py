import os
import numpy as np
import csv
import pickle
from dataset.OxfordRadar.utils import find_nearest_ndx


def read_synchronized_data(poses_filepath, lidar_filepath, radar_filepath, t_tolerance):
    system_timestamps = np.zeros((0,), dtype=np.int64)
    poses = np.zeros((0, 2), dtype=np.float64)
    with open(poses_filepath, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            system_timestamps = np.hstack((system_timestamps, np.array(int(row[0]))))
            poses = np.vstack((poses, np.array([float(row[8]), float(row[9])])))

    # Ensure timestamps and poses are sorted in ascending order
    sorted_ndx = np.argsort(system_timestamps, axis=0)
    system_timestamps = system_timestamps[sorted_ndx]
    poses = poses[sorted_ndx]

    # List LiDAR scan timestamps
    all_lidar_timestamps = [int(os.path.splitext(f)[0]) for f in os.listdir(lidar_filepath) if os.path.splitext(f)[1] == '.bin']
    all_lidar_timestamps.sort()

    # List Radar scan timestamps
    all_radar_timestamps = [int(os.path.splitext(f)[0]) for f in os.listdir(radar_filepath) if os.path.splitext(f)[1] == '.png']
    all_radar_timestamps.sort()

    synchronized_data_list = []
    count_rejected = 0

    for ndx, radar_ts in enumerate(all_radar_timestamps):
        synchronized_data_dict = dict()

        # Find index of the closest timestamp
        closest_pose_ts_ndx = find_nearest_ndx(radar_ts, system_timestamps)
        closest_lidar_ts_ndx = find_nearest_ndx(radar_ts, all_lidar_timestamps)
        system_ts = system_timestamps[closest_pose_ts_ndx]
        lidar_ts = all_lidar_timestamps[closest_lidar_ts_ndx]
        delta_p = abs(system_ts - radar_ts)
        delta_s = abs(lidar_ts - radar_ts)
        # Timestamp is in microseconds = 1e-6 second
        if (delta_p > t_tolerance * 1000000) | (delta_s > t_tolerance * 50000):
            # Reject point cloud without corresponding pose
            count_rejected += 1
            continue

        synchronized_data_dict['lidar_ts'] = lidar_ts
        synchronized_data_dict['radar_ts'] = radar_ts
        synchronized_data_dict['pose'] = poses[closest_pose_ts_ndx]

        synchronized_data_list.append(synchronized_data_dict)

    print(f'{len(synchronized_data_list)} samples with valid pose, {count_rejected} rejected due to unknown pose')
    return synchronized_data_list


def gen_info(sequence_name, poses_file, lidar_filepath, radar_filepath, t_tolerance):
    infos = list()
    synchronized_data_list = read_synchronized_data(poses_file, lidar_filepath, radar_filepath, t_tolerance)

    for synchronized_data_dict in synchronized_data_list:
        # store scene info
        info = dict()
        info['sequence_name'] = sequence_name
        info['timestamp'] = synchronized_data_dict['radar_ts']
        lidar_names = ['LIDAR_TOP']
        radar_names = ['RADAR_TOP']
        lidar_infos = dict()
        radar_infos = dict()

        for radar_name in radar_names:
            radar_info = dict()
            radar_info['ego_pose'] = synchronized_data_dict['pose']
            radar_info['filename'] = os.path.join(radar_filepath, str(synchronized_data_dict['radar_ts']) + '.png')
            radar_infos[radar_name] = radar_info

        for lidar_name in lidar_names:
            lidar_info = dict()
            lidar_info['ego_pose'] = synchronized_data_dict['pose']
            lidar_info['filename'] = os.path.join(lidar_filepath, str(synchronized_data_dict['lidar_ts']) + '.bin')
            lidar_infos[lidar_name] = lidar_info

        info['radar_infos'] = radar_infos
        info['lidar_infos'] = lidar_infos
        infos.append(info)

    return infos


if __name__ == '__main__':
    dataset_root = '/media/octane17/T7 Shield/OxfordRadar'
    info_root = '/home/octane17/LRFusionPR/data'
    pose_time_tolerance = 1.
    sequences = ['2019-01-11-13-24-51', '2019-01-16-14-15-33']

    all_infos = list()
    for sequence in sequences:
        sequence_path = os.path.join(dataset_root, sequence)
        assert os.path.exists(sequence_path), f'Cannot access sequence: {sequence_path}'

        pose_file = os.path.join(sequence_path, 'gps/gps.csv')
        assert os.path.exists(pose_file), f'Cannot access global pose file: {pose_file}'

        rel_lidar_path = os.path.join(sequence, 'velodyne_left')
        lidar_path = os.path.join(dataset_root, rel_lidar_path)
        assert os.path.exists(lidar_path), f'Cannot access lidar scans: {lidar_path}'

        rel_radar_path = os.path.join(sequence, 'radar')
        radar_path = os.path.join(dataset_root, rel_radar_path)
        assert os.path.exists(radar_path), f'Cannot access radar scans: {radar_path}'

        cur_infos = gen_info(sequence, pose_file, lidar_path, radar_path, pose_time_tolerance)
        all_infos.extend(cur_infos)

    with open(os.path.join(dataset_root, 'oxford_infos-2019-01-11-13-24-51.pkl'), 'wb') as f:
        pickle.dump(all_infos, f)
