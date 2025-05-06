import os
import numpy as np
import pickle
from dataset.MulRan.utils import find_nearest_ndx


FAULTY_POINTCLOUDS = [1566279795718079314]


def read_synchronized_data(poses_filepath, lidar_filepath, radar_filepath, t_tolerance):
    with open(poses_filepath, "r") as h:
        txt_poses = h.readlines()

    n = len(txt_poses)
    system_timestamps = np.zeros((n,), dtype=np.int64)
    poses = np.zeros((n, 4, 4), dtype=np.float64)       # 4x4 pose matrix

    for ndx, pose in enumerate(txt_poses):
        # Split by comma and remove whitespaces
        temp = [e.strip() for e in pose.split(',')]
        assert len(temp) == 13, f'Invalid line in global poses file: {temp}'
        system_timestamps[ndx] = int(temp[0])
        poses[ndx] = np.array([[float(temp[1]), float(temp[2]), float(temp[3]), float(temp[4])],
                               [float(temp[5]), float(temp[6]), float(temp[7]), float(temp[8])],
                               [float(temp[9]), float(temp[10]), float(temp[11]), float(temp[12])],
                               [0., 0., 0., 1.]])

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
        # Timestamp is in nanoseconds = 1e-9 second
        if (delta_p > t_tolerance * 1000000000) | (delta_s > t_tolerance * 50000000):
            # Reject point cloud without corresponding pose
            count_rejected += 1
            continue

        # Skip faulty point clouds
        if lidar_ts in FAULTY_POINTCLOUDS:
            continue

        synchronized_data_dict['lidar_ts'] = lidar_ts
        synchronized_data_dict['radar_ts'] = radar_ts
        synchronized_data_dict['pose'] = poses[closest_pose_ts_ndx]

        synchronized_data_list.append(synchronized_data_dict)

    print(f'{len(synchronized_data_list)} samples with valid pose, {count_rejected} rejected due to unknown pose')
    return synchronized_data_list


def gen_info_sequence(sequence_name, poses_file, lidar_filepath, radar_filepath, t_tolerance):
    infos = list()
    synchronized_data_list = read_synchronized_data(poses_file, lidar_filepath, radar_filepath, t_tolerance)

    for synchronized_data_dict in synchronized_data_list:
        # store scene info
        info = dict()
        info['sequence_name'] = sequence_name
        info['timestamp'] = synchronized_data_dict['radar_ts']
        info['prev_radar_files'] = [os.path.join(radar_filepath, str(iter_info['timestamp']) + '.png') for iter_info in infos[-min(len(infos), 6):]]
        info['prev_ego_poses'] = [iter_info['radar_infos']['RADAR_TOP']['ego_pose'] for iter_info in infos[-min(len(infos), 6):]]
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


def gen_info(dataset_root, sequences, pose_time_tolerance):
    all_infos = list()
    for sequence in sequences:
        sequence_path = os.path.join(dataset_root, sequence)
        assert os.path.exists(sequence_path), f'Cannot access sequence: {sequence_path}'

        pose_file = os.path.join(sequence_path, 'global_pose.csv')
        assert os.path.exists(pose_file), f'Cannot access global pose file: {pose_file}'

        rel_lidar_path = os.path.join(sequence, 'Ouster')
        lidar_path = os.path.join(dataset_root, rel_lidar_path)
        assert os.path.exists(lidar_path), f'Cannot access lidar scans: {lidar_path}'

        rel_radar_path = os.path.join(sequence, 'polar')
        radar_path = os.path.join(dataset_root, rel_radar_path)
        assert os.path.exists(radar_path), f'Cannot access radar scans: {radar_path}'

        cur_infos = gen_info_sequence(sequence, pose_file, lidar_path, radar_path, pose_time_tolerance)
        all_infos.extend(cur_infos)
    return all_infos


if __name__ == '__main__':
    data_root = '/media/octane17/T7 Shield/MulRan'
    info_root = '/home/octane17/LRFusionPR/data'
    pose_time_tolerance = 1.

    # --------------------------Sejong------------------------------
    sequences = ['Sejong01', 'Sejong02']
    all_infos = gen_info(data_root, sequences, pose_time_tolerance)
    with open(os.path.join(info_root, 'mulran_infos-sejong.pkl'), 'wb') as f:
        pickle.dump(all_infos, f)

    # -------------------------Riverside----------------------------
    sequences = ['Riverside01', 'Riverside02']
    all_infos = gen_info(data_root, sequences, pose_time_tolerance)
    with open(os.path.join(info_root, 'mulran_infos-riverside.pkl'), 'wb') as f:
        pickle.dump(all_infos, f)

    # ----------------------------DCC-------------------------------
    sequences = ['DCC01', 'DCC02']
    all_infos = gen_info(data_root, sequences, pose_time_tolerance)
    with open(os.path.join(info_root, 'mulran_infos-dcc.pkl'), 'wb') as f:
        pickle.dump(all_infos, f)
