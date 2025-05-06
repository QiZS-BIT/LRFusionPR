import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import random
import pickle
from nuscenes.nuscenes import NuScenes


def process_infos(nusc_trainval, nusc_test, infos_path, separate_th, ava_seq_len=5, dis_th_db=1.0, pos_th=9.0):
    with open(infos_path, 'rb') as f:
        infos = pickle.load(f)

    pos_whole = []
    timestamps = []
    available_ind = []
    euler_angles_list = []

    for i, info in enumerate(infos):
        pos = info['lidar_infos']['LIDAR_TOP']['ego_pose']['translation']
        pos_whole.append(pos[:2])
        timestamp = info['timestamp']
        timestamps.append(timestamp)

        from scipy.spatial.transform import Rotation as R
        rot = info['lidar_infos']['LIDAR_TOP']['ego_pose']['rotation']
        rotation = R.from_quat(rot)
        euler_angles = rotation.as_euler('zyx')
        euler_angles_deg = np.degrees(euler_angles)[2]
        euler_angles_list.append(euler_angles_deg)

        sample_tokens_in_scene = []
        scene_token = info['scene_token']
        try:
            scene = nusc_trainval.get('scene', scene_token)
            first_sample_token = scene['first_sample_token']
            current_sample_token = first_sample_token
            while current_sample_token != '':
                sample_tokens_in_scene.append(current_sample_token)
                current_sample = nusc_trainval.get('sample', current_sample_token)
                current_sample_token = current_sample['next']
        except:
            scene = nusc_test.get('scene', scene_token)
            first_sample_token = scene['first_sample_token']
            current_sample_token = first_sample_token
            while current_sample_token != '':
                sample_tokens_in_scene.append(current_sample_token)
                current_sample = nusc_test.get('sample', current_sample_token)
                current_sample_token = current_sample['next']
        nbr_samples = scene['nbr_samples']
        edge_frames_indices = set(range(0, ava_seq_len // 2, 1)) | set(range(nbr_samples - ava_seq_len // 2, nbr_samples, 1))
        sample_token = info['sample_token']
        sample_index = sample_tokens_in_scene.index(sample_token)
        if sample_index not in edge_frames_indices:
            available_ind.append(i)
    print("non-edge frames: ", len(available_ind))

    pos_whole = np.array(pos_whole, dtype=np.float32)
    timestamps = np.array(timestamps, dtype=np.float32).reshape(-1, 1)
    print('total frames: ', pos_whole.shape[0])

    timestamps = (timestamps - min(timestamps)) / (1e6 * 3600 * 24)  # unit: day
    fi_db_train, _ = np.where(timestamps < separate_th)
    fi_val_test, _ = np.where(timestamps >= separate_th)
    fi_db_train = set(fi_db_train) & set(available_ind)
    fi_val_test = set(fi_val_test) & set(available_ind)
    fi_db_train = np.array(list(fi_db_train))
    fi_val_test = np.array(list(fi_val_test))

    pos_whole = np.concatenate(
        (np.arange(len(pos_whole), dtype=np.int32).reshape(-1, 1), np.array(pos_whole)),
        axis=1).astype(np.float32)
    pos_db_train = pos_whole[fi_db_train]
    pos_db = pos_db_train[0, :].reshape(1, -1)  # add the first frame
    for i in range(1, pos_db_train.shape[0]):
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(pos_db[:, 1:3])
        dis, index = knn.kneighbors(pos_db_train[i, 1:3].reshape(1, -1), 1, return_distance=True)
        if dis > dis_th_db:
            pos_db = np.concatenate((pos_db, pos_db_train[i, :].reshape(1, -1)), axis=0)
    print("database frames: ", pos_db.shape[0])

    fi_db = pos_db[:, 0].astype(int)
    fi_test = fi_val_test

    pos_db = pos_whole[fi_db]
    pos_test = pos_whole[fi_test]

    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(pos_db[:, 1:3])
    pos_test_new = list()
    for i in range(len(pos_test)):
        dis, index = knn.kneighbors(pos_test[i, 1:3].reshape(1, -1), 1, return_distance=True)
        if dis < pos_th:
            pos_test_new.append(pos_test[i, :])
    pos_test = np.array(pos_test_new)
    print("test query frames: ", pos_test.shape[0])

    return pos_whole, pos_db, pos_test


def main():
    random.seed(0)
    np.random.seed(0)

    dataroot = '/home/octane17/LRFusionPR/data'
    nuscenes_root = '/media/octane17/T7ShieldNus/NuScenes'
    infos_son_path = os.path.join(dataroot, 'nuscenes_infos-son.pkl')
    infos_sq_path = os.path.join(dataroot, 'nuscenes_infos-sq.pkl')

    nuscenes_trainval = NuScenes(version='v1.0-trainval', dataroot=nuscenes_root, verbose=True)
    nuscenes_test = NuScenes(version='v1.0-test', dataroot=nuscenes_root, verbose=True)

    pos_whole_son, pos_son_db, pos_son_test_query = process_infos(nuscenes_trainval, nuscenes_test, infos_son_path, 30)
    pos_whole_sq, pos_sq_db, pos_sq_test_query = process_infos(nuscenes_trainval, nuscenes_test, infos_sq_path, 15)

    np.save(os.path.join(dataroot, 'son_whole.npy'), pos_whole_son)
    np.save(os.path.join(dataroot, 'son_db.npy'), pos_son_db)
    np.save(os.path.join(dataroot, 'son_test_query.npy'), pos_son_test_query)
    np.save(os.path.join(dataroot, 'sq_whole.npy'), pos_whole_sq)
    np.save(os.path.join(dataroot, 'sq_test_query.npy'), pos_sq_test_query)
    np.save(os.path.join(dataroot, 'sq_db.npy'), pos_sq_db)


if __name__ == '__main__':
    main()
