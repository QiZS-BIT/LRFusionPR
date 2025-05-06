import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import random
import pickle
from dataset.MulRan.utils import in_train_split, in_test_split
from tqdm import tqdm


def process_infos_sejong(infos_filepath, test_seq_name, pos_th=5.0):
    with open(infos_filepath, 'rb') as f:
        infos = pickle.load(f)

    pos_whole = []
    sequence_names = []

    for i, info in enumerate(infos):
        pos = info['lidar_infos']['LIDAR_TOP']['ego_pose']
        pos_whole.append(np.array([pos[0, 3], pos[1, 3]]))
        sequence_name = info['sequence_name']
        sequence_names.append(int(sequence_name[-2:]))

    pos_whole = np.array(pos_whole, dtype=np.float32)
    train_split_flags = in_train_split(pos_whole)
    test_split_flags = in_test_split(pos_whole)
    train_split_flags = train_split_flags[:, np.newaxis]
    test_split_flags = test_split_flags[:, np.newaxis]
    sequence_names = np.array(sequence_names, dtype=np.int32).reshape(-1, 1)
    print('total frames: ', pos_whole.shape[0])

    fi_db_train, _ = np.where((train_split_flags == 1) & (sequence_names != int(test_seq_name[-2:])))
    fi_query_train, _ = np.where((train_split_flags == 1) & (sequence_names == int(test_seq_name[-2:])))

    pos_whole = np.concatenate(
        (np.arange(len(pos_whole), dtype=np.int32).reshape(-1, 1), np.array(pos_whole)),
        axis=1).astype(np.float32)
    pos_db_train = pos_whole[fi_db_train]
    print("database frames: ", pos_db_train.shape[0])

    pos_query_train = pos_whole[fi_query_train]
    print("train query frames: ", pos_query_train.shape[0])

    fi_test_db, _ = np.where((test_split_flags == 1) & (sequence_names != int(test_seq_name[-2:])))
    fi_test_query, _ = np.where((test_split_flags == 1) & (sequence_names == int(test_seq_name[-2:])))
    pos_test_db = pos_whole[fi_test_db]
    pos_test_query = pos_whole[fi_test_query]
    print("test database frames: ", pos_test_db.shape[0])

    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(pos_test_db[:, 1:3])
    pos_test_query_new = list()
    for i in tqdm(range(len(pos_test_query))):
        dis, index = knn.kneighbors(pos_test_query[i, 1:3].reshape(1, -1), 1, return_distance=True)
        if dis < pos_th:
            pos_test_query_new.append(pos_test_query[i, :])
    pos_test_query = np.array(pos_test_query_new)
    print("test query frames: ", pos_test_query.shape[0])

    return pos_whole, pos_db_train, pos_query_train, pos_test_db, pos_test_query


def process_infos_tests(infos_filepath, test_seq_name, dis_th_db=1.0, pos_th=5.0):
    with open(infos_filepath, 'rb') as f:
        infos = pickle.load(f)

    pos_whole = []
    sequence_names = []

    for i, info in enumerate(infos):
        pos = info['lidar_infos']['LIDAR_TOP']['ego_pose']
        pos_whole.append(np.array([pos[0, 3], pos[1, 3]]))
        sequence_name = info['sequence_name']
        sequence_names.append(int(sequence_name[-2:]))

    pos_whole = np.array(pos_whole, dtype=np.float32)
    sequence_names = np.array(sequence_names, dtype=np.int32).reshape(-1, 1)
    print('total frames: ', pos_whole.shape[0])

    fi_db_train, _ = np.where(sequence_names != int(test_seq_name[-2:]))
    fi_val_test, _ = np.where(sequence_names == int(test_seq_name[-2:]))

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

    pos_val_test = pos_whole[fi_val_test]
    pos_val_test_dsp = pos_val_test[0, :].reshape(1, -1)  # add the first frame
    for i in range(1, pos_val_test.shape[0]):
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(pos_val_test_dsp[:, 1:3])
        dis, index = knn.kneighbors(pos_val_test[i, 1:3].reshape(1, -1), 1, return_distance=True)
        if dis > dis_th_db:
            pos_val_test_dsp = np.concatenate((pos_val_test_dsp, pos_val_test[i, :].reshape(1, -1)), axis=0)
    pos_val_test = pos_val_test_dsp

    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(pos_db[:, 1:3])
    pos_val_test_new = list()
    for i in range(len(pos_val_test)):
        dis, index = knn.kneighbors(pos_val_test[i, 1:3].reshape(1, -1), 1, return_distance=True)
        if dis < pos_th:
            pos_val_test_new.append(pos_val_test[i, :])
    pos_val_test = np.array(pos_val_test_new)
    print("test query frames: ", pos_val_test.shape[0])

    return pos_whole, pos_db, pos_val_test


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    dataroot = '/home/octane17/LRFusionPR/data'

    # --------------------------Sejong------------------------------
    infos_path = os.path.join(dataroot, 'mulran_infos-sejong.pkl')
    test_sequence_name = 'Sejong02'

    pos_whole, pos_db, pos_train_query, pos_test_db, pos_test_query = process_infos_sejong(infos_path, test_sequence_name, 9.0)

    np.save(os.path.join(dataroot, 'sejong_whole.npy'), pos_whole)
    np.save(os.path.join(dataroot, 'sejong_db.npy'), pos_db)
    np.save(os.path.join(dataroot, 'sejong_train_query.npy'), pos_train_query)
    np.save(os.path.join(dataroot, 'sejong_test_db.npy'), pos_test_db)
    np.save(os.path.join(dataroot, 'sejong_test_query.npy'), pos_test_query)

    # -------------------------Riverside-----------------------------
    infos_path = os.path.join(dataroot, 'mulran_infos-riverside.pkl')
    test_sequence_name = 'Riverside02'

    pos_whole, pos_db, pos_query = process_infos_tests(infos_path, test_sequence_name, 1.0, 9.0)

    np.save(os.path.join(dataroot, 'riverside_whole.npy'), pos_whole)
    np.save(os.path.join(dataroot, 'riverside_test_db.npy'), pos_db)
    np.save(os.path.join(dataroot, 'riverside_test_query.npy'), pos_query)

    # ----------------------------DCC--------------------------------
    infos_path = os.path.join(dataroot, 'mulran_infos-dcc.pkl')
    test_sequence_name = 'DCC02'

    pos_whole, pos_db, pos_query = process_infos_tests(infos_path, test_sequence_name, 10.0, 5.0)

    np.save(os.path.join(dataroot, 'dcc_whole.npy'), pos_whole)
    np.save(os.path.join(dataroot, 'dcc_test_db.npy'), pos_db)
    np.save(os.path.join(dataroot, 'dcc_test_query.npy'), pos_query)
