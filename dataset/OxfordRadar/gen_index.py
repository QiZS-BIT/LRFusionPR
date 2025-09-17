import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import random
import pickle


def process_infos(infos_filepath, test_seq_name, dis_th_db=3.0, pos_th=9.0):
    with open(infos_filepath, 'rb') as f:
        infos = pickle.load(f)

    pos_whole = []
    sequence_names = []

    for i, info in enumerate(infos):
        pos = info['lidar_infos']['LIDAR_TOP']['ego_pose']
        pos_whole.append(pos)
        sequence_name = info['sequence_name']
        sequence_names.append(sequence_name)

    pos_whole = np.array(pos_whole, dtype=np.float32)
    sequence_names = np.array(sequence_names).reshape(-1, 1)
    print('total frames: ', pos_whole.shape[0])

    fi_db_train, _ = np.where(sequence_names != test_seq_name)
    fi_val_test, _ = np.where(sequence_names == test_seq_name)

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


def main():
    random.seed(0)
    np.random.seed(0)

    dataroot = '/home/octane17/LRFusionPR/data'
    infos_path = os.path.join(dataroot, 'oxford_infos-2019-01-11-13-24-51.pkl')
    test_sequence_name = '2019-01-16-14-15-33'

    pos_whole, pos_db, pos_test_query = process_infos(infos_path, test_sequence_name, 0.2, 9.0)

    np.save(os.path.join(dataroot, 'oxford_infos-2019-01-11-13-24-51_whole.npy'), pos_whole)
    np.save(os.path.join(dataroot, 'oxford_infos-2019-01-11-13-24-51_db.npy'), pos_db)
    np.save(os.path.join(dataroot, 'oxford_infos-2019-01-11-13-24-51_test_query.npy'), pos_test_query)


if __name__ == '__main__':
    main()
