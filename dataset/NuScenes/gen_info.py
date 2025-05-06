import os
import pickle
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from utils import get_location_sample_tokens


def gen_info(nusc, sample_tokens, is_trainval):
    infos = list()
    for sample_token in tqdm(sample_tokens):
        # each info corresponds to a sample
        sample = nusc.get('sample', sample_token)
        # store scene info
        info = dict()
        info['sample_token'] = sample_token
        info['is_trainval'] = is_trainval
        info['prev'] = sample['prev']
        info['next'] = sample['next']
        info['timestamp'] = sample['timestamp']
        info['scene_token'] = sample['scene_token']
        lidar_names = ['LIDAR_TOP']
        radar_names = ['RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT', 'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT']
        lidar_infos = dict()
        radar_infos = dict()

        for radar_name in radar_names:
            radar_data = nusc.get('sample_data', sample['data'][radar_name])
            radar_info = dict()
            radar_info['sample_token'] = radar_data['sample_token']
            radar_info['ego_pose'] = nusc.get('ego_pose', radar_data['ego_pose_token'])
            radar_info['filename'] = radar_data['filename']
            radar_info['calibrated_sensor'] = nusc.get('calibrated_sensor', radar_data['calibrated_sensor_token'])
            radar_infos[radar_name] = radar_info

        for lidar_name in lidar_names:
            lidar_data = nusc.get('sample_data', sample['data'][lidar_name])
            lidar_info = dict()
            lidar_info['sample_token'] = lidar_data['sample_token']
            lidar_info['ego_pose'] = nusc.get('ego_pose', lidar_data['ego_pose_token'])
            lidar_info['filename'] = lidar_data['filename']
            lidar_info['calibrated_sensor'] = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
            lidar_infos[lidar_name] = lidar_info

        info['radar_infos'] = radar_infos
        info['lidar_infos'] = lidar_infos
        infos.append(info)

    return infos


if __name__ == '__main__':
    nuscenes_root = '/media/octane17/T7ShieldNus/NuScenes'
    dataroot = '/home/octane17/LRFusionPR/data'
    nusc_trainval = NuScenes(version='v1.0-trainval', dataroot=nuscenes_root, verbose=True)
    nusc_test = NuScenes(version='v1.0-test', dataroot=nuscenes_root, verbose=True)

    # ====================generate infos====================
    sample_tokens_trainval = get_location_sample_tokens(nusc_trainval, location='boston-seaport')
    sample_tokens_test = get_location_sample_tokens(nusc_test, location='boston-seaport')
    infos = gen_info(nusc_trainval, sample_tokens_trainval, is_trainval=True)
    infos.extend(gen_info(nusc_test, sample_tokens_test, is_trainval=False))
    with open(os.path.join(dataroot, 'nuscenes_infos-bs.pkl'), 'wb') as f:
        pickle.dump(infos, f)

    sample_tokens_trainval = get_location_sample_tokens(nusc_trainval, location='singapore-onenorth')
    sample_tokens_test = get_location_sample_tokens(nusc_test, location='singapore-onenorth')
    infos = gen_info(nusc_trainval, sample_tokens_trainval, is_trainval=True)
    infos.extend(gen_info(nusc_test, sample_tokens_test, is_trainval=False))
    with open(os.path.join(dataroot, 'nuscenes_infos-son.pkl'), 'wb') as f:
        pickle.dump(infos, f)

    sample_tokens_trainval = get_location_sample_tokens(nusc_trainval, location='singapore-queenstown')
    sample_tokens_test = get_location_sample_tokens(nusc_test, location='singapore-queenstown')
    infos = gen_info(nusc_trainval, sample_tokens_trainval, is_trainval=True)
    infos.extend(gen_info(nusc_test, sample_tokens_test, is_trainval=False))
    with open(os.path.join(dataroot, 'nuscenes_infos-sq.pkl'), 'wb') as f:
        pickle.dump(infos, f)
