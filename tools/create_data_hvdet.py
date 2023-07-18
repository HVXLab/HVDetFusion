# Copyright (c) OpenMMLab. All rights reserved.
import pickle
import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from tools.data_converter import nuscenes_converter as nuscenes_converter
from tools.radar_process.utils_radar import RadarPointCloudWithVelocity as RadarPointCloud

map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}
classes = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

def get_gt(info):  # 使用前置摄像头的ego作为gt——box的ego转换矩阵
    """Generate gt labels from info.

    Args:
        info(dict): Infos needed to generate gt labels.

    Returns:
        Tensor: GT bboxes.
        Tensor: GT labels.
    """
    ego2global_rotation = info['cams']['CAM_FRONT']['ego2global_rotation'] # 来自tools/data_converter/nuscenes_converter.py 中_fill_trainval_infos [line 179, line 263]
    ego2global_translation = info['cams']['CAM_FRONT'][
        'ego2global_translation']
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    gt_boxes = list()
    gt_labels = list()
    for ann_info in info['ann_infos']:
        # Use ego coordinate.
        if (map_name_from_general_to_detection[ann_info['category_name']]
                not in classes
                or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <= 0):
            continue
        box = Box(
            ann_info['translation'],
            ann_info['size'],
            Quaternion(ann_info['rotation']),
            velocity=ann_info['velocity'],
        )
        box.translate(trans)  # 转到CAM_FRONT时刻的ego坐标系下
        box.rotate(rot)  # 转到CAM_FRONT时刻的ego坐标系下
        box_xyz = np.array(box.center)
        box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
        box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
        box_velo = np.array(box.velocity[:2])
        gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
        gt_boxes.append(gt_box)
        gt_labels.append(
            classes.index(
                map_name_from_general_to_detection[ann_info['category_name']]))
    return gt_boxes, gt_labels  # 此时gt-boxes由原来的lidar数据中直接读取的替换成sample——annotation中的标注信息，并转到front cam ego坐标系下

def nuscenes_data_prep(root_path, info_prefix, version, max_sweeps=10):
    """Prepare data related to nuScenes dataset.
    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        max_sweeps (int, optional): Number of input consecutive frames. 连续帧的数量
            Default: 10
    """
    nuscenes_converter.create_nuscenes_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)

def add_ann_adj_info(extra_tag):
    nuscenes_version = 'v1.0-trainval'
    dataroot = './data/nuscenes/'
    nuscenes = NuScenes(nuscenes_version, dataroot)
    for set in ['train', 'val']:
        dataset = pickle.load(
            open('./data/nuscenes/%s_infos_%s.pkl' % (extra_tag, set), 'rb'))
        for id in range(len(dataset['infos'])):
            if id % 10 == 0:
                print('%d/%d' % (id, len(dataset['infos'])))
            info = dataset['infos'][id]

            # get sweep adjacent frame info
            sample = nuscenes.get('sample', info['token'])
            ann_infos = list()
            for ann in sample['anns']:
                ann_info = nuscenes.get('sample_annotation', ann)
                velocity = nuscenes.box_velocity(ann_info['token'])
                if np.any(np.isnan(velocity)):
                    velocity = np.zeros(3)
                ann_info['velocity'] = velocity
                ann_infos.append(ann_info)
            dataset['infos'][id]['ann_infos'] = ann_infos
            dataset['infos'][id]['ann_infos'] = get_gt(dataset['infos'][id])
            dataset['infos'][id]['scene_token'] = sample['scene_token']
        with open('./data/nuscenes/%s_infos_%s.pkl' % (extra_tag, set),
                  'wb') as fid:
            pickle.dump(dataset, fid)

def add_radar_info(extra_tag, sets=['train', 'val'], nuscenes_version='v1.0-trainval', dataroot='./data/nuscenes/'):
    '''
    Record radar datasets into pkl files，saved with RadarPointCloud class
    Args:
        extra_tag: tag，default: bevdetv2-nuscenes
        sets: [train, val] or [test],
        nuscenes_version: default: 'v1.0-trainval'
        dataroot: root path.
    Returns:
    '''
    USED_SENSOR = ["RADAR_FRONT_LEFT", "RADAR_FRONT", "RADAR_FRONT_RIGHT",
                   "RADAR_BACK_RIGHT","RADAR_BACK_LEFT"]
    # USED_SENSOR = ['LIDAR_TOP']
    nuscenes = NuScenes(nuscenes_version, dataroot)
    for set in sets:
        dataset = pickle.load(
            open('./data/nuscenes/%s_infos_%s.pkl' % (extra_tag, set), 'rb'))
        for id in range(len(dataset['infos'])):
            if id % 10 == 0:
                print('%d/%d' % (id, len(dataset['infos'])))
            info = dataset['infos'][id]

            # get sweep adjacent frame info
            sample = nuscenes.get('sample', info['token'])
            all_radar_pcs = RadarPointCloud(np.zeros((18, 0)))
            for sensor_name in sample['data']:
                if sensor_name in USED_SENSOR:
                    radar_pcs, _ = RadarPointCloud.from_file_multisweep(nuscenes,
                                        sample, sensor_name, ref_chan='CAM_FRONT', nsweeps=1)
                    all_radar_pcs.points = np.hstack((all_radar_pcs.points, radar_pcs.points))
            dataset['infos'][id]['radar_points'] = all_radar_pcs.points.tolist()
        with open('./data/nuscenes/%s_infos_%s.pkl' % (extra_tag, set),
                  'wb') as fid:
            pickle.dump(dataset, fid)


if __name__ == '__main__':
    dataset = 'nuscenes'
    version = 'v1.0'
    train_version = f'{version}-trainval'
    root_path = './data/nuscenes'
    extra_tag = 'bevdetv2-nuscenes'
    nuscenes_data_prep(
        root_path=root_path,
        info_prefix=extra_tag,
        version=train_version,
        max_sweeps=0)
    print('add_ann_infos')
    add_ann_adj_info(extra_tag)
    add_radar_info(extra_tag, dataroot=root_path)
