import copy
import cv2
import numpy as np
import os.path as osp
from functools import reduce
from pyquaternion import Quaternion
from typing import Tuple, List, Dict
from nuscenes.utils.data_classes import RadarPointCloud, LidarSegPointCloud
from nuscenes.utils.geometry_utils import view_points, transform_matrix
import os
from tools.radar_process.utils import _topk, _tranpose_and_gather_feat
import torch
import math
import logging as l
import time

class RadarPoints(object):
    def __init__(self, points_matrix):
        self.points = points_matrix

    def get_pillars(self):
        print('get pillars')


def get_radar_multisweep(nusc: 'NuScenes',
                         sample_rec: Dict,
                         chan: str,
                         ref_chan: str,
                         nsweeps: int = 5,
                         min_distance: float = 1.0) -> Tuple['PointCloud', np.ndarray]:
    """
    Return a point cloud that aggregates multiple sweeps.
    As every sweep is in a different coordinate frame, we need to map the coordinates to a single reference frame.
    As every sweep has a different timestamp, we need to account for that in the transformations and timestamps.
    :param nusc: A NuScenes instance.
    :param sample_rec: The current sample.
    :param chan: The lidar/radar channel from which we track back n sweeps to aggregate the point cloud.
    :param ref_chan: The reference channel of the current sample_rec that the point clouds are mapped to.
    :param nsweeps: Number of sweeps to aggregated.
    :param min_distance: Distance below which points are discarded.
    :return: (all_pc, all_times). The aggregated point cloud and timestamps.
    """
    # Init.
    all_pc = RadarPointCloud(np.zeros((18, 0)))
    all_times = np.zeros((1, 0))

    # Get reference pose and timestamp.
    ref_sd_token = sample_rec['data'][ref_chan]
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_time = 1e-6 * ref_sd_rec['timestamp']

    # Homogeneous transform from ego car frame to reference frame.
    ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)

    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                        inverse=True)

    # Aggregate current and previous sweeps.
    sample_data_token = sample_rec['data'][chan]
    current_sd_rec = nusc.get('sample_data', sample_data_token)
    for _ in range(nsweeps):
        # Load up the pointcloud and remove points close to the sensor.
        current_pc = RadarPointCloud.from_file(osp.join(nusc.dataroot, current_sd_rec['filename']))
        current_pc.remove_close(min_distance)

        # Get past pose.
        current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                            Quaternion(current_pose_rec['rotation']), inverse=False)

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                            inverse=False)

        # Fuse four transformation matrices into one and perform transform.
        trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
        current_pc.transform(trans_matrix)

        # Add time vector which can be used as a temporal feature.
        time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  # Positive difference.
        times = time_lag * np.ones((1, current_pc.nbr_points()))
        all_times = np.hstack((all_times, times))

        # Merge with key pc.
        all_pc.points = np.hstack((all_pc.points, current_pc.points))

        # Abort if there are no previous sweeps.
        if current_sd_rec['prev'] == '':
            break
        else:
            current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

    return all_pc, all_times


class RadarPointCloudWithVelocity(RadarPointCloud):

    @classmethod
    def rotate_velocity(cls, pointcloud, transform_matrix):
        n_points = pointcloud.shape[1]
        third_dim = np.zeros(n_points)
        pc_velocity = np.vstack((pointcloud[[8, 9], :], third_dim, np.ones(n_points)))
        pc_velocity = transform_matrix.dot(pc_velocity)

        ## in camera coordinates, x is right, z is front
        # pointcloud[[8, 9], :] = pc_velocity[[0, 2], :]
        pointcloud[[8, 9], :] = pc_velocity[[0, 1], :]

        return pointcloud

    @classmethod
    def from_file_multisweep(cls,
                             nusc: 'NuScenes',
                             sample_rec: Dict,
                             chan: str,
                             ref_chan: str='CAM_FRONT',
                             nsweeps: int = 5,
                             min_distance: float = 1.0) -> Tuple['PointCloud', np.ndarray]:
        """
        Return a point cloud that aggregates multiple sweeps.
        As every sweep is in a different coordinate frame, we need to map the coordinates to a single reference frame.
        As every sweep has a different timestamp, we need to account for that in the transformations and timestamps.
        :param nusc: A NuScenes instance.
        :param sample_rec: The current sample.
        :param chan: The lidar/radar channel from which we track back n sweeps to aggregate the point cloud.
        :param ref_chan: The reference channel of the current sample_rec that the point clouds are mapped to.
        :param nsweeps: Number of sweeps to aggregated.
        :param min_distance: Distance below which points are discarded.
        :return: (all_pc, all_times). The aggregated point cloud and timestamps.
      
        """
        # Init.
        points = np.zeros((cls.nbr_dims(), 0))
        all_pc = cls(points)
        all_radar_annos = list()
        all_times = np.zeros((1, 0))
        # Get reference pose and timestamp.
     
        ref_sd_token = sample_rec['data'][ref_chan]  # sample_rec 
        ref_sd_rec = nusc.get('sample_data', ref_sd_token)  # 
        ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])  # 

        ref_time = 1e-6 * ref_sd_rec['timestamp']

        # Homogeneous transform from ego car frame to reference frame.
        ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec[
            'calibrated_sensor_token'])  # calibrated_sensor：
        ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']),
                                        inverse=True) 
        ref_from_car_rot = transform_matrix([0.0, 0.0, 0.0], Quaternion(ref_cs_rec['rotation']), inverse=True)

        # Homogeneous transformation matrix from global to _current_ ego car frame.
        car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                           inverse=True)
        car_from_global_rot = transform_matrix([0.0, 0.0, 0.0], Quaternion(ref_pose_rec['rotation']),
                                               inverse=True)

        sample_data_token = sample_rec['data'][chan]  
        current_sd_rec = nusc.get('sample_data', sample_data_token)
        for _ in range(nsweeps):
            # Load up the pointcloud and remove points close to the sensor.
            # print(os.path.exists(osp.join(nusc.dataroot, current_sd_rec['filename'])))
            path = osp.join(nusc.dataroot, current_sd_rec['filename'])
            if not os.path.exists(path):
                print('invalid path:', path)
                continue
            current_pc = cls.from_file(path)  # 
        
            # Get past pose.
            current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])  # 
            global_from_car = transform_matrix(current_pose_rec['translation'],
                                               Quaternion(current_pose_rec['rotation']),
                                               inverse=False)  # 
            global_from_car_rot = transform_matrix([0.0, 0.0, 0.0],
                                                   Quaternion(current_pose_rec['rotation']), inverse=False)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = nusc.get('calibrated_sensor',
                                      current_sd_rec['calibrated_sensor_token'])  
            car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                inverse=False)
            car_from_current_rot = transform_matrix([0.0, 0.0, 0.0], Quaternion(current_cs_rec['rotation']),
                                                    inverse=False)

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(np.dot, [global_from_car,  #ref_from_car, car_from_global
                                           car_from_current])  
            velocity_trans_matrix = reduce(np.dot, [global_from_car_rot, #ref_from_car_rot, car_from_global_rot,
                                                    car_from_current_rot])  #

            # # # # Do the required rotations to the Radar velocity values
            current_pc.points = cls.rotate_velocity(current_pc.points, velocity_trans_matrix)

            # Merge with key pc.
            all_pc.points = np.hstack((all_pc.points, current_pc.points))

            # Abort if there are no previous sweeps.
            if current_sd_rec['prev'] == '':
                break
            else:
                current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])
  
        return all_pc, all_times


def get_valid_radar_feat(outputs, radar_pc, cfg):
    bbox_num = cfg.bbox_num
    ## Calculate values for the new pc_hm
    # radar-pc: b * n * 5
    # convert to bev grid
    feature_map_size = [(cfg['grid_config']['x'][1] - cfg['grid_config']['x'][0]) / cfg['grid_config']['x'][2],
                        (cfg['grid_config']['y'][1] - cfg['grid_config']['y'][0]) / cfg['grid_config']['y'][2]]
    if cfg['time_debug']:
        time_s = time.time()

    batch, _, height, width = outputs[0][0]['heatmap'].size()
    radar_feat = torch.zeros((batch, len(cfg['pc_feat_name']), height, width)).to(outputs[0][0]['heatmap'].device)
    for task_ind, task_out in enumerate(outputs):

        task_s = time.time()
        task_out = task_out[0]

        heat = task_out['heatmap'].detach()
        wh = cfg.img_feats_bbox_dims  

        wh[0] = wh[0] / cfg['voxel_size'][0] / cfg[  
                    'out_size_factor']
        wh[1] = wh[1] / cfg['voxel_size'][1] / cfg[
                    'out_size_factor']
        batch, cat, height, width = heat.size()
        reg = task_out['reg'].detach()
        scores, inds, clses, ys0, xs0 = _topk(heat, K=bbox_num)
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, bbox_num, 2)
        xs = xs0.view(batch, bbox_num, 1) + reg[:, :, 0].view(batch, bbox_num, 1)
        ys = ys0.view(batch, bbox_num, 1) + reg[:, :, 1].view(batch, bbox_num, 1)

        bboxes = torch.cat([torch.floor(xs - wh[0] / 2),  
                            torch.floor(ys - wh[1] / 2),
                            torch.ceil(xs + wh[0] / 2),
                            torch.ceil(ys + wh[1] / 2)], dim=2)  # B x K x 4
        bboxes[bboxes<0] = 0
        bboxes[bboxes>feature_map_size[0]] = feature_map_size[0]
        if cfg['time_debug']:                              
            bbox_gen_t = time.time()
            l.warning("    Generating img feat bboxes spend {} sec.".format(bbox_gen_t-task_s))
        # pick pointcloud which in bboxes
        for b_index, frame_radar_pc in enumerate(radar_pc):  
            if cfg['time_debug']:
                batch_t = time.time()
            bboxes_b = bboxes[b_index]
            # valid_radar_pc = pick_valid_points_in_imgfeat_bboxes(bboxes_b, frame_radar_pc, cfg)
            valid_radar_pc = pick_valid_points_in_imgfeat_bboxes_v2(bboxes_b, frame_radar_pc, cfg)
            if cfg['time_debug']:
                radar_pc_t = time.time()
                l.warning("        getting valid radar pc in each frame spends {} sec.".format(radar_pc_t-batch_t))
            if valid_radar_pc is None:  
                continue
            else:
                update_radar_featmap(b_index, radar_feat, valid_radar_pc, cfg)
    return radar_feat

def pick_valid_points_in_imgfeat_bboxes(bboxes_b, frame_radar_pc, cfg):
    '''
     Args:
        bboxes_b:
        frame_radar_pc:
    Returns: choose_pc
    '''
    x, y, z = frame_radar_pc[0].T, frame_radar_pc[1].T, frame_radar_pc[2].T
    coor_x = (x - cfg['point_cloud_range'][0]) / cfg['voxel_size'][0] / cfg['out_size_factor']
    coor_y = (y - cfg['point_cloud_range'][1]) / cfg['voxel_size'][1] / cfg['out_size_factor']
    coor_x = np.expand_dims(coor_x, 1)
    coor_y = np.expand_dims(coor_y, 1)
    center_points = np.hstack([coor_x, coor_y])  # n *
    # record for checking pc points coordination.
    # record_radar_pc_and_bbox_coor(bboxes_b, center)
    # pick unrepeat bbox, for claculating speed.
    valid_bboxes = list()
    feature_map_size = [(cfg['grid_config']['x'][1] - cfg['grid_config']['x'][0]) / cfg['grid_config']['x'][2],
                        (cfg['grid_config']['y'][1] - cfg['grid_config']['y'][0]) / cfg['grid_config']['y'][2]]
    for bbox in bboxes_b:
        bbox = bbox.cpu().detach().numpy().tolist()
        if bbox not in valid_bboxes:
            valid_bboxes.append(bbox) 
    choose_inds = list()
    cfg.change_pc_num = 0
    for i in range(center_points.shape[0]):
        center_p = center_points[i]
        for j, bbox in enumerate(valid_bboxes):
            x_bias = bbox[2] - bbox[0]
            y_bias = bbox[3] - bbox[1]
            if center_p[0] > feature_map_size[0] / 2: 
                x_bias *= 1
            else:
                x_bias *= -1
            if center_p[1] > feature_map_size[1] / 2:
                y_bias *= 1
            else:
                y_bias *= -1
            if bbox[0] <= (center_p[0] + x_bias) <= bbox[2] and \
                bbox[1] <= (center_p[1] + y_bias) <= bbox[3]:
                choose_inds.append(i)
                frame_radar_pc[0, i] += (x_bias * cfg['voxel_size'][0] * cfg['out_size_factor'])
                frame_radar_pc[1, i] += (y_bias * cfg['voxel_size'][1] * cfg['out_size_factor'])
                break
    del valid_bboxes
    if len(choose_inds) > 0:
        return frame_radar_pc[:, choose_inds]
    else:
        return None

def pick_valid_points_in_imgfeat_bboxes_v2(bboxes_b, frame_radar_pc, cfg):
    '''
     Args:
        bboxes_b:
        frame_radar_pc:
    Returns: choose_pc
    '''
    x, y, z = frame_radar_pc[0].T, frame_radar_pc[1].T, frame_radar_pc[2].T
    coor_x = (x - cfg['point_cloud_range'][0]) / cfg['voxel_size'][0] / cfg['out_size_factor']
    coor_y = (y - cfg['point_cloud_range'][1]) / cfg['voxel_size'][1] / cfg['out_size_factor']
    coor_x = np.expand_dims(coor_x, 1)
    coor_y = np.expand_dims(coor_y, 1)
    center_points = np.hstack([coor_x, coor_y])  # n *
    # record for checking pc points coordination.
    # record_radar_pc_and_bbox_coor(bboxes_b, center)
    # pick unrepeat bbox, for claculating speed.
    valid_bboxes = list()
    feature_map_size = [(cfg['grid_config']['x'][1] - cfg['grid_config']['x'][0]) / cfg['grid_config']['x'][2],
                        (cfg['grid_config']['y'][1] - cfg['grid_config']['y'][0]) / cfg['grid_config']['y'][2]]
    for bbox in bboxes_b:
        bbox = bbox.cpu().detach().numpy().tolist()
        if bbox not in valid_bboxes:
            valid_bboxes.append(bbox)  
    choose_inds = list()
    valid_mask = np.zeros((int(feature_map_size[0]), int(feature_map_size[1])))
    for j, bbox in enumerate(valid_bboxes):
        valid_tensor = np.ones((int(bbox[3]-bbox[1]), int(bbox[2]-bbox[0])))
        valid_mask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = valid_tensor
    
    for i in range(center_points.shape[0]):
        center_p = center_points[i]
        if max(center_p) >= feature_map_size[0]:
            continue
        if min(center_p) < 0:
            continue
        if valid_mask[int(center_p[0]), int(center_p[1])] > 0:
            choose_inds.append(i)
    del valid_bboxes, valid_mask
    if len(choose_inds) > 0:
        return frame_radar_pc[:, choose_inds]
    else:
        return None

def generate_radar_featmap(pc_3d, cfg):
    feature_map_size = [int((cfg['grid_config']['x'][1] - cfg['grid_config']['x'][0]) / cfg['grid_config']['x'][2]),
                        int((cfg['grid_config']['y'][1] - cfg['grid_config']['y'][0]) / cfg['grid_config']['y'][2])]
    bev_height, bev_width = feature_map_size
    pc_N = pc_3d.shape[1]
    pc_hm_feat = np.zeros((len(cfg['pc_feat_name']), bev_height, bev_width), np.float32)
    pillar_wh = create_pc_pillars(pc_3d, cfg)
    pc_feat_channels = {name: i for i, name in enumerate(cfg['pc_feat_name'])}
    # generate point cloud channels
    pc_range = cfg['point_cloud_range']
    for feat in cfg['pc_feat_name']:
        for i in range(pc_N - 1, -1, -1):
            point = pc_3d[:, i]
            x, y, z = point[0], point[1], point[2]

            coor_x = (x - pc_range[0]) / cfg['voxel_size'][0] / cfg['out_size_factor']
            coor_y = (y - pc_range[1]) / cfg['voxel_size'][1] / cfg['out_size_factor']

            ct = np.array([coor_x, coor_y])
            ct_int = ct.astype(np.int32)
            if not (0 <= ct_int[0] < bev_height
                    and 0 <= ct_int[1] < bev_width):
                continue
            # use pillars
            wh = pillar_wh[:, i]
            wh[0] = wh[0] / cfg['voxel_size'][0] / cfg['out_size_factor']  
            wh[1] = wh[1] / cfg['voxel_size'][1] / cfg['out_size_factor']
            b = [max(ct[0] - wh[0] / 2, 0),
                 min(ct[0] + wh[0] / 2, bev_width),
                 max(ct[1] - wh[1] / 2, 0),
                 min(ct[1] + wh[1] / 2, bev_height)]
            b = np.round(b).astype(np.int32)
            # coor_x, coor_y not coor_reg
            if feat == 'pc_x':
                channel = pc_feat_channels['pc_x']  # 0
                pc_hm_feat[channel, b[2]:max(b[3], b[2]+1), b[0]:max(b[1], b[2]+1)] = (coor_x - ct_int[0])

            if feat == 'pc_y':
                channel = pc_feat_channels['pc_y']  # 0
                pc_hm_feat[channel, b[2]:max(b[3], b[2]+1), b[0]:max(b[1], b[2]+1)] = (coor_y - ct_int[1])  

            if 'pc_vx' in cfg['pc_feat_name'] and feat == 'pc_vx': 
                vx = pc_3d[3, i]
                channel = pc_feat_channels['pc_vx']  # 1
                pc_hm_feat[channel, b[2]:max(b[3], b[2]+1), b[0]:max(b[1], b[2]+1)] = vx

            if 'pc_vy' in cfg['pc_feat_name'] and feat == 'pc_vy':
                vy = pc_3d[4, i]
                channel = pc_feat_channels['pc_vy']  # 2
                pc_hm_feat[channel, b[2]:max(b[3], b[2]+1), b[0]:max(b[1], b[2]+1)] = vy
    # print('')
    return pc_hm_feat

def update_radar_featmap_without_acceleration(b_index, radar_feat, pc_3d, cfg):
    feature_map_size = [int((cfg['grid_config']['x'][1] - cfg['grid_config']['x'][0]) / cfg['grid_config']['x'][2]),
                        int((cfg['grid_config']['y'][1] - cfg['grid_config']['y'][0]) / cfg['grid_config']['y'][2])]
    bev_height, bev_width = feature_map_size
    pc_N = pc_3d.shape[1]
    pc_hm_feat = radar_feat[b_index]
    pillar_wh = create_pc_pillars(pc_3d, cfg)
    pc_feat_channels = {name: i for i, name in enumerate(cfg['pc_feat_name'])}
    # generate point cloud channels
    pc_range = cfg['point_cloud_range']
    for i in range(pc_N - 1, -1, -1):
        point = pc_3d[:, i]
        x, y, z = point[0], point[1], point[2]

        coor_x = (x - pc_range[0]) / cfg['voxel_size'][0] / cfg['out_size_factor']
        coor_y = (y - pc_range[1]) / cfg['voxel_size'][1] / cfg['out_size_factor']

        ct = np.array([coor_x, coor_y])
        ct_int = ct.astype(np.int32)
        if not (0 <= ct_int[0] < bev_height
                and 0 <= ct_int[1] < bev_width):
            continue
        # use pillars
        wh = pillar_wh[:, i]
        wh[0] = wh[0] / cfg['voxel_size'][0] / cfg['out_size_factor']  
        wh[1] = wh[1] / cfg['voxel_size'][1] / cfg['out_size_factor']
        b = [max(ct[0] - wh[0] / 2, 0),
                min(ct[0] + wh[0] / 2, bev_width),
                max(ct[1] - wh[1] / 2, 0),
                min(ct[1] + wh[1] / 2, bev_height)]
        b = np.round(b).astype(np.int32)
        # coor_x, coor_y not coor_reg
        change_feat = torch.ones(4, b[3]-b[2], b[1]-b[0]).to(radar_feat.device)
        for feat in cfg['pc_feat_name']:
            if feat == 'pc_x':
                channel = pc_feat_channels['pc_x']  # 0
                change_value = torch.tensor(coor_x - ct_int[0]).to(radar_feat.device)

            elif feat == 'pc_y':
                channel = pc_feat_channels['pc_y']  # 1
                change_value = torch.tensor(coor_y - ct_int[1]).to(radar_feat.device)

            elif 'pc_vx' in cfg['pc_feat_name'] and feat == 'pc_vx': 
                vx = pc_3d[3, i]
                channel = pc_feat_channels['pc_vx']  # 2
                change_value = torch.tensor(vx).to(radar_feat.device)

            elif 'pc_vy' in cfg['pc_feat_name'] and feat == 'pc_vy':
                vy = pc_3d[4, i]
                channel = pc_feat_channels['pc_vy']  # 3
                change_value = torch.tensor(vy).to(radar_feat.device)
            else:
                raise('radar feat name is dismatched with name in radar config. please check radar feat name in radar config and if conditions.')
            change_feat[channel] *= change_value
        ori_tensor = pc_hm_feat[:, b[2]:b[3], b[0]:b[1]].clone()
        pc_hm_feat[:, b[2]:b[3], b[0]:b[1]] = torch.where(pc_hm_feat[:, b[2]:b[3], b[0]:b[1]]==0.0, change_feat, ori_tensor)
        del ori_tensor, change_feat

def update_radar_featmap(b_index, radar_feat, pc_3d, cfg):
    feature_map_size = [int((cfg['grid_config']['x'][1] - cfg['grid_config']['x'][0]) / cfg['grid_config']['x'][2]),
                        int((cfg['grid_config']['y'][1] - cfg['grid_config']['y'][0]) / cfg['grid_config']['y'][2])]
    bev_height, bev_width = feature_map_size
    pc_N = pc_3d.shape[1]
    pc_hm_feat = radar_feat[b_index]
    pillar_w = cfg.pillar_dims[0] / cfg['voxel_size'][0] / cfg['out_size_factor']
    pillar_h = cfg.pillar_dims[1] / cfg['voxel_size'][1] / cfg['out_size_factor']
    pillar_wh = [pillar_w, pillar_h]
    # generate point cloud channels
    pc_range = cfg['point_cloud_range']
    change_feat = torch.ones(4, int(pillar_wh[1]+1), int(pillar_wh[0]+1)).to(radar_feat.device)
    for i in range(pc_N - 1, -1, -1):
        point = pc_3d[:, i]
        x, y, z = point[0], point[1], point[2]

        coor_x = (x - pc_range[0]) / cfg['voxel_size'][0] / cfg['out_size_factor']
        coor_y = (y - pc_range[1]) / cfg['voxel_size'][1] / cfg['out_size_factor']

        ct = np.array([coor_x, coor_y])
        ct_int = ct.astype(np.int32)
        if not (0 <= ct_int[0] < bev_height
                and 0 <= ct_int[1] < bev_width):
            continue
        
        # coor_x, coor_y not coor_reg
        change_regx = coor_x - ct_int[0]
        change_feat[0] *= change_regx
        
        change_regy = coor_y - ct_int[1]
        change_feat[1] *= change_regy

        if 'pc_vx' in cfg['pc_feat_name']: 
            vx = pc_3d[3, i]
            change_feat[2] *= vx

        if 'pc_vy' in cfg['pc_feat_name']:
            vy = pc_3d[4, i]
            change_feat[3] *= vy
        pc_hm_feat[:, int(ct_int[1]-pillar_h/2):int(ct_int[1]-pillar_h/2)+int(pillar_h+1),
                      int(ct_int[0]-pillar_w/2):int(ct_int[0]-pillar_w/2)+int(pillar_w+1)] = change_feat
    del change_feat

def create_pc_pillars(pc_3d, cfg):
    pillar_wh = np.zeros((2, pc_3d.shape[1]))  # pc_3d:(18, n1) 
    pillar_dim = cfg['pillar_dims']  # [1.5, 0.2, 0.2]
    v = np.dot(np.eye(3), np.array([1, 0, 0]))  # [1. 0. 0.]
    ry = -np.arctan2(v[2], v[0])  # ry: -0.0

    for i, center in enumerate(pc_3d[:3, :].T):  # 
        # Create a 3D pillar at pc location for the full-size image
        box_3d = compute_box_3d(dim=pillar_dim, location=center, rotation_y=ry)
        box_2d = project_to_bev(box_3d).T  # [2x8]
        # get the bounding box in [xyxy] format
        bbox = [np.min(box_2d[0, :]),
                np.min(box_2d[1, :]),
                np.max(box_2d[0, :]),
                np.max(box_2d[1, :])]  # format: xyxy

        # store height and width of the 2D box
        pillar_wh[0, i] = bbox[2] - bbox[0]
        pillar_wh[1, i] = bbox[3] - bbox[1]

    return pillar_wh  

def comput_corners_3d(dim, rotation_y):
    # dim: 3,hwl
    # location: 3
    # rotation_y: 1
    # return: 8 x 3
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    w, l, h = dim[0], dim[1], dim[2]  
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]  # 

    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners).transpose(1, 0)
    return corners_3d  

def compute_box_3d(dim, location, rotation_y):
    # dim: 3
    # location: 3
    # rotation_y: 1
    # return: 8 x 3
    corners_3d = comput_corners_3d(dim, rotation_y)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(1, 3)
    return corners_3d

def project_to_bev(pts_3d):
    # pts_3d: n x 3
    pts_2d = pts_3d.copy()
    pts_2d[:, 2] = 0
    return pts_2d
