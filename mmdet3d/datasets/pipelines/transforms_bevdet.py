
import random
import warnings

import cv2
import mmcv
import numpy as np
import torch
import torchvision.transforms as transforms
from mmcv import is_tuple_of
from mmcv.utils import build_from_cfg
from PIL import Image

from mmdet3d.core import VoxelGenerator
from mmdet3d.core.bbox import (CameraInstance3DBoxes, DepthInstance3DBoxes,
                               LiDARInstance3DBoxes, box_np_ops)
from mmdet3d.datasets.pipelines.compose import Compose
from mmdet.datasets.pipelines import RandomCrop, RandomFlip, Rotate
from ..builder import OBJECTSAMPLERS, PIPELINES
from .data_augment_utils import noise_per_object_v3_


@PIPELINES.register_module()
class RandomScale_imgview():
    """
    bevdet 图像域和bev域分开,且涉及到到对数据增强参数进行mlp,所以图像域数据增强需要同时变换数据增强矩阵。
    模型推理完图像域后，默认将数据增强逆变换回来。所以真值不需要进行数据增强。
    Flip the points & bbox.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        sync_2d (bool, optional): Whether to apply flip according to the 2D
            images. If True, it will apply the same flip as that to 2D images.
            If False, it will decide whether to flip randomly and independently
            to that of 2D images. Defaults to True.
        flip_ratio_bev_horizontal (float, optional): The flipping probability
            in horizontal direction. Defaults to 0.0.
        flip_ratio_bev_vertical (float, optional): The flipping probability
            in vertical direction. Defaults to 0.0.
    """

    def __init__(self,
                 offline_save_result = True,
                 **kwargs):
        super(RandomScale_imgview, self).__init__(
             **kwargs)
        self.offline_save_result = offline_save_result

    def __call__(self, input_dict):
        """Call function to flip points, values in the ``bbox3d_fields`` and
        also flip 2D image and its annotations.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction',
                'pcd_horizontal_flip' and 'pcd_vertical_flip' keys are added
                into result dict.
        """
        # flip 2D image and its annotations
        if self.offline_save_result:
            if not input_dict['scale']:
                return input_dict
            input_dict['img_inputs'] = list(input_dict['img_inputs'])
            # trans = transforms.Compose([transforms.ToTensor()])
            # resize_dims = input_dict['scale'] #h,w
            h,w = input_dict['scale']
            img_nums, channel, img_h, img_w = input_dict['img_inputs'][0].shape
            resize = float(w) / float(img_w)
            results = torch.zeros((img_nums, channel, h, w), dtype=input_dict['img_inputs'][0].dtype)
            for index,img in enumerate(input_dict['img_inputs'][0]):
                # img =  transforms.ToPILImage()(img)
                # results[index] = trans(img.resize((w,h)))
                img = img.permute(1,2,0).contiguous().numpy()
                img = cv2.resize(img, (w,h))
                results[index] = torch.tensor(img).permute(2, 0, 1).contiguous()
            input_dict['img_inputs'][0] = results

            # scale的数据增强矩阵
            post_rot = input_dict['img_inputs'][4]
            post_rot[:,:2,:2] *= resize # 基于原点缩放
            input_dict['img_inputs'][4] = post_rot
            input_dict['img_inputs'] = tuple(input_dict['img_inputs'])
            return input_dict
        else:
            return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f' offline_save_result={self.offline_save_result})'
        return repr_str


@PIPELINES.register_module()
class RandomFlip_imgview(RandomFlip):
    """
    bevdet 图像域和bev域分开,且涉及到到对数据增强参数进行mlp,所以图像域数据增强需要同时变换数据增强矩阵。
    模型推理完图像域后，默认将数据增强逆变换回来。所以真值不需要进行数据增强。
    Flip the points & bbox.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        sync_2d (bool, optional): Whether to apply flip according to the 2D
            images. If True, it will apply the same flip as that to 2D images.
            If False, it will decide whether to flip randomly and independently
            to that of 2D images. Defaults to True.
        flip_ratio_bev_horizontal (float, optional): The flipping probability
            in horizontal direction. Defaults to 0.0.
        flip_ratio_bev_vertical (float, optional): The flipping probability
            in vertical direction. Defaults to 0.0.
    """

    def __init__(self,
                 offline_save_result = True,
                 **kwargs):
        super(RandomFlip_imgview, self).__init__(
             **kwargs)
        self.offline_save_result = offline_save_result

    def __call__(self, input_dict):
        """Call function to flip points, values in the ``bbox3d_fields`` and
        also flip 2D image and its annotations.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction',
                'pcd_horizontal_flip' and 'pcd_vertical_flip' keys are added
                into result dict.
        """
        # flip 2D image and its annotations
        # super(RandomFlip_imgview, self).__call__(input_dict)
        if self.offline_save_result:
            if not input_dict['flip']:
                return input_dict
            input_dict['img_inputs'] = list(input_dict['img_inputs'])
            # 方式1,耗时0.15s
            # input_dict['img_inputs'][0] = self.imflip(input_dict['img_inputs'][0], direction=input_dict['flip_direction']) # 测试两种方法是否相同？毕竟这个可以batchsize做，效率更高
            # 方式2 使用同aug相同的函数,耗时0.099s
            # trans = transforms.Compose([transforms.ToTensor()])
            for index,img in enumerate(input_dict['img_inputs'][0]):
                img = img.permute(1,2,0).contiguous().numpy()
                img = cv2.flip(img, 1) # 水平翻转
                input_dict['img_inputs'][0][index] = torch.tensor(img).permute(2, 0, 1).contiguous()

                # img =  transforms.ToPILImage()(img)
                # input_dict['img_inputs'][0][index] = trans(img.transpose(method=Image.FLIP_LEFT_RIGHT)) 
                
                # img = img.permute(1,2,0).contiguous().numpy()
                # img = Image.fromarray(img.astype('uint8'))
                # img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
                # img = np.array(img)
                # input_dict['img_inputs'][0][index] = torch.tensor(img).float().permute(2, 0, 1).contiguous()

            # flip的数据增强矩阵
            post_rot = input_dict['img_inputs'][4]
            post_tran = input_dict['img_inputs'][5]
            A = torch.Tensor([[-1, 0], [0, 1]]) # 水平flip
            b = torch.Tensor([input_dict['img_inputs'][0].shape[3], 0]) # fw，输入图像的宽,水平翻转后，再向右移动到原来的坐标位置
            post_rot = A.matmul(post_rot[:,:2,:2])
            post_tran = A.expand((post_rot.shape[0], 2,2)).matmul(post_tran[:,:2][:,:,None]).squeeze() + b 
            input_dict['img_inputs'][4][:, :2, :2] = post_rot
            input_dict['img_inputs'][5][:,:2] = post_tran
            input_dict['img_inputs'] = tuple(input_dict['img_inputs'])
            return input_dict
        else:
            return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f' offline_save_result={self.offline_save_result})'
        return repr_str


    def imflip(self, img, direction = 'horizontal'):
        """Flip an image horizontally or vertically.

        Args:
            img (tensor): Image to be flipped.
            direction (str): The flip direction, either "horizontal" or
                "vertical" or "diagonal".

        Returns:
            tensor: The flipped image.
        """
        # 54 3 256 704
        assert direction in ['horizontal', 'vertical', 'diagonal']
        if direction == 'horizontal':
            return torch.flip(img, [3]) # w
        elif direction == 'vertical':
            return torch.flip(img, [2]) # h
        else:
            return torch.flip(img, [2, 3])


@PIPELINES.register_module()
class RandomRot_imgview():
    """
    bevdet 图像域和bev域分开,且涉及到到对数据增强参数进行mlp,所以图像域数据增强需要同时变换数据增强矩阵。
    模型推理完图像域后，默认将数据增强逆变换回来。所以真值不需要进行数据增强。
    Flip the points & bbox.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        sync_2d (bool, optional): Whether to apply flip according to the 2D
            images. If True, it will apply the same flip as that to 2D images.
            If False, it will decide whether to flip randomly and independently
            to that of 2D images. Defaults to True.
        flip_ratio_bev_horizontal (float, optional): The flipping probability
            in horizontal direction. Defaults to 0.0.
        flip_ratio_bev_vertical (float, optional): The flipping probability
            in vertical direction. Defaults to 0.0.
    """

    def __init__(self,
                 offline_save_result = True,
                 **kwargs):
        super(RandomRot_imgview, self).__init__(
             **kwargs)
        self.offline_save_result = offline_save_result

    def __call__(self, input_dict):
        """Call function to flip points, values in the ``bbox3d_fields`` and
        also flip 2D image and its annotations.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction',
                'pcd_horizontal_flip' and 'pcd_vertical_flip' keys are added
                into result dict.
        """
        # flip 2D image and its annotations
        # super(RandomFlip_imgview, self).__call__(input_dict)
        if self.offline_save_result:
            if not input_dict['rot']:
                return input_dict

            input_dict['img_inputs'] = list(input_dict['img_inputs'])
            # trans = transforms.Compose([transforms.ToTensor()])
            for index,img in enumerate(input_dict['img_inputs'][0]):
                img = img.permute(1,2,0).contiguous().numpy()
                img = self.rotate(img, input_dict['rot']) #用hw还是h-1,w-1
                input_dict['img_inputs'][0][index] = torch.tensor(img).permute(2, 0, 1).contiguous() 
                # img =  transforms.ToPILImage()(img)
                # input_dict['img_inputs'][0][index] = trans(img.rotate(input_dict['rot'])) 

            h,w = input_dict['img_inputs'][0].shape[2:]
            # img = img.resize(resize_dims)

            # rotate的数据增强矩阵
            post_rot = input_dict['img_inputs'][4]
            post_tran = input_dict['img_inputs'][5]
            A = self.get_rot(input_dict['rot'] / 180 * np.pi)
            b = torch.Tensor([w-1, h-1]) / 2 # 小数 (shape-1)/2 (fw,fh)
            b = A.matmul(-b) + b

            # post_rot = A.matmul(post_rot)
            # post_tran = A.matmul(post_tran) + b
            # A = torch.Tensor([[-1, 0], [0, 1]]) # 水平flip
            # b = torch.Tensor([input_dict['img_inputs'][0].shape[3], 0]) # fw，输入图像的宽,水平翻转后，再向右移动到原来的坐标位置
            post_rot = A.matmul(post_rot[:,:2,:2])
            post_tran = A.expand((post_rot.shape[0], 2,2)).matmul(post_tran[:,:2][:,:,None]).squeeze() + b 
            input_dict['img_inputs'][4][:, :2, :2] = post_rot
            input_dict['img_inputs'][5][:,:2] = post_tran
            input_dict['img_inputs'] = tuple(input_dict['img_inputs'])
            return input_dict
        else:
            return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(offline_save_result={self.offline_save_result},'
        return repr_str

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])
    # 定义旋转rotate函数
    def rotate(self, image, angle, center=None, scale=1.0):
        # 获取图像尺寸
        (h, w) = image.shape[:2]
    
        # 若未指定旋转中心，则将图像中心设为旋转中心
        if center is None:
            center = (w / 2, h / 2)
    
        # 执行旋转
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))
    
        # 返回旋转后的图像
        return rotated


@PIPELINES.register_module()
class RandomFlip_bevview(RandomFlip):
    """
    bevdet 图像域和bev域分开,且涉及到到对数据增强参数进行mlp,所以图像域数据增强需要同时变换数据增强矩阵。
    模型推理完图像域后，默认将数据增强逆变换回来。所以真值不需要进行数据增强。
    Flip the points & bbox.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        sync_2d (bool, optional): Whether to apply flip according to the 2D
            images. If True, it will apply the same flip as that to 2D images.
            If False, it will decide whether to flip randomly and independently
            to that of 2D images. Defaults to True.
        flip_ratio_bev_horizontal (float, optional): The flipping probability
            in horizontal direction. Defaults to 0.0.
        flip_ratio_bev_vertical (float, optional): The flipping probability
            in vertical direction. Defaults to 0.0.
    """

    def __init__(self,
                 offline_save_result = True,
                 **kwargs):
        super(RandomFlip_bevview, self).__init__(
             **kwargs)
        self.offline_save_result = offline_save_result

    def __call__(self, input_dict):
        """Call function to flip points, values in the ``bbox3d_fields`` and
        also flip 2D image and its annotations.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction',
                'pcd_horizontal_flip' and 'pcd_vertical_flip' keys are added
                into result dict.
        """
        if self.offline_save_result:
            if not input_dict['bevflip']:
                return input_dict
            
            input_dict['img_inputs'] = list(input_dict['img_inputs'])
            # bevflip的数据增强矩阵
            bda_rot = input_dict['img_inputs'][6] # torch.Size([3, 3]) 默认对角线为1
            flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            if (input_dict['bevflip_direction']=='horizontal') or ('horizontal' in input_dict['bevflip_direction']):
                flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
            if (input_dict['bevflip_direction']=='vertical') or ('vertical' in input_dict['bevflip_direction']):
                flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
            bda_rot = flip_mat @ bda_rot
            input_dict['img_inputs'][6] = bda_rot
            input_dict['img_inputs'] = tuple(input_dict['img_inputs'])
            return input_dict
        else:
            return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'offline_save_result={self.offline_save_result})'
        return repr_str


@PIPELINES.register_module()
class RandomRot_bevview():
    """
    bevdet 图像域和bev域分开,且涉及到到对数据增强参数进行mlp,所以图像域数据增强需要同时变换数据增强矩阵。
    模型推理完图像域后，默认将数据增强逆变换回来。所以真值不需要进行数据增强。
    Flip the points & bbox.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.
    Args:
        
    """

    def __init__(self,
                 offline_save_result = True,
                 **kwargs):
        super(RandomRot_bevview, self).__init__(
             **kwargs)
        self.offline_save_result = offline_save_result

    def __call__(self, input_dict):
        """Call function to flip points, values in the ``bbox3d_fields`` and
        also flip 2D image and its annotations.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction',
                'pcd_horizontal_flip' and 'pcd_vertical_flip' keys are added
                into result dict.
        """
        if self.offline_save_result:
            if not input_dict['bevrot']:
                return input_dict
            
            input_dict['img_inputs'] = list(input_dict['img_inputs'])
            # bevflip的数据增强矩阵
            bda_rot = input_dict['img_inputs'][6] # torch.Size([3, 3]) 默认对角线为1

            rotate_angle = torch.tensor(input_dict['bevrot'][0] / 180 * np.pi)
            rot_sin = torch.sin(rotate_angle)
            rot_cos = torch.cos(rotate_angle)
            rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                    [0, 0, 1]])
            bda_rot = rot_mat @ bda_rot
            input_dict['img_inputs'][6] = bda_rot
            input_dict['img_inputs'] = tuple(input_dict['img_inputs'])
            return input_dict
        else:
            return input_dict

@PIPELINES.register_module()
class RandomScale_bevview():
    """
    bevdet 图像域和bev域分开,且涉及到到对数据增强参数进行mlp,所以图像域数据增强需要同时变换数据增强矩阵。
    模型推理完图像域后，默认将数据增强逆变换回来。所以真值不需要进行数据增强。
    Flip the points & bbox.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.
    Args:
        
    """

    def __init__(self,
                 offline_save_result = True,
                 **kwargs):
        super(RandomScale_bevview, self).__init__(
             **kwargs)
        self.offline_save_result = offline_save_result

    def __call__(self, input_dict):
        """Call function to flip points, values in the ``bbox3d_fields`` and
        also flip 2D image and its annotations.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction',
                'pcd_horizontal_flip' and 'pcd_vertical_flip' keys are added
                into result dict.
        """
        if self.offline_save_result:
            if not input_dict['bevscale_ratio']:
                return input_dict
            
            input_dict['img_inputs'] = list(input_dict['img_inputs'])
            # bevflip的数据增强矩阵
            bda_rot = input_dict['img_inputs'][6] # torch.Size([3, 3]) 默认对角线为1
            scale_ratio = input_dict['bevscale_ratio'][0]
            scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
            
            bda_rot = scale_mat @ bda_rot
            input_dict['img_inputs'][6] = bda_rot
            input_dict['img_inputs'] = tuple(input_dict['img_inputs'])
            return input_dict
        else:
            return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'offline_save_result={self.offline_save_result})'
        return repr_str