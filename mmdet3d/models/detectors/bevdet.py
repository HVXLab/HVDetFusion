# Copyright (c) Phigent Robotics. All rights reserved.
import torch
from mmdet.models import DETECTORS
from .. import builder
from .base import Base3DDetector

@DETECTORS.register_module()
class HVDetFusion(Base3DDetector):
    def __init__(self, img_backbone, img_view_transformer, pts_bbox_head, radar_cfg=None,
                 train_cfg=None, test_cfg=None, pretrained=None, num_adj=11, **kwargs):
        super(HVDetFusion, self).__init__(**kwargs)
        self.num_frame = num_adj + 1
        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_view_transformer is not None:
            self.img_view_transformer = builder.build_neck(img_view_transformer)
        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = builder.build_head(pts_bbox_head)
        if radar_cfg is not None:
            self.radar_cfg = radar_cfg  
        self.index = 0

    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        B, N, _, H, W = inputs[0].shape
        N = N // self.num_frame
        imgs = inputs[0].view(B, N, self.num_frame, 3, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        rots, trans, intrins, post_rots, post_trans, bda = inputs[1:7]
        extra = [
            rots.view(B, self.num_frame, N, 3, 3),
            trans.view(B, self.num_frame, N, 3),
            intrins.view(B, self.num_frame, N, 3, 3),
            post_rots.view(B, self.num_frame, N, 3, 3),
            post_trans.view(B, self.num_frame, N, 3)
        ]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans = extra
        return imgs, rots, trans, intrins, post_rots, post_trans, bda
    
    def aug_test(self):
        pass

    def extract_feat(self):
        pass

    def simple_test(self):
        pass

    def radar_head_result_serialize(self, outs):
        outs_ = []
        for out in outs:
            for key in ['sec_reg', 'sec_rot', 'sec_vel']:
                outs_.append(out[0][key])
        return outs_

    def radar_head_result_deserialize(self, outs):
        outs_ = []
        keys = ['sec_reg', 'sec_rot', 'sec_vel']
        for head_id in range(len(outs) // 3):
            outs_head = [dict()]
            for kid, key in enumerate(keys):
                outs_head[0][key] = outs[head_id * 3 + kid]
            outs_.append(outs_head)
        return outs_

    def pts_head_result_serialize(self, outs):
        outs_ = []
        for out in outs:
            for key in ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']:
                outs_.append(out[0][key])
        return outs_
    
    def pts_head_result_deserialize(self, outs):
        outs_ = []
        keys = ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']
        for head_id in range(len(outs) // 6):
            outs_head = [dict()]
            for kid, key in enumerate(keys):
                outs_head[0][key] = outs[head_id * 6 + kid]
            outs_.append(outs_head)
        return outs_
