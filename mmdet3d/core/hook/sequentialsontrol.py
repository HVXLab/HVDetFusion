# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks import HOOKS, Hook
from mmdet3d.core.hook.utils import is_parallel

__all__ = ['SequentialControlHook']


@HOOKS.register_module()
class SequentialControlHook(Hook):
    """ """

    def __init__(self, temporal_start_epoch=1):
        super().__init__()
        self.temporal_start_epoch=temporal_start_epoch

    def set_temporal_flag(self, runner, flag):
        if is_parallel(runner.model.module):
            runner.model.module.module.with_prev=flag
        else:
            runner.model.module.with_prev = flag

    def before_run(self, runner):
        self.set_temporal_flag(runner, False)

    def before_train_epoch(self, runner):
        for name, p in runner.model.named_parameters():
            if 'radar_bbox_head' not in name:
            # if 'img_backbone' in name:
                p.requires_grad = False
                layer = getattr(runner.model, name.split('.', 1)[0])
                layer.eval()
            if 'heatmap' in name:
                p.requires_grad = False
                layer = getattr(runner.model, name.split('.', 1)[0])
                layer.eval()
        #     else:
        #         p.requires_grad = True
        #         # print(p.grad)
        if runner.epoch > self.temporal_start_epoch:
            self.set_temporal_flag(runner, True)