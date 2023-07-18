import argparse

import torch.onnx
from mmengine.config import Config
# from mmdeploy.backend.tensorrt.utils import save, search_cuda_version

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

import os
from typing import Dict, Optional, Sequence, Union

# import h5py
import mmcv
import numpy as np
import onnx
import torch
import tqdm
from mmcv.runner import load_checkpoint
from packaging import version
from torch.utils.data import DataLoader

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from tools.misc.fuse_conv_bn import fuse_module
from tools.radar_process.utils_radar import get_valid_radar_feat
import onnxruntime
from mmdet3d.core import bbox3d2result


def parse_args():
    parser = argparse.ArgumentParser(description='Deploy BEVDet with Tensorrt')
    parser.add_argument('config', help='deploy config file path')
    parser.add_argument('onnx_dir', help='work dir to save file')
    parser.add_argument('--checkpoint', default='./checkpoint/backbone.pth', help='checkpoint file')
    parser.add_argument(
        '--prefix', default='bevdepth4d-radarfusion', help='prefix of the save file name')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument('--offline_eval', action='store_true', help='use pkl file to import results file.')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')


    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None

    cfg = compat_cfg(cfg)
    cfg.gpu_ids = [0]

    # build the dataloader
    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    if not args.offline_eval:
        # build the model and load checkpoint
        cfg.model.train_cfg = None
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        load_checkpoint(model, args.checkpoint, map_location='cpu')
        model_prefix = args.prefix
        if args.fuse_conv_bn:
            model_prefix = model_prefix + '_fuse'
            model = fuse_module(model)

        sess1_path = './' + args.onnx_dir + '/' + model_prefix + '_stage1.onnx'
        sess1_1_path = './' + args.onnx_dir + '/' + model_prefix + '_stage1_1.onnx'
        sess2_path = './' + args.onnx_dir + '/' + model_prefix + '_stage2.onnx'
        sess3_path = './' + args.onnx_dir + '/' + model_prefix + '_stage3.onnx'

        session1 = onnxruntime.InferenceSession(sess1_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        session1_1 = onnxruntime.InferenceSession(sess1_1_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        session2 = onnxruntime.InferenceSession(sess2_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        session3 = onnxruntime.InferenceSession(sess3_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        print('session start finished.')

        model.cuda()
        model.eval()
        results = []
        prog_bar = mmcv.ProgressBar(len(dataset))
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                inputs = [t.cuda() for t in data['img_inputs'][0]]
                imgs_meta = data['img_metas'][0].data[0]
                data['img_inputs'][0] = [t.cuda() for t in data['img_inputs'][0]]
                data['radar_feat'] = data['radar_feat'][0].data
                data['img_metas'] = data['img_metas'][0].data
                # out_bbox_list = model(return_loss=False, scale=False, **data)
                # print('------------------------- ', i, ' ------------------------------')
                # print('ori model:', out_bbox_list)
                imgs, rots, trans, intrins, post_rots, post_trans, bda = \
                    model.prepare_inputs(inputs)
        
                bev_feat_list = []
                for img, rot, tran, intrin, post_rot, post_tran in zip(
                        imgs, rots, trans, intrins, post_rots, post_trans):
            
                    B, N, C, imH, imW = img.shape
                    
                    img = img.view(B * N, C, imH, imW)
                    backbone_out = model.img_backbone(img)
                    mlp_input = model.img_view_transformer.get_mlp_input(
                        rots[0], trans[0], intrin, post_rot, post_tran, bda)
                    sess1_out_img_feat, sess1_out_depth, sess1_out_tran_feat = session1.run(['img_feat', 'depth', 'tran_feat'], 
                                                                                        {
                                                                                            'backbone_out0':backbone_out[0].cpu().numpy(),
                                                                                            'backbone_out1':backbone_out[1].cpu().numpy(),
                                                                                            'mlp_input':mlp_input.cpu().numpy(),
                                                                                    #     # dynamic_axes={'backbone_in':[0], 'backbone_out':[0]})
                                                                                        }) 
                    sess1_out_img_feat = torch.tensor(sess1_out_img_feat).to(rot.device)
                    sess1_out_depth = torch.tensor(sess1_out_depth).to(rot.device)
                    sess1_out_tran_feat = torch.tensor(sess1_out_tran_feat).to(rot.device)
                    inputs = [sess1_out_img_feat, rot, tran, intrin, post_rot, post_tran, bda, mlp_input]
                    bev_feat, _ = model.img_view_transformer.view_transform(inputs, sess1_out_depth, sess1_out_tran_feat)
                
                    sess1_1_bev_feat = session1_1.run(['out_bev_feat'],
                                                        {'bev_feat': bev_feat.cpu().numpy()})
                    bev_feat_list.append(sess1_1_bev_feat[0])

        
                multi_bev_feat = np.concatenate(bev_feat_list, axis=1)

                output_names=['bev_feat'] + [f'output_{j}' for j in range(36)]
                sess2_out = session2.run(output_names, 
                                            {
                                            'multi_bev_feat':multi_bev_feat,
                                            }) 
                for i in range(len(sess2_out)):
                    sess2_out[i] = torch.tensor(sess2_out[i]).cuda()
                bev_feat = sess2_out[0]
                pts_outs = sess2_out[1:]
            
                if cfg.radar_cfg is not None:
                    pts_out_dict = model.pts_head_result_deserialize(pts_outs)
                    radar_pc = data['radar_feat'][0]
        
                    radar_feat = get_valid_radar_feat(pts_out_dict, radar_pc, cfg.radar_cfg)
                    sec_feats = torch.cat([bev_feat, radar_feat], 1) 

                    output_names=[f'radar_out_{j}' for j in range(15)]

                    sess3_radar_out=session3.run(output_names, 
                                            {
                                            'sec_feat':sec_feats.cpu().numpy(),
                                            }) 
                    for i in range(len(sess3_radar_out)):
                        sess3_radar_out[i] = torch.tensor(sess3_radar_out[i]).to(pts_outs[0].device)
                    pts_outs = model.pts_head_result_deserialize(pts_outs)
                    sec_outs=model.radar_head_result_deserialize(sess3_radar_out)
    
                    for task_ind, task_out in enumerate(sec_outs):
                        ori_task_out = pts_outs[task_ind][0]
                        sec_task_out = task_out[0]
                        for k, v in ori_task_out.items():
                            sec_k = 'sec_' + k
                            if sec_k in sec_task_out.keys() and k != 'heatmap':
                                pts_outs[task_ind][0][k] = sec_task_out[sec_k]
                out_bbox_list = [dict() for _ in range(len(imgs_meta))]

                bbox_list = model.pts_bbox_head.get_bboxes(
                    pts_outs, imgs_meta, rescale=False)
                bbox_pts = [
                                bbox3d2result(bboxes, scores, labels)
                                for bboxes, scores, labels in bbox_list
                            ]
                # print(bbox_pts)
                # exit(0)
                for result_dict, pts_bbox in zip(out_bbox_list, bbox_pts):
                    result_dict['pts_bbox'] = pts_bbox
                results.extend(out_bbox_list)
                batch_size = len(out_bbox_list)
                for _ in range(batch_size):
                    prog_bar.update()
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(results, args.out)
    else:
        results = mmcv.load(args.out)
    
    
    kwargs = {} #if args.eval_options is None else args.eval_options
    if args.format_only:
        dataset.format_results(results, **kwargs)
    # results = mmcv.load//(args.out)
    print(results)
    if args.eval:
        eval_kwargs = cfg.get('evaluation', {}).copy()
        # hard-code way to remove EvalHook args
        for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule'
        ]:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval, **kwargs))
        tmp = dataset.evaluate(results, **eval_kwargs)
        print("ok")


if __name__ == '__main__':

    main()

