#! /bin/bash

python tools/HVDet_infer.py configs/hvdet/HVDetInfer_sim.py tools/convter2onnx/onnx_output --fuse-conv-bn --eval bbox  # --offline_eval --out ./res_pkl/test.pkl