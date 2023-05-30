# HVDetFusion

This is the official implementation of HVDetFusion. In this work, we integrates the radar inputs into a unified bev space based on BEVDepth. We use a novel bev-based method to associate the radar detections to their corresponding camera detections, which is modified from CenterFusion. Firstly, Objects in the bird's-eye view are detected using the BevDepth4D detection network. Then we use the spatial position and size information of the detected objects to filter the effective information in the radar detection data, and use the effective radar point cloud to generate radar-based feature maps. Finally, the radar feature map is fused with the feature information of the object detected in the corresponding image to enhance the regression accuracy of attributes such as object depth and velocity.

For more details, please refer to our papar, and our paper is comming soon.




## Acknowledgement
This work is built on the open-sourced [BevDet](https://github.com/HuangJunJie2017/BEVDet),[BevDepth](https://github.com/Megvii-BaseDetection/BEVDepth) and the published code of [CenterFusion](https://github.com/mrnabati/CenterFusion).

## License
This project is released under the Apache 2.0 license.
