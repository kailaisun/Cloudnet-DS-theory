# Cloudnet-DS-theory
Research on Multi-temporal Cloud Removal Using D-S Evidence Theory and Cloud Segmentation Model
We proposed a D-S evidence theory based multi-temporal cloud removal method and Cloud Segmentation Model, which is introduced in "Research on Multi-temporal Cloud Removal Using D-S Evidence Theory and Cloud Segmentation Model". We applied cloud-net[1] in our method. If you want to train the network yourself, you can find information in [here](https://github.com/SorourMo/Cloud-Net-A-semantic-segmentation-CNN-for-cloud-detection/tree/6c30ad6482847c855337baa5f17c24adaf5e5cda). The pretrained weights can be downloaded [here](https://vault.sfu.ca/index.php/s/2Xk6ZRbwfnjrOtu). The data we used were provided by government, so we can't publish our data. We are sorry for that.

## Requirements
Python 3.6<br>
Tensorflow 1.9.0, 1.10.0, 1.12.0<br>
Keras 2.2.4<br>
Scikit-image 0.15.0<br>
*If you are using latest GPU like RTX 3090, loading pretrained weights may cost long time. <br>

## How to remove cloud in images
1. Collect a series of cloudy pictures in same location.
2. Apply 'Cloud-Net\single_test' to get cloud segmentation results.
3. Follow instructions in D-S.ipynb to get final result.

This is not the final version of our method, we will improve it afterwards.

[1] S. Mohajerani and P. Saeedi, "Cloud-Net: An End-To-End Cloud Detection Algorithm for Landsat 8 Imagery," IGARSS 2019 - 2019 IEEE International Geoscience and Remote Sensing Symposium, Yokohama, Japan, 2019, pp. 1029-1032. doi: 10.1109/IGARSS.2019.8898776. Arxive URL: https://arxiv.org/pdf/1901.10077.pdf, IEEE URL: URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8898776&isnumber=8897702

