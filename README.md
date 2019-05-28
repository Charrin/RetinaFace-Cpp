# RetinaFace-Cpp
RetinaFace detector with C++

[official RetinaFace](https://github.com/deepinsight/insightface/tree/master/RetinaFace)

I convert [mobilenet-0.25 mxnet model](https://github.com/deepinsight/insightface/issues/669) (trained by [yangfly](https://github.com/yangfly)) to caffe model

* I have checked the output of the two models be the same.

* For same input images, the output of the two detector (python version and cpp version) is same.

(the code is too simple, only for reference 23333)

**Please replace your own inference code (caffe/ncnn/feather .etc) in the source code**

------

## Update 2019.5.28
I convert R50 mxnet model to caffe model [BaiDuYun](https://pan.baidu.com/s/1By24gkB1a76qJvxsg-gIgQ) | [Google Drive](https://drive.google.com/drive/folders/1hA5x3jCYFdja3PXLl9EcmucipRmVAj3W?usp=sharing)

Because of the GPU memory limited, I set max(width, height) to 1000 and test it on WiderFace_val set for **SINGLE SCALE, NO MULTI-SCALE, NO FLIP**, results as follows:

| wider val | easy | medium | hard |
| ------ | ------ | ------ | ------ |
| python version | 92.25 | 88.86 | 64.02 |
| cpp version | 91.36 | 87.15 | 62.22 |

## Update 2019.5.27
I test on WiderFace_val set for **SINGLE SCALE, NO MULTI-SCALE, NO FLIP**, results as follows:

| wider val | easy | medium | hard |
| ------ | ------ | ------ | ------ |
| python version | 83.28 | 77.02 | 39.52 |
| cpp version | 83.04 | 76.84 | 39.43 |

You can see the two detect results as **results/cpp_detect** and **results/python_detect**. The main cause of the difference in test results is the numerical precision difference between C++ and python.


## TODO
* implement bbox_vote
* multi-scale test
* speed benchmark (CPU/ARM)
* new models
