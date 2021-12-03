<h1><p align="center" style="color: bisque">PL-OCR</p></h1>

## 简介

基于PytorchLightning的OCR检测识别以及分类。

## 文档

本项目基于[PytorchLightning](https://www.pytorchlightning.ai/)以及torch 1.8开发, 训练部分可支持PytorchLighting一切功能。

### 快速开始

#### 训练
##### DB训练
修改train.bash中第六行为 `python $(dirname $(readlink -f "$0"))/trainDet.py`
使用`bash train.bash`进行训练

##### CRNN训练
修改train.bash第六行为 `python $(dirname $(readlink -f "$0"))/trainCRNN.py`
使用 `bash train.bash`进行训练

##### 分类训练
修改train.bash第六行为 `python $(dirname $(readlink -f "$0"))/trainClassify.py`
使用 `bash train.bash`进行训练

### 参数配置

#### 检测网络可用编码器

<details>
<summary style="margin-left: 25px;">ResNet</summary>
<div style="margin-left: 25px;">

|Encoder                         |
|--------------------------------|
|resnet18                        |
|resnet34                        |
|resnet50                        |
|resnet101                       |
|resnet152                       |
|resnext50_32x4d                 |
|resnext101_32x4d                |
|resnext101_32x8d                |
|resnext101_32x16d               |
|resnext101_32x32d               |
|resnext101_32x48d               |

</div>
</details>


<details>
<summary style="margin-left: 25px;">mobilenet_v2</summary>
<div style="margin-left: 25px;">

|Encoder                     |
|----------------------------|
|mobilenet_v2                |

</div>
</details>

<details>
<summary style="margin-left: 25px;">DPN</summary>
<div style="margin-left: 25px;">

|Encoder                     |
|----------------------------|
|dpn68                |
|dpn68b                |
|dpn92                |
|dpn98                |
|dpn107                |
|dpn131                |

</div>
</details>

<details>
<summary style="margin-left: 25px;">SENet</summary>
<div style="margin-left: 25px;">

|Encoder                     |
|----------------------------|
|se_resnet50                |
|se_resnet101                |
|se_resnet152                |
|se_resnext50_32x4d                |
|se_resnext101_32x4d                |
|senet154                |

</div>
</details>

#### 识别网络可用编码器

<details>
<summary style="margin-left: 25px;">ResNet</summary>
<div style="margin-left: 25px;">

|Encoder                           |
|----------------------------------|
|resnet18vd                        |
|resnet34vd                        |
|resnet50vd                        |
|resnet101vd                       |
|resnet152vd                       |
|resnet200vd                       |

</div>
</details>

## 致谢

本项目在开发过程中参考了 [PyTorchOCR](https://github.com/WenmuZhou/PytorchOCR/)以及 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR),感谢大佬们的开源~



