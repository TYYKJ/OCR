<h1><p align="center" style="color: bisque">OCR</p></h1>

## 简介

基于PL的OCR检测识别。

## 文档

本项目基于[PytorchLightning](https://www.pytorchlightning.ai/)以及torch 1.8开发, 训练部分可支持PytorchLighting一切功能。

### 快速开始

#### 检测模型

```python
from det import FPN

model = FPN(encoder_name='resnet50')
```

#### 识别模型
```python
from rec import CRNN

model = CRNN(
    classes=21,
    encoder_name='vgg19_bn'
)
```

识别模型的Dataloader需要返回`x, y, y_len`

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
<summary style="margin-left: 25px;">VGG</summary>
<div style="margin-left: 25px;">

|Encoder                     |
|----------------------------|
|vgg11                       |
|vgg11_bn                    |
|vgg13                       |
|vgg13_bn                    |
|vgg16                       |
|vgg16_bn                    |
|vgg19                       |
|vgg19_bn                    |

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

<details>
<summary style="margin-left: 25px;">VGG</summary>
<div style="margin-left: 25px;">

|Encoder                     |
|----------------------------|
|vgg11                       |
|vgg11_bn                    |
|vgg13                       |
|vgg13_bn                    |
|vgg16                       |
|vgg16_bn                    |
|vgg19                       |
|vgg19_bn                    |

</div>
</details>

<details>
<summary style="margin-left: 25px;">MobilenetV3</summary>
<div style="margin-left: 25px;">

|Encoder                     |
|----------------------------|
|mobilenetV3_small           |
|mobilenetV3_large           |

</div>
</details>

#### 优化器

本项目提供五种优化器方案, 具体配置可在实例化模型的时候填写, 默认优化器为Adam。 可用优化器:

- adam
- sparseadam
- adamw
- radam
- plainradam

#### 损失函数

目前仅支持CTC Loss, 需要注意的是, `blank_idx`需要设置为字典第一个数据

### 工具包

工具包中包含字典生成和数据集内容查看以及系统可用字体查看。 在使用matplotlib查看数据的时候,中文乱码问题可以使用系统自带的文字输入法来规避中文字符的乱码问题。

#### 数据生成工具

[数据生成文档](https://github.com/TYYKJ/limapOCR/blob/master/tools/generateBoat/README.md)

## 致谢

本项目在开发过程中参考了 [PyTorchOCR](https://github.com/WenmuZhou/PytorchOCR/)项目以及 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)项目,感谢大佬们的开源~



