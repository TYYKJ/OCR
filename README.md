<h1><p align="center" style="color: bisque">OCR文字识别</p></h1>

## 简介

内部自用OCR识别框架。

## 注意

本项目基于torch 1.8开发

## 文档

本项目模型基于[PytorchLightning](https://www.pytorchlightning.ai/)开发, 训练部分可支持PytorchLighting一切功能。

### 快速开始
```python
from ocr import CRNN

model = CRNN(
    classes=21,
    encoder_name='vgg19_bn'
)
```

### 参数配置

#### 可用编码器

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
