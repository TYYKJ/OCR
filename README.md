<h1><p align="center">OCR文字识别</p></h1>

## 简介

内部自用OCR识别框架。

## 注意

本项目基于torch 1.8开发

## 文档

本项目模型基于[PytorchLightning](https://www.pytorchlightning.ai/)开发, 训练部分可支持PytorchLighting一切功能。

### Train Custom Data

- 通过ocr包导入CRNN模型, 设置classes参数和字典地址
- 自定义DataLoader
  - 通过继承`torch.utils.data`的DataSet类来自定义DataSet
  - 自定义`DataLoader`
    - 继承`LightningDataModule`类实现DataLoader
    - 实现以下四个抽象方法
      - `setup`
      - `train_dataloader`
      - `val_dataloader`
      - `test_dataloader`
### 其他参数配置

#### 优化器
本项目提供五种优化器方案, 具体配置可在实例化模型的时候填写, 默认优化器为Adam。
可用优化器:
- adam
- sparseadam
- adamw
- radam
- plainradam

#### 损失函数

目前仅支持CTC Loss, 需要注意的是, `blank_idx`需要设置为字典第一个数据

### 工具包

工具包中包含字典生成和数据集内容查看以及系统可用字体查看。
在使用matplotlib查看数据的时候,中文乱码问题可以使用系统自带的文字输入法来规避中文字符的乱码问题。
