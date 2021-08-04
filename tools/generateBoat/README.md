<h1><p align="center">渲染产生船牌数据</p></h1>

## 简介

通过渲染船牌加上畸变、噪声、与自然环境结合生成车牌的样本。

## 文档

### Generate Boat Plate

- 通过generateBoat包里的两个文件generate_scense.py和PlateCommom.py实现渲染产生船牌图片数据
- 自定义类GenPlateScence
  - 自定义`GenPlateScence`
    - 实现一下四个方法 
      - `draw` 设置船牌文字位置
      - `generate` 对船牌号文字和图片进行旋转变形等操作
      - `gen_batch` 批量生成图片
      - `gen_plate_string` 生成船牌号码字符串
- PlateCommon.py
 - 自定义文件PlateCommon.py，对生成船牌加上畸变、旋转、调灰度等功能
    - 实现以下七个方法     
      - `rot` 对图片进行旋转
      - `rotRandrom` 仿射畸变
      - `tfactor` 调灰度，调颜色
      - `random_scene` 将车牌放入自然场景图像中，并返回该图像和位置信息
      - `GenCh` 生成汉字
      - `GenCh1` 生成英文字符
      - `r` 产生随机数字
- 注意
    -font文件夹里的文件需要解压
## 运行
- 运行generate_scense.py
