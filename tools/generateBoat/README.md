<h1><p align="center">渲染产生船牌数据</p></h1>

## 简介

通过渲染船牌加上畸变、噪声、与自然环境结合生成车牌的样本。

## 文档

### Generate Boat Plate

- 通过generateBoat包里的两个文件genplate_scene.py和plate_Commom.py实现渲染产生船牌图片数据
- 自定义类GenPlateScene
  - 自定义`GenPlateScene`
    - 实现一下四个方法 
      - `draw` 设置船牌文字位置
      - `generate` 对船牌号文字和图片进行旋转变形等操作
      - `gen_batch` 批量生成图片
      - `gen_plate_string` 生成船牌号码字符串
     
- plate_common.py
 - 自定义文件plate_common.py，对生成船牌加上畸变、旋转、调灰度等功能
    - 实现以下七个方法     
      - `rot` 对图片进行旋转
      - `image_distortion` 仿射畸变
      - `change_gray_and_color` 调灰度，调颜色
      - `random_scene` 将车牌放入自然场景图像中，并返回该图像和位置信息
      - `generate_ch_characters` 生成汉字
      - `generate_en_characters` 生成英文字符
      - `random_seed` 产生随机数字
      
### genplate_scene.py 参数配置
- `--bg_dir` 船牌的场景图片文件夹
- `--out_dir` 生成渲染船牌图片后保存文件夹
- `--make_num` 产生多少张船牌图片

## 注意
    -  font文件夹里的文件需要解压
    
## 运行
- 运行genplate_scene.py
