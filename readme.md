# YOLOV5：You Only Look Once目标检测模型 pytorch实现（edition v5.0 in Ultralytics）
简介：这是一个基于pytorch实现的Yolov5算法的项目，作为Yolov5算法的复现。可以帮助读者更好的理解
它的网络结构、训练流程、损失计算等。读者也可以使用该仓库训练自己的数据集

Yolov5算法介绍，参考博客：https://blog.csdn.net/qq_40980981/article/details/138960623?spm=1001.2014.3001.5502

项目的详细介绍使用，参考博客：

检测结果示例：

## 环境准备
torch==1.2.0

pip install -r requirements.txt
## 数据集准备
仓库使用VOC07+12的数据集作为示例，读者可以根据该数据集格式训练自己的数据集

### 下载
VOC数据集官网下载地址：

VOC数据集包含目标检测和分割标注，本仓库只需要用到目标检测部分，需要用到下面几个文件夹数据

Annotations:图片目标检测标注信息

ImageSets:存放不同任务的划分的数据集（本项目会从新划分训练验证）

JPEGImages:这里存放的就是JPG格式的原图，包含17125张彩色图片，2913张是用于分割的

### 处理
数据集下载好需要用voc_annotation.py脚本来划分训练验证，修改脚本中VOCdevkit_path指向数据集目录，
该脚本执行完会，会在数据集目录创建2007_train.txt和2007_val.txt，用于训练。也可以直接从百度网盘下载作者处理好的数据集

百度网盘地址：

## 模型训练
train.py中训练配置了训练voc数据集的模型参数，直接运行python train.py即可。模型训练的结果数据会保存在--save_dir参数制定的文件中

## 模型预测
_todo_

## Reference
https://github.com/bubbliiiing/yolov5-pytorch