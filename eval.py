import argparse
import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_map
from yolo import YOLO

"""
VOC测试集验证，得到模型性能指标
"""

def main(args):
    image_ids = open(os.path.join(args.VOCdevkit_path, "ImageSets/Main/test.txt"), encoding='utf-8').read().strip().split()

    if not os.path.exists(args.map_out_path):
        os.makedirs(args.map_out_path)
    if not os.path.exists(os.path.join(args.map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(args.map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(args.map_out_path, 'detection-results')):
        os.makedirs(os.path.join(args.map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(args.map_out_path, 'images-optional')):
        os.makedirs(os.path.join(args.map_out_path, 'images-optional'))

    class_names, _ = get_classes(args.classes_path)

    print("Load model.")
    yolo = YOLO(model_path=args.model_path, classes_path=args.classes_path, anchors_path=args.anchors_path,
                anchors_mask=args.anchors_mask, input_shape=args.input_shape, phi=args.phi, confidence=args.confidence,
                nms_iou=args.nms_iou, letterbox_image=args.letterbox_image, cuda=args.cuda)
    print("Load model done.")

    # 获取测试集图片预测结果
    print("Get predict result.")
    for image_id in tqdm(image_ids):
        image_path = os.path.join(args.VOCdevkit_path, "JPEGImages/" + image_id + ".jpg")
        image = Image.open(image_path)
        if args.map_vis:
            image.save(os.path.join(args.map_out_path, "images-optional/" + image_id + ".jpg"))
        yolo.get_map_txt(image_id, image, class_names, args.map_out_path)
    print("Get predict result done.")

    # 获取测试集图片标签
    print("Get ground truth result.")
    for image_id in tqdm(image_ids):
        with open(os.path.join(args.map_out_path, "ground-truth/" + image_id + ".txt"), "w", encoding='utf-8') as new_f:
            root = ET.parse(os.path.join(args.VOCdevkit_path, "Annotations/" + image_id + ".xml")).getroot()
            for obj in root.findall('object'):
                difficult_flag = False
                if obj.find('difficult') != None:
                    difficult = obj.find('difficult').text
                    if int(difficult) == 1:
                        difficult_flag = True
                obj_name = obj.find('name').text
                if obj_name not in class_names:
                    continue
                bndbox = obj.find('bndbox')
                left = bndbox.find('xmin').text
                top = bndbox.find('ymin').text
                right = bndbox.find('xmax').text
                bottom = bndbox.find('ymax').text

                if difficult_flag:
                    new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                else:
                    new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
    print("Get ground truth result done.")

    # 计算map
    print("Get map.")
    get_map(args.MINOVERLAP, True, path=args.map_out_path)
    print("Get map done.")


if __name__ == "__main__":
    # argparse模块
    parser = argparse.ArgumentParser(description="----------------yolov5 predict.py-----------------")

    # 模型相关
    parser.add_argument('--model_path', default='./model_data/yolov5_s.pth', help='location of model path')
    parser.add_argument('--classes_path', default='./model_data/coco_classes.txt', help='location of classes path')
    parser.add_argument('--anchors_path', default='./model_data/yolo_anchors.txt', help='location of anchors path')
    parser.add_argument('--anchors_mask', default="[[6, 7, 8], [3, 4, 5], [0, 1, 2]]", help='')
    parser.add_argument('--input_shape', default=640, type=int, help='model input size')
    parser.add_argument('--phi', default="s", help='model type s, m, l, or x')
    parser.add_argument('--confidence', default=0.5, type=float, help='object confidence')
    parser.add_argument('--nms_iou', default=0.3, type=float, help='nms iou threshold')
    parser.add_argument('--letterbox_image', default='True', help='use letterbox for input image or not')
    parser.add_argument('--cuda', default='True', help='use cuda')

    #   MINOVERLAP用于指定想要获得的mAP0.x,比如计算mAP0.75，可以设定MINOVERLAP = 0.75。
    parser.add_argument('--MINOVERLAP', default=0.5, type=float, help='')
    #   指向VOC数据集所在的文件夹, 默认指向根目录下的VOC数据集
    parser.add_argument('--VOCdevkit_path', default='./data/VOC2007/', help='')
    # map_vis用于指定是否开启VOC_map计算的可视化
    parser.add_argument('--map_vis', default='False', help='')
    #   结果输出的文件夹，默认为map_out
    parser.add_argument('--map_out_path', default='map_out', help='')

    args = parser.parse_args()
    # 参数转换
    args.anchors_mask = eval(args.anchors_mask)
    args.input_shape = [args.input_shape, args.input_shape]
    args.letterbox_image = eval(args.letterbox_image)
    args.cuda = eval(args.cuda)
    args.map_vis = eval(args.map_vis)
    print(args)
    # 调用主函数
    main(args)
