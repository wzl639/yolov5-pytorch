import os
import random
import xml.etree.ElementTree as ET
from utils.utils import get_classes

"""
voc数据集生成处理脚本，用于生成 ImageSets/Main下的.txt 和训练用的2007_train.txt、2007_val.txt
"""

classes_path = 'model_data/voc_classes.txt'

trainval_percent = 0.9  # (训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1

train_percent = 0.9  # (训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1

VOCdevkit_path = './data/VOC2007'  # 指向VOC数据集所在的文件夹

VOCdevkit_sets = [('2007', 'train'), ('2007', 'val')]

classes, _ = get_classes(classes_path)


def convert_annotation(year, image_id, list_file):
    in_file = open(os.path.join(VOCdevkit_path, 'Annotations/%s.xml' % (image_id)), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


if __name__ == "__main__":
    random.seed(0)
    print("Generate txt in ImageSets.")
    xmlfilepath = os.path.join(VOCdevkit_path, 'Annotations')
    print(VOCdevkit_path)
    saveBasePath = os.path.join(VOCdevkit_path, 'ImageSets/Main')
    temp_xml = os.listdir(xmlfilepath)
    total_xml = []
    for xml in temp_xml:
        if xml.endswith(".xml"):
            total_xml.append(xml)

    num = len(total_xml)
    list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)

    print("train and val size", tv)
    print("train size", tr)
    ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w', encoding='utf-8')
    ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w', encoding='utf-8')
    ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w', encoding='utf-8')
    fval = open(os.path.join(saveBasePath, 'val.txt'), 'w', encoding='utf-8')

    for i in list:
        name = total_xml[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
    print("Generate txt in ImageSets done.")


    print("Generate 2007_train.txt and 2007_val.txt for train.")
    for year, image_set in VOCdevkit_sets:
        image_ids = open(os.path.join(VOCdevkit_path, 'ImageSets/Main/%s.txt' % (image_set)),
                         encoding='utf-8').read().strip().split()
        list_file = open(os.path.join(VOCdevkit_path, '%s_%s.txt'% (year, image_set)), 'w', encoding='utf-8')
        for image_id in image_ids:
            list_file.write('%s/JPEGImages/%s.jpg' % (os.path.abspath(VOCdevkit_path), image_id))
            convert_annotation(year, image_id, list_file)
            list_file.write('\n')
        list_file.close()
    print("Generate 2007_train.txt and 2007_val.txt for train done.")
