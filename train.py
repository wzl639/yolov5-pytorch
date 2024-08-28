import argparse
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.yolo import YoloBody
from models.yolo_loss import YOLOLoss
from utils.train_fc import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import LossHistory
from dataset.yolo_dataset import YoloDataset, yolo_dataset_collate
from utils.utils import get_anchors, get_classes, get_lr


def fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen,
                  gen_val, Epoch, cuda, save_period, save_dir):
    """
    单个epoch训练逻辑
    """
    loss = 0
    val_loss = 0

    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break

            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    images = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
            # 清零梯度
            optimizer.zero_grad()
            # 前向传播
            outputs = model_train(images)

            loss_value_all = 0
            # 计算损失
            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all += loss_item
            loss_value = loss_value_all

            # 反向传播
            loss_value.backward()
            optimizer.step()

            loss += loss_value.item()

            pbar.set_postfix(**{'loss': loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    # 没训练完一轮验证一次
    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    images = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
                #  清零梯度
                optimizer.zero_grad()
                #  前向传播
                outputs = model_train(images)

                loss_value_all = 0
                #  计算损失
                for l in range(len(outputs)):
                    loss_item = yolo_loss(l, outputs[l], targets)
                    loss_value_all += loss_item
                loss_value = loss_value_all

            val_loss += loss_value.item()
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    print('Finish Validation')

    loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (
            epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))


def main(args):
    print(args)
    # 获取数据集类别数，获取anchor
    class_names, num_classes = get_classes(args.classes_path)
    anchors, num_anchors = get_anchors(args.anchors_path)

    # 模型定于，初始化权重
    model = YoloBody(args.anchors_mask, num_classes, args.phi)
    weights_init(model)
    if args.model_path != '':
        print('Load weights {}.'.format(args.model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # 构建损失
    yolo_loss = YOLOLoss(anchors, num_classes, args.input_shape, args.cuda, args.anchors_mask, args.label_smoothing)
    loss_history = LossHistory(args.save_dir, model, input_shape=args.input_shape)

    model_train = model.train()
    if args.cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    # 读数据集
    with open(args.train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(args.val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    # 构建数据集加载器
    train_dataset = YoloDataset(train_lines, args.input_shape, num_classes, epoch_length=args.epochs, mosaic=args.mosaic,
                                train=True)
    val_dataset = YoloDataset(val_lines, args.input_shape, num_classes, epoch_length=args.epochs, mosaic=False,
                              train=False)
    gen = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
                     drop_last=True, collate_fn=yolo_dataset_collate)
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)

    # 判断当前batch_size，自适应调整学习率
    nbs = 64
    lr_limit_max = 1e-3 if args.optimizer_type == 'adam' else 5e-2
    lr_limit_min = 3e-4 if args.optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(args.batch_size / nbs * args.Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(args.batch_size / nbs * args.Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    #  根据optimizer_type选择优化器
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)
    optimizer = {
        'adam': optim.Adam(pg0, Init_lr_fit, betas=(args.momentum, 0.999)),
        'sgd': optim.SGD(pg0, Init_lr_fit, momentum=args.momentum, nesterov=True)
    }[args.optimizer_type]
    optimizer.add_param_group({"params": pg1, "weight_decay": args.weight_decay})
    optimizer.add_param_group({"params": pg2})

    #  获得学习率下降的公式
    lr_scheduler_func = get_lr_scheduler(args.lr_decay_type, Init_lr_fit, Min_lr_fit, args.epochs)

    #  判断每一个世代的长度
    epoch_step = num_train // args.batch_size
    epoch_step_val = num_val // args.batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

    # 开始模型训练
    for epoch in range(args.epochs):
        gen.dataset.epoch_now = epoch
        gen_val.dataset.epoch_now = epoch
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val,
                      gen, gen_val, args.epochs, args.cuda, args.save_period, args.save_dir)


if __name__ == '__main__':
    # argparse模块，当字典一样用，方便传参
    parser = argparse.ArgumentParser(description="----------------yolov5 train-----------------")

    parser.add_argument('--cuda', default='True', help='use cuda')
    # 模型损失相关
    parser.add_argument('--classes_path', default='./model_data/voc_classes.txt', help='location of classes path')
    parser.add_argument('--anchors_path', default='./model_data/yolo_anchors.txt', help='location of anchors path')
    parser.add_argument('--anchors_mask', default="[[6, 7, 8], [3, 4, 5], [0, 1, 2]]", help='')
    parser.add_argument('--model_path', default='./model_data/yolov5_s.pth', help='location of model path')
    parser.add_argument('--phi', default="s", help='model type s, m, l, or x')
    parser.add_argument('--label_smoothing', default=0.0, type=float, help='label smoothing value')
    # 数据集相关
    parser.add_argument('--num_workers', default=4, type=int, help='nums of data load thread')
    parser.add_argument('--mosaic', default='True', help='use mosaic data enhancement or not')
    parser.add_argument('--input_shape', default=640, type=int, help='model input size')
    parser.add_argument('--train_annotation_path', default='./data/VOC2007/2007_train.txt', help='location of train annotation path')
    parser.add_argument('--val_annotation_path', default="./data/VOC2007/2007_val.txt", help='location of val annotation path')
    # 优化器相关
    parser.add_argument('--Init_lr', default=1e-2, type=float, help='train total epochs')
    parser.add_argument('--Min_lr', default=0.00001, type=float, help='batch size')
    parser.add_argument('--optimizer_type', default="sgd", help='optimizer type, adam or sgd')
    parser.add_argument('--momentum', default=0.937, type=float, help='momentum for optimizer')
    parser.add_argument('--weight_decay', default=5e-4, help='weight decay')
    parser.add_argument('--lr_decay_type', default="cos", help='lr decay , cos or step')
    # 训练相关
    parser.add_argument('--epochs', default=100, type=int, help='train total epochs')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--save_period', default=1, type=int, help='save period')
    parser.add_argument('--save_dir', default='./logs/', help='location of checkpoint')
    args = parser.parse_args()
    # 参数转换
    args.anchors_mask = eval(args.anchors_mask)
    args.mosaic = eval(args.mosaic)
    args.cuda = eval(args.cuda)
    args.input_shape = [args.input_shape, args.input_shape]
    # print(args, type(args.cuda))
    # 调用主函数
    main(args)
