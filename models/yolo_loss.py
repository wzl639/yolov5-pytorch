import math
from functools import partial
import numpy as np
import torch
import torch.nn as nn


class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, cuda, anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 label_smoothing=0):
        super(YOLOLoss, self).__init__()
        # -----------------------------------------------------------#
        #   13x13的特征层对应的anchor是[142, 110],[192, 243],[459, 401]
        #   26x26的特征层对应的anchor是[36, 75],[76, 55],[72, 146]
        #   52x52的特征层对应的anchor是[12, 16],[19, 36],[40, 28]
        # -----------------------------------------------------------#
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask
        self.label_smoothing = label_smoothing

        self.threshold = 4

        self.balance = [0.4, 1.0, 4]
        self.box_ratio = 0.05
        self.obj_ratio = 1 * (input_shape[0] * input_shape[1]) / (640 ** 2)
        self.cls_ratio = 0.5 * (num_classes / 80)
        self.cuda = cuda

    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def MSELoss(self, pred, target):
        return torch.pow(pred - target, 2)

    def BCELoss(self, pred, target):
        epsilon = 1e-7
        pred = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        output = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output

    def box_giou(self, b1, b2):
        """
        输入为：
        ----------
        b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

        返回为：
        -------
        giou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
        """
        # ----------------------------------------------------#
        #   求出预测框左上角右下角
        # ----------------------------------------------------#
        b1_xy = b1[..., :2]
        b1_wh = b1[..., 2:4]
        b1_wh_half = b1_wh / 2.
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half
        # ----------------------------------------------------#
        #   求出真实框左上角右下角
        # ----------------------------------------------------#
        b2_xy = b2[..., :2]
        b2_wh = b2[..., 2:4]
        b2_wh_half = b2_wh / 2.
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half

        # ----------------------------------------------------#
        #   求真实框和预测框所有的iou
        # ----------------------------------------------------#
        intersect_mins = torch.max(b1_mins, b2_mins)
        intersect_maxes = torch.min(b1_maxes, b2_maxes)
        intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        union_area = b1_area + b2_area - intersect_area
        iou = intersect_area / union_area

        # ----------------------------------------------------#
        #   找到包裹两个框的最小框的左上角和右下角
        # ----------------------------------------------------#
        enclose_mins = torch.min(b1_mins, b2_mins)
        enclose_maxes = torch.max(b1_maxes, b2_maxes)
        enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
        # ----------------------------------------------------#
        #   计算对角线距离
        # ----------------------------------------------------#
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        giou = iou - (enclose_area - union_area) / enclose_area

        return giou

    def smooth_labels(self, y_true, label_smoothing, num_classes):
        """
        标签类别平滑
        """
        return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

    def forward(self, l, input, targets=None):
        """
        yolov5单层输出损失计算
        l: 代表使用的是第几个有效特征层
        input: 模型当前层输出， bs, 3*(5+num_classes), 13, 13
        targets: 真实框的标签情况 list [batch_size, num_gt, 5]
        return: 当前层计算得到的损失
        """
        # 获得当前批次图片数量，特征层的高和宽
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)

        # 计算步长， stride_h = stride_w = 32、16、8
        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w

        # 将原图anchor缩放到特征图大小，此时获得的scaled_anchors大小是相对于特征层的
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]

        # 调整模型输出，将每个预测框的预测信息拆分出来
        prediction = input.view(
            bs, len(self.anchors_mask[l]),
            self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()  # [b, 3, 20, 20, 25(5 + num_classes)]
        # 先验框的中心位置的调整参数
        x = torch.sigmoid(prediction[..., 0])  # [b, 3, 20, 20]
        y = torch.sigmoid(prediction[..., 1])
        # 先验框的宽高调整参数
        w = torch.sigmoid(prediction[..., 2])
        h = torch.sigmoid(prediction[..., 3])
        # 获得置信度，是否有物体
        conf = torch.sigmoid(prediction[..., 4])
        # 种类置信度
        pred_cls = torch.sigmoid(prediction[..., 5:])

        # 获得网络应该有的预测结果, 这里包含正样本匹配的过程
        # 网络应该有的预测结果:y_true: [1, 3, 20, 20, 25(5 + num_classes)]，用于后续loss计算
        # noobj_mask: [1, 3, 20, 20] noobj_mask代表无目标的特征点，暂时没有用到
        y_true, noobj_mask = self.get_target(l, targets, scaled_anchors, in_h, in_w)

        # 将预测结果进行解码, 方便后面计算giou损失
        pred_boxes = self.get_pred_boxes(l, x, y, h, w, targets, scaled_anchors, in_h, in_w)  # [1, 3, 20, 20, 4]

        if self.cuda:
            y_true = y_true.cuda()
            # noobj_mask = noobj_mask.cuda()

        # loss计算
        loss = 0
        n = torch.sum(y_true[..., 4] == 1)  # 统计当前批次数据中是否有正样本
        if n != 0:
            # 当前batch数据有目前，计算正样本的位置和类别损失
            giou = self.box_giou(pred_boxes, y_true[..., :4])  # [1, 3, 20, 20]
            loss_loc = torch.mean((1 - giou)[y_true[..., 4] == 1])  # 只用正样本anchor计算
            loss_cls = torch.mean(self.BCELoss(pred_cls[y_true[..., 4] == 1],
                                               self.smooth_labels(y_true[..., 5:][y_true[..., 4] == 1],
                                                                  self.label_smoothing,
                                                                  self.num_classes)))  # 只用正样本anchor计算
            loss += loss_loc * self.box_ratio + loss_cls * self.cls_ratio

            #   计算置信度的标签，这里就是将正样本anchor预测框和真实框的giou值作为置信度，giou值越大执行度越大
            #   torch.where(condition, x, y), 若满足条件，则取x中元素 若不满足条件，则取y中元素
            tobj = torch.where(y_true[..., 4] == 1, giou.detach().clamp(0), torch.zeros_like(y_true[..., 4]))
        else:
            # 当前batch数据没有目标，只计算置信度损失
            tobj = torch.zeros_like(y_true[..., 4])  # [1, 3, 20, 20] tobj是物体置信度label
        loss_conf = torch.mean(self.BCELoss(conf, tobj))

        loss += loss_conf * self.balance[l] * self.obj_ratio
        # if n != 0:
        #     print(loss_loc * self.box_ratio, loss_cls * self.cls_ratio, loss_conf * self.balance[l] * self.obj_ratio)
        return loss

    def get_near_points(self, x, y, i, j):
        sub_x = x - i
        sub_y = y - j
        if sub_x > 0.5 and sub_y > 0.5:
            return [[0, 0], [1, 0], [0, 1]]
        elif sub_x < 0.5 and sub_y > 0.5:
            return [[0, 0], [-1, 0], [0, 1]]
        elif sub_x < 0.5 and sub_y < 0.5:
            return [[0, 0], [-1, 0], [0, -1]]
        else:
            return [[0, 0], [1, 0], [0, -1]]

    def get_target(self, l, targets, anchors, in_h, in_w):
        """
        # 获得网络应该有的预测结果, 这里包含正样本匹配的过程
        l: 代表使用的是第几个有效特征层
        targets: 真实框的标签情况 list [batch_size, num_gt, 5]
        anchor: 原图anchor缩放到特征图大小，相对于特征层的
        in_h: 当前特征层的高
        in_w: 当前特征层的宽
        return:
            y_true: 网络应该有的预测结果, [1, 3, 20, 20, 25(5 + num_classes)]，用于后续loss计算
            noobj_mask: 代表无目标的特征点 [1, 3, 20, 20]
        """
        # -----------------------------------------------------#
        #   计算一共有多少张图片
        # -----------------------------------------------------#
        bs = len(targets)
        # -----------------------------------------------------#
        #   用于选取哪些先验框不包含物体
        # -----------------------------------------------------#
        noobj_mask = torch.ones(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad=False)
        # -----------------------------------------------------#
        #   anchors_best_ratio
        # -----------------------------------------------------#
        box_best_ratio = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad=False)
        # -----------------------------------------------------#
        #   batch_size, 3, 13, 13, 5 + num_classes
        # -----------------------------------------------------#
        y_true = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, self.bbox_attrs, requires_grad=False)
        for b in range(bs):
            if len(targets[b]) == 0:
                continue
            batch_target = torch.zeros_like(targets[b])
            # -------------------------------------------------------#
            #   计算出正样本在特征层上的中心点
            # -------------------------------------------------------#
            batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * in_w
            batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * in_h
            batch_target[:, 4] = targets[b][:, 4]
            batch_target = batch_target.cpu()

            # -------------------------------------------------------#
            #   batch_target            : num_true_box, 4
            #   anchors                 : 9, 2
            #
            #   ratios_of_gt_anchors    : num_true_box, 9, 2
            #   ratios_of_anchors_gt    : num_true_box, 9, 2
            #
            #   ratios                  : num_true_box, 9, 4
            #   max_ratios              : num_true_box, 9
            # -------------------------------------------------------#
            ratios_of_gt_anchors = torch.unsqueeze(batch_target[:, 2:4], 1) / torch.unsqueeze(
                torch.FloatTensor(anchors), 0)
            ratios_of_anchors_gt = torch.unsqueeze(torch.FloatTensor(anchors), 0) / torch.unsqueeze(
                batch_target[:, 2:4], 1)
            ratios = torch.cat([ratios_of_gt_anchors, ratios_of_anchors_gt], dim=-1)
            max_ratios, _ = torch.max(ratios, dim=-1)

            for t, ratio in enumerate(max_ratios):
                # -------------------------------------------------------#
                #   ratio : 9
                # -------------------------------------------------------#
                over_threshold = ratio < self.threshold
                over_threshold[torch.argmin(ratio)] = True
                for k, mask in enumerate(self.anchors_mask[l]):
                    if not over_threshold[mask]:
                        continue
                    # ----------------------------------------#
                    #   获得真实框属于哪个网格点
                    # ----------------------------------------#
                    i = torch.floor(batch_target[t, 0]).long()
                    j = torch.floor(batch_target[t, 1]).long()

                    offsets = self.get_near_points(batch_target[t, 0], batch_target[t, 1], i, j)
                    for offset in offsets:
                        local_i = i + offset[0]
                        local_j = j + offset[1]

                        if local_i >= in_w or local_i < 0 or local_j >= in_h or local_j < 0:
                            continue

                        if box_best_ratio[b, k, local_j, local_i] != 0:
                            if box_best_ratio[b, k, local_j, local_i] > ratio[mask]:
                                y_true[b, k, local_j, local_i, :] = 0
                            else:
                                continue

                        # ----------------------------------------#
                        #   取出真实框的种类
                        # ----------------------------------------#
                        c = batch_target[t, 4].long()

                        # ----------------------------------------#
                        #   noobj_mask代表无目标的特征点
                        # ----------------------------------------#
                        noobj_mask[b, k, local_j, local_i] = 0
                        # ----------------------------------------#
                        #   tx、ty代表中心调整参数的真实值
                        # ----------------------------------------#
                        y_true[b, k, local_j, local_i, 0] = batch_target[t, 0]
                        y_true[b, k, local_j, local_i, 1] = batch_target[t, 1]
                        y_true[b, k, local_j, local_i, 2] = batch_target[t, 2]
                        y_true[b, k, local_j, local_i, 3] = batch_target[t, 3]
                        y_true[b, k, local_j, local_i, 4] = 1
                        y_true[b, k, local_j, local_i, c + 5] = 1
                        # ----------------------------------------#
                        #   获得当前先验框最好的比例
                        # ----------------------------------------#
                        box_best_ratio[b, k, local_j, local_i] = ratio[mask]

        return y_true, noobj_mask

    def get_pred_boxes(self, l, x, y, h, w, targets, scaled_anchors, in_h, in_w):
        """
        预测结果解码
        """
        # -----------------------------------------------------#
        #   计算一共有多少张图片
        # -----------------------------------------------------#
        bs = len(targets)

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        # -----------------------------------------------------#
        #   生成网格，先验框中心，网格左上角
        # -----------------------------------------------------#
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(y.shape).type(FloatTensor)

        # 生成先验框的宽高
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]
        anchor_w = FloatTensor(scaled_anchors_l).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors_l).index_select(1, LongTensor([1]))

        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        # -------------------------------------------------------#
        #   计算调整后的先验框中心与宽高
        # -------------------------------------------------------#
        pred_boxes_x = torch.unsqueeze(x * 2. - 0.5 + grid_x, -1)
        pred_boxes_y = torch.unsqueeze(y * 2. - 0.5 + grid_y, -1)
        pred_boxes_w = torch.unsqueeze((w * 2) ** 2 * anchor_w, -1)
        pred_boxes_h = torch.unsqueeze((h * 2) ** 2 * anchor_h, -1)
        pred_boxes = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], dim=-1)
        return pred_boxes



if __name__ == '__main__':
    anchors = np.array(
        [[10., 13.], [16., 30.], [33., 23.], [30., 61.], [62., 45.], [59., 119.], [116., 90.], [156., 198.],
         [373., 326.]])
    num_classes = 20
    input_shape = [640, 640]
    Cuda = True
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    label_smoothing = 0
    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask, label_smoothing)

    y = torch.rand((2, 3 * (5 + 20), 32, 32))
