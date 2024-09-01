# -----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# -----------------------------------------------------------------------#
import argparse
import time
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

"""
单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能,通过指定mode进行模式的修改。
"""


def main(args):
    # 初始化检测模型
    yolo = YOLO(model_path=args.model_path, classes_path=args.classes_path, anchors_path=args.anchors_path,
                anchors_mask=args.anchors_mask, input_shape=args.input_shape, phi=args.phi, confidence=args.confidence,
                nms_iou=args.nms_iou, letterbox_image=args.letterbox_image, cuda=args.cuda)

    if args.mode == "predict":
        '''
        1、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
        2、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        3、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        在原图上利用矩阵的方式进行截取。
        4、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
        比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
        '''
        image = Image.open(args.img_path)
        r_image = yolo.detect_image(image, crop=args.crop)
        r_image.show()

    elif args.mode == "video":
        capture = cv2.VideoCapture(args.video_path)
        if args.video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(args.video_save_path, fourcc, args.video_fps, size)

        fps = 0.0
        while (True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(yolo.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if args.video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if args.video_save_path != "":
            print("Save processed video to the path :" + args.video_save_path)
            out.release()
        cv2.destroyAllWindows()
    elif args.mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(args.dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(args.dir_origin_path, img_name)
                image = Image.open(image_path)
                r_image = yolo.detect_image(image)
                if not os.path.exists(args.dir_save_path):
                    os.makedirs(args.dir_save_path)
                r_image.save(os.path.join(args.dir_save_path, img_name.replace(".jpg", ".png")),
                             quality=95, subsampling=0)

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', or 'dir_predict'.")


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

    # predict表示单张图片预测,video表示视频检测, dir_predict表示遍历文件夹进行检测并保存
    parser.add_argument('--mode', default='predict', help='input mode, predict, video, or dir_predict')

    # crop指定了是否在单张图片预测后对目标进行截取, crop仅在mode='predict'时有效
    parser.add_argument('--img_path', default='./data/VOC2007/JPEGImages/2007_000027.jpg', help='location of image path')
    parser.add_argument('--crop', default='False', help='')

    # video_path用于指定视频的路径,video_save_path表示视频保存的路径，当video_save_path=""时表示不保存
    # video_fps用于保存的视频的fps,video_path、video_save_path和video_fps仅在mode='video'时有效
    parser.add_argument('--video_path', default="test.mp4", help='')
    parser.add_argument('--video_save_path', default='test_result.mp4', help='')
    parser.add_argument('--video_fps', default=25, type=int, help='')

    # dir_origin_path指定了用于检测的图片的文件夹路径,dir_save_path指定了检测完图片的保存路径
    # dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    parser.add_argument('--dir_origin_path', default="./imgs/", help='')
    parser.add_argument('--dir_save_path', default='./result/', help='')

    args = parser.parse_args()
    # 参数转换
    args.anchors_mask = eval(args.anchors_mask)
    args.input_shape = [args.input_shape, args.input_shape]
    args.letterbox_image = eval(args.letterbox_image)
    args.cuda = eval(args.cuda)
    args.crop = eval(args.crop)
    print(args)
    # 调用主函数
    main(args)
