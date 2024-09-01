from PIL import Image
from torch.utils.data import DataLoader
from dataset.yolo_dataset import YoloDataset, yolo_dataset_collate

from yolo import YOLO


# 测试数据加载
def test_dataset():
    with open("./2007_train.txt", encoding='utf-8') as f:
        train_lines = f.readlines()
    train_dataset = YoloDataset(train_lines, [640, 640], 20, epoch_length=100, mosaic=True,
                                train=True)
    gen = DataLoader(train_dataset, shuffle=True, batch_size=4, num_workers=2, pin_memory=True,
                     drop_last=True, collate_fn=yolo_dataset_collate)
    for iteration, batch in enumerate(gen):
        images, targets = batch[0], batch[1]
        print(type(images), images.shape)
        print(len(targets), targets)
        break


def test_model():
    yolo = YOLO(model_path="./model_data/yolov5_s.pth")
    image = Image.open("./data/VOC2007/JPEGImages/2007_000027.jpg")
    r_image = yolo.detect_image(image, crop=False)
    r_image.show()


if __name__ == '__main__':
    # test_dataset()
    test_model()
