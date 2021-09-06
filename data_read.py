import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random

class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        #初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, "image/*.png"))
        self.lab_path = glob.glob(os.path.join(data_path, "label/*.png"))

    #数据集数量还可以，这里先不进行增强
    # def augment(self, image, flipCode):
    #     # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
    #     flip = cv2.flip(image, flipCode)
    #     return flip

    def __getitem__(self, index):
        #根据索引读取图片
        image_path = self.imgs_path[index]
        #根据image_path生成lable_path
        #实际只替换了图片的名称，代表的是同一张图片

        ###问题出在这里
        label_path = self.lab_path[index]
        #用lable替换image
        #读取训练图片和标签图片(dicom图片，用pydicom去读取)

        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        print(label)
        #将数据（image和lable）转为单通道的图片（转为灰度图片）
        image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label=cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        #shape[0]表示数组的行数，shape[1]表示数组的列数
        #reshape函数将数据变为1(1为通道数)*.shape[0]*.shape[1]的三维矩阵
        image=image.reshape(1, image.shape[0], image.shape[1])
        label=label.reshape(1, label.shape[0], label.shape[1])

        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
        # 随机进行数据增强，为2时不做处理
        # flipCode = random.choice([-1, 0, 1, 2])
        # if flipCode != 2:
        #     image = self.augment(image, flipCode)
        #     label = self.augment(label, flipCode)
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


if __name__ == "__main__":
    #isbi_dataset = ISBI_Loader("D:/pycharm/zhiqian")
    isbi_dataset = ISBI_Loader(r"C:\Users\17286\Desktop\try")
    print("数据个数：", len(isbi_dataset))
    #（batch_size=2,每次训练两个样本）
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=2,
                                               #shuffle将数据置乱
                                               shuffle=True)

    for image, label in train_loader:
        #通过访问矩阵类的成员变量的shape值
        print(image.shape)



