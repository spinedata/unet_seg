import glob
import numpy as np
import torch
import os
import cv2
from projectdiyi.unet_model import UNet

if __name__=="__main__":
    #选择设备，有cuda用cuda，没有cuda用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #加载网络，图片单通道，分类为1.
    net = UNet(n_channels=1, n_classes=1)
    #将网络拷贝到device中
    net.to(device=device)
    #加载模型参数
    net.load_state_dict(torch.load('best_model.pth', map_location=device))
    #测试模式
    net.eval()
    #print("#")
    #读取所有图片路径
    tests_path = glob.glob('D:/pycharm/projectdiyi/test/*.png')
    #result_path =
    #遍历所有图片
    for test_path in tests_path:
        #保存结果地址
        save_res_path = test_path.split('.')[0]+'res.png'

        #读取图片
        img = cv2.imread(test_path)
        #转化为灰度
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #转为batch为1，通道为1，大小为512*512的数组,img.shape[0]为图像的长
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        #将numpy数组转为tensor
        img_tensor = torch.from_numpy(img)
        #将tensor拷贝到device中，只用cpu就拷贝到cpu中，用cuda就拷贝到cuda中
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        #预测
        pred = net(img_tensor)
        #提取结果
        pred =np.array(pred.data.cpu()[0][0])
        #处理结果

        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        #保存图片

        cv2.imwrite(save_res_path, pred)

