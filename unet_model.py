import torch.nn.functional as F
from projectdiyi.unet_parts import *

class UNet(nn.Module):
    #n_classes：希望获得的每个像素的概率数，对于一个类和背
    # 景，使用n_classes=1，这里输出的就是黑白对照，（所以使
    # 用1）；（若）n_channels=3是因为输入的图片是RGB 图像，因此是三维
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        #classes最后输出的通道数
        self.n_classes = n_classes
        self.bilinear = bilinear

        #四次下采样
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        #四次上采样
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        #n_classes最后输出的通道数
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        #下采样部分
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        #上采样部分
        #上采样时需要拼接起来
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        #输出预测，这里的大小跟输入是一致的
        return logits

if __name__=='__main__':
    net = UNet(n_channels=3, n_classes=1)
    print(net)


