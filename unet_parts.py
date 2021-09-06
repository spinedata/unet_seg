import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):#（两次卷积）
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            #nn.Conv2d的功能是：对由多个输入平面组成的输入信号进行二维卷积
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            #batchmorm2d做归一化处理
            nn.BatchNorm2d(out_channels),
            #inplace选择是否进行覆盖运算，是否将计算得到的新值覆盖之前原来的值
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #使用卷积进行2倍下采样，通道数不变
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self,x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # 上采样，特征图扩大两倍，通道数减半
        if bilinear:
            #采用双线性插值
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            #采用反卷积
            self.up = nn.ConvTranspose2d(in_channels//2, in_channels//2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        #x1接收上采样数据，x2接收特征融合数据
        x1 = self.up(x1)
        #，skip_connection两边的图像大小不一样，根据要求，需要将decoder部分
        #扩展到encoder部分的大小。
        diffY = x2.size()[2]-x1.size()[2]#第一个参数代表下采样
        #中图片的height，第二个参数代表目标图片的大小。
        diffX = x2.size()[3]-x1.size()[3]#第一个参数代表下采样
        #中图片的weight，第二个参数代表目标图片的大小。

        #需扩展的像素（暂时这么说，对上采样的像素进行扩展，扩展到下采样图像的大小）
        #[ , , , ]四个参数分别是左边填充数，右边填充数，上边填充数与下边填充数
        x1 = F.pad(x1, [diffX//2, diffX-diffX//2,
                        diffY//2, diffY-diffY//2])
        #拼接，当前采样的，和之前下采样的
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):#最后一步，输出
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        #最后输出卷积核大小为1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    #forward方法接受张量作为输入，返回张量作为输出
    def forward(self, x):
        return self.conv(x)


