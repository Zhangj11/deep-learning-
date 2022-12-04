#网络模型框架
import torch
from torch import nn

#定义一个网络模型类
class MyLeNet5(nn.Module):
    #初始化网络
    def __init__(self):
        super(MyLeNet5,self).__init__()
        #输入大小为32*32，输出大小为28*28,输入通道为1，输出为6，卷积核大小为5,步长为1
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        #sigmoid激活函数
        self.Sigmoid= nn.Sigmoid()
        #平均池化
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.c5 = nn.Conv2d(in_channels=16,out_channels=120,kernel_size=5)
        #展开
        self.flatten = nn.Flatten()
        self.f6 = nn.Linear(120,84)
        self.output = nn.Linear(84,10)

    def forward(self,x):
        #输入x为32*32*1,输出为28*28*6
        x = self.Sigmoid(self.c1(x))
        #输入为28*28*6,输出为14*14*6
        x = self.s2(x)
        # 输入为14*14*6,输出为10*10*16
        x = self.Sigmoid(self.c3(x))
        # 输入为10*10*16,输出为5*5*16
        x = self.s4(x)
        # 输入为5*5*16,输出为1*1*120
        x = self.c5(x)
        x = self.flatten(x)
        # 输入为120，输出为84
        x = self.f6(x)
        # 输入为84，输出为10
        x = self.output(x)
        return x

if __name__=="__main__":
    x = torch.rand([1,1,28,28])#任意产生一个张量,批次1，通道为1，大小为28*28
    model = MyLeNet5()#网络实例化
    y = model(x) #输出结果