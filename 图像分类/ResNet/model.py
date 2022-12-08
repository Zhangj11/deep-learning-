import torch.nn as nn
import torch

class BasicBlock(nn.Module):
    #对应18层和34层的残差块
    expansion = 1
    def __init__(self,in_channel,out_channel,stride=1,downsample=None,**kwargs):
        super(BasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,
                               kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel,out_channels=out_channel,
                               kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self,x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out +=identity#跨层连接
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    #适用于50，101，152层的
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4
    def __int__(self,in_channel,out_channel,stride=1,downsample=None):
        super(Bottleneck,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,
                               kernel_size=1,stride=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel,out_channels=out_channel,
                               kernel_size=3,stride=stride,bias=False,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel,out_channels=out_channel*self.expansion,
                               kernel_size=1,stride=1,bias=False)#扩展维度
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)#inplace = True ,会改变输入数据的值,节省反复申请与释放内存的空间与时间,只是将原来的地址传递,效率更好
        self.downsample = downsample

    def forward(self,x):
        identity = x #跨层连接的x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self,block,blocks_num,num_classes=1000,include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64


        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            '初始化权重'
            if isinstance(m,nn.Conv2d):
                '随机矩阵显式创建权重'
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')


    def _make_layer(self,block,channel,block_num,stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:#表示层数是大于50的
            '''
            表示虚线的残差结构，需要进行维度扩展，一般是每一层的第一个残差结构
            第一层(conv2_x)的虚线残差结构只需要扩展维度
            而后面层的虚线残差结构还需要下采样将图像大小缩小一般
            '''
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        '放入第一块残差结构'
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride))
        self.in_channel = channel * block.expansion

        '放入剩余的残差块'
        for _ in range(1, block_num):
        #实线残差结构，不需要维度扩展
            layers.append(block(self.in_channel,
                                channel))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

def resnet34(num_classes=1000,include_top=True):
    '用于18，34层'
    return ResNet(BasicBlock,[3,4,6,3],num_classes=num_classes,include_top=include_top)


def resnet101(num_classes=1000,include_top=True):
    '用于50，101，152层，只需要将括号的数字改了即可'
    return ResNet(BasicBlock,[3,4,23,3],num_classes=num_classes,include_top=include_top)

if __name__=="__main__":
    #没有固定的输入大小，因为有自适应池化层，但这里统一用输入为224*224
    x = torch.rand([1, 3, 224, 224])
    model = resnet34(num_classes=5)
    y = model(x)
    print(y)

    #统计模型参数
    sum = 0
    for name, param in model.named_parameters():
        num = 1
        for size in param.shape:
            num *= size
        sum += num
        #print("{:30s} : {}".format(name, param.shape))
    print("total param num {}".format(sum))#total param num 21,287,237






