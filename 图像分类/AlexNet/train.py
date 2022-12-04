#修改后加进度条的代码
import json
import torch
from torch import nn
from NET import MyAlexNet
import numpy as np

from tqdm import tqdm#用于画进度条

from torch.optim import lr_scheduler

import os
import sys

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 如果显卡可用，则用显卡进行训练
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("using {} device".format(device))


# 将图像RGB三个通道的像素值分别减去0.5,再除以0.5.从而将所有的像素值固定在[-1,1]范围内
#normalize = transforms.Normalize(std=[0.5,0.5,0.5],mean=[0.5,0.5,0.5])#image=(image-mean)/std
data_transform = {
    "train":transforms.Compose([
            transforms.Resize((224,224)),#裁剪为224*224
            transforms.RandomVerticalFlip(),#随机垂直旋转
            transforms.ToTensor(),#将0-255范围内的像素转为0-1范围内的tensor
            transforms.Normalize(std=[0.5,0.5,0.5],mean=[0.5,0.5,0.5])#归一化
]),
    "val":transforms.Compose([
          transforms.Resize((224,224)),#裁剪为224*224
          transforms.ToTensor(),#将0-255范围内的像素转为0-1范围内的tensor
          transforms.Normalize(std=[0.5,0.5,0.5],mean=[0.5,0.5,0.5])#归一化
])}

#数据集路径
ROOT_TRAIN = 'data/train'
ROOT_TEST = 'data/val'

batch_size = 16

train_dataset = ImageFolder(ROOT_TRAIN,transform=data_transform["train"])#ImageFolder()根据文件夹名来对图像添加标签
val_dataset = ImageFolder(ROOT_TEST,transform=data_transform["val"])#可以利用print(val_dataset.imgs)对象查看,返回列表形式('data/val\\cat\\110.jpg', 0)
#print(val_dataset.imgs)

# nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
# print('Using {} dataloader workers every process'.format(nw))

train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
val_dataloader = DataLoader(val_dataset,batch_size=batch_size,shuffle=True)

flow_list = train_dataset.class_to_idx#转换维字典,train_dataset里有这个对象
# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
cla_dict = dict((val,key) for key,val in flow_list.items())#键值对转换
#{0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)  # 保存json文件(好处，方便转换为其它类型数据)用于预测用

train_num = len(train_dataset)
val_num = len(val_dataset)
print("using {} images for training, {} images for validation.".format(train_num,val_num))

# 调用net里面的定义的网络模型， 如果GPU可用则将模型转到GPU
model = MyAlexNet(num_classes=5).to(device)

#加载预训练模型
# weights_path = "save_model/best_model.pth"
# assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
# missing_keys, unexpected_keys = net.load_state_dict(torch.load(weights_path,),strict=False)

#定义损失函数
loss_fn = nn.CrossEntropyLoss()

#定义优化器
optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9)#googlenet用的是adam
# 学习率每隔10epoch变为原来的0.5
lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)

#定义训练函数
def train(dataloader,model,loss_fn,optimizer,i,epoch):
    model.train()
    loss,current,n = 0.0,0.0,0
    train_bar = tqdm(dataloader,file=sys.stdout)#输出方式，默认为sys.stderr
    for batch,(x,y) in enumerate(train_bar):#enumerate()默认两个参数，第一个用于记录序号，默认0开始，第二个参数(x,y)才是需要遍历元素(dataloder)的值
        #前向传播
        image,y = x.to(device),y.to(device)
        output = model(image)
        cur_loss = loss_fn(output,y)
        _,pred = torch.max(output,axis=-1)
        cur_acc = torch.sum(y==pred)/output.shape[0]
        #反向传播
        optimizer.zero_grad()#梯度归零
        cur_loss.backward()
        optimizer.step()
        loss += cur_loss
        current += cur_acc
        n += 1
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(i + 1, epoch, cur_loss)
    train_loss = loss / n
    train_acc = current / n
    print(f'train_loss:{train_loss}')
    print(f'train_acc:{train_acc}')
    return  train_loss,train_acc

#定义验证函数
def val(dataloader,model,loss_fn,i,epcho):
    #转换为验证模型
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
        val_bar = tqdm(dataloader,file=sys.stdout)
        for batch, (x, y) in enumerate(val_bar):  # enumerate()默认两个参数，第一个用于记录序号，默认0开始，第二个参数(x,y)才是需要遍历元素(dataloder)的值
            # 前向传播
            image, y = x.to(device), y.to(device)
            output = model(image)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=-1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss
            current += cur_acc
            n += 1
            val_bar.desc = "val epoch[{}/{}] loss:{:.3f}".format(i + 1, epoch, cur_loss)
        val_loss = loss / n
        val_acc = current / n
        print(f'val_loss:{val_loss}')
        print(f'val_acc:{val_acc}')
        return val_loss, val_acc

#画图函数
def matplot_loss(train_loss,val_loss):
    plt.figure()  # 声明一个新画布，这样两张图像的结果就不会出现重叠
    plt.plot(train_loss,label='train_loss')#画图
    plt.plot(val_loss, label='val_loss')
    plt.legend(loc='best')#图例
    plt.ylabel('loss',fontsize=12)
    plt.xlabel('epoch',fontsize=12)
    plt.title("训练集和验证集loss对比图")
    folder = 'result'
    if not os.path.exists(folder):
        os.mkdir('result')
    plt.savefig('result/loss.jpg')

def matplot_acc(train_acc,val_acc):
    plt.figure()  # 声明一个新画布，这样两张图像的结果就不会出现重叠
    plt.plot(train_acc, label='train_acc')  # 画图
    plt.plot(val_acc, label='val_acc')
    plt.legend(loc='best')  # 图例
    plt.ylabel('acc', fontsize=12)
    plt.xlabel('epoch', fontsize=12)
    plt.title("训练集和验证集acc对比图")
    plt.savefig('result/acc.jpg')

#开始训练
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []

epoch = 3
max_acc = 0

for i in range(epoch):
    lr_scheduler.step()#学习率迭代，10epoch变为原来的0.5
    train_loss,train_acc = train(train_dataloader,model,loss_fn,optimizer,i,epoch)
    val_loss,val_acc = val(val_dataloader,model,loss_fn,i,epoch)

    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)
    #保存最好的模型权重
    if val_acc >max_acc:
        folder = 'save_model'
        if not os.path.exists(folder):
            os.mkdir('save_model')
        max_acc = val_acc
        print(f'save best model,第{i+1}轮')
        torch.save(model.state_dict(),'save_model/best_model.pth')#保存
    #保存最后一轮
    if i == epoch - 1:
        torch.save(model.state_dict(), 'save_model/last_model.pth')  # 保存
print("done")

#画图
matplot_loss(train_loss_list,val_loss_list)
matplot_acc(train_acc_list,val_acc_list)
