import torch
from torch import nn
from torchvision import transforms,datasets
from torch import optim
from torch.optim import lr_scheduler
from net import vgg
import os
import sys
import json
from torch.utils.data import DataLoader
from tqdm import tqdm#用于画进度条
import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 如果显卡可用，则用显卡进行训练
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("using {} device".format(device))
print(device)

data_transform = {
    "train":transforms.Compose([
        transforms.RandomResizedCrop(224),#随机裁剪
        transforms.RandomVerticalFlip(),#随机垂直翻转
        transforms.ToTensor(),#转换为tensor格式
        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))#RGB三通道
                                ]),
    "val":transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])
}
#数据集路径
ROOT_TRAIN = 'data/train'
ROOT_TEST = 'data/val'

batch_size = 16

train_dataset = datasets.ImageFolder(ROOT_TRAIN,transform=data_transform["train"])
val_dataset = datasets.ImageFolder(ROOT_TEST,transform=data_transform["val"])

train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
val_dataloader = DataLoader(val_dataset,batch_size=batch_size,shuffle=True)

train_num = len(train_dataset)#计数
val_num = len(val_dataset)
print("using {} images for training, {} images for validation.".format(train_num,val_num))

#将#{'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}键值对值反转，并保存
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)#保存json文件(好处，方便转换为其它类型数据)用于预测用

model_name = "vgg16"
model = vgg(model_name,num_classes=5,init_weights=True)

# 加载预训练模型
model_weights_path = './vgg16.pth'
ckpt = torch.load(model_weights_path)
ckpt.pop('classifier.6.weight')
ckpt.pop('classifier.6.bias')
missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)

model.to(device)

loss_fn = nn.CrossEntropyLoss()
#定义优化器
optimizer = optim.SGD(model.parameters(),lr=0.003)
#学习率每隔10epoch变为原来的0.1
lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)

#定义训练函数
def train(dataloader,model,loss_fn,optimizer):
    model.train()
    loss,current,n = 0.0,0.0,0
    train_bar = tqdm(dataloader, file=sys.stdout)  # 输出方式，默认为sys.stder
    for batch,(x,y) in enumerate(train_bar):
        #前向传播
        image,y = x.to(device),y.to(device)
        output = model(image)
        cur_loss = loss_fn(output,y)
        _,pred = torch.max(output,axis=-1)
        cur_acc = torch.sum(y==pred)/output.shape[0]
        #反向传播
        optimizer.zero_grad()#梯度清零
        cur_loss.backward()
        optimizer.step()
        loss += cur_loss
        current += cur_acc
        n += 1
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(i + 1, epoch, cur_loss)
    train_loss = loss / n
    train_acc = current / n
    print(f"tran_loss:{train_loss}")
    print(f"tran_acc:{train_acc}")
    return train_loss,train_acc

def val(dataloader,model,loss_fn):
    #验证模式
    model.eval()
    loss, current,n = 0.0, 0.0,0
    with torch.no_grad():
        val_bar = tqdm(dataloader, file=sys.stdout)
        for batch, (x, y) in enumerate(val_bar):
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
        print(f"val_loss:{val_loss}")
        print(f"val_acc:{val_acc}")
        return val_loss,val_acc

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

epoch = 1

max_acc = 0

for i in range(epoch):
    lr_scheduler.step()#学习率迭代，10epoch变为原来的0.1
    train_loss,train_acc=train(train_dataloader,model,loss_fn,optimizer)
    val_loss,val_acc=val(val_dataloader,model,loss_fn)

    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)
    # 保存最好的模型权重
    if val_acc > max_acc:
        folder = 'save_model'
        if not os.path.exists(folder):
            os.mkdir('save_model')
        max_acc = val_acc
        print(f'save best model,第{i + 1}轮')
        torch.save(model.state_dict(), 'save_model/best_model.pth')  # 保存网络权重
    # 保存最后一轮
    if i == epoch - 1:
        torch.save(model.state_dict(), 'save_model/last_model.pth')  # 保存
print("done")

# 画图
matplot_loss(train_loss_list, val_loss_list)
matplot_acc(train_acc_list, val_acc_list)
