import os
import sys
import json
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from model import resnet34

def main():
    # 如果显卡可用，则用显卡进行训练
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("using {} device".format(device))
    print(torch.cuda.get_device_name(0))

    data_transform = {
        "train":transforms.Compose([
                                   transforms.RandomResizedCrop(224),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ]),
        "val":transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    }

    #数据集路径
    ROOT_TRAIN = 'data/train'
    ROOT_TEST = 'data/val'

    batch_size = 16
    #加载数据集并处理
    train_dataset = datasets.ImageFolder(ROOT_TRAIN,transform=data_transform["train"])
    val_dataset = datasets.ImageFolder(ROOT_TEST,transform=data_transform["val"])
    # 划成一批批乱序数据集
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_dataloader = DataLoader(val_dataset,batch_size=batch_size,shuffle=True)
    #计算数据数量
    train_num = len(train_dataset)
    val_num = len(val_dataset)
    print("using {} images for training,{} images for validation.".format(train_num,val_num))

    #将{'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}键值对值反转，并保存
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val,key) for key,val in flower_list.items())
    #将键值对写入json文件
    json_str = json.dumps(cla_dict,indent=4)
    with open('class_indices.json','w')as json_file:
        json_file.write(json_str)#保存json文件(好处，方便转换为其它类型数据)用于预测用

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    model = resnet34()
    #加载预训练权重
    model_weight_path = "save_model/best_model.pth"
    assert os.path.exists(model_weight_path),"file {} does not exist.".format(model_weight_path)
    model.load_state_dict(torch.load(model_weight_path,map_location='cpu'))

    #change fc layer structure
    # in_channel = model.fc.in_features
    # model.fc = nn.Linear(in_channel,5)
    model.to(device)

    #损失函数
    loss_function = nn.CrossEntropyLoss()
    #优化器
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    # 学习率每隔10epoch变为原来的0.1
    lr_s = lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)

    #定义训练函数
    def train(dataloader,model,loss_fn,optimizer):
        model.train()
        loss,acc,n = 0.0,0.0,0
        train_bar = tqdm(dataloader,file=sys.stdout)
        for batch,(x,y) in enumerate(train_bar):
            #前向传播
            x,y = x.to(device),y.to(device)
            output = model(x)
            cur_loss = loss_fn(output,y)
            _,pred = torch.max(output,axis=-1)
            cur_acc = torch.sum(y==pred)/output.shape[0]
            #反向传播
            optimizer.zero_grad()#梯度清零
            cur_loss.backward()
            optimizer.step()
            loss += cur_loss.item()
            acc += cur_acc.item()
            n += 1
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(i+1,epoch,cur_loss)
        train_loss = loss / n
        train_acc = acc / n

        print(f"train_loss:{train_loss}")
        print(f"train_acc:{train_acc}")
        return train_loss,train_acc

    #定义验证函数
    def val(dataloader,model,loss_fn):
        model.eval()
        loss,acc,n = 0.0,0.0,0
        val_bar = tqdm(dataloader,file=sys.stdout)
        for batch,(x,y) in enumerate(val_bar):
            #前向传播
            x,y = x.to(device),y.to(device)
            output = model(x)
            cur_loss = loss_fn(output,y)
            _,pred = torch.max(output,axis=-1)
            cur_acc = torch.sum(y==pred)/output.shape[0]
            loss += cur_loss.item()
            acc += cur_acc.item()
            n += 1
            val_bar.desc = "val epoch[{}/{}] loss:{:.3f}".format(i+1,epoch,cur_loss)
        val_loss = loss / n
        val_acc = acc / n

        print(f"val_loss:{val_loss}")
        print(f"val_acc:{val_acc}")
        return val_loss,val_acc

    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 画图函数
    def matplot_loss(train_loss, val_loss):
        plt.figure()  # 声明一个新画布，这样两张图像的结果就不会出现重叠
        plt.plot(train_loss, label='train_loss')  # 画图
        plt.plot(val_loss, label='val_loss')
        plt.legend(loc='best')  # 图例
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylabel('loss', fontsize=12)
        plt.xlabel('epoch', fontsize=12)
        plt.title("训练集和验证集loss对比图")
        folder = 'result'
        if not os.path.exists(folder):
            os.mkdir('result')
        plt.savefig('result/loss.jpg')

    def matplot_acc(train_acc, val_acc):
        plt.figure()  # 声明一个新画布，这样两张图像的结果就不会出现重叠
        plt.plot(train_acc, label='train_acc')  # 画图
        plt.plot(val_acc, label='val_acc')
        plt.legend(loc='best')  # 图例
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylabel('acc', fontsize=12)
        plt.xlabel('epoch', fontsize=12)
        plt.title("训练集和验证集acc对比图")
        plt.savefig('result/acc.jpg')

    #开始训练
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    epoch = 5
    max_acc = 0

    wandb.init(project='ResNet',name='resnet34.1')

    for i in range(epoch):
        lr_s.step()
        train_loss,train_acc=train(train_dataloader,model,loss_function,optimizer)
        wandb.log({'train_loss': train_loss, 'train_acc': train_acc})
        val_loss,val_acc=val(val_dataloader,model,loss_function)
        wandb.log({'val_loss': val_loss, 'val_acc': val_acc})

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        #保存最好的模型权重
        if val_acc > max_acc:
            folder = 'save_model'
            if not os.path.exists(folder):
                os.mkdir('save_model')
            max_acc = val_acc
            print('save best model')
            torch.save(model.state_dict(), "save_model/best_model.pth")
        # 保存最后一轮
        # if i == epoch - 1:
        #     torch.save(model.state_dict(), 'save_model/last_model.pth')

    print("Finished Training")
    #画图
    # matplot_loss(train_loss_list,val_loss_list)
    # matplot_acc(train_acc_list,val_acc_list)

if __name__=='__main__':
    main()





