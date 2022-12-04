import json
import torch
from torch import nn
from torchvision import transforms,datasets,utils
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm#用于画进度条
from model import GoogLeNet
import matplotlib.pyplot as plt
import os
import sys



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    #训练集以及测试集路径
    ROOT_TRAIN = 'data/train'
    ROOT_TEST = 'data/val'

    batch_size = 16

    train_dataset = datasets.ImageFolder(root=ROOT_TRAIN,transform=data_transform["train"])
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

    train_num = len(train_dataset)

    flow_list = train_dataset.class_to_idx#转换维字典,train_dataset里有这个对象
    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    cla_dict = dict((val,key) for key,val in flow_list.items())#键值对转换
    #{0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)  # 保存json文件(好处，方便转换为其它类型数据)用于预测用

    val_dataset = datasets.ImageFolder(root=ROOT_TEST,transform=data_transform["val"])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    val_num = len(val_dataset)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    net = GoogLeNet(num_classes=5,aux_logits=True,init_weights=True)

    #加载预训练模型
    # weights_path = "save_model/best_model.pth"
    # assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    # missing_keys, unexpected_keys = net.load_state_dict(torch.load(weights_path,),strict=False)

    net.to(device)
    loss_fc = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0003)
    # 学习率每隔10epoch变为原来的0.1
    lr_s = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    #定义训练函数
    def train(dataloader, net, loss_fn, optimizer,i,epoch):
        net.train()
        loss, current,n = 0.0, 0.0,0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for batch, (x, y) in enumerate(train_bar):
            # 前向传播
            image, y = x.to(device), y.to(device)
            logits,aux_logits1,aux_logits2 = net(image)
            loss0 = loss_fn(logits,y)
            loss1 = loss_fn(aux_logits1,y)
            loss2 = loss_fn(aux_logits2,y)
            cur_loss = loss0 + loss1*0.3 + loss2*0.3#在论文中辅助分类器权重为0.3
            _, pred = torch.max(logits, axis=-1)
            cur_acc = torch.sum(y == pred) / logits.shape[0]
            # 反向传播
            optimizer.zero_grad()  # 梯度清零
            cur_loss.backward()
            optimizer.step()
            loss += cur_loss
            current += cur_acc
            n +=1
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(i + 1, epoch, cur_loss)
        train_loss = loss / n
        train_acc = current / n

        print(f"tran_loss:{train_loss}")
        print(f"tran_acc:{train_acc}")
        return train_loss, train_acc

    def val(dataloader,net,loss_fn):
        #验证模式
        net.eval()
        loss, current ,n = 0.0, 0.0,0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for batch,(x,y) in enumerate(val_bar):
                #前向传播
                image,y = x.to(device),y.to(device)
                output = net(image)
                cur_loss = loss_fn(output,y)
                _,pred = torch.max(output,axis=-1)
                cur_acc = torch.sum(y==pred)/output.shape[0]
                loss += cur_loss
                current += cur_acc
                val_bar.desc = "val epoch[{}/{}] loss:{:.3f}".format(i + 1, epoch, cur_loss)
                n +=1
            val_loss = loss / n
            val_acc = current / n
            print(f"val_loss:{val_loss}")
            print(f"val_acc:{val_acc}")
            return val_loss,val_acc

    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 画图函数
    def matplot_loss(train_loss, val_loss):
        plt.figure()
        plt.plot(train_loss, label='train_loss')  # 画图
        plt.plot(val_loss, label='val_loss')
        plt.legend(loc='best')  # 图例
        plt.ylabel('loss', fontsize=12)
        plt.xlabel('epoch', fontsize=12)
        plt.title("训练集和验证集loss对比图")
        folder = 'result'
    if not os.path.exists(folder):
        os.mkdir('result') 
        plt.savefig('result/loss.jpg')

    def matplot_acc(train_acc, val_acc):
        plt.figure()#声明一个新画布，这样两张图像的结果就不会出现重叠
        plt.plot(train_acc, label='train_acc')  # 画图
        plt.plot(val_acc, label='val_acc')
        plt.legend(loc='best')  # 图例
        plt.ylabel('acc', fontsize=12)
        plt.xlabel('epoch', fontsize=12)
        plt.title("训练集和验证集acc对比图")
        plt.savefig('result/acc.jpg')

    # 开始训练
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    epoch = 60
    max_acc = 0

    for i in range(epoch):
        lr_s.step()#学习率优化，10epoch变为原来的0.5

        train_loss,train_acc = train(train_loader,net,loss_fc,optimizer,i,epoch)

        val_loss,val_acc = val(val_loader,net,loss_fc)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)

        # 保存最好的模型权重
        if val_acc > max_acc:
            folder = 'save_model'
            if not os.path.exists(folder):
                os.mkdir('save_model')
            max_acc = val_acc
            print(f'save best model,第{i + 1}轮')
            torch.save(net.state_dict(), 'save_model/best_model.pth')  # 保存网络权重
        # 保存最后一轮
        if i == epoch - 1:
            torch.save(net.state_dict(), 'save_model/last_model.pth')  # 保存

    print("done")

    #画图
    matplot_loss(train_loss_list,val_loss_list)
    matplot_acc(train_acc_list,val_acc_list)
if __name__=="__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    main()

