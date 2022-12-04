#训练
import torch
from torch import nn
from net import MyLeNet5
from torch.optim import  lr_scheduler
from torchvision import datasets,transforms
import os


#将数据转化为tensor格式
data_transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载训练数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
# 给训练集创建一个数据加载器, shuffle=True用于打乱数据集，每次都会以不同的顺序返回。
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
# 加载训练数据集
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
# 给训练集创建一个数据加载器, shuffle=True用于打乱数据集，每次都会以不同的顺序返回。
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)


# 如果显卡可用，则用显卡进行训练
device = "cuda" if torch.cuda.is_available() else 'cpu'

#调用net文件的模型，果GPU可用则将模型转到GPU
model = MyLeNet5().to(device)

#定义损失函数，交叉熵损失
loss_fn = nn.CrossEntropyLoss()

#定义优化器SGD,随机梯度下降
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

#学习率每10个epoch变为原来的0.1
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

#定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    loss, current, n = 0.0, 0.0, 0
    # enumerate返回为数据和标签还有批次
    for batch, (X, y) in enumerate(dataloader):
        # 前向传播
        X, y = X.to(device), y.to(device)
        output = model(X)
        cur_loss = loss_fn(output, y)
        # torch.max返回每行最大的概率和最大概率的索引,由于批次是16，所以返回16个概率和索引
        _, pred = torch.max(output, axis=1)

        # 计算每批次的准确率， output.shape[0]为该批次的多少
        cur_acc = torch.sum(y == pred) / output.shape[0]
        # print(cur_acc)
        # 反向传播
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()
        # 取出loss值和精度值
        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1

    print('train_loss:' + str(loss / n))
    print('train_acc:' + str(current / n))


#定义验证函数
def val(dataloader,model,loss_fn):
    # 将模型转为验证模式
    model.eval()
    loss, acc, n = 0.0, 0.0, 0
    # enumerate返回为数据和标签还有批次
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            # 前向传播
            x, y = x.to(device), y.to(device)
            output = model(x)
            cur_loss = loss_fn(output, y)
            # torch.max返回每行最大的概率和最大概率的索引,由于批次是16，所以返回16个概率和索引
            _, pred = torch.max(output, axis=1)

            # 计算每批次的准确率， output.shape[0]为该批次的多少
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            acc += cur_acc.item()#取出单元素张量的元素值并返回该值
            n += 1  # 记录有多少批次
        print('test_loss:' + str(loss / n))
        print('test_acc:' + str(acc / n))

        return acc/n

#开始训练
epoch = 30
max_acc = 0
for t in range(epoch):
    lr_scheduler.step()#学习率调整
    print(f"epoch{t+1}\n-------------------")#加f表示格式化字符串，加f后可以在字符串里面使用用花括号括起来的变量和表达式
    train(train_dataloader, model, loss_fn, optimizer)#调用train函数
    a = val(test_dataloader,model,loss_fn)
    #保存最后的模型权重文件
    if a > max_acc:
        folder = 'save_model'
        if not os.path.exists(folder):
            os.mkdir('save_model')
        max_acc = a
        print('save best model')
        torch.save(model.state_dict(),"save_model/best_model.pth")
    #保存最后的文件
    if t == epoch - 1:
        torch.save(model.state_dict(),"save_model/last_model.pth")
print('Done')

