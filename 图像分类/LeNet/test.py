#测试
import torch
from net import MyLeNet5
from torch.autograd import Variable
from torchvision import datasets,transforms
from torchvision.transforms import ToPILImage

# 将数据转化为tensor格式
data_transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载训练数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
# 给训练集创建一个数据加载器, shuffle=True用于打乱数据集，每次都会以不同的顺序返回。
#train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
# 加载训练数据集
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
# 给训练集创建一个数据加载器, shuffle=True用于打乱数据集，每次都会以不同的顺序返回。
#test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

#  如果显卡可用，则用显卡进行训练
device = "cuda" if torch.cuda.is_available() else 'cpu'

# 调用net里面定义的模型，如果GPU可用则将模型转到GPU
model = MyLeNet5().to(device)

#加载train.py里训练好的模型
model.load_state_dict(torch.load(("D:/python/LeNet-5/save_model/best_model.pth")))#填写权重路径

#获取预测结果

classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

# 把tensor转换成Image，方便可视化
show = ToPILImage()

#进入验证阶段
model.eval()
# 对test_dataset手写数字图片进行推理
for i in range(10):
    x,y = test_dataset[i][0],test_dataset[i][1]
    #可视化
    #show(x).show()
    # 扩展张量维度为4维
    x = Variable(torch.unsqueeze(x,dim=0).float(),requires_grad=False).to(device)
    with torch.no_grad():
        pred = model(x)
        # 得到预测类别中最高的那一类，再把最高的这一类对应的标签输出
        predicted,actual = classes[torch.argmax(pred[0])],classes[y]
        print(f'predicted:"{predicted},actual:{actual}"')
