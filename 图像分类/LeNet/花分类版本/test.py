import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from net import MyLeNet5

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])
    #load image
    img_path = "data/val/dandelion/2522454811_f87af57d8b.jpg"
    assert os.path.exists(img_path),"file:'{}' dose not exist. ".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)

    #[N, C, H, W]归一化
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img,dim=0)


    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path,"r") as f:
        class_indict = json.load(f)

    #实例化模型
    model = MyLeNet5(num_classes=5).to(device)

    #加载权重
    weights_path = "save_model/best_model.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    missing_keys,unexpected_keys = model.load_state_dict(torch.load(weights_path,map_location=device),
                                                         strict=False)
    model.eval()
    with torch.no_grad():
        #预测
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    #最大概率结果
    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    #前10个最大概率的结果
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()
if __name__=="__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()

