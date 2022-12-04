## 该文件夹是用来存放训练数据的目录
### 使用步骤如下：
* （1）点击链接下载花分类数据集 [https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)
* （2）解压得到数据集到flower_photos文件，总共5类，5个文件夹
* （4）执行"huafen.py"脚本自动将数据集划分成训练集train和验证集val并保存在data文件夹下,可以自己设置划分比例    

```
划分后的数据，默认9:1
├── data（解压的数据集文件夹，3670个样本）    
    ├── train（生成的训练集，3306个样本）  
    └── val（生成的验证集，364个样本） 
```
