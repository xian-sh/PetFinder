#from PetFinder.model import *
from PetFinder.models.vgg16 import *
from PetFinder.dataset.dataset import *
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# 准备数据集
# 这里把PetFInder作为根目录，右击该文件夹->Make Directory as...->root
root_dir = "../dataset"
image_dir = "train"
label1_dir = "train_data.csv"
label2_dir = "val_data.csv"
dataset_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # 随机长宽比裁剪为224*224
    transforms.RandomHorizontalFlip(),  # 依概率p=0.5水平翻转
    transforms.ToTensor(),  # 转化为张量并归一化为[0-1]
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
train_dataset = MyData(root_dir, image_dir, label1_dir, transform=dataset_transform)
val_dataset = MyData(root_dir, image_dir, label2_dir, transform=dataset_transform)

# length 长度
train_data_size = len(train_dataset)
val_data_size = len(val_dataset)
# 如果train_data_size=10, 训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
#print("验证数据集的长度为：{}".format(val_data_size))

# 格式转换
# 利用 DataLoader 来加载数据集
batch = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch, drop_last=True)
# 创建网络模型
model = Model()
model.load_state_dict(torch.load('../checkpoint/VGG16_Cats_Dogs_loss1.pth'))
model = model.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
# 优化器

optimizer = torch.optim.SGD(model.classifier.parameters(), lr=0.01, momentum=0.5)

# 保存每个epoch后的Accuracy Loss Val_Accuracy
Accuracy = []
Loss = []
Val_Accuracy = []
BEST_VAL_ACC = np.loadtxt('../checkpoint/vgg_val_acc.txt')
BEST_TR_ACC = np.loadtxt('../checkpoint/vgg_train_acc.txt')
Min_Loss = np.loadtxt('../checkpoint/vgg_loss_acc.txt')
# 训练

writer = SummaryWriter("../logs_loss")
for epoch in range(5):
    print("---------------------第 {} 轮训练开始---------------------".format(epoch + 1))
    since = time.time()
    total_train_loss = 0.
    total_accuracy = 0.
    model.train()
    for i,(imgs, feature, labels) in enumerate(train_dataloader, 0):
        imgs = imgs.to(device)
        feature = feature.to(device)
        labels = labels.to(device)

        # 优化器优化模型
        optimizer.zero_grad()
        outs = model(imgs, feature)
        loss = loss_fn(outs, labels)
        loss.backward()
        optimizer.step()

        #      输出状态
        total_train_loss = total_train_loss + loss.item()
        _, prediction = torch.max(outs, 1)
        total_accuracy += (prediction == labels).sum().item()

        if i % 20 == 19:
            print('epoch {}, iter{}, loss: {:.4f}'.format(epoch+1,i+1, loss.item() / batch))
        writer.add_scalar("train_loss", loss.item(), i)
    Loss.append(total_train_loss / train_data_size)
    Accuracy.append(total_accuracy / train_data_size)
    print("epoch {}在整体训练集上的平均损失: {:.4f}".format(epoch+1,total_train_loss / train_data_size))
    print("epoch {}在整体训练集上的正确率: {:.4f}%".format(epoch+1,100*total_accuracy / train_data_size))
    writer.add_scalar("train_loss", total_train_loss / train_data_size, epoch)
    writer.add_scalar("train_accuracy", total_accuracy / train_data_size, epoch)
    if Loss[epoch] < Min_Loss:
        print('Find Better Model and Saving it...')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(model.state_dict(), '../checkpoint/VGG16_Cats_Dogs_loss1.pth')
        Min_Loss = Loss[epoch]
        ACC = []
        ACC.append(Min_Loss)
        np.savetxt('../checkpoint/vgg_loss_acc.txt',ACC ,fmt='%.04f')
        print('Saved!')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Now the minimum loss is {:.4f}'.format(Min_Loss))

writer.close()




