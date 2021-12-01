from PetFinder.models.resnet50 import *
from PetFinder.dataset.dataset import *
from torchvision import transforms
import torch
from torch import nn
from torch.utils.data import DataLoader
import time
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
print("验证数据集的长度为：{}".format(val_data_size))

# 格式转换
# 利用 DataLoader 来加载数据集
batch = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch, drop_last=True)
# 创建网络模型
model = Model()
#model.load_state_dict(torch.load('../checkpoint/ResNet_Cats_Dogs_val.pth'))
model = model.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
# 优化器
optimizer = torch.optim.SGD(model.add_block.parameters(), lr=0.01, momentum=0.5)

# 保存每个epoch后的Accuracy Loss Val_Accuracy
Accuracy = []
Loss = []
Val_Accuracy = []
BEST_VAL_ACC = np.loadtxt('../checkpoint/resnet_val_acc.txt')
BEST_TR_ACC = np.loadtxt('../checkpoint/resnet_train_acc.txt')

# 训练


for epoch in range(200):
    since = time.time()
    train_loss = 0.
    train_accuracy = 0.
    run_accuracy = 0.
    run_loss = 0.
    total = 0.
    model.train()
    for i,(imgs, feature, labels) in enumerate(train_dataloader, 0):
        imgs = imgs.to(device)
        feature = feature.to(device)
        labels = labels.to(device)

        # 优化器优化模型
        optimizer.zero_grad()
        outs = model(imgs, feature)
        loss = loss_fn(outs, labels.long())
        loss.backward()
        optimizer.step()

        #      输出状态
        total += labels.size(0)
        run_loss += loss.item()
        _, prediction = torch.max(outs, 1)
        run_accuracy += (prediction == labels).sum().item()

        if i % 20 == 19:
            print('epoch {}, iter{},train accuracy:{:4f}% loss: {:.4f}'.format(epoch, i + 1, 100 * run_accuracy / (
                        labels.size(0) * 20), run_loss / 20))
            train_accuracy += run_accuracy
            train_loss += run_loss
            run_accuracy, run_loss = 0., 0.

    Loss.append(train_loss / total)
    Accuracy.append(100 * train_accuracy / total)
    # 这部分只根据训练集的准确度决定保留训练模型（容易过拟合）
    # if Accuracy[epoch] > BEST_TR_ACC:
    #     print('Find Better Model and Saving it...')
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(model.state_dict(), '../checkpoint/VGG16_Cats_Dogs.pth')
    #     BEST_TR_ACC = Accuracy[epoch]
    #     ACC = []
    #     ACC.append(BEST_TR_ACC)
    #     np.savetxt('../checkpoint/train_acc.txt',ACC ,fmt='%.04f')
    #     print('Saved!')
    # time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Now the best train Acc is {:.4f}%'.format(BEST_TR_ACC))
    # # 可视化训练过程（这部分先不要加上，碍事）
    # fig1, ax1 = plt.subplots(figsize=(11, 8))
    # ax1.plot(range(0, epoch + 1, 1), Accuracy)
    # ax1.set_title("Average trainset accuracy vs epochs")
    # ax1.set_xlabel("Epoch")
    # ax1.set_ylabel("Avg. train. accuracy")
    # plt.savefig('Train_accuracy_vs_epochs.png')
    # plt.clf()
    # plt.close()
    #
    # fig2, ax2 = plt.subplots(figsize=(11, 8))
    # ax2.plot(range(epoch + 1), Loss)
    # ax2.set_title("Average trainset loss vs epochs")
    # ax2.set_xlabel("Epoch")
    # ax2.set_ylabel("Current loss")
    # plt.savefig('loss_vs_epochs.png')
    #
    # plt.clf()
    # plt.close()

    #   验证
    acc = 0.
    model.eval()
    print('waitting for val...')
    with torch.no_grad():
        accuracy = 0.
        total = 0
        for data in val_dataloader:
            imgs, feature, labels = data
            imgs = imgs.to(device)
            feature = feature.to(device)
            labels = labels.to(device)
            out = model(imgs, feature)
            _, prediction = torch.max(out, 1)
            total += labels.size(0)
            accuracy += (prediction == labels).sum().item()
            acc = 100. * accuracy / total
    print('epoch {} The ValSet accuracy is {:.4f}% \n'.format(epoch, acc))
    Val_Accuracy.append(acc)
    if acc > BEST_VAL_ACC or (acc == BEST_VAL_ACC and run_accuracy > BEST_TR_ACC):
        print('Find Better Model and Saving it...')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(model.state_dict(), '../checkpoint/ResNet50_Cats_Dogs_val.pth')
        BEST_VAL_ACC = acc
        ACC_val = []
        ACC_val.append(BEST_VAL_ACC)
        np.savetxt('../checkpoint/resnet_val_acc.txt',ACC_val ,fmt='%.04f')
        BEST_TR_ACC = run_accuracy
        ACC_train = []
        ACC_train.append(BEST_VAL_ACC)
        np.savetxt('../checkpoint/resnet_train_acc.txt', ACC_train, fmt='%.04f')
        print('Saved!')
    #
    # # fig3, ax3 = plt.subplots(figsize=(11, 8))
    # #
    # # ax3.plot(range(epoch + 1), Val_Accuracy)
    # # ax3.set_title("Average Val accuracy vs epochs")
    # # ax3.set_xlabel("Epoch")
    # # ax3.set_ylabel("Current Val accuracy")
    # #
    # # plt.savefig('val_accuracy_vs_epoch.png')
    # # plt.close()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Now the best val Acc is {:.4f}%'.format(BEST_VAL_ACC))
    print('Now the train Acc is {:.4f}%'.format(Accuracy[epoch]))
