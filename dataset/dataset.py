from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import numpy as np
from torchvision import transforms
#测试
# data = pd.read_csv("PetFinder/dataset/train.csv", header=0)
# print(data.iloc[1])
class MyData(Dataset):

    def __init__(self, root_dir, image_dir, label_dir, transform=None):
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.label_path = os.path.join(self.root_dir, self.label_dir)
        self.image_path = os.path.join(self.root_dir, self.image_dir)
        self.image_list = os.listdir(self.image_path)
        self.transform = transform

        with open(self.label_path, 'r') as f:
            self.data = f.readlines()  # 去掉跳过文件头行
        f.close()

    def __getitem__(self, idx):
        img_name = self.data[idx].rsplit('\n')[0].split(',')[0] + '.jpg'
        img_item_path = os.path.join(self.root_dir, self.image_dir, img_name)
        img = Image.open(img_item_path)
        feature = [int(x) for x in self.data[idx].rsplit('\n')[0].split(',')[1:13]]
        target = int(self.data[idx].rsplit('\n')[0].split(',')[13])-1
        img = self.transform(img)
        # trans_norm = transforms.Normalize([0.5, 0.5, 0.5])
        # trans_norm1 = transforms.Normalize([0.5,])
        # img = trans_norm(img)
        feature = torch.as_tensor(feature)
        feature = torch.reshape(feature, (1, 1, -1))
        # feature = trans_norm1(feature)
        label = torch.as_tensor(target)
        return img, feature, label

    def __len__(self):
        return len(self.data)


# 测试
# root_dir = "dataset"
# image_dir = "train"
# label1_dir = "train_dataset.csv"
# label2_dir = "test_dataset.csv"
# dataset_transform = torchvision.transforms.Compose([
#     torchvision.transforms.ToTensor()
# ])
# train_dataset = MyData(root_dir,image_dir,label1_dir,transform=dataset_transform)
# test_dataset = MyData(root_dir,image_dir,label2_dir,transform=dataset_transform)
#
# print(train_dataset[0])
# print(test_dataset[0])