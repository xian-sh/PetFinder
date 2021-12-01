#可以手动打乱原始数据集，直接在excel表中操作
import os
import csv
import random

#数据集地址
data_dir = '../PetFinder/dataset'
origin_dir = os.path.join(data_dir,'train.csv')
train_dir = os.path.join(data_dir,'train_data.csv')
val_dir = os.path.join(data_dir,'val_data.csv')

train_file = open(train_dir,'w',newline='')
val_file = open(val_dir,'w',newline='')
#读取源数据地址，在train.csv文件夹下
with open(origin_dir,'r') as f:
    lines = f.readlines()[1:] #  去掉跳过文件头行
f.close()
#打乱读入的每一行数据（样例顺序打乱）
random.shuffle(lines)
#写文件
train_writer = csv.writer(train_file)
val_writer = csv.writer(val_file)

for i in range(len(lines)):
    temp = lines[i].rsplit('\n')[0].split(',')
    # 取大约80%的样例为训练集，剩余的为测试集
    if i < 7830:
        train_writer.writerow(temp)
    else:
        val_writer.writerow(temp)

train_file.close()
val_file.close()
