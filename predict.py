from PetFinder.dataset.dataset import *
from torch.utils.data import DataLoader
import os
from PetFinder.models.vgg16 import *
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#准备数据集
root_dir = "../PetFinder/dataset"
label3_dir = "test.csv"
dataset_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # 随机长宽比裁剪为224*224
    transforms.RandomHorizontalFlip(),  # 依概率p=0.5水平翻转
    transforms.ToTensor(),  # 转化为张量并归一化为[0-1]
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test = MyData(root_dir,"test",label3_dir,transform=dataset_transform)
test_dataloader = DataLoader(test, batch_size=8, drop_last=True)

# 创建网络模型
model = Model()
model.load_state_dict(torch.load('./checkpoint/VGG16_Cats_Dogs_val.pth'))
model = model.to(device)
# 测试
id_list = []
pred_list = []
with open(os.path.join(root_dir, label3_dir), 'r') as f:
    test_data = f.readlines()  # 去掉跳过文件头行
f.close()



model.eval()
with torch.no_grad():
    for data in test_dataloader:
        imgs, feature, targets = data
        imgs = imgs.to(device)
        feature = feature.to(device)
        targets = targets.to(device)
        out = model(imgs, feature)
        _, prediction = torch.max(out, 1)
id_list = []
for i in range(len(test_data)):
    id_list.append(test_data[i].rsplit('\n')[0].split(',')[0])
pred_list = prediction.cuda().data.cpu().numpy()
targets = targets.cuda().data.cpu().numpy()
print(targets)
print(pred_list)
# res = pd.DataFrame({
#     'Id': id_list,
#     'Pawpularity': pred_list
# })
# res.to_csv('submission.csv', index=False)



