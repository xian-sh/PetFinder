import torchvision

from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        ##导入训练好的VGG16网络
        vgg16 = torchvision.models.vgg16(pretrained=True)
        vgg16.classifier = nn.Sequential()  # 将分类层置空，下面将改变我们的分类层
        ##获取VGG16的特征提取层
        vgg = vgg16.features
        ##将VGG16的特征提取层参数进行冻结，不对其进行更新
        for param in vgg.parameters():
            param.requires_grad_(False)
        ##预训练的VGG16特征提取层
        self.features = vgg
        ##将VGG16的特征提取层参数进行冻结，不对其进行更新
        for param in vgg.parameters():
            param.requires_grad_(False)
        self.classifier = nn.Sequential(  # 定义自己的分类层
            nn.Linear(512 * 7 * 7+12, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 100)
        )

    def forward(self, input1, input2):
        input2 = input2.view(input2.size(0), -1)
        input1 = self.features(input1)
        input1 = input1.view(input1.size(0), -1)
        input = torch.cat([input1, input2], dim=1)
        output = self.classifier(input)
        return output

#测试
# VGG = Model()
# print(VGG)
# input1 = torch.ones((64,3,224,224))
# input2 = torch.ones((64,1,1,12))
#
# output = VGG(input1,input2)
# print(output.shape)
#
# writer = SummaryWriter("../logs_seq")
# writer.add_graph(VGG, input_to_model=(input1,input2))
# writer.close()