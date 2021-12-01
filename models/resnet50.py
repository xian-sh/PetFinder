import torchvision

from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        BackBone = torchvision.models.__dict__['resnet50'](pretrained=True)
        add_block = []
        add_block += [nn.Linear(1012, 512)]
        add_block += [nn.ReLU(True)]
        add_block += [nn.Linear(512, 128)]
        add_block += [nn.ReLU(True)]
        add_block += [nn.Linear(128, 100)]
        add_block = nn.Sequential(*add_block)
        self.BackBone = BackBone
        for param in self.BackBone.parameters():
            param.requires_grad_(False)

        self.add_block = add_block

    def forward(self, input1, input2):
        input2 = input2.view(input2.size(0), -1)
        input1 = self.BackBone(input1)
        input1 = input1.view(input1.size(0), -1)
        input = torch.cat([input1, input2], dim=1)
        output = self.add_block(input)
        return output


# #测试
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