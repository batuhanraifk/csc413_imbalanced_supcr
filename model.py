import torch
from torch import nn
from torchvision.models import resnet18

class AgeNet(nn.Module):
    def __init__(self):
        super(AgeNet, self).__init__()

        resnet = resnet18()
        # All resnet18 except final layer
        modules = list(resnet.children())[:-1]

        self.resnet = nn.Sequential(*modules)

        #Replace final layer with regressor
        self.lin_reg = nn.Linear(512, 1)
    
    def forward(self, samples):
        x = self.resnet(samples)
        x = x.squeeze()
        out = self.lin_reg(x)
        return out, x

if __name__ == '__main__':
    net = AgeNet()
    net.resnet.requires_grad_(False)
    net.resnet.requires_grad_(False)