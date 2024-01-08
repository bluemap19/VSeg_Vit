# net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


# stage one ,unsupervised learning
class SimCLRStage1(nn.Module):
    def __init__(self, feature_dim=512):
        super(SimCLRStage1, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                # print('*************')
                module = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                # print('------------')
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 1024, bias=False),
                               nn.BatchNorm1d(1024),
                               nn.ReLU(inplace=True),
                               nn.Linear(1024, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


# stage two ,supervised learning
class SimCLRStage2(torch.nn.Module):
    def __init__(self, num_class):
        super(SimCLRStage2, self).__init__()
        # encoder
        self.f = SimCLRStage1().f
        # classifier
        self.fc = nn.Linear(2048, num_class, bias=True)

        for param in self.f.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out


class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss,self).__init__()

    def forward(self, out_1, out_2, batch_size, temperature=0.5):
        # 分母 ：X.X.T，再去掉对角线值，分析结果一行，可以看成它与除了这行外的其他行都进行了点积运算（包括out_1和out_2）,
        # 而每一行为一个batch的一个取值，即一个输入图像的特征表示，
        # 因此，X.X.T，再去掉对角线值表示，每个输入图像的特征与其所有输出特征（包括out_1和out_2）的点积，用点积来衡量相似性
        # 加上exp操作，该操作实际计算了分母
        # [2*B, D]
        print('out_1 shape:{}'.format(out_1.shape))
        out = torch.cat([out_1, out_2], dim=0)
        print('out shape is :{}'.format(out.shape))
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        print('sim_matrix shape is :{}'.format(sim_matrix.shape))
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        print('mask shape is :{}'.format(mask.shape))
        # torch.eye:生成对角线全1，其余部分全0的二维数组
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
        print('sim_matrix shape is :{}'.format(sim_matrix.shape))

        # 分子： *为对应位置相乘，也是点积
        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        print('pos_sim shape is :{}'.format(pos_sim.shape))
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        print('pos_sim shape is :{}'.format(pos_sim.shape))
        return (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()


if __name__=="__main__":
    a = SimCLRStage1()
    print(a)

