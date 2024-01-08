
import torch
import torch.nn as nn
import torch.nn.functional as F
import vision_transformer as vits

def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize
        z = F.normalize(z, dim=1) # l2-normalize
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception

class vit_simsiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, in_chans=2, patch_size=8):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(vit_simsiam, self).__init__()

        # self.encoder = vits.__dict__['vit_small'](patch_size=patch_size, drop_path_rate=0.1, in_chans=in_chans)
        self.arch_model = vits.__dict__['vit_small'](patch_size=patch_size, drop_path_rate=0.1, in_chans=in_chans)
        # student = vits.__dict__['vit_small'](patch_size=16, drop_path_rate=0.1)
        # teacher = vits.__dict__['vit_small'](patch_size=16)


        # embed_dim = self.encoder.embed_dim
        embed_dim = self.arch_model.embed_dim

        out_dim = 384
        prev_dim = 256
        # out_dim = 512
        # prev_dim = 384
        use_bn_in_head = False

        # print(self.encoder)

        # print(self.encoder)
        self.arch_model.head = nn.Sequential(nn.Linear(out_dim, out_dim, bias=False),
        # self.encoder.head = nn.Sequential(nn.Linear(out_dim, out_dim, bias=False),
                                        nn.BatchNorm1d(out_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(out_dim, out_dim, bias=False),
                                        nn.BatchNorm1d(out_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(out_dim, out_dim, bias=True),
                                        nn.BatchNorm1d(out_dim, affine=False)) # output layer


        self.arch_model.head[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN
        # self.encoder.head[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN


        self.predictor = nn.Sequential(nn.Linear(out_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(prev_dim, out_dim)) # output layer


    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        # print(z1.shape, z2.shape)

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        # print(p1.shape, p2.shape)

        L = D(p1, z2) / 2 + D(p2, z1) / 2

        return {'loss': L, 'z1': z1, 'z2': z2}



# a = vit_simsiam()
# a.arch_model.head[6].bias.requires_grad = False
# print(a)
# for name, param in a.named_parameters():
#     print(name)
#     if "arch_model.head.4.bias" in name:
#         print('requires_grad set no')
#         param.requires_grad = False
# vit_simsiam().arch_model.head[6].bias.requires_grad = False
# b = torch.rand((2, 2, 224, 224))
# c = torch.rand((2, 2, 224, 224))
# print(a(b,  c).shape)
# import torchvision.models as models
# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))
#
