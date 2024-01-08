from simsiam.ELE_data import get_train_data

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import TensorDataset

from simsiam.pred_answer_validate import get_charter_acc, get_multi_charter_acc
from simsiam.sim_model.model_stage3 import Model_simple_linear


class dataset(Dataset):
    def __init__(self):
        # data = pd.read_csv('c:/Users/Administrator/Desktop/iris/iris.data',
        #                    names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'y'])
        # data['new_y'] = data.iloc[:, [4]].replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [0, 1, 2])

        # ###### 这个是载入那几口井的预处理好的常规九条的测井数据
        # data = get_train_data()
        # self.X = torch.FloatTensor(data[:, :-1])
        # self.y = torch.FloatTensor(data[:, -1])

        ##### 这个载入的是 6维图像特征 ＋ 10维颜色特征 + 四维储层类型
        data = np.loadtxt(r'D:\GitHubProj\dino\simsiam\feature_all_10_512_res50.txt', dtype=np.float32)
        # data = np.loadtxt(r'D:\GitHubProj\dino\simsiam\feature_all_2_512_res50.txt', dtype=np.float32)
        # data = np.loadtxt(r'D:\GitHubProj\dino\simsiam\feature_all_2_20_res50.txt', dtype=np.float32)
        self.X = torch.FloatTensor(data[:, :-4])
        self.y = torch.FloatTensor(data[:, -4:-1])

        self.len = data.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


data = dataset()

# 随机分成训练集和测试集，训练集占70%
train_set, test_set = random_split(data, [int(data.len * 0.7), data.len - int(data.len * 0.7)])

# 加载训练集
train_loader = DataLoader(dataset=train_set,
                          batch_size=64,
                          shuffle=True, )

# 加载测试集
test_loader = DataLoader(dataset=test_set,
                         batch_size=64,
                         shuffle=True, )

# model = Model_simple_linear(in_dim=16, out_dim=3)
model = Model_simple_linear(in_dim=data.X.shape[1], out_dim=data.y.shape[1])

# 使用BCE(Binary Cross Entropy)二元交叉熵损失函数
# criterion = nn.BCELoss()
criterion = nn.BCEWithLogitsLoss()

# 使用Adam优化算法
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 用于存放loss
loss_list = []

# 对整个样本训练10次
for epoch in range(10):
    # 每次训练一个minibatch
    for i, (X, y) in enumerate(train_loader):
        # print('input_x shape is:{}, target y shape is :{}'.format(X.shape, y.shape))

        # 进行预测，也就是做了一次前向传播
        y_pred = model(X)
        # print(y_pred.shape, y.shape)

        # 计算损失
        loss = criterion(y_pred.view(-1), y.view(-1))

        # 梯度归0
        optimizer.zero_grad()

        # 反向传播
        loss.backward()

        # 更新梯度
        optimizer.step()

        if i%5 == 0:
            # 记录损失
            loss_list.append(loss.data.item())

# # 画出损失下降的图像
# plt.plot(np.linspace(0, 100, len(loss_list)), loss_list)
# plt.show()

# # 查看当前的训练参数，也就是w和b
# print(model.state_dict())

# 使用测试集验证
model.eval()



# TP_N_A, TN_N_A, FN_N_A, FP_N_A, TL_N_A, FL_N_A, ALL_N_A = 0, 0, 0, 0, 0, 0, 0
# for batch, (X, y) in enumerate(test_loader):
#     # 进行预测，也就是做了一次前向传播
#     y_pred = model(X)
#     # y_pred = y_pred.data.item()
#
#     y_pred = y_pred.cpu().detach().numpy()
#     y = y.cpu().detach().numpy()
#     TP_N, TN_N, FN_N, FP_N, TL_N, FL_N, ALL_N = get_charter_acc(y_pred, y, binary_use=True)
#     # ACC_FL_t = get_multi_charter_acc(y_pred, y, binary_use=True)
#     # print(ACC_FL_t)
#
#     # ACC_FL +=  ACC_FL_t
#     TP_N_A += TP_N
#     TN_N_A += TN_N
#     FN_N_A += FN_N
#     FP_N_A += FP_N
#     TL_N_A += TL_N
#     FL_N_A += FL_N
#     ALL_N_A += ALL_N
#
# ACC = (TP_N_A + TN_N_A)/ALL_N_A
# print(TP_N_A/TL_N_A, TN_N_A/FL_N_A, FN_N_A/TL_N_A, FP_N_A/FL_N_A)
# print(ACC)





ACC_FL = np.zeros((3, 7))
for batch, (X, y) in enumerate(test_loader):
    # 进行预测，也就是做了一次前向传播
    y_pred = model(X)
    # y_pred = y_pred.data.item()

    y_pred = y_pred.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    # TP_N, TN_N, FN_N, FP_N, TL_N, FL_N, ALL_N = get_charter_acc(y_pred, y, binary_use=True)
    ACC_FL_t = get_multi_charter_acc(y_pred, y, binary_use=True)
    # print(ACC_FL_t)

    ACC_FL +=  ACC_FL_t
    # TP_N_A += TP_N
    # TN_N_A += TN_N
    # FN_N_A += FN_N
    # FP_N_A += FP_N
    # TL_N_A += TL_N
    # FL_N_A += FL_N
    # ALL_N_A += ALL_N

# ACC = (TP_N_A + TN_N_A)/ALL_N_A
# print(TP_N_A/TL_N_A, TN_N_A/FL_N_A, FN_N_A/TL_N_A, FP_N_A/FL_N_A)
print(ACC_FL)

for i in range(y_pred.shape[1]):
    TP_N_A, TN_N_A, FN_N_A, FP_N_A, TL_N_A, FL_N_A, ALL_N_A = ACC_FL[i]
    ACC = (TP_N_A + TN_N_A) / ALL_N_A
    I = TP_N_A/(TP_N_A + FP_N_A + FN_N_A)
    P = TP_N_A/(TP_N_A + FP_N_A)
    R = TP_N_A/(TP_N_A + FN_N_A)
    D = 2*TP_N_A/(2*TP_N_A + FP_N_A + FN_N_A)
    print(I, P, R, D)

torch.save({
    'state_dict': model.state_dict()
}, 'checkpoint_predictor_batch{}_dim{}_epoch{:04d}.pth.tar'.format(64, 3, 10))


