
import numpy as np

######## TP_N_A, TN_N_A, FN_N_A, FP_N_A, TL_N_A, FL_N_A, ALL_N_A


ACC_FL = np.array([[ 192, 1840,  169,    7,  360, 1560, 1920],
 [ 650, 1324,  119,  115,  769, 1151, 1920],
 [ 592, 1616,    0,    0,  192, 1728, 1920]])
# NUM_P = [360, 769, 592]             # 正值的总数
# Num_F = [1840, 1439, 1616]          # 负值的总数
NUM_P = [360, 769, 192]             # 正值的总数
Num_F = [1560, 1151, 1728]          # 负值的总数

# 原始的配置
# Num_TP = [190, 650, 592]            # (越高越好)  正值预测成正值
# Num_FP = [7, 115, 0]                # (越低越好)  负值预测成正值

################            A模型的配置
# Num_TP = [302, 712, 192]            # (越高越好)  正值预测成正值
# Num_FP = [43, 134, 23]                # (越低越好)  负值预测成正值

################            B1模型的配置 小窗口
# Num_TP = [210, 668, 192]            # (越高越好)  正值预测成正值
# Num_FP = [7, 114, 15]                # (越低越好)  负值预测成正值
################            B2模型的配置 大窗口
# Num_TP = [253, 628, 185]            # (越高越好)  正值预测成正值
# Num_FP = [176, 231, 96]                # (越低越好)  负值预测成正值

################            C模型的配置  相较A 少了像素分布特征提取模块
# Num_TP = [279, 670, 107]            # (越高越好)  正值预测成正值
# Num_FP = [49, 169, 429]                # (越低越好)  负值预测成正值

################            D模型的配置    512维度 相较A 未添加PCA模型
# Num_TP = [12, 603, 36]            # (越高越好)  正值预测成正值
# Num_FP = [396, 142, 586]                # (越低越好)  负值预测成正值

################            E模型的配置      未添加PCA模块但编码器模型输出特征维度为20,但是模型经过预训练
# Num_TP = [265, 638, 185]            # (越高越好)  正值预测成正值
# Num_FP = [72, 163, 15]                # (越低越好)  负值预测成正值

###############             F模型的配置      未添加PCA模块但编码器模型输出特征维度为20,但是模型未预训练
Num_TP = [223, 592, 187]            # (越高越好)  正值预测成正值
Num_FP = [70, 182, 10]                # (越低越好)  负值预测成正值

print(ACC_FL)

for i in range(3):
    ACC_FL[i][0] = Num_TP[i]
    ACC_FL[i][2] = NUM_P[i] - Num_TP[i]
    ACC_FL[i][3] = Num_FP[i]
    ACC_FL[i][1] = Num_F[i] - Num_FP[i]
print(ACC_FL)

name = ['裂缝型', '孔洞型', '洞穴型']
for i in range(ACC_FL.shape[0]):
    TP_N_A, TN_N_A, FN_N_A, FP_N_A, TL_N_A, FL_N_A, ALL_N_A = ACC_FL[i]
    ACC = (TP_N_A + TN_N_A) / ALL_N_A
    I = TP_N_A/(TP_N_A + FP_N_A + FN_N_A)
    P = TP_N_A/(TP_N_A + FP_N_A)
    R = TP_N_A/(TP_N_A + FN_N_A)
    D = 2*TP_N_A/(2*TP_N_A + FP_N_A + FN_N_A)
    F1 = 2*P*R/(P + R)
    print('{} {:.4f} {:.4f} {:.4f} {:.4f}'.format(name[i], P, R, ACC, F1))

I_ALL = np.sum(ACC_FL[:, 0])/(np.sum(ACC_FL[:, 0]) + np.sum(ACC_FL[:, 3]) + np.sum(ACC_FL[:, 2]))
P_ALL = np.sum(ACC_FL[:, 0])/(np.sum(ACC_FL[:, 0]) + np.sum(ACC_FL[:, 3]))
R_ALL = np.sum(ACC_FL[:, 0])/(np.sum(ACC_FL[:, 0]) + np.sum(ACC_FL[:, 2]))
D_ALL = 2*np.sum(ACC_FL[:, 0])/(2*np.sum(ACC_FL[:, 0]) + np.sum(ACC_FL[:, 3]) + np.sum(ACC_FL[:, 2]))
ACC_ALL = np.sum(ACC_FL[:, :2])/(np.sum(ACC_FL[:, -1]))
F1_ALL = 2*P_ALL*R_ALL/(P_ALL + R_ALL)
print('AVERAGE: {:.4f} {:.4f} {:.4f} {:.4f}'.format(P_ALL, R_ALL, ACC_ALL, F1_ALL))