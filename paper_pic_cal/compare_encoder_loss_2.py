import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl

# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# #处理中文乱码
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False


# #设置绘图风格
# print(plt.style.available)
# plt.style.use('ggplot')
# plt.style.use('seaborn-paper')


#坐标轴负号的处理
plt.rcParams['axes.unicode_minus']=False

# loss_all = np.loadtxt('loss.txt', dtype=np.float32, delimiter='\t', skiprows=0)
loss_all = np.loadtxt('loss_real_512dim.txt', dtype=np.float32, delimiter='\t', skiprows=0)
# loss_all = np.loadtxt('loss_real_20dim.txt', dtype=np.float32, delimiter='\t', skiprows=0)
print(loss_all.shape)


# x_axis_data = [1, 2, 3, 4, 5, 6, 7]
fig = plt.figure(figsize=(16, 10), dpi=80)

linestyle_list = [':', '-.', '--']
# color_list = ['darkblue', 'orange', 'r']
color_list = ['lightcoral', 'sandybrown', 'skyblue']
marker_list = ['^', 'o', '*']
scale = [28, 14, 7]
markeredgecolor_list = ['black', 'black', 'black']                              # 点的边框色
# markerfacecolor_list = ['ivory', 'cyan', 'violet']                            # 点的填充色
markerfacecolor_list = ['lightcoral', 'sandybrown', 'skyblue']
# label_list = ['M1:512dim_64batch', 'M1:512dim_128batch', 'M3:512dim_256batch']
label_list = ['M1:20dim_64batch', 'M1:20dim_128batch', 'M3:20dim_256batch']
# markersize_list = [3, 4, 5]
markersize_list = [5, 6, 7]
# markersize_list = [7, 8, 9]
fontsize_t = 30

for i in range(loss_all.shape[1]-1):
    plt.plot(loss_all[:, 0] / scale[i],  # x轴数据
             loss_all[:, i+1],  # y轴数据
             linestyle=linestyle_list[i],  # 折线类型
             linewidth=3,  # 折线宽度
             color=color_list[i],  # 折线颜色
             marker=marker_list[i],  # 折线图中添加圆点
             markersize=markersize_list[i],  # 点的大小
             markeredgecolor=markeredgecolor_list[i],  # 点的边框色
             markerfacecolor=markerfacecolor_list[i],  # 点的填充色
             label=label_list[i])

plt.xlabel('epoch', fontsize=fontsize_t)
plt.ylabel('loss', fontsize=fontsize_t)
plt.legend(fontsize=fontsize_t)

plt.xlim(0, 6)

#对于X轴，只显示x中各个数对应的刻度值
plt.xticks(fontsize=fontsize_t)  #改变x轴文字值的文字大小
# 添加y轴标签
plt.yticks(fontsize=fontsize_t)

# 添加图形标题
plt.title('不同参数的模型训练过程loss对比', fontsize=fontsize_t)
# 显示图形
plt.show()