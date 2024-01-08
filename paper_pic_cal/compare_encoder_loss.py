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
print(loss_all.shape)


# x_axis_data = [1, 2, 3, 4, 5, 6, 7]
fig = plt.figure(figsize=(16, 10), dpi=80)

# plt.plot(loss_all[:, 0]/12, # x轴数据
#          loss_all[:, 3], # y轴数据
#          linestyle = '-.', # 折线类型
#          linewidth = 2, # 折线宽度
#          color = 'darkblue', # 折线颜色
#          marker = '^', # 折线图中添加圆点
#          markersize = 4, # 点的大小
#          markeredgecolor='black', # 点的边框色
#          markerfacecolor='brown', # 点的填充色
#          label='line3')

plt.plot(loss_all[:, 0]/12, # x轴数据
         loss_all[:, 4], # y轴数据
         linestyle = '-.', # 折线类型
         linewidth = 3, # 折线宽度
         color = 'turquoise', # 折线颜色
         marker = 'o', # 折线图中添加圆点
         markersize = 5, # 点的大小
         markeredgecolor='black', # 点的边框色
         markerfacecolor='turquoise', # 点的填充色
         label='M1:128_20')


# plt.plot(loss_all[:, 0]/12, # x轴数据
#          loss_all[:, 1], # y轴数据
#          linestyle = '-', # 折线类型
#          linewidth = 2, # 折线宽度
#          color = 'steelblue', # 折线颜色
#          marker = 'o', # 折线图中添加圆点
#          markersize = 4, # 点的大小
#          markeredgecolor='black', # 点的边框色
#          markerfacecolor='brown', # 点的填充色
#          label='line1')

plt.plot(loss_all[:, 0]/12, # x轴数据
         loss_all[:, 2], # y轴数据
         linestyle = '-.', # 折线类型
         linewidth = 3, # 折线宽度
         color = 'chocolate', # 折线颜色
         marker = '^', # 折线图中添加圆点
         markersize = 5, # 点的大小
         markeredgecolor='black', # 点的边框色
         markerfacecolor='chocolate', # 点的填充色
         label='M2:128_512')

# plt.plot(loss_all[:, 0]/12, # x轴数据
#          loss_all[:, 5], # y轴数据
#          linestyle = '-', # 折线类型
#          linewidth = 2, # 折线宽度
#          color = 'tomato', # 折线颜色
#          marker = '+', # 折线图中添加圆点
#          markersize = 4, # 点的大小
#          markeredgecolor='black', # 点的边框色
#          markerfacecolor='brown', # 点的填充色
#          label='line5')

plt.plot(loss_all[:, 0]/12, # x轴数据
         loss_all[:, 6], # y轴数据
         linestyle = ':', # 折线类型
         linewidth = 3, # 折线宽度
         color = 'slateblue', # 折线颜色
         marker = 'o', # 折线图中添加圆点
         markersize = 5, # 点的大小
         markeredgecolor='black', # 点的边框色
         markerfacecolor='slateblue', # 点的填充色
         label='M3:256_20')

plt.plot(loss_all[:, 0]/12, # x轴数据
         loss_all[:, 10], # y轴数据
         linestyle = '--', # 折线类型
         linewidth = 3, # 折线宽度
         color = 'red', # 折线颜色
         marker = '^', # 折线图中添加圆点
         markersize = 5, # 点的大小
         markeredgecolor='black', # 点的边框色
         markerfacecolor='red', # 点的填充色
         label='M4:256_512')
plt.xlabel('epoch', fontsize=18)
plt.ylabel('loss', fontsize=18)
plt.legend(fontsize=18)


#对于X轴，只显示x中各个数对应的刻度值
plt.xticks(fontsize=16)  #改变x轴文字值的文字大小
# 添加y轴标签
plt.yticks(fontsize=16)

# 添加图形标题
plt.title('不同参数的模型训练过程loss对比', fontsize=16)
# 显示图形
plt.show()