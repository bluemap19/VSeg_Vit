from src_ele.file_operation import get_test_ele_data
import numpy as np
import matplotlib.pyplot as plt

from src_ele.pic_opeeration import get_pic_distribute, show_Pic

data_img_dyna, data_img_stat, data_depth = get_test_ele_data()
print(data_img_dyna.shape, data_img_stat.shape, data_depth.shape)

dist_l = 64

pic_dist = get_pic_distribute(data_img_stat, dist_length=dist_l, min_V=0, max_V=256)

a = []
# print(a)
for i in range(dist_l):
    bei = 256//dist_l
    a.append(i*bei)
plt.bar(a, pic_dist, width=3)
plt.show()
# show_Pic([data_img_stat], pic_order='11')
