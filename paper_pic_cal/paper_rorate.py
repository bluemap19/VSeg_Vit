import cv2

from simCLR.pic_pre_process import get_random_RB_all, pic_rotate_by_Rb, get_random_RB_curve_2
from simsiam.ELE_data import Ele_data_img
import numpy as np
# 禁用科学计数法
np.set_printoptions(suppress=True)
from src_ele.pic_opeeration import show_Pic, pic_enhence_random
import matplotlib.pyplot as plt


######## 这个是用来查看图像绕井壁旋转，图像增强效果图的
path_t = r'C:\Users\Administrator\Desktop\pic_rorate_show_LG7-4'
a = Ele_data_img(path_t)
print(a.img_dyna.shape)


data_depth = a.depth_data
RB_index = np.random.randint(0, 2)
dyna_org = a.img_dyna
stat_org = a.img_static

# rb_random, RB_index = get_random_RB_all(depth=data_depth, RB_index=RB_index)
rb_random = get_random_RB_curve_2(data_depth, dep_inf=[140, 160])

# plt.figure(figsize=(10, 7))
# plt.plot(data_depth.ravel(), rb_random.ravel())
# plt.show()

pic_dyna = pic_rotate_by_Rb(pic=dyna_org, Rb=rb_random)
pic_stat = pic_rotate_by_Rb(pic=stat_org, Rb=rb_random)

# input, windows_shape=3, ratio_top=0.2, ratio_migration=0.6, random_times=3
pic_dyna_enhance = pic_enhence_random(pic_dyna, windows_shape=3, ratio_top=0.3, ratio_migration=0.5, random_times=2)
# pic_dyna_enhance = cv2.equalizeHist(np.uint8(pic_dyna))
# pic_dyna_enhance = cv2.bilateralFilter(np.uint8(pic_dyna), 5, 75, 75)
# pic_dyna_enhance = cv2.GaussianBlur(np.uint8(pic_dyna), (5, 5), 0)


show_Pic([dyna_org, stat_org, pic_dyna, pic_stat, pic_dyna_enhance], '15')



# print(data_depth.shape)
rb_random = rb_random.reshape((-1, 1))
rb_random_save = np.append(data_depth, rb_random, axis=1)

# print(rb_random.shape, data_depth.shape)
pic_dyna_rorate = np.append(data_depth, pic_dyna, axis=1)
pic_stat_rorate = np.append(data_depth, pic_stat, axis=1)
pic_dyna_rorate_enhance = np.append(data_depth, pic_dyna, axis=1)


np.savetxt('{}/{}'.format(path_t, 'rb_random.txt'), rb_random_save, fmt='%.4f', delimiter='\t', comments='',
        header='WELLNAME= LG7-4\nSTDEP\t= {}\nENDEP\t= {}\nLEV\t= 0.0025\nUNIT\t= meter\nCURNAMES= {}\n\n#DEPTH\t{}'.
           format(data_depth[0, 0], data_depth[-1, 0], 'RB_random', 'RB_random'))

np.savetxt('{}/{}'.format(path_t, 'pic_dyna_rorate.txt'), pic_dyna_rorate, fmt='%.4f', delimiter='\t', comments='',
        header='WELLNAME= LG7-4\nSTDEP\t= {}\nENDEP\t= {}\nLEV\t= 0.0025\nUNIT\t= meter\nCURNAMES= {}\n\n#DEPTH\t{}'.
           format(data_depth[0, 0], data_depth[-1, 0], 'IMAGE.DYNA_RORATE', 'IMAGE.DYNA_RORATE'))

np.savetxt('{}/{}'.format(path_t, 'pic_stat_rorate.txt'), pic_stat_rorate, fmt='%.4f', delimiter='\t', comments='',
        header='WELLNAME= LG7-4\nSTDEP\t= {}\nENDEP\t= {}\nLEV\t= 0.0025\nUNIT\t= meter\nCURNAMES= {}\n\n#DEPTH\t{}'.
           format(data_depth[0, 0], data_depth[-1, 0], 'IMAGE.STAT_RORATE', 'IMAGE.STAT_RORATE'))

np.savetxt('{}/{}'.format(path_t, 'pic_dyna_rorate_enhance.txt'), pic_dyna_rorate_enhance, fmt='%.4f', delimiter='\t', comments='',
        header='WELLNAME= LG7-4\nSTDEP\t= {}\nENDEP\t= {}\nLEV\t= 0.0025\nUNIT\t= meter\nCURNAMES= {}\n\n#DEPTH\t{}'.
           format(data_depth[0, 0], data_depth[-1, 0], 'IMAGE.DYNA_RORATE_ENHANCE', 'IMAGE.DYNA_RORATE_ENHANCE'))