import cv2

from simCLR.pic_pre_process import pic_rotate_by_Rb
from src_ele.dir_operation import traverseFolder
from src_ele.file_operation import get_ele_data_from_path
from src_ele.pic_opeeration import show_Pic
import numpy as np

def start(path_in=r'D:\Data\pic_seg_choices\org+mask1'):
    file_list = traverseFolder(path_in)
    # print(file_list)

    for i in range(len(file_list)):
        if file_list[i].__contains__('_mask_rotate'):
            path_mask_t = file_list[i]
            print(file_list[i].split('\\')[-1])
        else:
            continue
        img_mask_t, depth = get_ele_data_from_path(path_mask_t)
        print(img_mask_t.shape)

        RB_set = np.zeros_like(depth)
        print(RB_set.shape)

        for j in range(RB_set.shape[0]):
            RB_set[j][0] = -40

        img_n = pic_rotate_by_Rb(pic=img_mask_t, Rb=RB_set)
        print(img_n.shape)
        # show_Pic([img_mask_t, img_n], '12')
        # img_name = path_dyna_t.split('/')[-1].split('_')[0] + '_' + str(i) + '_' + str(depth[0][0]) + '_' + str(depth[-1][0])
        # print(path_out+'\\'+img_name+'_dyna.png')
        cv2.imwrite(path_mask_t.replace('_mask_rotate', '_mask_rr'), img_n)
        # cv2.imwrite(img_name+'_stat.png', img_stat_t)
    pass



if __name__ == '__main__':
    start()