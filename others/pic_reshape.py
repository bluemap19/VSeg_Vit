import cv2
from src_ele.dir_operation import traverseFolder
from src_ele.file_operation import get_ele_data_from_path
from src_ele.pic_opeeration import show_Pic
import numpy as np


def start_reshape(path_in=r'D:\Data\pic_seg_choices\DATA_NEW_ADD\mask_alter\org_paint', path_out=r'D:\Data\pic_seg_choices\DATA_NEW_ADD\mask_alter\org_paint_reshape'):
    file_list = traverseFolder(path_in)

    for i in range(len(file_list)):
        path_mask_t = file_list[i]
        print(file_list[i].split('\\')[-1])

        img_mask_t, depth = get_ele_data_from_path(path_mask_t)
        print(img_mask_t.shape)

        img_mask_t = cv2.resize(img_mask_t, (224, 224))
        # show_Pic([img_mask_t, img_n], '12')
        # img_name = path_dyna_t.split('/')[-1].split('_')[0] + '_' + str(i) + '_' + str(depth[0][0]) + '_' + str(depth[-1][0])
        # print(path_out+'\\'+img_name+'_dyna.png')
        cv2.imwrite(path_out+ '/' + path_mask_t.split('/')[-1], img_mask_t)
        # cv2.imwrite(img_name+'_stat.png', img_stat_t)
    pass


if __name__ == '__main__':
    start_reshape()