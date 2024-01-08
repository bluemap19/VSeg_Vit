import cv2

from src_ele.dir_operation import traverseFolder
from src_ele.file_operation import get_ele_data_from_path
from src_ele.pic_opeeration import show_Pic


def start(path_in=r'C:\Users\Administrator\Desktop\paper\分割\分割裂缝源数据', path_out=r'C:\Users\Administrator\Desktop\paper\分割\分割裂缝源图像'):
    file_list = traverseFolder(path_in)
    # print(file_list)

    for i in range(len(file_list)):
        # print(file_list[i])
        if file_list[i].__contains__('dyna'):
            path_dyna_t = file_list[i]
            path_stat_t = path_dyna_t.replace('dyna', 'stat')
        else:
            continue
        img_dyna_t, depth = get_ele_data_from_path(path_dyna_t)
        img_stat_t, depth = get_ele_data_from_path(path_stat_t)

        print(path_dyna_t.split('/')[-1].split('_')[0])
        print(img_dyna_t.shape, img_stat_t.shape)

        # show_Pic([img_dyna_t, img_stat_t], '12')
        img_name = path_dyna_t.split('/')[-1].split('_')[0] + '_' + str(i) + '_' + str(depth[0][0]) + '_' + str(depth[-1][0])
        print(path_out+'\\'+img_name+'_dyna.png')
        cv2.imwrite(img_name+'_dyna.png', img_dyna_t)
        cv2.imwrite(img_name+'_stat.png', img_stat_t)
    pass



if __name__ == '__main__':
    start()