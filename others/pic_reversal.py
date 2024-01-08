import cv2

from src_ele.dir_operation import traverseFolder
from src_ele.file_operation import get_ele_data_from_path
from src_ele.pic_opeeration import show_Pic

# path_show = r'D:\Data\pic_seg_choices\data_paper'
# path_list = traverseFolder(path_show)
# print(path_list)

def traverse_pic(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = 255-img[i][j]
    return img


# for i in range(len(path_list)):
#     if path_list[i].__contains__('stat'):
#         print(path_list[i])
#     else:
#         continue
#
#     img_dyna, _ = get_ele_data_from_path(path_list[i])
#     # print(img_dyna.shape)
#
#     img_dyna = cv2.resize(img_dyna, (224, 224))
#     img_dyna = traverse_pic(img_dyna)
#
#     show_Pic([img_dyna], pic_order='11')
