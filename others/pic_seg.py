import cv2

from src_ele.file_operation import get_ele_data_from_path
from src_ele.pic_opeeration import show_Pic

path_dyna = r'D:\Data\pic_seg_choices\data_paper\LG7-4_6_5169.5_5171.6_dyna.png'
path_stat = path_dyna.replace('dyna', 'stat')
path_mask = path_dyna.replace('dyna', 'mask')


print(path_dyna, path_mask, path_stat)

img_dyna, depth = get_ele_data_from_path(path_dyna)
img_stat, _ = get_ele_data_from_path(path_stat)
img_mask, _ = get_ele_data_from_path(path_mask)

index_t = 500
t1 = 45
t2 = 23
print(depth[index_t+t1, 0], depth[index_t-t2, 0])
# show_Pic([img_dyna[:index_t+t1, :], img_stat[:index_t+t1, :], img_mask[:index_t+t1, :]], '13')
# show_Pic([img_dyna[index_t-t2:, :], img_stat[index_t-t2:, :], img_mask[index_t-t2:, :]], '13')


# cv2.imwrite(path_dyna.replace('5171.6', '5170.9'), img_dyna[:index_t+t1, :])
# cv2.imwrite(path_stat.replace('5171.6', '5170.9'), img_stat[:index_t+t1, :])
# cv2.imwrite(path_mask.replace('5171.6', '5170.9'), img_mask[:index_t+t1, :])


cv2.imwrite(path_dyna.replace('5169.5', '5170.7'), img_dyna[index_t-t2:, :])
cv2.imwrite(path_stat.replace('5169.5', '5170.7'), img_stat[index_t-t2:, :])
cv2.imwrite(path_mask.replace('5169.5', '5170.7'), img_mask[index_t-t2:, :])