import cv2

from src_ele.file_operation import get_ele_data_from_path
from src_ele.pic_opeeration import show_Pic

path = r'D:\Data\target_stage1_small_big_mix\guan17-11_128_3644.0025_3645.2525_stat.png'
path = r'D:\Data\target_stage1_small_big_mix\guan17-11_181_3670.5025_3671.1275_stat.png'

pic, _ = get_ele_data_from_path(path)
pic = cv2.resize(pic, (224, 224))

print(pic.shape)
show_Pic([pic], pic_order='11')
