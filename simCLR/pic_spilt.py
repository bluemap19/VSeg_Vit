import cv2
import numpy as np
from tqdm import trange

from src_ele.dir_operation import traverseFolder
from src_ele.file_operation import get_ele_data_from_path

path = r'D:\Data\target4'
path_all = traverseFolder(path)

print(path_all)

pic_index = 0

step = 320
for i in (path_all):
    print(i)
    img, depth_data = get_ele_data_from_path(i)

    num_pic = (img.shape[0]-600)//step + 1

    print(img.shape, num_pic)
    for j in trange(num_pic):
        length_pic = np.random.randint(400, 500)
        index_start = j * step

        pic = img[index_start:index_start+length_pic, :]
        depth_temp = depth_data[index_start:index_start+length_pic, :]

        pic_index += 1
        cv2.imwrite("img_temp/pic_{}_{}_{}_.jpg".format(pic_index, depth_temp[0, 0], depth_temp[-1, 0]), pic)