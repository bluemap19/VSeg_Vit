import os
import cv2
import numpy as np
from src_ele.file_operation import get_ele_data_from_path


def check_and_make_dir(dir_path):
    if os.path.exists(dir_path):
        return
        # print("{} already exists!!!".format(dir_path))
        # exit()
    else:
        os.makedirs(dir_path)
        print('successfully create dir:{}'.format(dir_path))
        assert os.path.exists(dir_path), dir_path

def traverseFolder(path):
    FilePath = []
    for path in os.walk(path):
        for file_name in path[2]:
            FilePath.append(path[0].replace('\\', '/')+'/' + file_name)

    return FilePath

def traverseFolder_folder(path):
    path_folder = []
    for path in os.walk(path):
        for file_name in path[1]:
            path_folder.append(path[0].replace('\\', '/')+'/' + file_name)

    return path_folder


def folder_search_by_charter(path=r'D:/1111/Input/fanxie184', target_file_charter=['data_org'], end_token=['.txt']):
    file_list = traverseFolder(path)
    file_list_tmp = []
    for i in file_list:
        for j in end_token:
         if i.__contains__(j):
             file_list_tmp.append(i)

    target_file_list = []

    for i in range(len(target_file_charter)):
        for j in range(len(file_list)):
            if file_list[j].__contains__(target_file_charter[i]):
                target_file_list.append(file_list[j])

    return target_file_list
# if __name__ == '__main__':
#     save_dir = r'D:\1111\PicData\PicOrg\class'
#     # check_and_make_dir(save_dir)
#     # print(traverseFolder_folder(save_dir).__len__())
#     save_file_as_xlsx([np.zeros((10, 10)), np.zeros((20, 20)), np.zeros((7, 7))], sheet_name=['ssdf', 'sasfa'])