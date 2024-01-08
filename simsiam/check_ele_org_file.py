import argparse

from src_ele.dir_operation import traverseFolder, traverseFolder_folder

parser = argparse.ArgumentParser(description='*****************')
parser.add_argument('--path_unsupervised_in', default=r'D:\Data\target_stage3_small_p', type=str,
                    help='path need to be checked')
parser.add_argument('--path_in', default=r'D:\Data\target_stage3_small_p', type=str,
                    help='path need to be checked')
parser.add_argument('--charter', default='store_true', type=str,
                    help='charter folder....')

def main():
    args = parser.parse_args()

    file_list = traverseFolder(args.path_unsupervised_in)

    # print(file_list)

    for i in file_list:
        if i.__contains__('cif_dyna_full'):
            pass
        elif i.__contains__('cif_stat_full'):
            pass
        elif i.__contains__('layer_info'):
            pass
        else:
            print('unnecessary file:{}'.format(i))

    folder_list = traverseFolder_folder(args.path_in)
    for i in folder_list:
        file_list_temp = traverseFolder(i)
        # print(file_list_temp)
        if (file_list_temp[0].__contains__('cif_dyna_full')) & (file_list_temp[1].__contains__('cif_stat_full')) \
                & (file_list_temp[2].__contains__('layer_info')):
            print('check successfully:{}'.format(file_list_temp[0].split('/')[-2]))
            pass
        else:
            print('{}'.format(file_list_temp))


if __name__ == '__main__':
    main()
