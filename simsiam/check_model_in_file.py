import argparse
import os.path

from src_ele.dir_operation import traverseFolder, traverseFolder_folder

parser = argparse.ArgumentParser(description='*****************')
parser.add_argument('--path_in', default=r'D:\Data\target_stage3_small_p', type=str,
                    help='path need to be checked')
parser.add_argument('--charter', default='store_true', type=str,
                    help='charter folder....')

def main():
    args = parser.parse_args()


    folder_list = traverseFolder(args.path_in)
    for i in folder_list:
        # file_list_temp = traverseFolder(i)
        # print(file_list_temp)
        # if (file_list_temp[0].__contains__('cif_dyna_full')) & (file_list_temp[1].__contains__('cif_stat_full')) \
        #         & (file_list_temp[2].__contains__('layer_info')):
        #     print('check successfully:{}'.format(file_list_temp[0].split('/')[-2]))
        #     pass
        # else:
        #     print('{}'.format(file_list_temp))
        if i.__contains__('dyna'):
            if os.path.isfile(i.replace('dyna', 'stat')):
                pass
            else:
                print('no vice:{}'.format(i))
                exit(0)
        elif i.__contains__('stat'):
            if os.path.isfile(i.replace('stat', 'dyna')):
                pass
            else:
                print('no vice:{}'.format(i))
                exit(0)
        else:
            print('whats this file:{}'.format(i))
            exit(0)

if __name__ == '__main__':
    main()
