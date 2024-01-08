import os
import time
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import nn
import torch.nn.functional as F
from tqdm import trange

from simsiam.ele_seg_2.dataloader_ele_seg import dataloader_bottle, \
    get_windows_pic_from_feature_maps_FULLY, windows_chip_2_full_pic_repeated, dataloader_base, dataloader_up1, \
    dataloader_up2, dataloader_up3
from simsiam.ele_seg_2.model_fracture_seg import model_S, model_VL, model_V
from simsiam.pred_answer_validate import multi_pic_seg_acc_cal, single_pic_seg_acc_cal
from src_ele.dir_operation import traverseFolder
from src_ele.pic_opeeration import show_Pic


def test_model_by_para(model, data_path=r'D:\Data\img_seg_data_in\train\1'):
    # data_path = r'D:\Data\img_seg_data_in\train\1'
    # data_path = r'D:\Data\target_stage3_small_p\pic_seg\1_o_fracture'

    # model = WV_model(num_in_dim=384, num_out_dim=1)

    # print('Load weights {}.'.format(model_path))
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # pretrained_dict = torch.load(model_path, map_location=device)
    # model.load_state_dict(pretrained_dict['state_dict'])
    # windows_length = pretrained_dict['windows_length']
    # windows_step = pretrained_dict['windows_step']
    # # windows_step = windows_length
    # layer_index = pretrained_dict['layer_index']
    # Init_lr = pretrained_dict['Init_lr']
    # print('current model {} para:\nlayer_index:{}\twindows_length:{}\twindows_step:{}\tInit_lr:{}'.format(
    #     model_str, layer_index, windows_length, windows_step, Init_lr))
    # 'windows_length': windows_length,
    # 'windows_step': windows_step,
    # 'Init_lr': Init_lr,
    # 'layer_index': layer_index,
    # 'batch_size': batch_size,


    # 开启测试模式
    model_eval = model.eval()
    # 测试很快，就不放到GPU上了
    # if torch.cuda.is_available():
    #     # model_eval = torch.nn.DataParallel(model)
    #     cudnn.benchmark = True
    #     model_eval = model_eval.cuda()
    # else:
    #     # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #     device = torch.device("cpu")
    #     model_eval = model.to(device)


    # path_temp = r'D:\Data\target_stage3_small_p\train\1'
    if True:
        # train_dataset = dataloader_base(path=data_path, pre_process=False)
        # train_dataset = dataloader_up1(path=data_path, windows_length=windows_length, windows_step=windows_step)
        train_dataset = dataloader_up2(path=data_path, windows_length=windows_length, windows_step=windows_step)
        # train_dataset = dataloader_up3(path=data_path, windows_length=windows_length, windows_step=windows_step)
        print('current train dataset length is:{}'.format(train_dataset.length))

        acc_list = []
        miou_list = []
        F1_list = []

        for i in trange(train_dataset.length):
            pic_all_New, feature_in, feature_in_split, mask_org_split = train_dataset[i]
            # pic_all_New, embedding_pic_split, mask_split = train_dataset[i]

            shape_t = (feature_in.shape[-2], feature_in.shape[-1])
            # print(feature_in.shape, feature_in_split.shape, mask_org_split.shape, shape_t)

            in_pic_split, mask_out = get_windows_pic_from_feature_maps_FULLY(
                feature_in, pic_all_New[-1, :, :], windows_length=windows_length, step=windows_step)

            # print('in_pic_split:{}, mask_out:{}'.format(in_pic_split.shape, mask_out.shape))
            a, b, c = mask_out.shape
            x = torch.from_numpy(in_pic_split).type(torch.FloatTensor).reshape(a, -1, b, c)
            # y = torch.from_numpy(mask_out).type(torch.FloatTensor)

            outputs = model_eval(x)

            # pic_mask = mask_out
            outputs = outputs.reshape(-1, b, c).cpu().detach().numpy()
            out_mask_full = windows_chip_2_full_pic_repeated(outputs, windows_length=windows_length, step=windows_step, full_pic_shape=shape_t)
            # print(outputs.shape)

            # for i in range(outputs.shape[0]):
            #     for j in range(outputs.shape[1]):
            #         for k in range(outputs.shape[2]):
            #             if outputs[i][j][k] < 0.5:
            #                 outputs[i][j][k] = 0
            #             else:
            #                 outputs[i][j][k] = 1
            #
            #             if pic_mask[i][j][k] < 0.5:
            #                 pic_mask[i][j][k] = 0
            #             else:
            #                 pic_mask[i][j][k] = 1
            correct_prediction, MIOU, F1 = single_pic_seg_acc_cal(out_mask_full, cv2.resize(pic_all_New[-1, :, :], shape_t))
            acc_list.append(correct_prediction)
            miou_list.append(MIOU)
            F1_list.append(F1)
            # print('pic seg miu and acc、F1 is:{}, {}, {}'.format(correct_prediction, MIOU, F1))

            # show_Pic([pic_all_New[-1, :, :]*256, out_mask_full*256,
            #           outputs[0, :, :]*256, outputs[1, :, :]*256,
            #           outputs[-2, :, :]*256, outputs[-1, :, :]*256,
            #           mask_out[0, :, :]*256, mask_out[1, :, :]*256,
            #           mask_out[-2, :, :]*256, mask_out[-1, :, :]*256], pic_order='52')
            # exit(0)

        # print('acc list:{}'.format(acc_list))
        # print('miou list:{}'.format(miou_list))
        # print('F1 list:{}'.format(F1_list))
        print('final acc、mIou、F1 mean:{:.4f}、{:.4f}、{:.4f}'.format(np.mean(acc_list), np.mean(miou_list), np.mean(F1_list)))
    pass




if __name__ == "__main__":
    # path_model_folder = r'D:\GitHubProj\dino\simsiam\ele_seg_2\model_list\lr'
    # path_model_folder = r'D:\GitHubProj\dino\simsiam\ele_seg_2\model_list\win_len_VL'
    # path_model_folder = r'D:\GitHubProj\dino\simsiam\ele_seg_2\model_list\layer_index'
    # path_model_folder = r'D:\GitHubProj\dino\simsiam\ele_seg_2\model_list\win_len_V_Up2'
    path_model_folder = r'D:\GitHubProj\dino\simsiam\ele_seg_2\model_list\up2_choices'
    path_model_list = traverseFolder(path_model_folder)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    for i in range(len(path_model_list)):
        # print(path_model_list[i])

        path_t = path_model_list[i]
        if path_t.__contains__('model_VL_'):
            # model = model_VL(384, 1)
            model = model_VL(38, 1)
            model_str = 'model_VL'
        elif path_t.__contains__('model_V_'):
            # model = model_V(384, 1)
            # model = model_V(38, 1)
            model = model_V(8, 1)
            model_str = 'model_V'
        elif path_t.__contains__('model_S_'):
            # model = model_S(num_in_dim=384, num_out_dim=1)
            model = model_S(num_in_dim=38, num_out_dim=1)
            model_str = 'model_S'

        print('Load weights {}.'.format(path_t))
        pretrained_dict = torch.load(path_t, map_location=device)
        model.load_state_dict(pretrained_dict['state_dict'])
        windows_length = pretrained_dict['windows_length']
        windows_step = pretrained_dict['windows_step']
        # windows_step = windows_length
        # layer_index = pretrained_dict['layer_index']
        Init_lr = pretrained_dict['Init_lr']
        print('current model {} para:\nwindows_length:{}\twindows_step:{}\tInit_lr:{}'.format(
            model_str, windows_length, windows_step, Init_lr))

        # test_data_path = r'D:\Data\img_seg_data_in\train\1'
        # test_data_path = r'D:\Data\pic_seg_choices\data_all'

        # test_data_path = r'D:\Data\pic_seg_choices\data_way1_test\data_single'
        test_data_path = r'D:\Data\pic_seg_choices\data_way1_test\data_mix'
        test_model_by_para(model, data_path=test_data_path)

