import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from simCLR.pic_pre_process import get_pic_random
from simsiam.ele_seg_2.model_fracture_seg import model_S, model_V
from simsiam.sim_model.model_stage_1_vit import vit_simsiam
from src_ele.dir_operation import traverseFolder
from src_ele.file_operation import get_ele_data_from_path
import vision_transformer as vits
import math
from src_ele.pic_opeeration import show_Pic


path_folder_model = r'D:\Data\target_answer\model_list'
# dataloader中的所有处理都尽量放在GPU上处理，出进来的不论是什么参数，都尽量先往GPU上放
# 同样的，模型也要尽量放在GPU上
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# 配置 训练好的模型参数
if torch.cuda.is_available():
    pretrained_weights = r''
else:
    pretrained_weights = path_folder_model+'/checkpoint_VIT_8patch_56_0014.pth.tar'
    # pretrained_weights = r'D:\Data\target_answer\vit_small-64\checkpoint_vit_64_0014.pth.tar'

image_size = (224, 224)
patch_size = 8
model = vits.__dict__['vit_small'](patch_size=patch_size, in_chans=2, num_classes=0)
model_temp = vit_simsiam()
for p in model.parameters():
    p.requires_grad = False

if os.path.isfile(pretrained_weights):
    state_dict = torch.load(pretrained_weights, map_location=device)
    model_temp.load_state_dict(state_dict['model_dict'])

    dict_temp = {}
    for name, param in model.named_parameters():
        for name_t, param_t in model_temp.named_parameters():
            if name_t.__contains__(name):
                dict_temp.update({name: param_t})
                break
    model.load_state_dict(dict_temp)
else:
    print('error pretrained_weights path')
    exit(0)
# model.eval()就是帮我们一键搞定Dropout层和batch normalization层
model.eval()
model.to(device)
del model_temp

# 获取最后一层的attention的feature map
def get_attention_maps(imgs = np.array([])):
    imgs = imgs.reshape(1, 2, imgs.shape[-2], imgs.shape[-1])

    img = torch.from_numpy(imgs).float()
    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size
    attentions = model.get_last_selfattention(img.to(device))
    nh = attentions.shape[1]  # number of head       # nh = 6
    # # # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    attentions = attentions.reshape(nh, w_featmap, h_featmap)

    # print(attentions.shape)
    if torch.cuda.is_available():
        attentions = attentions.cpu().numpy()
    else:
        attentions = attentions.numpy()

    return attentions
    # # torch.unsqueeze扩展维度
    # # 返回一个新的张量，对输入的既定位置插入维度 1
    # attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()


# 根据feature pics获取零散的feature pics，把其打碎
# 刻意强调了，碎片的迭代顺序，使用了顺序结构对碎片进行叠加，
# 但是丢弃了右边缘、下边缘的数据
# 这个函数适合用来收集模型训练的数据集
# 输入1 * dim * w * h
def get_windows_pic_from_feature_maps(feature, pic_mask, windows_length=7, step=1):
    pic_mask = cv2.resize(pic_mask, (feature.shape[-2], feature.shape[-1]))
    x_num = (feature.shape[-2]-windows_length)//step+1
    y_num = (feature.shape[-1]-windows_length)//step+1
    a, b, c, d = feature.shape

    for i in range(x_num):
        for j in range(y_num):
            index_x = i*step
            index_y = j*step
            if (index_x == 0) & (index_y == 0):
                in_pic_split = feature[:, :, index_x:index_x+windows_length, index_y:index_y+windows_length].reshape((a, b, windows_length, windows_length))
                mask_out = pic_mask[index_x:index_x+windows_length, index_y:index_y+windows_length].reshape((-1, windows_length, windows_length))
            else:
                p_t = feature[:, :, index_x:index_x+windows_length, index_y:index_y+windows_length]
                m_t = pic_mask[index_x:index_x+windows_length, index_y:index_y+windows_length].reshape((-1, windows_length, windows_length))
                in_pic_split = np.append(in_pic_split, p_t, axis=0)
                mask_out = np.append(mask_out, m_t, axis=0)

    return in_pic_split, mask_out



# 这个函数将feature pics图片打碎
# 函数强调了对右边缘、下边缘的处理，使其尽可能提取全部的图片信息
# 这个函数适合用来进行图像分割预测、测试、整图生成
# 输入1 * dim * w * h
def get_windows_pic_from_feature_maps_FULLY(feature, pic_mask, windows_length=7, step=3):
    pic_mask = cv2.resize(pic_mask, (feature.shape[-2], feature.shape[-1]))
    x_num = math.ceil((feature.shape[-2]-windows_length)/step) + 1
    y_num = math.ceil((feature.shape[-1]-windows_length)/step) + 1
    a, b, c, d = feature.shape

    mask_out = []
    in_pic_split = []
    for i in range(x_num):
        for j in range(y_num):
            index_x = i*step
            index_y = j*step
            if (i == 0) & (j == 0):
                pic_t = feature[:, :, index_x:index_x+windows_length, index_y:index_y+windows_length].reshape((a, b, windows_length, windows_length))
                mask_t = pic_mask[index_x:index_x+windows_length, index_y:index_y+windows_length].reshape((-1, windows_length, windows_length))
                in_pic_split = pic_t
                mask_out = mask_t
                continue
            elif (i == x_num-1) & (j != y_num-1):
                p_t = feature[:, :, -windows_length:, index_y:index_y+windows_length].reshape((a, b, windows_length, windows_length))
                m_t = pic_mask[-windows_length:, index_y:index_y+windows_length].reshape((-1, windows_length, windows_length))
            elif (i != x_num-1) & (j == y_num-1):
                p_t = feature[:, :, index_x:index_x+windows_length, -windows_length:].reshape((a, b, windows_length, windows_length))
                m_t = pic_mask[index_x:index_x+windows_length, -windows_length:].reshape((-1, windows_length, windows_length))
            elif (i == x_num-1) & (j == y_num-1):
                p_t = feature[:, :, -windows_length:, -windows_length:].reshape((a, b, windows_length, windows_length))
                m_t = pic_mask[-windows_length:, -windows_length:].reshape((-1, windows_length, windows_length))
            else:
                p_t = feature[:, :, index_x:index_x+windows_length, index_y:index_y+windows_length].reshape((a, b, windows_length, windows_length))
                m_t = pic_mask[index_x:index_x+windows_length, index_y:index_y+windows_length].reshape((-1, windows_length, windows_length))

            in_pic_split = np.append(in_pic_split, p_t, axis=0)
            mask_out = np.append(mask_out, m_t, axis=0)

    in_pic_split = np.array(in_pic_split)
    mask_out = np.array(mask_out)

    return in_pic_split, mask_out



# 这个函数，将碎片化的 图像mask拼接到一起，这些碎片的重复部分不多，重复部分直接覆盖处理了，非常不推荐这种方式，
# 输入num * win_l * win_l
def windows_chip_2_full_pic_no_repeate(maskarray, windows_length=14, full_pic_shape=(28, 28)):
    step = windows_length
    x_num = math.ceil(full_pic_shape[0]/windows_length)
    y_num = math.ceil(full_pic_shape[1]/windows_length)
    mask_full = np.zeros(full_pic_shape)
    for i in range(x_num):
        for j in range(y_num):
            index_x = i*step
            index_y = j*step
            index_mkar = i*x_num+j
            if (i==x_num-1) & (j==y_num-1):
                mask_full[-windows_length:, -windows_length:] = maskarray[index_mkar]
            elif (i==x_num-1) & (j!=y_num-1):
                mask_full[-windows_length:, index_y:index_y+windows_length] = maskarray[index_mkar]
            elif (i!=x_num-1) & (j==y_num-1):
                mask_full[index_x:index_x+windows_length, -windows_length:] = maskarray[index_mkar]
            else:
                mask_full[index_x:index_x+windows_length, index_y:index_y+windows_length] = maskarray[index_mkar]

    return mask_full


# 输入num * win_l * win_l
# 这个函数，将碎片化的 图像mask拼接到一起，这些碎片的重复部分较多，重复部分取平均值处理，推荐这种方式，
def windows_chip_2_full_pic_repeated(maskarray, windows_length=14, step=14, full_pic_shape=(28, 28)):
    # print('maskarray shape:{}'.format(maskarray.shape))

    x_num = math.ceil((full_pic_shape[0]-windows_length)/step)+1
    y_num = math.ceil((full_pic_shape[1]-windows_length)/step)+1

    mask_full = np.zeros(full_pic_shape)
    mask_weight = np.zeros(full_pic_shape)
    windows_weight = np.ones((windows_length, windows_length))

    for i in range(x_num):
        for j in range(y_num):
            index_x = i * step
            index_y = j * step
            index_mkar = i * x_num + j
            if (i == x_num - 1) & (j == y_num - 1):
                mask_full[-windows_length:, -windows_length:] += maskarray[index_mkar, :, :]
                mask_weight[-windows_length:, -windows_length:] += windows_weight
            elif (i == x_num - 1) & (j != y_num - 1):
                mask_full[-windows_length:, index_y:index_y + windows_length] += maskarray[index_mkar, :, :]
                mask_weight[-windows_length:, index_y:index_y + windows_length] += windows_weight
            elif (i != x_num - 1) & (j == y_num - 1):
                mask_full[index_x:index_x + windows_length, -windows_length:] += maskarray[index_mkar, :, :]
                mask_weight[index_x:index_x + windows_length, -windows_length:] += windows_weight
            else:
                mask_full[index_x:index_x + windows_length, index_y:index_y + windows_length] += maskarray[index_mkar, :, :]
                mask_weight[index_x:index_x + windows_length, index_y:index_y + windows_length] += windows_weight

    return mask_full/mask_weight


# 这个函数，将碎片化的 图像mask拼接到一起，这些碎片全都是重复的
# in :torch.Size([16, 1, 14, 14])
def windows_chip_2_full_pic_no_repeate(maskarray, windows_length=14, step=3, full_pic_shape=(28, 28)):
    step = windows_length
    x_num = math.ceil(full_pic_shape[0] / windows_length)
    y_num = math.ceil(full_pic_shape[1] / windows_length)
    mask_full = np.zeros(full_pic_shape)
    for i in range(x_num):
        for j in range(y_num):
            index_x = i * step
            index_y = j * step
            index_mkar = i * x_num + j
            if (i == x_num - 1) & (j == y_num - 1):
                mask_full[-windows_length:, -windows_length:] = maskarray[index_mkar]
            elif (i == x_num - 1) & (j != y_num - 1):
                mask_full[-windows_length:, index_y:index_y + windows_length] = maskarray[index_mkar]
            elif (i != x_num - 1) & (j == y_num - 1):
                mask_full[index_x:index_x + windows_length, -windows_length:] = maskarray[index_mkar]
            else:
                mask_full[index_x:index_x + windows_length, index_y:index_y + windows_length] = maskarray[index_mkar]

    return mask_full


# 获取中间transformer层的所有 embedding feature
# small Vit一共是12层
def get_inter_feature_maps(imgs = np.array([])):
    imgs = imgs.reshape(1, 2, imgs.shape[-2], imgs.shape[-1])

    img = torch.from_numpy(imgs).float()
    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size
    f_a = model.get_all_layers_feature(img.to(device))

    B, N, C = f_a[0].shape                      # 1*785*384
    embedding_pic_list = []
    pic_len = int(pow(N, 0.5))                  # 28

    for i in range(len(f_a)):
        d_t = f_a[i].transpose(1, 2)[:, :, 1:].reshape(B, C, pic_len, pic_len)          # 1*384*28*28
        d_t = d_t.cpu().numpy()
        embedding_pic_list.append(d_t)

    return np.array(embedding_pic_list)

# 图形装配，用来将tensor图像转换为可显示的list格式
# def pic_allocated(embedding_pic, show_N=9):
#     num, dim, W, L = embedding_pic.shape
#     a = embedding_pic.reshape(-1, W, L)
#
#     p_l = []
#     for i in range(show_N):
#         p_l.append(a[i*dim, :, :].cpu().numpy()*256)
#
#     return p_l

# 返回的是，文件夹下的，随机一张图片的，经过处理后的两张图像增强照片，用于阶段1的无监督学习的对比学习
class dataloader_bottle(Dataset):
    def __init__(self, path=r'D:\Data\img_seg_data_in\train', windows_length=14, windows_step=2, layer_index=8):
        super().__init__()
        self.list_all_file = traverseFolder(path)
        self.length = len(self.list_all_file)//3
        self.RB_index = np.random.randint(0, 2)
        self.windows_length = windows_length
        self.windows_step = windows_step
        self.layer_index = layer_index

    def __getitem__(self, index):
        path_temp = self.list_all_file[index*3]
        path_temp_stat = ''
        # print(path_temp)

        if path_temp.__contains__('dyna'):
            path_temp_stat = path_temp.replace('dyna', 'stat')
            path_temp_mask = path_temp.replace('dyna', 'mask')
        elif path_temp.__contains__('stat'):
            path_temp_stat = path_temp
            path_temp = path_temp_stat.replace('stat', 'dyna')
            path_temp_mask = path_temp_stat.replace('stat', 'mask')

        # print(path_temp)

        pic_dyna, depth = get_ele_data_from_path(path_temp)
        pic_stat, depth = get_ele_data_from_path(path_temp_stat)
        pic_mask, _ = get_ele_data_from_path(path_temp_mask)
        pic_mask = cv2.resize(pic_mask, (pic_dyna.shape[1], pic_dyna.shape[0]))

        # show_Pic([pic_dyna, pic_stat], pic_order='12', pic_str=['', ''], save_pic=False, path_save='')
        self.pic_all = np.array([pic_dyna, pic_stat, pic_mask])

        pic_shape = (224, 224)
        pic_all_New, self.RB_index = get_pic_random(self.pic_all, depth, self.RB_index, pic_shape=pic_shape)

        # # 图像缩放、将图像进行2、4、8、16倍数的缩放
        # pic_zoom = []
        # mi_t = 1
        # for i in range(4):
        #     shape_t = (pic_shape[0]//mi_t, pic_shape[1]//mi_t)
        #     p1 = cv2.resize(pic_all_New[0, :, :], dsize=shape_t)
        #     p2 = cv2.resize(pic_all_New[0, :, :], dsize=shape_t)
        #     p3 = cv2.resize(pic_all_New[0, :, :], dsize=shape_t)
        #     pic_n_t = np.array([p1, p2])
        #     mi_t = mi_t * 2
        #     pic_zoom.append(pic_n_t)

        # 获取图片经attention模型的所有中间层的特征
        all_feature_list = get_inter_feature_maps(pic_all_New[:2, :, :])

        # # 对所有中间层特征进行图片分割
        # embedding_pic_split = []
        # mask_split = []
        # for i in range(all_feature_list.shape[0]):
        #     x_t, y_t = get_windows_pic_from_feature_maps(all_feature_list[i], pic_all_New[-1, :, :],
        #                                                  windows_length=self.windows_length, step=self.windows_step)
        #     embedding_pic_split.append(x_t)
        #     mask_split.append(y_t)
        # embedding_pic_split = np.array(embedding_pic_split)
        # mask_split = np.array(mask_split)
        # # 仅对指定的self.layer_index层进行特征分割
        # all_feature_list[self.layer_index].shape  :
        embedding_pic_split, mask_split = get_windows_pic_from_feature_maps(all_feature_list[self.layer_index], pic_all_New[-1, :, :],
                                                                windows_length=self.windows_length, step=self.windows_step)

        return pic_all_New, embedding_pic_split, mask_split

    def __len__(self):
        return self.length


# 返回的是，文件夹下的，随机一张图片的，经过处理后的两张图像增强照片，用于阶段1的无监督学习的对比学习
class dataloader_base(Dataset):
    def __init__(self, path=r'D:\Data\img_seg_data_in\train', pre_process=True):
        super().__init__()
        self.list_all_file = traverseFolder(path)
        self.length = len(self.list_all_file)//3
        self.RB_index = np.random.randint(0, 2)
        self.pre_process = pre_process

    def __getitem__(self, index):
        path_temp = self.list_all_file[index*3]
        path_temp_stat = ''
        # print(path_temp)

        if path_temp.__contains__('dyna'):
            path_temp_stat = path_temp.replace('dyna', 'stat')
            path_temp_mask = path_temp.replace('dyna', 'mask')
        elif path_temp.__contains__('stat'):
            path_temp_stat = path_temp
            path_temp = path_temp_stat.replace('stat', 'dyna')
            path_temp_mask = path_temp_stat.replace('stat', 'mask')

        # print(path_temp)

        pic_dyna, depth = get_ele_data_from_path(path_temp)
        pic_stat, depth = get_ele_data_from_path(path_temp_stat)
        pic_mask, _ = get_ele_data_from_path(path_temp_mask)
        pic_mask = cv2.resize(pic_mask, (pic_dyna.shape[1], pic_dyna.shape[0]))

        # show_Pic([pic_dyna, pic_stat], pic_order='12', pic_str=['', ''], save_pic=False, path_save='')
        self.pic_all = np.array([pic_dyna, pic_stat, pic_mask])

        pic_shape = (224, 224)
        if self.pre_process:
            pic_all_New, self.RB_index = get_pic_random(self.pic_all, depth, self.RB_index, pic_shape=pic_shape)
        else:
            pic_all_New = np.array([cv2.resize(pic_dyna, pic_shape)/256, cv2.resize(pic_stat, pic_shape)/256, cv2.resize(pic_mask, pic_shape)/256])

        # # 图像缩放、将图像进行2、4、8、16倍数的缩放
        # pic_zoom = []
        # mi_t = 1
        # for i in range(4):
        #     shape_t = (pic_shape[0]//mi_t, pic_shape[1]//mi_t)
        #     p1 = cv2.resize(pic_all_New[0, :, :], dsize=shape_t)
        #     p2 = cv2.resize(pic_all_New[0, :, :], dsize=shape_t)
        #     p3 = cv2.resize(pic_all_New[0, :, :], dsize=shape_t)
        #     pic_n_t = np.array([p1, p2])
        #     mi_t = mi_t * 2
        #     pic_zoom.append(pic_n_t)

        # # 获取图片经attention模型的所有中间层的特征
        all_feature_list = get_inter_feature_maps(pic_all_New[:2, :, :])

        return pic_all_New, all_feature_list

    def __len__(self):
        return self.length



if torch.cuda.is_available():
    pass
else:
    model_bottle_folder_path = path_folder_model+'/bottle_choices'

model_bottle_path_list = traverseFolder(model_bottle_folder_path)

win_len_l = []
win_step_l = []
layer_index_l = []
model_bottle_list = []
for i in range(len(model_bottle_path_list)):
    pretrained_dict = torch.load(model_bottle_path_list[i], map_location=device)

    if model_bottle_path_list[i].__contains__('model_V_'):
        model_t = model_V(num_in_dim=384, num_out_dim=1).eval().to(device)

    model_t.load_state_dict(pretrained_dict['state_dict'])
    model_bottle_list.append(model_t)
    win_len_l.append(pretrained_dict['windows_length'])
    win_step_l.append(pretrained_dict['windows_step'])
    layer_index_l.append(pretrained_dict['layer_index'])


# 返回的是，文件夹下的，随机一张图片的，经过处理后的两张图像增强照片，用于阶段1的无监督学习的对比学习
class dataloader_up1(Dataset):
    def __init__(self, path=r'D:\Data\pic_seg_choices\data_train', windows_length=20, windows_step=4):
        super().__init__()
        self.list_all_file = traverseFolder(path)
        self.length = len(self.list_all_file)//3
        self.RB_index = np.random.randint(0, 2)
        self.windows_length = windows_length
        self.windows_step = windows_step

    def __getitem__(self, index):
        path_temp = self.list_all_file[index*3]
        path_temp_stat = ''
        # print(path_temp)

        if path_temp.__contains__('dyna'):
            path_temp_stat = path_temp.replace('dyna', 'stat')
            path_temp_mask = path_temp.replace('dyna', 'mask')
        elif path_temp.__contains__('stat'):
            path_temp_stat = path_temp
            path_temp = path_temp_stat.replace('stat', 'dyna')
            path_temp_mask = path_temp_stat.replace('stat', 'mask')

        pic_dyna, depth = get_ele_data_from_path(path_temp)
        pic_stat, depth = get_ele_data_from_path(path_temp_stat)
        pic_mask, _ = get_ele_data_from_path(path_temp_mask)
        pic_mask = cv2.resize(pic_mask, (pic_dyna.shape[1], pic_dyna.shape[0]))

        # show_Pic([pic_dyna, pic_stat], pic_order='12', pic_str=['', ''], save_pic=False, path_save='')
        self.pic_all = np.array([pic_dyna, pic_stat, pic_mask])

        pic_shape = (224, 224)
        pic_all_New, self.RB_index = get_pic_random(self.pic_all, depth, self.RB_index, pic_shape=pic_shape)

        # 图像缩放、将图像进行2、4、8、16倍数的缩放
        # up1阶段(28*28)--》(56*56)阶段，这里只需把(224*224)的图片缩放4倍，无需其他倍率缩放
        shape_t = (pic_shape[0]//4, pic_shape[1]//4)
        p1 = cv2.resize(pic_all_New[0, :, :], dsize=shape_t)
        p2 = cv2.resize(pic_all_New[1, :, :], dsize=shape_t)
        p3 = cv2.resize(pic_all_New[2, :, :], dsize=shape_t)
        # pic_all_new_scaled = np.array([p1, p2, p3])

        # 获取图片经attention模型的所有中间层的特征
        all_feature_list = get_inter_feature_maps(pic_all_New[:2, :, :])

        # embedding_pic_split, mask_split = get_windows_pic_from_feature_maps(all_feature_list[self.layer_index], pic_all_New[-1, :, :],
        #                                                         windows_length=self.windows_length, step=self.windows_step)

        mask_bottle_list = []
        for i in range(len(model_bottle_list)):
            in_pic_split, mask_out = get_windows_pic_from_feature_maps_FULLY(
                all_feature_list[layer_index_l[i]], pic_all_New[-1, :, :], windows_length=win_len_l[i], step=win_step_l[i])
            a, b, c = mask_out.shape
            x = torch.from_numpy(in_pic_split).type(torch.FloatTensor).reshape(a, -1, b, c)

            outputs = model_bottle_list[i](x.to(device))
            outputs = outputs.reshape(-1, b, c).cpu().detach().numpy()
            out_mask_full = windows_chip_2_full_pic_repeated(outputs, windows_length=win_len_l[i], step=win_step_l[i], full_pic_shape=(28, 28))
            # print(i, out_mask_full.shape, 2*out_mask_full.shape)

            # 如果要缩小图像，建议选择：cv2.INTER_AREA；如果要放大图像，cv2.INTER_CUBIC效果更好但是速度慢，cv2.INTER_LINEAR效果尚可且速度快。
            # cv2.INTER_NEAREST 最近邻插值                      cv2.INTER_LINEAR    双线性插值（默认）
            # cv2.INTER_AREA    使用像素区域关系进行重采样。
            # cv2.INTER_CUBIC   4x4像素邻域的双3次插值           cv2.INTER_LANCZOS4  8x8像素邻域的Lanczos插值
            # 这里使用两种方式进行上采样，增加上采样特征图像的多样性
            shape_n = (2*out_mask_full.shape[0], 2*out_mask_full.shape[1])
            out_mask_full_up1 = cv2.resize(out_mask_full, dsize=shape_n, interpolation=cv2.INTER_CUBIC)
            out_mask_full_up2 = cv2.resize(out_mask_full, dsize=shape_n, interpolation=cv2.INTER_LINEAR)

            # show_Pic([pic_dyna, pic_stat, pic_mask, p1*256, p2*256, p3*256, out_mask_full*256, out_mask_full_up1*256, out_mask_full_up2*256], pic_order='33')
            mask_bottle_list.append(np.array([out_mask_full_up1, out_mask_full_up2]))

        # 把缩放后的原始成像也放到这个feature list里
        mask_bottle_list.append(np.array([p1, p2]))
        feature_in = np.array(mask_bottle_list).reshape(1, -1, 56, 56)           # (3, 2, 56, 56)

        # print(feature_in.shape)
        feature_in_split, mask_org_split = get_windows_pic_from_feature_maps(
            feature_in, pic_all_New[-1, :, :], windows_length=self.windows_length, step=self.windows_step)
        # print(feature_in_split.shape, mask_org_split.shape)
        return pic_all_New, feature_in, feature_in_split, mask_org_split

    def __len__(self):
        return self.length



if torch.cuda.is_available():
    pass
else:
    model_bottle_folder_path = path_folder_model+'/up1_choices'
model_up1_path_list = traverseFolder(model_bottle_folder_path)

win_len_l_up1 = []
win_step_l_up1 = []
model_up1_list = []
for i in range(len(model_up1_path_list)):
    pretrained_dict = torch.load(model_up1_path_list[i], map_location=device)

    if model_up1_path_list[i].__contains__('model_V_'):
        model_t = model_V(num_in_dim=38, num_out_dim=1).eval().to(device)

    model_t.load_state_dict(pretrained_dict['state_dict'])
    model_up1_list.append(model_t)
    win_len_l_up1.append(pretrained_dict['windows_length'])
    win_step_l_up1.append(pretrained_dict['windows_step'])
# 返回的是，文件夹下的，随机一张图片的，经过处理后的两张图像增强照片，用于阶段1的无监督学习的对比学习
class dataloader_up2(Dataset):
    def __init__(self, path=r'D:\Data\pic_seg_choices\data_train', windows_length=40, windows_step=8):
        super().__init__()
        self.list_all_file = traverseFolder(path)
        self.length = len(self.list_all_file)//3
        self.RB_index = np.random.randint(0, 2)
        self.windows_length = windows_length
        self.windows_step = windows_step

    def __getitem__(self, index):
        path_temp = self.list_all_file[index*3]
        path_temp_stat = ''
        # print(path_temp)

        if path_temp.__contains__('dyna'):
            path_temp_stat = path_temp.replace('dyna', 'stat')
            path_temp_mask = path_temp.replace('dyna', 'mask')
        elif path_temp.__contains__('stat'):
            path_temp_stat = path_temp
            path_temp = path_temp_stat.replace('stat', 'dyna')
            path_temp_mask = path_temp_stat.replace('stat', 'mask')

        pic_dyna, depth = get_ele_data_from_path(path_temp)
        pic_stat, depth = get_ele_data_from_path(path_temp_stat)
        pic_mask, _ = get_ele_data_from_path(path_temp_mask)
        pic_mask = cv2.resize(pic_mask, (pic_dyna.shape[1], pic_dyna.shape[0]))

        # show_Pic([pic_dyna, pic_stat], pic_order='12', pic_str=['', ''], save_pic=False, path_save='')
        self.pic_all = np.array([pic_dyna, pic_stat, pic_mask])

        pic_shape = (224, 224)
        pic_all_New, self.RB_index = get_pic_random(self.pic_all, depth, self.RB_index, pic_shape=pic_shape)

        # 图像缩放、将图像进行2、4、8、16倍数的缩放
        # up1阶段(28*28)--》(56*56)阶段，这里只需把(224*224)的图片缩放4倍，无需其他倍率缩放
        shape_t = (pic_shape[0]//4, pic_shape[1]//4)
        p1 = cv2.resize(pic_all_New[0, :, :], dsize=shape_t)
        p2 = cv2.resize(pic_all_New[1, :, :], dsize=shape_t)
        p3 = cv2.resize(pic_all_New[2, :, :], dsize=shape_t)
        # pic_all_new_scaled = np.array([p1, p2, p3])

        # 获取图片经attention模型的所有中间层的特征
        all_feature_list = get_inter_feature_maps(pic_all_New[:2, :, :])

        # embedding_pic_split, mask_split = get_windows_pic_from_feature_maps(all_feature_list[self.layer_index], pic_all_New[-1, :, :],
        #                                                         windows_length=self.windows_length, step=self.windows_step)

        mask_bottle_list = []
        for i in range(len(model_bottle_list)):
            in_pic_split, mask_out = get_windows_pic_from_feature_maps_FULLY(
                all_feature_list[layer_index_l[i]], pic_all_New[-1, :, :], windows_length=win_len_l[i], step=win_step_l[i])
            a, b, c = mask_out.shape
            x = torch.from_numpy(in_pic_split).type(torch.FloatTensor).reshape(a, -1, b, c)

            outputs = model_bottle_list[i](x.to(device))
            outputs = outputs.reshape(-1, b, c).cpu().detach().numpy()
            out_mask_full = windows_chip_2_full_pic_repeated(outputs, windows_length=win_len_l[i], step=win_step_l[i], full_pic_shape=(28, 28))
            # print(i, out_mask_full.shape, 2*out_mask_full.shape)

            # 如果要缩小图像，建议选择：cv2.INTER_AREA；如果要放大图像，cv2.INTER_CUBIC效果更好但是速度慢，cv2.INTER_LINEAR效果尚可且速度快。
            # cv2.INTER_NEAREST 最近邻插值                      cv2.INTER_LINEAR    双线性插值（默认）
            # cv2.INTER_AREA    使用像素区域关系进行重采样。
            # cv2.INTER_CUBIC   4x4像素邻域的双3次插值           cv2.INTER_LANCZOS4  8x8像素邻域的Lanczos插值
            # 这里使用两种方式进行上采样，增加上采样特征图像的多样性
            shape_n = (2*out_mask_full.shape[0], 2*out_mask_full.shape[1])
            out_mask_full_up1 = cv2.resize(out_mask_full, dsize=shape_n, interpolation=cv2.INTER_CUBIC)
            out_mask_full_up2 = cv2.resize(out_mask_full, dsize=shape_n, interpolation=cv2.INTER_LINEAR)

            # show_Pic([pic_dyna, pic_stat, pic_mask, p1*256, p2*256, p3*256, out_mask_full*256, out_mask_full_up1*256, out_mask_full_up2*256], pic_order='33')
            mask_bottle_list.append(np.array([out_mask_full_up1, out_mask_full_up2]))

        # 把缩放后的原始成像也放到这个feature list里
        mask_bottle_list.append(np.array([p1, p2]))
        feature_in = np.array(mask_bottle_list).reshape(1, -1, 56, 56)           # (3, 2, 56, 56)

        # # print(feature_in.shape)
        # feature_in_split, mask_org_split = get_windows_pic_from_feature_maps(
        #     feature_in, pic_all_New[-1, :, :], windows_length=self.windows_length, step=self.windows_step)

        shape_t_up1 = (pic_shape[0] // 2, pic_shape[1] // 2)
        p1_up1 = cv2.resize(pic_all_New[0, :, :], dsize=shape_t_up1)
        p2_up1 = cv2.resize(pic_all_New[1, :, :], dsize=shape_t_up1)
        p3_up1 = cv2.resize(pic_all_New[-1, :, :], dsize=shape_t_up1)

        mask_up1_list = []
        for i in range(len(model_up1_list)):
            in_pic_split, mask_out = get_windows_pic_from_feature_maps_FULLY(
                feature_in, pic_all_New[-1, :, :], windows_length=win_len_l_up1[i], step=win_step_l_up1[i])
            a, b, c = mask_out.shape
            x = torch.from_numpy(in_pic_split).type(torch.FloatTensor).reshape(a, -1, b, c)

            outputs = model_up1_list[i](x.to(device))
            outputs = outputs.reshape(-1, b, c).cpu().detach().numpy()
            out_mask_full = windows_chip_2_full_pic_repeated(outputs, windows_length=win_len_l_up1[i], step=win_step_l_up1[i],
                                                             full_pic_shape=(56, 56))

            # 如果要缩小图像，建议选择：cv2.INTER_AREA；如果要放大图像，cv2.INTER_CUBIC效果更好但是速度慢，cv2.INTER_LINEAR效果尚可且速度快。
            # cv2.INTER_NEAREST 最近邻插值                      cv2.INTER_LINEAR    双线性插值（默认）
            # cv2.INTER_AREA    使用像素区域关系进行重采样。
            # cv2.INTER_CUBIC   4x4像素邻域的双3次插值           cv2.INTER_LANCZOS4  8x8像素邻域的Lanczos插值
            # 这里使用两种方式进行上采样，增加上采样特征图像的多样性
            shape_n = (2 * out_mask_full.shape[0], 2 * out_mask_full.shape[1])
            out_mask_full_up1 = cv2.resize(out_mask_full, dsize=shape_n, interpolation=cv2.INTER_CUBIC)
            out_mask_full_up2 = cv2.resize(out_mask_full, dsize=shape_n, interpolation=cv2.INTER_LINEAR)

            # show_Pic([pic_all_New[0, :, :]*256, pic_all_New[1, :, :]*256, pic_all_New[-1, :, :]*256,
            #           p1_up1*256, p2_up1*256, p3_up1*256,
            #           out_mask_full*256, out_mask_full_up1*256, out_mask_full_up2*256], pic_order='33')
            mask_up1_list.append(np.array([out_mask_full_up1, out_mask_full_up2]))


        mask_up1_list.append(np.array([p1_up1, p2_up1]))
        feature_in = np.array(mask_up1_list).reshape(1, -1, 112, 112)           # (3, 2, 56, 56)

        feature_in_split, mask_org_split = get_windows_pic_from_feature_maps(
            feature_in, pic_all_New[-1, :, :], windows_length=self.windows_length, step=self.windows_step)

        return pic_all_New, feature_in, feature_in_split, mask_org_split

    def __len__(self):
        return self.length



if torch.cuda.is_available():
    pass
else:
    model_bottle_folder_path = path_folder_model+'/up2_choices'
model_up2_path_list = traverseFolder(model_bottle_folder_path)

win_len_l_up2 = []
win_step_l_up2 = []
model_up2_list = []
for i in range(len(model_up2_path_list)):
    pretrained_dict = torch.load(model_up2_path_list[i], map_location=device)

    if model_up2_path_list[i].__contains__('model_V_'):
        model_t = model_V(num_in_dim=8, num_out_dim=1).eval().to(device)

    model_t.load_state_dict(pretrained_dict['state_dict'])
    model_up2_list.append(model_t)
    win_len_l_up2.append(pretrained_dict['windows_length'])
    win_step_l_up2.append(pretrained_dict['windows_step'])
# 返回的是，文件夹下的，随机一张图片的，经过处理后的两张图像增强照片，用于阶段1的无监督学习的对比学习
class dataloader_up3(Dataset):
    def __init__(self, path=r'D:\Data\pic_seg_choices\data_all', preprocess=False, windows_length=40, windows_step=8):
    # def __init__(self, path=r'D:\Data\pic_seg_choices\temp', preprocess=False, windows_length=40, windows_step=8):
        super().__init__()
        self.list_all_file = traverseFolder(path)
        self.length = len(self.list_all_file)//3
        self.RB_index = np.random.randint(0, 2)
        self.windows_length = windows_length
        self.windows_step = windows_step
        self.pre_process = preprocess


    def __getitem__(self, index):
        path_temp = self.list_all_file[index*3]
        path_temp_stat = ''
        # print(path_temp)

        # 文件路径载入
        if path_temp.__contains__('dyna'):
            path_temp_stat = path_temp.replace('dyna', 'stat')
            path_temp_mask = path_temp.replace('dyna', 'mask')
        elif path_temp.__contains__('stat'):
            path_temp_stat = path_temp
            path_temp = path_temp_stat.replace('stat', 'dyna')
            path_temp_mask = path_temp_stat.replace('stat', 'mask')

        # 根据文件路径，载入文件图像数据
        pic_dyna, depth = get_ele_data_from_path(path_temp)
        pic_stat, depth = get_ele_data_from_path(path_temp_stat)
        pic_mask, _ = get_ele_data_from_path(path_temp_mask)
        pic_mask = cv2.resize(pic_mask, (pic_dyna.shape[1], pic_dyna.shape[0]))

        # 动态、静态、mask 三数据合并为一个
        # show_Pic([pic_dyna, pic_stat], pic_order='12', pic_str=['', ''], save_pic=False, path_save='')
        self.pic_all = np.array([pic_dyna, pic_stat, pic_mask])

        # pic_shape = (224, 224)
        # pic_all_New, self.RB_index = get_pic_random(self.pic_all, depth, self.RB_index, pic_shape=pic_shape)
        # 根据输入数据形状格式 重置 数据的形状
        pic_shape = (224, 224)
        if self.pre_process:
            pic_all_New, self.RB_index = get_pic_random(self.pic_all, depth, self.RB_index, pic_shape=pic_shape)
        else:
            pic_all_New = np.array([cv2.resize(pic_dyna, pic_shape) / 256, cv2.resize(pic_stat, pic_shape) / 256,
                                    cv2.resize(pic_mask, pic_shape) / 256])

        # 图像缩放、将图像进行2、4、8、16倍数的缩放
        # up1阶段(28*28)--》(56*56)阶段，这里只需把(224*224)的图片缩放4倍，无需其他倍率缩放
        shape_t = (pic_shape[0]//4, pic_shape[1]//4)
        p1 = cv2.resize(pic_all_New[0, :, :], dsize=shape_t)
        p2 = cv2.resize(pic_all_New[1, :, :], dsize=shape_t)
        p3 = cv2.resize(pic_all_New[2, :, :], dsize=shape_t)
        # pic_all_new_scaled = np.array([p1, p2, p3])

        # 获取图片经attention模型的所有中间层的特征
        all_feature_list = get_inter_feature_maps(pic_all_New[:2, :, :])

        # embedding_pic_split, mask_split = get_windows_pic_from_feature_maps(all_feature_list[self.layer_index], pic_all_New[-1, :, :],
        #                                                         windows_length=self.windows_length, step=self.windows_step)

        mask_bottle_list = []
        # 根据model_bottle_list里的model 使用不同参数配置来处理图像
        for i in range(len(model_bottle_list)):
            in_pic_split, mask_out = get_windows_pic_from_feature_maps_FULLY(
                # 根据 窗长win_len_l[i], 步长win_step_l[i] 来把第layer_index_l[i]层的 特征数据all_feature_list 进行切片，方便后续处理
                all_feature_list[layer_index_l[i]], pic_all_New[-1, :, :], windows_length=win_len_l[i], step=win_step_l[i])
            a, b, c = mask_out.shape
            x = torch.from_numpy(in_pic_split).type(torch.FloatTensor).reshape(a, -1, b, c)

            outputs = model_bottle_list[i](x.to(device))
            outputs = outputs.reshape(-1, b, c).cpu().detach().numpy()
            # 把碎片的mask进行合并，整合成完整的mask
            out_mask_full = windows_chip_2_full_pic_repeated(outputs, windows_length=win_len_l[i], step=win_step_l[i], full_pic_shape=(28, 28))

            # 如果要缩小图像，建议选择：cv2.INTER_AREA；如果要放大图像，cv2.INTER_CUBIC效果更好但是速度慢，cv2.INTER_LINEAR效果尚可且速度快。
            # cv2.INTER_NEAREST 最近邻插值                      cv2.INTER_LINEAR    双线性插值（默认）
            # cv2.INTER_AREA    使用像素区域关系进行重采样。
            # cv2.INTER_CUBIC   4x4像素邻域的双3次插值           cv2.INTER_LANCZOS4  8x8像素邻域的Lanczos插值
            # 这里使用两种方式进行上采样，增加上采样特征图像的多样性
            shape_n = (2*out_mask_full.shape[0], 2*out_mask_full.shape[1])
            out_mask_full_up1 = cv2.resize(out_mask_full, dsize=shape_n, interpolation=cv2.INTER_CUBIC)
            out_mask_full_up2 = cv2.resize(out_mask_full, dsize=shape_n, interpolation=cv2.INTER_LINEAR)

            # show_Pic([pic_dyna, pic_stat, pic_mask, p1*256, p2*256, p3*256, out_mask_full*256, out_mask_full_up1*256, out_mask_full_up2*256], pic_order='33')
            mask_bottle_list.append(np.array([out_mask_full_up1, out_mask_full_up2]))

        # 把缩放后的原始成像也放到这个feature list里
        mask_bottle_list.append(np.array([p1, p2]))
        feature_in = np.array(mask_bottle_list).reshape(1, -1, 56, 56)           # (3, 2, 56, 56)

        # # print(feature_in.shape)
        # feature_in_split, mask_org_split = get_windows_pic_from_feature_maps(
        #     feature_in, pic_all_New[-1, :, :], windows_length=self.windows_length, step=self.windows_step)

        shape_t_up1 = (pic_shape[0] // 2, pic_shape[1] // 2)
        p1_up1 = cv2.resize(pic_all_New[0, :, :], dsize=shape_t_up1)
        p2_up1 = cv2.resize(pic_all_New[1, :, :], dsize=shape_t_up1)
        p3_up1 = cv2.resize(pic_all_New[-1, :, :], dsize=shape_t_up1)

        mask_up1_list = []
        for i in range(len(model_up1_list)):
            in_pic_split, mask_out = get_windows_pic_from_feature_maps_FULLY(
                feature_in, pic_all_New[-1, :, :], windows_length=win_len_l_up1[i], step = win_step_l_up1[i])
            a, b, c = mask_out.shape
            x = torch.from_numpy(in_pic_split).type(torch.FloatTensor).reshape(a, -1, b, c)

            outputs = model_up1_list[i](x.to(device))
            outputs = outputs.reshape(-1, b, c).cpu().detach().numpy()
            out_mask_full = windows_chip_2_full_pic_repeated(outputs, windows_length=win_len_l_up1[i], step = win_step_l_up1[i],
                                                             full_pic_shape=(56, 56))

            shape_n = (2 * out_mask_full.shape[0], 2 * out_mask_full.shape[1])
            out_mask_full_up1 = cv2.resize(out_mask_full, dsize=shape_n, interpolation=cv2.INTER_CUBIC)
            out_mask_full_up2 = cv2.resize(out_mask_full, dsize=shape_n, interpolation=cv2.INTER_LINEAR)

            # show_Pic([pic_all_New[0, :, :]*256, pic_all_New[1, :, :]*256, pic_all_New[-1, :, :]*256,
            #           p1_up1*256, p2_up1*256, p3_up1*256,
            #           out_mask_full*256, out_mask_full_up1*256, out_mask_full_up2*256], pic_order='33')
            mask_up1_list.append(np.array([out_mask_full_up1, out_mask_full_up2]))


        mask_up1_list.append(np.array([p1_up1, p2_up1]))
        feature_in = np.array(mask_up1_list).reshape(1, -1, 112, 112)           # (3, 2, 56, 56)

        # up2阶段
        mask_up2_list = []
        for i in range(len(model_up2_list)):
            in_pic_split, mask_out = get_windows_pic_from_feature_maps_FULLY(
                feature_in, pic_all_New[-1, :, :], windows_length=win_len_l_up2[i], step = win_step_l_up2[i])
            a, b, c = mask_out.shape
            x = torch.from_numpy(in_pic_split).type(torch.FloatTensor).reshape(a, -1, b, c)

            outputs = model_up2_list[i](x.to(device))
            outputs = outputs.reshape(-1, b, c).cpu().detach().numpy()
            out_mask_full = windows_chip_2_full_pic_repeated(outputs, windows_length=win_len_l_up2[i],
                                                             step=win_step_l_up2[i],
                                                             full_pic_shape=(112, 112))

            # 这里使用两种方式进行上采样，增加上采样特征图像的多样性
            shape_n = (2 * out_mask_full.shape[0], 2 * out_mask_full.shape[1])
            out_mask_full_up1 = cv2.resize(out_mask_full, dsize=shape_n, interpolation=cv2.INTER_CUBIC)
            out_mask_full_up2 = cv2.resize(out_mask_full, dsize=shape_n, interpolation=cv2.INTER_LINEAR)

            # # show_Pic([pic_all_New[0, :, :]*256, pic_all_New[1, :, :]*256, pic_all_New[-1, :, :]*256,
            # #           out_mask_full*256, out_mask_full_up1*256, out_mask_full_up2*256], pic_order='23')
            name_1 = '{}_{}_up3.png'.format(index, i)
            cv2.imwrite(name_1, out_mask_full*256)
            name_2 = '{}_{}_up3_X2_1.png'.format(index, i)
            cv2.imwrite(name_2, out_mask_full_up1*256)
            name_3 = '{}_{}_up3_X2_2.png'.format(index, i)
            cv2.imwrite(name_3, out_mask_full_up2*256)
            name_4 = '{}_dyna.png'.format(index)
            cv2.imwrite(name_4, pic_all_New[0, :, :]*256)

            mask_up2_list.append(np.array([out_mask_full_up1, out_mask_full_up2]))

        shape_t_up1 = (pic_shape[0], pic_shape[1])
        p1_up2 = cv2.resize(pic_all_New[0, :, :], dsize=shape_t_up1)
        p2_up2 = cv2.resize(pic_all_New[1, :, :], dsize=shape_t_up1)

        mask_up2_list.append(np.array([p1_up2, p2_up2]))
        feature_in = np.array(mask_up2_list).reshape(1, -1, 224, 224)  # (3, 2, 56, 56)

        feature_in_split, mask_org_split = get_windows_pic_from_feature_maps(
            feature_in, pic_all_New[-1, :, :], windows_length=self.windows_length, step=self.windows_step)

        return pic_all_New, feature_in, feature_in_split, mask_org_split

    def __len__(self):
        return self.length

# a = dataloader_bottle()
# print('pic_seg_data_train_in num is :{}'.format(a.length))
# pic_all_New, all_feature_list, embedding_pic_split, mask_split = a[0]
# print('pic_all_New:{}, all_feature_list:{}, embedding_pic_split:{}, mask_split:{}'.format(
#     pic_all_New.shape, all_feature_list.shape, embedding_pic_split.shape, mask_split.shape))
# show_Pic([embedding_pic_split[0][0, 0, :, :].cpu().numpy()*256, mask_split[0][0, 0, :, :].cpu().numpy()*256], '12')
# show_Pic([embedding_pic_split[0][-1, 0, :, :].cpu().numpy()*256, mask_split[0][-1, 0, :, :].cpu().numpy()*256], '12')


# # 查看12个通道，哪一个通道是顶层，哪一个通道时底层
# # 0是顶层，-1是底层
# for i in range(all_feature_list.shape[0]):
#     a = []
#     for j in range(9):
#         a.append(all_feature_list[i][0][j]*256)
#     show_Pic(a, pic_order='33')


# # 查看图像的分割效果
# a = []
# a.append(pic_all_New[0, :, :]*256)
# a.append(pic_all_New[1, :, :]*256)
# a.append(pic_all_New[2, :, :]*256)
# for i in range(3):
#     a.append(embedding_pic_split[-1, i*4, 0, :, :]*256)
#     a.append(mask_split[0, i*4, :, :]*256)
# show_Pic(a, pic_order='33')


# (a.length)
# print(a[0][0].shape)

