import copy
import math
import random
import cv2
import numpy as np
from src_ele.file_operation import get_test_ele_data
from src_ele.pic_opeeration import show_Pic, pic_enhence_random, pic_scale_normal


# def show_curve(curve_list, curve_order='12', pic_str=[], save_pic=False, path_save=''):
#     if len(curve_order) != 2:
#         print('pic order error:{}'.format(curve_order))
#
#     num = int(curve_order[0]) * int(curve_order[1])
#
#     if num != len(curve_list):
#         print('pic order num is not equal to pic_list num:{},{}'.format(len(pic_list), curve_order))
#
#     # print(num)
#     while( len(pic_str) < len(curve_list)):
#         pic_str.append('pic_str'+str(len(curve_list)-len(pic_str)))
#     # print(pic_str)
#
#     for i in range(len(curve_list)):
#         for j in range(curve_list[i].shape[0]):
#             for k in range(curve_list[i].shape[1]):
#                 if curve_list[i][j][k] < 0:
#                     curve_list[i][j][k] = 0
#
#     plt.close('all')
#     fig = plt.figure(figsize=(16, 9))
#     for i in range(len(curve_list)):
#         order_str = int(curve_order+str(i+1))
#         ax = fig.add_subplot(order_str)
#         ax.set_title(pic_str[i])
#         plt.axis('off')
#         ax.imshow(curve_list[i], cmap='hot')
#         # ax.imshow(curve_list[i], cmap='afmhot')
#         # ax.imshow(curve_list[i], cmap='gist_heat')
#     plt.show()
#
#     if save_pic:
#         if path_save == '':
#             plt.savefig('temp.png')
#         else:
#             plt.savefig(path_save)
#         plt.close()

# 获取随机的方位角曲线，为了接下来的进行图像绕井壁旋转
def get_random_RB_curve(depth, start_angle=np.random.randint(-60, 60)):
    max_rotate_angle = 1
    Rb_random = np.zeros(depth.shape)

    for i in range(depth.shape[0]):
        np.random.seed()
        rotate_angle = (np.random.random()-0.5)*max_rotate_angle
        if i == 0:
            Rb_random[i][0] = start_angle
        else:
            Rb_random[i][0] = max(min(Rb_random[i-1][0] + rotate_angle, 180), -180)

    return Rb_random


def get_random_RB_curve_2(depth, dep_inf=[-1, -1]):

    if dep_inf[0] < 0:
        # 计算 图像能够旋转的 最大值
        ratate_angle = min(int(depth.shape[0] * 0.075) + 2, 80)
        start_angle = np.random.randint(-(180-ratate_angle-30), 180-ratate_angle-30)
        end_angle = start_angle + np.random.randint(-ratate_angle, ratate_angle)
    else:
        start_angle, end_angle = dep_inf
    # np.random.seed()
    # print('ratate_angle:{}, start_angle:{}, end_angle:{}'.format(ratate_angle, start_angle, end_angle))

    Rb_random = np.zeros_like(depth)

    rotate_angle = (end_angle-start_angle)/depth.shape[0]
    # print('rotate_angle + {}'.format(rotate_angle))
    for i in range(depth.shape[0]):
        if i == 0:
            Rb_random[i][0] = start_angle
        else:
            Rb_random[i][0] = max(min(Rb_random[i-1][0] + rotate_angle + np.random.randint(-60, 60)/depth.shape[0], 180), -180)

    return Rb_random

# 根据RB曲线进行图像旋转
def pic_rotate_by_Rb(pic=np.zeros((10,10)), Rb=np.zeros((10, 1))):
    if pic.shape[0] != Rb.shape[0]:
        print('pic length is not equal to depth length:{}, {}'.format(pic.shape, Rb.shape))
        exit(0)

    pic_new = np.zeros(pic.shape)
    # print(pic_new.shape)
    temp = 360/pic.shape[1]
    for i in range(pic.shape[0]):
        pixel_rotate = int(Rb[i][0] / temp)
        # print(pixel_rotate)
        if pixel_rotate != 0:
            pic_new[i, pixel_rotate:] = pic[i, :-pixel_rotate]
            pic_new[i, :pixel_rotate] = pic[i, -pixel_rotate:]
        else:
            pic_new[i, :] = pic[i, :]

    return pic_new


def get_random_RB_all(depth=np.zeros((5, 1)), RB_index=-1, ratio=np.random.random()):
    rb_random = np.array([])
    # RB_index_new = 0
    if RB_index == -1:
        if ratio < 0.3:
            # print('way 1, R:{}'.format(ratio))
            rb_random = get_random_RB_curve(depth)
            RB_index = 0
        elif ratio < 0.6:
            # print('way 2, R:{}'.format(ratio))
            rb_random = get_random_RB_curve_2(depth)
            RB_index = 1
        else:
            # print('way 3, R:{}'.format(ratio))
            rb_random = np.zeros((depth.shape[0], 1))
            rb_random.fill(np.random.randint(-120, 120))
            RB_index = 2
        return rb_random, RB_index

    else:
        if RB_index == 0:
            if ratio < 0.5:
                rb_random = get_random_RB_curve_2(depth)
                RB_index = 1
            else:
                rb_random = np.zeros((depth.shape[0], 1))
                rb_random.fill(np.random.randint(-120, 120))
                RB_index = 2

        elif RB_index == 1:
            if ratio < 0.5:
                rb_random = get_random_RB_curve(depth)
                RB_index = 0
            else:
                rb_random = np.zeros((depth.shape[0], 1))
                rb_random.fill(np.random.randint(-120, 120))
                RB_index = 2

        elif RB_index == 2:
            if ratio < 0.5:
                rb_random = get_random_RB_curve(depth)
                RB_index = 0
            else:
                rb_random = get_random_RB_curve_2(depth)
                RB_index = 1
        else:
            print('RB_index error:{}'.format(RB_index))
            exit(0)

        return rb_random, RB_index

# 图像 生成随机RB曲线 并旋转
def pic_rotate_random(pic=np.zeros((5, 5)), depth=np.zeros((5, 1)), ratio=np.random.random()):
    # print('depth shape is :{}'.format(depth.shape))
    if ratio < 0.4:
        # print('way 1, R:{}'.format(ratio))
        rb_random = get_random_RB_curve(depth)
    elif ratio < 0.95:
        # print('way 2, R:{}'.format(ratio))
        rb_random = get_random_RB_curve_2(depth)
    else:
        # print('way 3, R:{}'.format(ratio))
        rb_random = np.zeros((pic.shape[0], 1))
        return pic, rb_random
    pic_new = pic_rotate_by_Rb(pic, rb_random)
    return pic_new, rb_random


# 获取图像的 像素分布
def image_hist(img):
    if img.shape.__len__() >= 3:
        print('pic_dim_shift fun dim error:{}'.format(img.shape))
        exit(0)

    x_range = np.zeros((64))
    y_range = np.zeros((64))
    for i in range(x_range.shape[0]):
        x_range[i] = 4*(i + 1)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            index = int(img[i][j])//4
            y_range[index] += 1
    y_range = y_range/(img.shape[0] * img.shape[1])
    # print(x_range, y_range)
    return x_range, y_range


# 图像色域偏移
def pic_dim_shift(pic=np.random.random((5, 5))*255):
    # print(pic.shape, pic.shape.__len__())

    ran_int = np.random.randint(-30, 30)

    if pic.shape.__len__() >= 3:
        print('pic_dim_shift fun dim error:{}'.format(pic.shape))
        exit(0)

    pic = pic + ran_int

    pic = np.clip(pic, 0, 255)
    return pic


# 图像去噪处理，图像增强处理
def pic_denoise(pic=np.zeros((5, 5)), ratio=np.random.random(), k_size=np.random.randint(1, 2)*2+1):
    if ratio < 0.15:
        # 高斯滤波
        pic_new = cv2.GaussianBlur(np.uint8(pic), (k_size, k_size), 0)
    elif ratio < 0.3:
        # 中值滤波
        pic_new = cv2.medianBlur(np.uint8(pic), k_size)
    elif ratio < 0.6:
        # 双边滤波
        pic_new = cv2.bilateralFilter(np.uint8(pic), k_size+2, 75, 75)
    elif ratio < 0.8:
        # 直方图均衡化
        pic_new = cv2.equalizeHist(np.uint8(pic))
    elif ratio < 0.85:
        # 随机偏移图像增强
        pic_new = pic_enhence_random(pic, windows_shape=k_size)
    else:
        pic_new = pic
    return pic_new


# 图像添加噪声
def pic_add_noise(pic=np.zeros((5, 5)), ratio=np.random.random()):
    # print(ratio)
    pic_new = np.zeros((10, 10))
    if ratio < 0.2:
        # 添加高斯噪声
        # 产生高斯随机数
        # print('添加高斯噪声')
        noise = np.random.normal(0, 50, size=pic.size).reshape(pic.shape[0], pic.shape[1])
        # 加上噪声
        pic_new = pic + noise
        pic_new = np.clip(pic_new, 0, 255)
    elif ratio < 0.4:
        # 添加泊松噪声
        # 产生泊松噪声
        # print('添加泊松噪声')
        noise = np.random.poisson(lam=10, size=pic.shape).astype('uint8')
        # 加上噪声
        pic_new = pic + noise
        pic_new = np.clip(pic_new, 0, 255)
    elif ratio < 0.6:
        # print('添加椒盐噪声')
        # 添加椒盐噪声
        # 转化成向量
        x = pic.reshape(1, -1)
        # 设置信噪比
        SNR = 0.90
        # 得到要加噪的像素数目
        noise_num = x.size * (1 - SNR)
        # 得到需要加噪的像素值的位置
        list = random.sample(range(0, x.size), int(noise_num))

        for i in list:
            if random.random() >= 0.5:
                x[0][i] = 0
            else:
                x[0][i] = 255
        pic_new = x.reshape(pic.shape)
    else:
        pic_new = pic

    return pic_new

def motion_blur(image, degree=8, angle=20):

    image = np.array(image)
    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred

# 图像添加噪声
def pic_add_blur(pic=np.zeros((5, 5)), ratio=np.random.random()):
    # print(ratio)
    pic_new = np.zeros((10, 10))
    if ratio < 0.2:
        # 添加高斯模糊
        # print('添加高斯模糊')
        pic_new = cv2.GaussianBlur(pic, (9, 9), 0)  # 高斯模糊
    elif ratio < 0.4:
        pic_new = motion_blur(pic)  # 运动模糊
    elif ratio < 0.6:
        # 此为均值模糊
        # （30,1）为一维卷积核，指在x，y方向偏移多少位
        pic_new = cv2.blur(pic, (8, 1))
    elif ratio < 0.8:
        # 此为中值模糊，常用于去除椒盐噪声
        pic_new = cv2.medianBlur(pic, 5)
    else:
        # pic_new = cv2.bilateralFilter(pic, 0, 25, 25)
        pic_new = pic

    return pic_new



# 图像镜像
def pic_mirror(pic=np.zeros((5, 5)), ratio=np.random.random()):
    if len(pic.shape)==2:
        pic_new = np.ones_like(pic)

        if ratio < 0.25:
            # print('上下镜像')
            for i in range(pic.shape[0]):
                pic_new[i] = pic[-i]
        elif ratio < 0.5:
            # print('左右镜像')
            for j in range(pic.shape[1]):
                pic_new[:, j] = pic[:, -j]
        elif ratio < 0.75:
            # print('上下左右镜像')
            for i in range(pic.shape[0]):
                for j in range(pic.shape[1]):
                    pic_new[i, j] = pic[-i, -j]
        else:
            # print('原图像')
            pic_new = pic
        return pic_new
    elif len(pic.shape)==3:
        pic_new = np.zeros_like(pic)

        if ratio < 0.1:
            # pic_new = pic
            # # print('上下镜像')
            for i in range(pic.shape[0]):
                for j in range(pic.shape[1]):
                    pic_new[i, j, :] = pic[i, -j, :]
        elif ratio < 0.4:
            # print('左右镜像')
            for i in range(pic.shape[0]):
                for k in range(pic.shape[2]):
                    pic_new[i, :, k] = pic[i, :, -k]
        elif ratio < 0.75:
            pic_new = pic
            # # print('上下左右镜像')
            for i in range(pic.shape[0]):
                for j in range(pic.shape[1]):
                    for k in range(pic.shape[2]):
                        pic_new[i, j, k] = pic[i, -j, -k]
        else:
            # print('原图像')
            pic_new = pic
        return pic_new
    else:
        print('error tailor shape:{}'.format(pic.shape))
        exit(0)


# 图像随机裁剪
def pic_tailor(pic=np.zeros((5, 5)), pic_shape_ex=[224, 224]):
    if len(pic.shape) <= 2:
        if (pic.shape[0] < pic_shape_ex[0]) | (pic.shape[1] < pic_shape_ex[1]):
            print('shape error ,pic shape is smaller than target shape:{},{}'.format(pic.shape, pic_shape_ex))
            exit(0)

        x_windows_pixel = np.random.randint(pic_shape_ex[0]+5, pic.shape[0]-5)
        y_windows_pixel = np.random.randint(pic_shape_ex[1]+5, pic.shape[1]-5)

        x_index_start = np.random.randint(0, pic.shape[0]-x_windows_pixel-1)
        y_index_start = np.random.randint(0, pic.shape[1]-y_windows_pixel-1)

        pic_new = pic[x_index_start:x_index_start+x_windows_pixel, y_index_start:y_index_start+y_windows_pixel]
        return pic_new
    else:
        # print(pic.shape)
        pix_drop_x = int(pic.shape[-2]/10)
        # print(pix_drop_x)
        x_windows_pixel = np.random.randint(pic.shape[-2]-pix_drop_x, pic.shape[-2]-5)
        y_windows_pixel = np.random.randint(pic_shape_ex[1]+5, pic.shape[-1]-5)

        x_index_start = np.random.randint(0, pic.shape[-2] - x_windows_pixel - 1)
        y_index_start = np.random.randint(0, pic.shape[-1] - y_windows_pixel - 1)

        pic_new = pic[:, x_index_start:x_index_start + x_windows_pixel, y_index_start:y_index_start + y_windows_pixel]
        return pic_new


def get_pic_random(data_img_o, data_depth, RB_index=-1, pic_shape=(224, 224)):
    data_img = copy.deepcopy(data_img_o)

    if len(data_img.shape) == 2:
        # data_img, data_depth = get_test_ele_data()
        # data_img = data_img[1200:1600, :]
        # data_depth = data_depth[1200:1600, :]
        # data_depth = np.zeros((data_img.shape[0], 1))

        # 图片绕井壁 按RB曲线 旋转
        pic_new, rb_random = pic_rotate_random(data_img, depth=data_depth)
        # print('111{}'.format(pic_new.shape))
        # plt.figure(figsize=(10, 7))
        # plt.plot(data_depth.ravel(), rb_random.ravel())
        # plt.show()

        # 图片像素偏移
        pic_new = pic_dim_shift(pic_new)
        # print('222{}'.format(pic_new.shape))

        # 图像降噪 以及 图像去噪
        ratio = np.random.random()
        if ratio < 0.7:
            pic_new = pic_denoise(pic_new)
        elif ratio < 0.8:
            pic_new = pic_add_noise(pic_new)
        # print('333{}'.format(pic_new.shape))

        # 图像镜像
        pic_new = pic_mirror(pic_new)
        # print('444{}'.format(pic_new.shape))
        pic_shape = (224, 224)
        # 图像裁剪
        pic_new = np.clip(pic_tailor(pic_new, pic_shape), 0, 255)
        # print('555{}'.format(pic_new.shape))

        # 图像缩放以及 归一化
        pic_new = cv2.resize(pic_new, pic_shape)/256
        # pic_new = pic_scale_normal(pic_new, pic_shape)/256
        return pic_new
    elif len(data_img.shape) == 3:
        pic_new = np.zeros_like(data_img)

        # 获取随机的RB曲线 并对成像数据进行 绕井壁旋转
        np.random.seed()
        rb_random, RB_index = get_random_RB_all(depth=data_depth, RB_index=RB_index)
        # print(rb_random.shape)
        for i in range(data_img.shape[0]):
            pic_new[i, :, :] = pic_rotate_by_Rb(pic=data_img[i, :, :], Rb=rb_random)
        # show_Pic([pic_new[0], pic_new[1]], pic_order='12')

        np.random.seed()
        # 图像降噪 以及 图像去噪
        # 只对 第一维 数据进行 图像的 降噪、模糊等 处理
        ratio = np.random.random()
        if ratio < 0.6:
            pic_new[0, :, :] = pic_denoise(pic_new[0, :, :])
        elif ratio < 0.7:
            pic_new[0, :, :] = pic_add_noise(pic_new[0, :, :])
        elif ratio < 0.8:
            pic_new[0, :, :] = pic_add_blur(pic_new[0, :, :])
        # print('333{}'.format(pic_new.shape))

        # np.random.seed()
        # # 图像镜像
        # pic_new = pic_mirror(pic_new)
        # print('444{}'.format(pic_new.shape))

        # # 图像裁剪
        pic_new = pic_tailor(pic_new, pic_shape)

        # 图像缩放以及 归一化
        answer = []
        for i in range(pic_new.shape[0]):
            answer.append(cv2.resize(pic_new[i, :, :], pic_shape))
        # pic_new = pic_scale_normal(pic_new, pic_shape)/256
        return np.array(answer)/256, RB_index
        # rb_random = get_random_RB_all(depth=data_depth)
        #
        # for i in range(data_img.shape[0]):
        #     data_img[i, :, :] = pic_rotate_by_Rb(pic=data_img[i, :, :], Rb=rb_random)
        #
        # return data_img
    else:
        print('wrong img shape:{}'.format(data_img.shape))
        exit(0)


def get_pic_random_VIT_teacher(data_img_o, data_depth, RB_index=-1):
    data_img = copy.deepcopy(data_img_o)

    if len(data_img.shape) == 2:

        # 图片绕井壁 按RB曲线 旋转
        pic_new, rb_random = pic_rotate_random(data_img, depth=data_depth)

        # 图片像素偏移
        pic_new = pic_dim_shift(pic_new)

        # 图像降噪 以及 图像去噪
        ratio = np.random.random()
        if ratio < 0.7:
            pic_new = pic_denoise(pic_new)
        elif ratio < 0.8:
            pic_new = pic_add_noise(pic_new)

        # 图像镜像
        pic_new = pic_mirror(pic_new)
        pic_shape = (196, 196)
        # 图像裁剪
        pic_new = np.clip(pic_tailor(pic_new, pic_shape), 0, 255)

        # 图像缩放以及 归一化
        pic_new = cv2.resize(pic_new, pic_shape)
        return pic_new
    elif len(data_img.shape) == 3:
        pic_new = np.zeros_like(data_img)

        np.random.seed()
        rb_random, RB_index = get_random_RB_all(depth=data_depth, RB_index=RB_index)

        # from matplotlib import pyplot as plt
        # plt.plot(data_depth, rb_random)
        # plt.show()

        for i in range(data_img.shape[0]):
            pic_new[i, :, :] = pic_rotate_by_Rb(pic=data_img[i, :, :], Rb=rb_random)

        np.random.seed()
        # 图像降噪 以及 图像去噪
        ratio = np.random.random()
        if ratio < 0.6:
            pic_new[0, :, :] = pic_denoise(pic_new[0, :, :])
        elif ratio < 0.7:
            pic_new[0, :, :] = pic_add_noise(pic_new[0, :, :])
        elif ratio < 0.8:
            pic_new[0, :, :] = pic_add_blur(pic_new[0, :, :])
        # print('333{}'.format(pic_new.shape))

        np.random.seed()
        # 图像镜像
        pic_new = pic_mirror(pic_new)
        # print('444{}'.format(pic_new.shape))

        # pic_shape = (224, 224)

        # # # # 图像裁剪
        # # pic_new = pic_tailor(pic_new, pic_shape)
        #
        # # 图像缩放以及 归一化
        # answer = []
        # for i in range(pic_new.shape[0]):
        #     answer.append(cv2.resize(pic_new[i, :, :], pic_shape))
        #     # answer.append(pic_new[i, :, :])
        # # pic_new = pic_scale_normal(pic_new, pic_shape)/256
        return pic_new.astype(np.float32), RB_index

    else:
        print('wrong img shape:{}'.format(data_img.shape))
        exit(0)


def get_pic_random_VIT_student(data_img_o, data_depth, RB_index=-1):
    data_img = copy.deepcopy(data_img_o)

    if len(data_img.shape) == 2:

        # 图片绕井壁 按RB曲线 旋转
        pic_new, rb_random = pic_rotate_random(data_img, depth=data_depth)

        # 图片像素偏移
        pic_new = pic_dim_shift(pic_new)

        # 图像降噪 以及 图像去噪
        ratio = np.random.random()
        if ratio < 0.7:
            pic_new = pic_denoise(pic_new)
        elif ratio < 0.8:
            pic_new = pic_add_noise(pic_new)

        # 图像镜像
        pic_new = pic_mirror(pic_new)
        pic_shape = (196, 196)
        # 图像裁剪
        pic_new = np.clip(pic_tailor(pic_new, pic_shape), 0, 255)

        # 图像缩放以及 归一化
        pic_new = cv2.resize(pic_new, pic_shape)
        return pic_new
    elif len(data_img.shape) == 3:
        pic_new = np.zeros_like(data_img)

        np.random.seed()
        rb_random, RB_index = get_random_RB_all(depth=data_depth, RB_index=RB_index)

        for i in range(data_img.shape[0]):
            pic_new[i, :, :] = pic_rotate_by_Rb(pic=data_img[i, :, :], Rb=rb_random)

        np.random.seed()
        # 图像降噪 以及 图像去噪
        ratio = np.random.random()
        if ratio < 0.3:
            pic_new[0, :, :] = pic_denoise(pic_new[0, :, :])
        elif ratio < 0.35:
            pic_new[0, :, :] = pic_add_noise(pic_new[0, :, :])
        elif ratio < 0.4:
            pic_new[0, :, :] = pic_add_blur(pic_new[0, :, :])
        # print('333{}'.format(pic_new.shape))

        # np.random.seed()
        # # 图像镜像
        # pic_new = pic_mirror(pic_new)
        # # print('444{}'.format(pic_new.shape))

        pic_shape = (96, 96)

        # # 图像裁剪
        pic_new = pic_tailor(pic_new, pic_shape)

        # 图像缩放以及 归一化
        answer = []
        for i in range(pic_new.shape[0]):
            answer.append(cv2.resize(pic_new[i, :, :], pic_shape))
            # answer.append(pic_new[i, :, :])
        # pic_new = pic_scale_normal(pic_new, pic_shape)/256
        return np.array(answer).astype(np.float32), RB_index

    else:
        print('wrong img shape:{}'.format(data_img.shape))
        exit(0)

# pic_new = pic_denoise(data_img)

# Rb_random = get_random_RB_curve_2(depth=data_depth)
# # plt.figure(figsize=(10, 7))
# # plt.plot(data_depth.ravel(), Rb_random.ravel())
# # plt.ylim([-180, 180])
# # plt.show()
# pic_new = pic_rotate_by_Rb(data_img, Rb_random)
# pic_new = pic_dim_shift(pic_new)
# pic_new = pic_denoise(pic_new)


# pic_new = pic_mirror(data_img)

# data_img_dyna, data_img_stat, data_depth = get_test_ele_data()
# pic_dyna = data_img_dyna.reshape((1, data_img_dyna.shape[0], data_img_dyna.shape[1]))
# pic_stat = data_img_stat.reshape((1, data_img_stat.shape[0], data_img_stat.shape[1]))
# # print(pic_dyna.shape)
# pic_all = np.append(pic_dyna, pic_stat, axis=0)
# pic_all = np.array(pic_all)
#
# p_n_1, RB_index = get_pic_random_vit(pic_all, data_depth)
# p_n_2, RB_index = get_pic_random_vit(pic_all, data_depth, RB_index)
#
# show_Pic([pic_all[0]*256, pic_all[1]*256, p_n_1[0]*256, p_n_2[0]*256], '22')

# data_img = np.array(pic_all)
# pic_show = [data_img_dyna]
# for i in range(5):
#     ratio = 0.1 + 0.2*i
#     pic_t = pic_add_blur(data_img_dyna, ratio=ratio)
#     print(pic_t.shape)
#     pic_show.append(pic_t)
#
# show_Pic(pic_show, '23')
# print(len(pic_show))





# data_img = data_img[1200:1600, :]
# data_depth = data_depth[1200:1600, :]
# show_Pic([data_img, get_pic_random(), get_pic_random()], pic_str=['img_org', 'img_random1', 'img_random2'], pic_order='13')
# RB_info = np.hstack((data_depth, Rb_random+180))
# np.savetxt('pic_rotate_by_rb_test/Rb_info_{}.txt'.format('test_rb'), RB_info, fmt='%.4f', delimiter='\t', comments='',
#            header='WELLNAME={}\nSTDEP\t= {}\nENDEP\t= {}\nLEV\t= {:.4f}\nUNIT\t= meter\nCURNAMES= {}\n\n#DEPTH\t{}'.format(
#                'test_1', data_depth[0, 0], data_depth[-1, 0], data_depth[1, 0] - data_depth[0, 0], 'Rb_random', 'Rb_random'))
# np.savetxt('pic_rotate_by_rb_test/pic_org_info_{}.txt'.format('test_rb'), np.hstack((data_depth, data_img)), fmt='%.4f', delimiter='\t', comments='',
#            header='WELLNAME={}\nSTDEP\t= {}\nENDEP\t= {}\nLEV\t= {:.4f}\nUNIT\t= meter\nCURNAMES= {}\n\n#DEPTH\t{}'.format(
#                'test_1', data_depth[0, 0], data_depth[-1, 0], data_depth[1, 0] - data_depth[0, 0], 'img_org', 'img_org'))
# np.savetxt('pic_rotate_by_rb_test/pic_rotate_info_{}.txt'.format('test_rb'), np.hstack((data_depth, pic_new)), fmt='%.4f', delimiter='\t', comments='',
#            header='WELLNAME={}\nSTDEP\t= {}\nENDEP\t= {}\nLEV\t= {:.4f}\nUNIT\t= meter\nCURNAMES= {}\n\n#DEPTH\t{}'.format(
#                'test_1', data_depth[0, 0], data_depth[-1, 0], data_depth[1, 0] - data_depth[0, 0], 'img_rotate', 'img_rotate'))


# data_img_shifted = pic_dim_shift(data_img)
# show_Pic([data_img, data_img_shifted], pic_order='12', pic_str=['原始成像', '色域偏移成像'], save_pic=False, path_save='')
# x_range, y_range = image_hist(data_img)
# x_range_shifted, y_range_shifted = image_hist(data_img_shifted)
# print(x_range, y_range)
# # 全部的线段风格
# styles = ['c:s', 'y:8', 'r:^', 'r:v', 'g:D', 'm:X', 'b:p', ':>'] # 其他可用风格 ':<',':H','k:o','k:*','k:*','k:*'
# # # 获取全部的图例
# # columns = [i[:-2] for i in data.columns]
# # n,m = data.shape
# plt.figure(figsize=(10, 7))
#
# # 设置字体
# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams.update({'font.size': 22})
# plt.rc('legend', fontsize=15)
#
# plt.plot(x_range, y_range, styles[2], markersize=4, label='img_org')
#
# # 设置图片的x,y轴的限制，和对应的标签
# plt.xlim([0, 256])
# plt.ylim([0, 0.04])
# plt.xlabel("pixel value distribution")
# plt.ylabel("percentage")
#
# # 设置图片的方格线和图例
# plt.grid()
# plt.legend(loc='lower right', framealpha=0.7)
# plt.tight_layout()
# # plt.show()
#
# # 如果想保存图片，请把plt.show注释掉，然后把下面这行代码打开注释
# plt.savefig("img.png", dpi=800)
# plt.plot(x_range_shifted, y_range_shifted, styles[0], markersize=4, label='img_shifted')
#
# # 设置图片的x,y轴的限制，和对应的标签
# plt.xlim([0, 256])
# plt.ylim([0, 0.04])
# plt.xlabel("pixel value distribution")
# plt.ylabel("percentage")
#
# # 设置图片的方格线和图例
# plt.grid()
# plt.legend(loc='lower right', framealpha=0.7)
# plt.tight_layout()
# # plt.show()
#
# # 如果想保存图片，请把plt.show注释掉，然后把下面这行代码打开注释
# plt.savefig("img2.png", dpi=800)epth[1, 0] - data_depth[0, 0], 'img_org', 'img_org'))
# np.savetxt('pic_rotate_by_rb_test/pic_rotate_info_{}.txt'.format('test_rb'), np.hstack((data_depth, pic_new)), fmt='%.4f', delimiter='\t', comments='',
#            header='WELLNAME={}\nSTDEP\t= {}\nENDEP\t= {}\nLEV\t= {:.4f}\nUNIT\t= meter\nCURNAMES= {}\n\n#DEPTH\t{}'.format(
#                'test_1', data_depth[0, 0], data_depth[-1, 0], data_depth[1, 0] - data_depth[0, 0], 'img_rotate', 'img_rotate'))


# data_img_shifted = pic_dim_shift(data_img)
# show_Pic([data_img, data_img_shifted], pic_order='12', pic_str=['原始成像', '色域偏移成像'], save_pic=False, path_save='')
# x_range, y_range = image_hist(data_img)
# x_range_shifted, y_range_shifted = image_hist(data_img_shifted)
# print(x_range, y_range)
# # 全部的线段风格
# styles = ['c:s', 'y:8', 'r:^', 'r:v', 'g:D', 'm:X', 'b:p', ':>'] # 其他可用风格 ':<',':H','k:o','k:*','k:*','k:*'
# # # 获取全部的图例
# # columns = [i[:-2] for i in data.columns]
# # n,m = data.shape
# plt.figure(figsize=(10, 7))
#
# # 设置字体
# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams.update({'font.size': 22})
# plt.rc('legend', fontsize=15)
#
# plt.plot(x_range, y_range, styles[2], markersize=4, label='img_org')
#
# # 设置图片的x,y轴的限制，和对应的标签
# plt.xlim([0, 256])
# plt.ylim([0, 0.04])
# plt.xlabel("pixel value distribution")
# plt.ylabel("percentage")
#
# # 设置图片的方格线和图例
# plt.grid()
# plt.legend(loc='lower right', framealpha=0.7)
# plt.tight_layout()
# # plt.show()
#
# # 如果想保存图片，请把plt.show注释掉，然后把下面这行代码打开注释
# plt.savefig("img.png", dpi=800)
# plt.plot(x_range_shifted, y_range_shifted, styles[0], markersize=4, label='img_shifted')
#
# # 设置图片的x,y轴的限制，和对应的标签
# plt.xlim([0, 256])
# plt.ylim([0, 0.04])
# plt.xlabel("pixel value distribution")
# plt.ylabel("percentage")
#
# # 设置图片的方格线和图例
# plt.grid()
# plt.legend(loc='lower right', framealpha=0.7)
# plt.tight_layout()
# # plt.show()
#
# # 如果想保存图片，请把plt.show注释掉，然后把下面这行代码打开注释
# plt.savefig("img2.png", dpi=800)