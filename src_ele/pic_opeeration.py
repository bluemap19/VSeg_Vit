import copy
import math
import random
import numpy as np
import cv2
import os
from src_ele.file_operation import get_test_ele_data



# 图片的一些操作
# 1.show_Pic(pic_list, pic_order='12', pic_str=[], save_pic=False, path_save='')
# 展示图片，无返回
# 2.WindowsDataZoomer(SinglePicWindows, ExtremeRatio=0.02)
# 数据缩放，把电阻的数据域映射到图像的数据域，返回原图片数组大的图片数组[m,n] int
# 3.GetPicContours(PicContours, threshold = 4000)
# 对图片进行分割，threshold代表了目标区域需要保留的最小面积大小
# 返回的 contours_Conform, contours_Drop, contours_All 代表了目标轮廓信息list，被丢掉的轮廓信息list，总的轮廓信息list
# 轮廓信息包括，轮廓面积数值，轮廓描述（即是轮廓的存放），轮廓的质心[x, y]
# 4.GetBinaryPic(ProcessingPic)
# 有点问题，别用这个函数
# 5.pic_enhence(input, windows_shape = 7, ratio_top = 0.33, ratio_migration = 5/6)
# 图片的增强函数，用的是局部梯度偏移
# 6.pic_enhence_random(input, windows_shape=3, ratio_top=0.2, ratio_migration=0.6, random_times=3)
# 图片的增强函数，用的是随机局部梯度偏移
# 7.pic_scale(input, windows_shape=3, center_ratio=0.5, x_size=100.0, y_size=100.0, ratio_top=0.1)
# 图片缩放函数，用的是局部梯度增强缩放
# 8.test_pic_enhance_effect()
# 测试图片的增强效果
# 9.adjust_gamma(image, gamma=1.0)
# 图片的 伽马增强
# 10.save_img_data(dep, data, path='')
# 保存图片
# 11.pic_smooth_effect_compare()
# 图片的增强效果对比函数，对比上面的随机局部梯度偏移、伽马增强、直方图均衡增强效果

def get_pic_distribute(pic=np.random.randint(1,256,(2,2)), dist_length=9, min_V=0, max_V=256):
    pic_mean = np.mean(pic)
    pic_s2 = np.var(pic)

    if len(pic.shape)==2:
        step = (max_V-min_V)/dist_length
        pic_dist = np.zeros(dist_length)
        for i in range(pic.shape[0]):
            for j in range(pic.shape[1]):
                index_t = math.floor((pic[i][j]-min_V)/step)
                pic_dist[index_t] += 1

        pic_dist = pic_dist/pic.size
        # return pic_dist, np.array([pic_mean, pic_s2])
        return pic_dist
    else:
        print('wrong pic shape:{}'.format(pic.shape))
        exit(0)

    # print(pic_dist, pic)

# get_pic_distribute()


def show_Pic(pic_list, pic_order='12', pic_str=[], save_pic=False, path_save=''):
    from matplotlib import pyplot as plt
    if len(pic_order) != 2:
        print('pic order error:{}'.format(pic_order))
    # 计算图像总数，并判断是否与输入数据相匹配
    num = int(pic_order[0]) * int(pic_order[1])
    if num != len(pic_list):
        print('pic order num is not equal to pic_list num:{},{}'.format(len(pic_list), pic_order))


    while( len(pic_str) < len(pic_list)):
        pic_str.append('pic_str'+str(len(pic_list)-len(pic_str)))


    for i in range(len(pic_list)):
        for j in range(pic_list[i].shape[0]):
            for k in range(pic_list[i].shape[1]):
                if pic_list[i][j][k] < 0:
                    pic_list[i][j][k] = 0
                if pic_list[i][j][k] > 256:
                    pic_list[i][j][k] = 255

    plt.close('all')
    fig = plt.figure(figsize=(16, 9))
    for i in range(len(pic_list)):
        pic_temp = pic_list[i]
        # print(pic_temp.shape)
        if len(pic_list[i].shape) == 3:
            if pic_list[i].shape[0] == 3:
                pic_temp = pic_temp.transpose(1, 2, 0)
                # print(pic_temp.shape)

        order_str = int(pic_order+str(i+1))
        a = int(pic_order[0])
        b = int(pic_order[1])
        c = i + 1
        # print(order_str)
        # 当a,b,c大于等于10时 .add_subplot(a, b, c)
        ax = fig.add_subplot(a, b, c)
        # ax = fig.add_subplot(order_str)
        ax.set_title(pic_str[i])
        plt.axis('off')
        if pic_temp.shape[-1] == 3:
            ax.imshow(pic_temp.astype(np.int))
        else:
            ax.imshow(pic_temp.astype(np.int), cmap='hot')
        # ax.imshow(pic_list[i], cmap='afmhot')
        # ax.imshow(pic_list[i], cmap='gist_heat')
    plt.show()

    if save_pic:
        if path_save == '':
            plt.savefig('temp.png')
        else:
            plt.savefig(path_save)
        plt.close()

def WindowsDataZoomer_PicList(pic_list, ExtremeRatio=0.02, USE_EXTRE=False, Max_V=-1, Min_V=-1):
    pic_list_numpy = np.array(pic_list)
    ExtremePointNum = int(pic_list_numpy.size * ExtremeRatio)
    bigTop = np.max(pic_list_numpy)
    smallTop = np.min(pic_list_numpy)
    pic_list_N = copy.deepcopy(pic_list)

    if USE_EXTRE:
        bigTop = np.mean(np.sort(pic_list_numpy.reshape(1, -1)[0])[-ExtremePointNum:])
        smallTop = np.mean(np.sort(pic_list_numpy.reshape(1, -1)[0])[:ExtremePointNum])

    if Max_V > 0:
        bigTop = Max_V
        smallTop = Min_V

    if bigTop - smallTop < 0.001:
        print("Error........bigTop == smallTop")
        exit(0)
    Step = 256 / (bigTop - smallTop)

    for n in range(len(pic_list)):
        for j in range(pic_list[n].shape[0]):
            for k in range(pic_list[n].shape[1]):
                pic_list_N[n][j][k] = (pic_list[n][j][k] - smallTop) * Step
                if pic_list_N[n][j][k] < 0:
                    pic_list_N[n][j][k] = 0
                elif pic_list_N[n][j][k] > 255:
                    pic_list_N[n][j][k] = 255

    return pic_list_N, Step, smallTop


# 数据缩放，把电阻的数据域映射到图像的数据域
def WindowsDataZoomer(SinglePicWindows, ExtremeRatio=0.02):
    """
    数据缩放，把电阻的数据域映射到图像的数据域
    通过计算5%的极大值、极小值来完成，会修改原本的数组，数组依旧是小数
    修改原数据
    :param SinglePicWindows:2d np.array
    :return:no change original data
    """
    # print('Windows Data Zoomer......')
    # tem = np.argsort(SinglePicWindows.reshape(1, -1)[0], axis=-1, kind='quicksort', order=None)
    # print(np.sort(SinglePicWindows.reshape(1, -1)[0]))
    ExtremePointNum = int(SinglePicWindows.size*ExtremeRatio)
    # print('缩放的最大最小值的窗口大小：%s'%ExtremePointNum)
    # bigTop = np.mean(np.sort(SinglePicWindows.reshape(1, -1)[0])[-ExtremePointNum:])
    bigTop = np.max(SinglePicWindows)
    # print('大的一段%.5f'%bigTop)
    # smallTop = np.mean(np.sort(SinglePicWindows.reshape(1, -1)[0])[:ExtremePointNum])
    smallTop = np.min(SinglePicWindows)
    # print('小的一段%.5f'%smallTop)
    if bigTop - smallTop < 0.000001:
        print("Error........bigTop == smallTop")
        exit(0)
    Step = 256 / (bigTop - smallTop)
    # print('缩放的倍数：%.5f'%Step)
    # print(SinglePicWindows[:5, :5])
    # print('缩放前子图平均数：%.5f'%(np.mean(SinglePicWindows)))
    SinglePicWindows_new = np.copy(SinglePicWindows)
    for j in range(SinglePicWindows.shape[0]):
        for k in range(SinglePicWindows.shape[1]):
            SinglePicWindows_new[j][k] = (SinglePicWindows[j][k] - smallTop) * Step
            if SinglePicWindows_new[j][k] < 0:
                SinglePicWindows_new[j][k] = 0
            elif SinglePicWindows_new[j][k] > 255:
                SinglePicWindows_new[j][k] = 255
    # print(SinglePicWindows[:5, :5])
    # print('缩放后子图平均数：%.5f'%(np.mean(SinglePicWindows)))
    # SinglePicWindows = np.array(SinglePicWindows, dtype=np.int)
    # SinglePicWindows = np.array(SinglePicWindows, dtype=np.float)
    return SinglePicWindows_new, Step, smallTop


def GetPicContours(PicContours, threshold = 4000):
    # findContours函数第二个参数表示轮廓的检索模式
    # cv2.RETR_EXTERNAL 表示只检测外轮廓
    # cv2.RETR_LIST     检测的轮廓不建立等级关系
    # cv2.RETR_CCOMP    建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
    # cv2.RETR_TREE     建立一个等级树结构的轮廓。
    # 第三个参数method为轮廓的近似办法
    # cv2.CHAIN_APPROX_NONE     存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1 - x2），abs（y2 - y1）） == 1
    # cv2.CHAIN_APPROX_SIMPLE   压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
    # cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh - Chinl chain近似算法
    contours, hierarchy = cv2.findContours(PicContours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE )

    contours_Conform = [[], [], []]       # 存储符合要求的轮廓 顺序为 # 面积，轮廓，质心
    contours_Drop = [[], [], []]          # 存储不符合要求的轮廓
    contours_All = [[], [], []]           # 存储所有轮廓
    for i in range(len(contours)):
        # contour_S 为轮廓面积
        contour_S = cv2.contourArea(contours[i])
        M = cv2.moments(contours[i])
        # mc为质心
        mc = [int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])]

        if contour_S > threshold:         # 筛选出面积大于4000的轮廓
            # print('第%d个轮廓面积：' % i + str(temp))
            contours_Conform[0].append(contour_S)
            contours_Conform[1].append(contours[i])
            contours_Conform[2].append(mc)
        else:                           # 剩下的为不合格的轮廓
            # print('第%d个轮廓面积：'%i + str(temp))
            contours_Drop[0].append(contour_S)
            contours_Drop[1].append(contours[i])
            contours_Drop[2].append(mc)

        # 记录全部的轮廓信息
        contours_All[0].append(contour_S)
        contours_All[1].append(contours[i])
        contours_All[2].append(mc)
    return contours_Conform, contours_Drop, contours_All


def GetBinaryPic(ProcessingPic):
    Blur_Average = cv2.blur(ProcessingPic, (7, 5))
    Blur_Gauss = cv2.GaussianBlur(ProcessingPic, (7, 5), 0)
    Blur_Median = cv2.medianBlur(ProcessingPic, 5)

    ProcessingPic = Blur_Gauss
    firstLevel = 40
    ret, img_binary_Level1 = cv2.threshold(ProcessingPic, firstLevel, 255, cv2.THRESH_BINARY)
    ret, img_binary_Level2 = cv2.threshold(ProcessingPic, firstLevel + 10, 255, cv2.THRESH_BINARY)
    ret, img_binary_Level3 = cv2.threshold(ProcessingPic, firstLevel + 20, 255, cv2.THRESH_BINARY)
    ret, img_binary_Level4 = cv2.threshold(ProcessingPic, firstLevel + 30, 255, cv2.THRESH_BINARY)
    ret, img_binary_Level5 = cv2.threshold(ProcessingPic, firstLevel + 40, 255, cv2.THRESH_BINARY)
    ret, img_binary_Level6 = cv2.threshold(ProcessingPic, firstLevel + 50, 255, cv2.THRESH_BINARY)

    ProcessingPic = img_binary_Level3
    Kernel_Rect = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))  # 生成形状为矩形5x5的卷积核
    Kernel_Ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 3))  # 椭圆形9x9
    kernel = np.ones((5, 5), np.uint8)
    targetKernel = Kernel_Ellipse
    Pic_erosion = cv2.erode(ProcessingPic, targetKernel, iterations=1)
    Pic_dilation = cv2.dilate(ProcessingPic, targetKernel, iterations=1)
    Pic_opening = cv2.morphologyEx(ProcessingPic, cv2.MORPH_OPEN, targetKernel)
    Pic_closing = cv2.morphologyEx(ProcessingPic, cv2.MORPH_CLOSE, targetKernel)
    Pic_opening_closing = cv2.morphologyEx(Pic_opening, cv2.MORPH_CLOSE, targetKernel)
    Pic_closing_opening = cv2.morphologyEx(Pic_closing, cv2.MORPH_CLOSE, targetKernel)
    ProcessingPic = Pic_opening_closing
    t, Pic_To_Count_Contours = cv2.threshold(ProcessingPic, 0, 255, cv2.THRESH_BINARY_INV)  # 通过阀值将其反色为白图 二值化图像反转

    return Pic_To_Count_Contours


def process_pix(index_x, index_y, input, windows_shape, max_pixel, ratio_top, ratio_migration):
    # 寻找窗口的index
    start_index_x = max(index_x-windows_shape//2, 0)
    end_index_x = min(index_x+windows_shape//2 + 1, input.shape[0])
    start_index_y = max(index_y-windows_shape//2, 0)
    end_index_y = min(index_y+windows_shape//2 + 1, input.shape[1])

    # 根据窗口index 获得窗口的 数据
    data_windows = copy.deepcopy(input[start_index_x:end_index_x, start_index_y:end_index_y]).ravel()

    value = input[index_x][index_y]

    # 根据窗口周边数据情况，计算像素移动方向， 正的为 增大，负的为 减小
    direction = -1
    if (np.sum(data_windows)-value) > (max_pixel/2) * (windows_shape*windows_shape-1):
        direction = 1
    # direction = ((np.sum(data_windows)-value)//(windows_shape*windows_shape-1))-(max_pixel//2)

    # ordered_list = sorted(data_windows)
    # small_top = np.mean(ordered_list[:int(len(ordered_list)*ratio_top)])
    # big_top = np.mean(ordered_list[-int(len(ordered_list)*ratio_top):])
    # print(small_top, big_top)
    small_top = np.min(data_windows)
    big_top = np.max(data_windows)

    if direction < 0:
        return (value - (value - small_top)*ratio_migration)
    else:
        return (value + (big_top - value)*ratio_migration)



def pic_enhence(input, windows_shape = 7, ratio_top = 0.33, ratio_migration = 5/6):
    max_pixel = np.max(input)
    data_new = copy.deepcopy(input)
    if (windows_shape%2) != 1:
        print('windows shape error...........')
        exit()

    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            data_new[i][j] = process_pix(i, j, input, windows_shape, max_pixel, ratio_top, ratio_migration)

    return data_new

def shuffle(lis):
    for i in range(len(lis) - 1, 0, -1):
        p = random.randrange(0, i + 1)
        lis[i], lis[p] = lis[p], lis[i]
    return lis




def pic_enhence_random(input, windows_shape=3, ratio_top=0.2, ratio_migration=0.6, random_times=3):
    if ((windows_shape % 2) != 1) | (windows_shape < 0):
        print('windows shape error...........')
        exit()
    if len(input.shape) >= 3:
        print('转换成灰度图再运行')
        exit()

    max_pixel = np.max(input)
    data_new = copy.deepcopy(input)
    all_times = input.shape[0] * input.shape[1]

    a = list(range(all_times))
    r = shuffle(a)
    # print(r)

    # for i in range(all_times):
    #     x = random.randint(0, input.shape[0]-1)
    #     y = random.randint(0, input.shape[1]-1)
    #
    #     data_new[x][y] = process_pix(x, y, input, windows_shape, max_pixel, ratio_top, ratio_migration)

    for j in range(random_times):
        for i in r:
            # print(i)
            x = i // input.shape[1]
            y = i % input.shape[1]
            # print(i, x, y)

            data_new[x][y] = process_pix(x, y, input, windows_shape, max_pixel, ratio_top, ratio_migration)

    return data_new

# 图像缩放
def pic_scale(input, windows_shape=3, center_ratio=0.5, x_size=100.0, y_size=100.0, ratio_top=0.1):
    if x_size <= 1.0:
        x_size = int(x_size * input.shape[0])
    else:
        x_size = int(x_size)
    if y_size <= 1.0:
        y_size = int(y_size * input.shape[1])
    else:
        y_size = int(y_size)

    pic_new = np.zeros((x_size, y_size)).astype('uint8')

    if x_size>input.shape[0] | y_size>input.shape[1]:
        print('size error pic processing..............')
        exit()

    if windows_shape%2 != 1:
        print('windows shape error, must be single......')
        exit()


    for i in range(x_size):
        for j in range(y_size):
            index_x = int(i/x_size * (input.shape[0] - 1))
            index_y = int(j/y_size * (input.shape[1] - 1))

            # 寻找窗口的index
            start_index_x = max(index_x - windows_shape // 2, 0)
            end_index_x = min(index_x + windows_shape // 2 + 1, input.shape[0])
            start_index_y = max(index_y - windows_shape // 2, 0)
            end_index_y = min(index_y + windows_shape // 2 + 1, input.shape[1])

            # 根据窗口index 获得窗口的 数据
            data_windows = copy.deepcopy(input[start_index_x:end_index_x, start_index_y:end_index_y]).ravel()

            value = input[index_x][index_y]

            windows_mean = np.mean(data_windows)

            ordered_list = sorted(data_windows)
            small_top = np.mean(ordered_list[:int(len(ordered_list) * ratio_top)])
            big_top = np.mean(ordered_list[-int(len(ordered_list) * ratio_top):])

            if windows_mean > int(center_ratio * 256):
                pic_new[i][j] = int(max(value, big_top))
            elif windows_mean < int(center_ratio * 256):
                pic_new[i][j] = int(min(value, small_top))

    return pic_new


# 图像缩放
def pic_scale_normal(input, shape=(196, 196)):
    if len(input.shape) == 2:
        if (shape[0] < input.shape[0]) | (shape[1] < input.shape[1]):
            print('pic scale fun error:shape {}&{}'.format(shape,input.shape))
            exit(0)

        pic_new = np.zeros(shape).astype('uint8')

        for i in range(shape[0]):
            for j in range(shape[1]):
                index_x = int(i/shape[0] * (input.shape[0] - 1))
                index_y = int(j/shape[1] * (input.shape[1] - 1))

                pic_new[i][j] = input[index_x, index_y]

        return pic_new
    elif len(input.shape) == 3:
        img_tar = []
        for i in range(input.shape[0]):
            img_tar.append(cv2.resize(input[i], shape))
        return np.array(img_tar)
    else:
        print('error shape:{}'.format(input.shape))
        exit(0)
# def get_pixel_normal(pic_t=np.random.random((9, 17))):
#     index_1 = -1
#     index_2 = -1
#     for j in range(pic_t.shape[1]):
#         if (pic_t[0][j] < 0) :
#             index_2 = j
#             if index_1 == -1:
#                 index_1 = j
#
#     # print(index_1, index_2)
#     a = pic_t[:, :index_1]
#     b = pic_t[:, index_2+1:]
#
#     # print(a, b)
#
#     if a.shape[1] > b.shape[1]:
#         return np.mean(a)
#     else:
#         return np.mean(b)


def test_pic_enhance_effect():
    data_img, data_depth = get_test_ele_data()
    print(data_img.shape, data_depth.shape)

    processing_pic = data_img[0:600, :]
    pic_EH = pic_enhence_random(processing_pic, windows_shape=3, ratio_top=0.1, ratio_migration=0.3, random_times=1)
    pic_equalizeHist = cv2.equalizeHist(processing_pic)  # 直方图均衡化


    show_Pic([processing_pic, pic_EH], save_pic=False, pic_order='12', pic_str=['pic_org', 'pic_enhance'])

    # hist_o = cv2.calcHist([np.uint8(processing_pic)], [0], None, [256], [0, 256])
    # hist_EH = cv2.calcHist([np.uint8(pic_EH)], [0], None, [256], [0, 256])
    # plt.subplot(2, 2, 1)
    # plt.plot(hist_o/processing_pic.size, label="原图灰度直方图", linestyle="--", color='g')
    # plt.legend()
    # plt.subplot(2, 2, 2)
    # plt.plot(hist_EH/pic_EH.size, label="增强后灰度直方图", linestyle="--", color='r')
    # plt.legend()
    # # plt.show()
    #
    # plt.subplot(2, 2, 3)
    # plt.imshow(processing_pic)
    # plt.subplot(2, 2, 4)
    # plt.imshow(pic_EH)
    # plt.savefig("pic_enhance_effect.jpg")
    # plt.show()


# a = cv2.imread('1_2.png')
# print(a.shape)
# show_Pic([a , a, a, a], pic_order='22', save_pic=True, path_save='121212.png')



def metrological_performance():
    img1 = cv2.imread('messi5.jpg')
    e1 = cv2.getTickCount()
    for i in range(5, 49, 2):
        img1 = cv2.medianBlur(img1, i)
    e2 = cv2.getTickCount()
    t = (e2 - e1) / cv2.getTickFrequency()
    # print(t)


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0/gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)
    table = np.array(table).astype("uint8")
    return cv2.LUT(image, table)

def save_img_data(dep, data, path=''):
    dep = np.reshape(dep, (-1, 1))
    data = np.hstack((dep, data))

    np.savetxt(path, data, fmt='%.4f', delimiter='\t', comments='',
               header='WELLNAME={}\nSTDEP\t= {}\nENDEP\t= {}\nLEV\t= {:.4f}\nUNIT\t= meter\nCURNAMES= {}\n#DEPTH\t{}'.format(
                   'Temp_well', dep[0, 0], dep[-1, 0], dep[1, 0]-dep[0, 0], 'Img_data', 'Img_data'))



def pic_smooth_effect_compare():

    # import cv2 as cv
    # import numpy as np
    from matplotlib import pyplot as plt
    # img = cv.imread('opencv-logo-white.png')
    os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'
    plt.rcParams['font.family'] = 'SimHei'
    data_img_dyna, data_img_stat, data_depth = get_test_ele_data()
    print('data_image shape:{}'.format(data_img_dyna.shape))

    data_img = data_img_dyna

    index_1, index_2 = 1400, 1800

    # img = np.uint8(data_img[index_1:index_2, :])
    img = np.uint8(data_img)
    dep_img = data_depth[index_1:index_2, :]
    avg_blur = cv2.blur(img, (5, 5))
    # plt.subplot(121), plt.imshow(img), plt.title('Original')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(blur), plt.title('Blurred')
    # plt.xticks([]), plt.yticks([])
    # plt.show()

    guass_blur = cv2.GaussianBlur(img, (5, 5), 0)

    median_blur = cv2.medianBlur(img, 5)

    bil_fil_blur = cv2.bilateralFilter(img, 9, 75, 75)

    windows_shape = [3, 5, 7, 9]
    ratio_mig = [0.6, 0.6, 0.6, 0.6]
    random_times = [1, 1, 1, 1]

    pic_EH_3 = pic_enhence_random(img, windows_shape=windows_shape[0], ratio_migration=ratio_mig[0], random_times=random_times[0])
    pic_EH_5 = pic_enhence_random(img, windows_shape=windows_shape[1], ratio_migration=ratio_mig[1], random_times=random_times[1])
    pic_EH_7 = pic_enhence_random(img, windows_shape=windows_shape[2], ratio_migration=ratio_mig[2], random_times=random_times[2])
    pic_EH_9 = pic_enhence_random(img, windows_shape=windows_shape[3], ratio_migration=ratio_mig[3], random_times=random_times[3])

    pic_equalizeHist = cv2.equalizeHist(img)  # 直方图均衡化

    # 对图像进行局部直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))  # 对图像进行分割，10*10
    pic_local_equalizeHist = clahe.apply(img)  # 进行直方图均衡化

    imgGrayNorm = img / 255
    gamma = 2
    img_gamma = np.power(imgGrayNorm, gamma) * 256

    img_gamma_1 = adjust_gamma(img, 2)

    # show_Pic([img, avg_blur, guass_blur, median_blur, bil_fil_blur, pic_EH], pic_order='23',
    #          pic_str=['original image', 'Average blur', 'Gaussian blur image', 'median blur image', 'bilateral filter image', 'Random shift enhance'])

    # save_img_data(dep_img, img, 'org.txt')
    # save_img_data(dep_img, pic_EH, 'my.txt')
    # save_img_data(dep_img, pic_equalizeHist, 'hist.txt')
    # save_img_data(dep_img, img_gamma, 'gama.txt')
    # plt.imshow(img, cmap='hot')
    # plt.axis('off')
    # plt.show()
    # plt.imshow(pic_EH, cmap='hot')
    # plt.axis('off')
    # plt.show()
    # plt.imshow(img_gamma, cmap='hot')
    # plt.axis('off')
    # plt.show()
    # plt.imshow(pic_equalizeHist, cmap='hot')
    # plt.axis('off')
    # plt.show()

    # # plt.subplot(2, 2, 1)
    # plt.subplot(1, 1, 1)
    # # plt.plot(hist_o, label="原始电成像图像灰度直方图", linestyle="--", color='g')
    # plt.plot(hist_o, linestyle="--", color='g')
    # plt.xlabel("像素值")
    # plt.ylabel("频率分布")
    # plt.legend()
    # plt.show()
    #
    # # plt.subplot(2, 2, 2)
    # plt.subplot(1, 1, 1)
    # # plt.plot(hist_EH, label="随即迁移增强后的灰度直方图", linestyle="--", color='r')
    # plt.plot(hist_EH, linestyle="--", color='r')
    # plt.xlabel("像素值")
    # plt.ylabel("频率分布")
    # plt.legend()
    # plt.show()
    #
    # # plt.subplot(2, 2, 3)
    # plt.subplot(1, 1, 1)
    # # plt.plot(hist_gamma, label="伽马变换增强后的灰度直方图", linestyle="--", color='g')
    # plt.plot(hist_gamma, linestyle="--", color='g')
    # plt.xlabel("像素值")
    # plt.ylabel("频率分布")
    # plt.legend()
    # plt.show()
    #
    # # plt.subplot(2, 2, 4)
    # plt.subplot(1, 1, 1)
    # # plt.plot(hist_equalize_hist, label="直方图均衡增强后的灰度直方图", linestyle="--", color='g')
    # plt.plot(hist_equalize_hist, linestyle="--", color='g')
    # plt.xlabel("像素值")
    # plt.ylabel("频率分布")
    # plt.legend()
    # plt.savefig("pic_enhance_effect.jpg")
    # plt.show()

    show_Pic([img, pic_EH_3, pic_EH_5, pic_EH_7], pic_order='22',
             pic_str=['原始电成像图像', '像素值偏移增图像效果:n=5', '像素值偏移增图像效果:n=7', '像素值偏移增图像效果:n=9'])

    # hist_o = cv2.calcHist([np.uint8(img)], [0], None, [256], [0, 256])/img.size
    # hist_EH = cv2.calcHist([np.uint8(pic_EH_3)], [0], None, [256], [0, 256])/img.size
    # hist_gamma = cv2.calcHist([np.uint8(img_gamma)], [0], None, [256], [0, 256])/img.size
    # # hist_gamma_1 = cv2.calcHist([np.uint8(img_gamma_1)], [0], None, [256], [0, 256])/img.size
    # hist_equalize_hist = cv2.calcHist([np.uint8(pic_equalizeHist)], [0], None, [256], [0, 256])/img.size

    # plt.subplot(2, 2, 1)
    # plt.plot(hist_o, label="原始电成像图像灰度直方图", linestyle="--", color='g')
    # plt.xlabel("像素值")
    # plt.ylabel("频率分布")
    # plt.legend()
    # plt.subplot(2, 2, 2)
    # plt.plot(hist_EH, label="随机迁移增强后的灰度直方图", linestyle="--", color='r')
    # plt.xlabel("像素值")
    # plt.ylabel("频率分布")
    # plt.legend()
    # plt.subplot(2, 2, 3)
    # plt.plot(hist_gamma, label="伽马变换增强后的灰度直方图", linestyle="--", color='g')
    # plt.xlabel("像素值")
    # plt.ylabel("频率分布")
    # plt.legend()
    # plt.subplot(2, 2, 4)
    # plt.plot(hist_equalize_hist, label="直方图均衡增强后的灰度直方图", linestyle="--", color='g')
    # plt.xlabel("像素值")
    # plt.ylabel("频率分布")
    # plt.legend()
    # # plt.savefig("pic_enhance_effect.jpg")
    # plt.show()

# pic_smooth_effect_compare()







# 对二维数据 进行 简单的数据缩放
def pic_scale_simple(pic=np.array([]), pic_shape=[0,0]):
    if len(pic.shape) >= 3:
        print('only process two dim pic& pic shape is:{}'.format(pic.shape))
        exit(0)

    if pic_shape[0] <= 0:
        print('shape error...:{}'.format(pic_shape))
        exit(0)
    elif (pic_shape[0]>1) & (pic_shape[1]>1):
        x_size = pic_shape[0]
        y_size = pic_shape[1]
    elif (pic_shape[0] <= 1) & (pic_shape[0] > 0) & (pic_shape[1] <= 1) & (pic_shape[1] > 0):
        x_size = int(pic_shape[0] * pic.shape[0])
        y_size = int(pic_shape[1] * pic.shape[1])
    elif (pic_shape[0] > pic.shape[0]) | (pic_shape[1] > pic.shape[1]):
        print('target pic shape is {},org shape is {}'.format(pic_shape, pic.shape))
        exit(0)
    else:
        print('pic shape error:{}'.format(pic_shape))
        exit(0)

    pic_new = np.zeros((x_size, y_size))
    for i in range(x_size):
        for j in range(y_size):
            index_x = int(i/x_size*pic.shape[0])
            index_y = int(j/y_size*pic.shape[1])
            pic_new[i][j] = pic[index_x][index_y]

    return pic_new

def pic_repair_normal(pic):
    PicDataWhiteStripe = copy.deepcopy(pic)

    # 手动空白带提取
    for i in range(pic.shape[0]):
        for j in range(pic.shape[1]):
            if pic[i][j] <= 0.01:
                PicDataWhiteStripe[i][j] = 255
            else:
                PicDataWhiteStripe[i][j] = 0
    # 空白带提取
    # ret, PicDataWhiteStripe = cv2.threshold(pic, 0, 1, cv2.THRESH_BINARY_INV)

    PicDataWhiteStripe = np.uint8(PicDataWhiteStripe)
    pic = np.uint8(pic)

    # TELEA 图像修复
    PIC_Repair_dst_TELEA = cv2.inpaint(pic, PicDataWhiteStripe, 5, cv2.INPAINT_TELEA)
    # NS 图像修复
    PIC_Repair_dst_NS = cv2.inpaint(pic, PicDataWhiteStripe, 5, cv2.INPAINT_NS)

    return PIC_Repair_dst_TELEA, PIC_Repair_dst_NS, PicDataWhiteStripe
# pic_new = pic_scale_simple(pic_shape=[0.5, 0.5])
# print(pic_new)


def pic_seg_by_kai_bi():
    path_in = r'C:\Users\Administrator\Desktop\paper_f\unsupervised_segmentation\fracture\LN11-4_367_5444.3994_5445.0244_dyna.png'
    # path_in = r'D:\Data\target_stage3_small_p\train\2\LG701-H1_358_5380.5020_5381.1695_dyna.png'
    path_in = r'D:\Data\target_stage3_small_p\train\1\LG7-4_301_5259.5002_5260.1627_dyna.png'
    path_in = r'D:\Data\target_stage3_small_p\train\1\LG7-4_132_5171.0002_5171.6427_dyna.png'
    path_in = r'D:\Data\target_stage3_small_p\train\1\LG701_205_5224.5000_5225.1325_dyna.png'
    path_in = r'D:\Data\target_stage3_small_p\train\1\LG7-4_129_5169.5002_5170.1252_dyna.png'
    path_in = r'D:\Data\target_stage3_small_p\train\1\LG7-4_116_5162.0002_5162.6252_dyna.png'
    path_in = r'D:\Data\target_stage3_small_p\train\1\LG701_189_5215.5000_5216.1725_dyna.png'
    path_in = r'D:\Data\target_stage3_small_p\train\1\LG701_194_5218.0000_5218.6275_dyna.png'
    path_in = r'D:\Data\target_stage3_small_p\train\1\LG701-H1_104_5248.0020_5248.6370_dyna.png'
    path_in = r'D:\Data\target_stage3_small_p\train\1\LG7-4_107_5157.5002_5158.1277_dyna.png'
    path_in = r'D:\Data\target_stage3_small_p\train\1\LG701-H1_93_5242.5020_5243.1645_dyna.png'
    path_in = r'D:\Data\target_stage3_small_p\train\1\LG7-4_301_5259.5002_5260.1627_dyna.png'
    path_in = r'D:\Data\target_stage3_small_p\train\1\LG701_126_5183.0000_5183.6600_dyna.png'
    # path_in = r'D:\Data\target_stage3_small_p\train\2\LG701-H1_241_5320.0020_5320.6595_dyna.png'
    # path_in = r'D:\Data\target_stage3_small_p\train\2\LG701-H1_250_5324.5020_5325.1645_dyna.png'
    # path_in = r'D:\Data\target_stage3_small_p\train\2\LG701-H1_249_5324.0020_5324.6745_dyna.png'
    # path_in = r'D:\Data\target_stage3_small_p\train\2\LG701-H1_148_5271.5020_5272.1295_dyna.png'
    # path_in = r'D:\Data\target_stage3_small_p\train\2\LG7-4_336_5278.0002_5278.6477_dyna.png'

    pic = cv2.imread(path_in, cv2.IMREAD_GRAYSCALE)
    print(pic.shape)

    ProcessingPic = copy.deepcopy(pic)

    Blur_Average = cv2.blur(ProcessingPic, (7, 5))
    Blur_Gauss = cv2.GaussianBlur(ProcessingPic, (7, 5), 0)
    Blur_Median = cv2.medianBlur(ProcessingPic, 5)

    firstLevel = 40
    ret, img_binary_Level1 = cv2.threshold(ProcessingPic, firstLevel, 255, cv2.THRESH_BINARY)
    ret, img_binary_Level2 = cv2.threshold(ProcessingPic, firstLevel + 10, 255, cv2.THRESH_BINARY)
    ret, img_binary_Level3 = cv2.threshold(ProcessingPic, firstLevel + 20, 255, cv2.THRESH_BINARY)
    ret, img_binary_Level4 = cv2.threshold(ProcessingPic, firstLevel + 30, 255, cv2.THRESH_BINARY)
    ret, img_binary_Level5 = cv2.threshold(ProcessingPic, firstLevel + 40, 255, cv2.THRESH_BINARY)
    ret, img_binary_Level6 = cv2.threshold(ProcessingPic, firstLevel + 130, 255, cv2.THRESH_BINARY)
    ProcessingPic = img_binary_Level6

    # cv2.THRESH_BINARY：二值阈值处理，只有大于阈值的像素值为最大值，其他像素值为最小值。
    # cv2.THRESH_BINARY_INV：反二值阈值处理，只有小于阈值的像素值为最大值，其他像素值为最小值。
    # cv2.THRESH_TRUNC：截断阈值处理，大于阈值的像素值被赋值为阈值，小于阈值的像素值保持原值不变。
    # cv2.THRESH_TOZERO：置零阈值处理，只有大于阈值的像素值被置为0，其他像素值保持原值不变。
    # cv2.THRESH_TOZERO_INV：反置零阈值处理，只有小于阈值的像素值被置为0，其他像素值保持原值不变。

    Kernel_Rect = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))  # 生成形状为矩形5x5的卷积核
    Kernel_Rect2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    Kernel_Rect3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    Kernel_Ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 3))  # 椭圆形9x9
    kernel = np.ones((5, 5), np.uint8)
    targetKernel = Kernel_Rect3


    Pic_erosion = cv2.erode(ProcessingPic, targetKernel, iterations=1)
    Pic_dilation = cv2.dilate(ProcessingPic, targetKernel, iterations=1)
    Pic_opening = cv2.morphologyEx(ProcessingPic, cv2.MORPH_OPEN, targetKernel)
    Pic_closing = cv2.morphologyEx(ProcessingPic, cv2.MORPH_CLOSE, targetKernel)
    Pic_opening_closing = cv2.morphologyEx(Pic_opening, cv2.MORPH_CLOSE, targetKernel)
    Pic_closing_opening = cv2.morphologyEx(Pic_closing, cv2.MORPH_OPEN, targetKernel)

    ProcessingPic = Pic_opening_closing

    contours_Conform, contours_Drop, contours_All = GetPicContours(ProcessingPic, threshold=500)
    img_white = np.zeros((ProcessingPic.shape[0], ProcessingPic.shape[1]), np.uint8)
    img_white2 = np.zeros((ProcessingPic.shape[0], ProcessingPic.shape[1]), np.uint8)
    # img_white = np.zeros_like(ProcessingPic).astype(np.uint8).fill(0)
    # img_white = copy.deepcopy(ProcessingPic).astype(np.uint8).fill(0)
    print(ProcessingPic.shape[0], ProcessingPic.shape[1])
    # print(img_white)
    cv2.drawContours(img_white, contours_Conform[1], -1, 255, thickness=-1)
    # print(img_mask)

    # show_Pic([pic, ProcessingPic, img_white, img_white2], pic_order='14', pic_str=[], save_pic=False)

    cv2.imwrite(path_in.replace('dyna', 'mask2'), img_white)
    cv2.imwrite(path_in.replace('dyna', 'mask'), ProcessingPic)


# pic_seg_by_kai_bi()