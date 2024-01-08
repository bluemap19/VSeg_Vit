import math
import numpy as np
import matplotlib.pyplot as plt
import scipy

from src_ele.file_operation import get_ele_data_from_path

# rarray = np.random.random(size=10000)
# rarray = rarray*1000
# rarray = np.reshape(rarray, newshape=(100, 100))
# print(rarray, rarray.shape)


def spilt_num(num):
    a = math.floor(math.log(num, 10))
    b = math.floor(num * math.pow(10, -a))
    c = math.floor(num * math.pow(10, -(a-1)))%10
    # print(num, a, b, c)
    return a, b, c

def get_x_y_smooth_value(x, y, step=0.9):
    if (x.shape != (3,)) | (y.shape != (3,)):
        print('x or y shape error.....')
        print('x shape:{}, y shape:{}'.format(x.shape, y.shape))
        exit(0)

    y_should = y[0] + (y[2]-y[0])/(x[2]-x[0])*(x[1]-x[0])
    y_n = y_should + (1-step)*(y[1]-y_should)
    return y_n

def line_smooth(x, y, step=0.9):
    print(x.shape, y.shape[0])
    if x.shape != y.shape:
        print('smooth shape error.......')
        exit(0)

    y_new = np.zeros((y.shape[0], ))
    for i in range(x.shape[0]):
        if (i==0) | (i==(x.shape[0]-1)):
            y_new[i] = y[i]
            pass
        else:
            y_new[i] = get_x_y_smooth_value(x[i-1:i+2], y[i-1:i+2], step=step)

    return y_new

def get_ele_data_distribute(pic, dimension=5, dis_start=0.1):
    distribute_dimension = []
    distribute_dimension_detailed = []
    dis_start_still = dis_start
    # 获得成像谱的 统计x用 数列 包括两组，详尽的和粗略的
    for i in range(dimension):
        for j in range(9):
            distribute_dimension.append(dis_start_still * (j+1))
            distribute_dimension_detailed.append(dis_start_still * (j+1))
            for k in range(9):
                distribute_dimension_detailed.append(dis_start_still * (j+1+0.1*(k+1)))
        dis_start_still *= 10
    dis_short = np.zeros((len(distribute_dimension),))
    dis_long = np.zeros((len(distribute_dimension_detailed),))
    # print(np.reshape(distribute_dimension_detailed, (-1, 9)))
    # print(len(distribute_dimension), len(distribute_dimension_detailed))

    # 对数据进行统计
    mi_start = math.floor(math.log(dis_start, 10))
    # print(mi_start)
    num_pixel_sum = 0
    for i in range(pic.shape[0]):
        for j in range(pic.shape[1]):
            if pic[i][j] < 0:
                continue
            else:
                num_pixel_sum += 1
            mi, ge_wei, shifen_wei = spilt_num(pic[i][j])
            mi -= mi_start
            # if mi < 2:
            #     print(mi, ge_wei, shifen_wei)
            index_short = 9 * mi + ge_wei - 1
            index_long = 90 * mi + 10 * (ge_wei-1) + shifen_wei
            if index_short >= len(dis_short):
                index_short = len(dis_short) - 1
            if index_long >= len(dis_long):
                index_long = len(dis_long) - 1
            # if mi < 2:
            #     print(pic[i][j], mi, ge_wei, shifen_wei, index_short, index_long, distribute_dimension[index_short], distribute_dimension_detailed[index_long])
            dis_short[index_short] += 1
            dis_long[index_long] += 1
    dis_long = dis_long/num_pixel_sum
    dis_short = dis_short/num_pixel_sum
    # print(np.reshape(distribute_dimension_detailed, (-1, 10)))

    # 获取画图用x谱
    # space_x_scale_sum = [0, 0.3010, 0.4771, 0.6021, 0.6990, 0.7782, 0.8451, 0.9031, 0.9542, 1.0000]
    space_x_scale_sum = [0.0, 0.28906483, 0.45815691, 0.57812965, 0.67118774, 0.74722174, 0.81150756, 0.86719448, 0.91631382, 0.96025257]
    space_x_scale = [0.3010, 0.1761, 0.125, 0.0969, 0.0792, 0.0669, 0.0580, 0.0511, 0.0458]
    x_scale_short = [0]
    x_scale_long = []
    for i in range(len(dis_short)):
        temp = i % 9
        x_scale_short.append(x_scale_short[i] + space_x_scale[temp])
        for j in range(10):
            # x_scale_long.append(x_scale_long[-1] + space_x_scale[temp]*space_x_scale[j])
            x_scale_long.append(x_scale_short[-2] + space_x_scale[temp] * space_x_scale_sum[j])
            # print(x_scale_short[-2], space_x_scale[temp], space_x_scale_sum[j], space_x_scale[temp]*space_x_scale_sum[j])
    x_scale_short.pop(0)
    # x_scale_short = np.array(x_scale_short).reshape((dimension, 9))
    # x_scale_long = np.array(x_scale_long).reshape((dimension, 9, -1))
    x_scale_short = np.array(x_scale_short) * math.pow(10, dimension) * dis_start
    x_scale_long = np.array(x_scale_long) * math.pow(10, dimension) * dis_start
    print(x_scale_short)
    print(x_scale_long)

    ynew = line_smooth(x_scale_long, dis_long, step=0.5)
    # from scipy import interpolate
    # func = interpolate.interp1d(x_scale_long, dis_long, kind='cubic')
    # # 利用xnew和func函数生成ynew，xnew的数量等于ynew数量
    # ynew = func(x_scale_long)
    plt.semilogx(np.array(distribute_dimension_detailed), dis_long)
    plt.semilogx(np.array(distribute_dimension_detailed), ynew)
    plt.show()

    pass

def pic_to_text(path=r'D:\1111\Input\zg112-r\zg112-1-r_6207.3027_6571.2021_XRMI_EXCELL2000_.png'):
    img_data, depth_data = get_ele_data_from_path(path)
    print(img_data.shape, depth_data.shape)
    charter = path.split('_')[0].split('\\')[-1]
    lev = (depth_data[-1, 0] - depth_data[0, 0])/depth_data.shape[0]
    len_path = len(path)

    print(charter, lev, path[:len_path-4])
    np.savetxt(path[:len_path-4]+'.txt', np.hstack((depth_data, img_data)),
        header='WELLNAME= {}\nSTDEP\t= {}\nENDEP\t= {}\nLEV\t= {}\nUNIT\t= meter\nCURNAMES= {}\n\n#DEPTH'.format(
        charter, depth_data[0][0], depth_data[-1][0], lev, 'IMAGE.DYNA_FULL'), delimiter='\t', comments='', fmt='%.4f')
    pass

# get_ele_data_distribute(rarray)
# line_test()

# pic_to_text()