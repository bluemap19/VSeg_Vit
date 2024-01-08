import copy
import numpy as np
from matplotlib import pyplot as plt


# 数据合并，把几种带有深度的 测井信息进行 通过深度的 数据合并导出，按统一深度把数据合并，方便数据统一后进行无监督聚类
# curve_zip为 曲线集合，曲线必须带与深度，且深度信息放在第一列
def data_combine(curve_zip, depth=[-1, 0], step='MIN'):

    # 计算最小的step， 当作整体的step, 计算最小的end_dep， 当作整体的end_dep, 计算最大的start_dep， 当作整体的start_dep
    step_final = 10
    start_depth = -1
    end_depth = 999999999
    # 总的曲线条数
    curve_num = 0

    # 遍历曲线集合，统计曲线条数，统计曲线的顶深底深，用来新建合并后的曲线信息
    for i in range(len(curve_zip)):
        # print(curve_zip[i].shape)
        # 计算 该数据集的 曲线 step
        step_n = (curve_zip[i][-1, 0] - curve_zip[i][0, 0]) / curve_zip[i].shape[0]
        # 计算 该数据集的 曲线条数
        curve_num += (curve_zip[i].shape[1] - 1)
        if step == 'MIN':
            step_final = min(step_n, step_final)
        else:
            step_final = step
        # 统计各自的顶底信息，并选取最大的顶深start_depth，最小的底深end_depth
        start_depth = max(start_depth, curve_zip[i][0, 0])
        end_depth = min(end_depth, curve_zip[i][-1, 0])

        # print(curve_zip[i][-1, 0], curve_zip[i][0, 0])
        # print(start_depth, end_depth)
        # print(step_final)

    # 如果深度初始化了，进行深度初始化的赋值
    if (depth[0] >= 0) & (depth[1] >= 0):
        if (min(depth) < start_depth) | (max(depth) > end_depth):
            print('dep pre value error:合理区间{}， 实际区间{}'.format([start_depth, end_depth], depth))
            exit()
        start_depth = max(min(depth), start_depth)
        end_depth = min(max(depth), end_depth)

    step_num = int((end_depth - start_depth)/step_final)

    # print(start_depth, end_depth, step_final, step_num, curve_num)

    all_curve_np = np.zeros((step_num, 1))

    index_curve = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    values = [[], [], [], [], [], [], [], [], [], []]
    for i in range(step_num):
        temp_dep = start_depth + step_final * i
        all_curve_np[i][0] = temp_dep
        for j in range(len(curve_zip)):
            while index_curve[j] < curve_zip[j].shape[0]-1:
                if (temp_dep >= curve_zip[j][index_curve[j], 0]) & (temp_dep <= curve_zip[j][index_curve[j]+1, 0]) | (abs(temp_dep-curve_zip[j][index_curve[j], 0]) <= step_final/2):
                    values[j].append(curve_zip[j][index_curve[j], 1:])
                    # print(index_curve[j])
                    index_curve[j] -= 1
                    break
                else:
                    index_curve[j] += 1

        # print(index_curve[:3])

    for i in range(len(curve_zip)):
        # print(np.array(values[i]).shape)
        all_curve_np = np.hstack((all_curve_np, np.array(values[i])))
        # print(all_curve_np.shape)

    # print(values[0])
    # print(all_curve_np[:10, :])
    return all_curve_np


# 把 深度-类别 信息 合并成 顶深-底深-类别 信息
def layer_table_combine(layer_inf_str):
    layer_class = []
    layer_inf_final = []

    i = 0
    while i < layer_inf_str.shape[0]:
        new_lis = []
        new_lis.append(i)
        i += 1
        while layer_inf_str[i][-1] == layer_inf_str[i-1][-1]:
            new_lis.append(i)
            i+=1
            if i == layer_inf_str.shape[0]:
                break
        layer_class.append(new_lis)

    # print(layer_class)
    for i in range(len(layer_class)):
        # print(layer_class[i])
        dep_start = layer_inf_str[layer_class[i][0]][0]
        dep_end = layer_inf_str[layer_class[i][-1]][1]
        dep_class = layer_inf_str[layer_class[i][0]][-1]

        layer_inf_final.append(np.array([dep_start, dep_end, dep_class]))

    return np.array(layer_inf_final)


# 对聚类结果进行画图
from pylab import mpl
# 设置中文显示字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False
def two_disScatter(KmeansPred, X_Input, Y_Input, pltTitle='', img_path=r'', skip_point = -1,
                   class_name=[], x_label='x_label', y_label='y_label'):
    X_Input_N = []
    Y_Input_N = []
    KmeansPred_N = []
    if skip_point > 0:
        for i in range(X_Input.shape[0]):
            if i % skip_point == 0:
                X_Input_N.append(X_Input[i])
                Y_Input_N.append(Y_Input[i])
                KmeansPred_N.append(KmeansPred[i])

        X_Input = np.array(X_Input_N)
        Y_Input = np.array(Y_Input_N)
        KmeansPred = np.array(KmeansPred_N).astype(np.int64)

    print('Drawing Scatter..........')
    ClassNum = int(np.max(KmeansPred))+1
    # ColorList = ['black', 'red', 'chartreuse', 'springgreen', 'orange', 'dodgerblue', 'fuchsia', 'cornflowerblue', 'maroon', 'slategray']
    ColorList = ['black', 'darkred', 'darkslateblue', 'm', 'seagreen', 'cadetblue', 'tan', 'olivedrab', 'peru', 'slategray']
    # MarkerList = [',', '.', '1', '2', '3', '4', '8', 's', 'o', 'v', '+', 'x', '^', '<', '>', 'p']
    MarkerList = ['v', '*', 'X', '+', 'D', 'h', '1', 's', 'o', '.', 'd', '>', '^', '<', 'p']
    for i in range(ClassNum):
        if i>= len(class_name):
            class_name.append('label{}'.format(i+1))

    # 基于类绘制图形
    pltPic = []
    for i in range(KmeansPred.shape[0]):
        a = []
        for j in range(ClassNum):
            if KmeansPred[i] == j:
                a = plt.scatter(X_Input[i], Y_Input[i], c=ColorList[j], marker=MarkerList[j], s=20)
        pltPic.append(a)

    plt.xlabel(x_label.replace('chu', '/'))
    plt.ylabel(y_label.replace('chu', '/'))
    plt.legend(pltPic, class_name[:ClassNum])
    plt.title(pltTitle.replace('chu', '/'))
    plt.savefig(img_path+'/{}.png'.format(pltTitle))
    plt.clf()
    plt.close()
    # plt.savefig()
    # plt.show()


# 数据标准化
def data_normalized(data_normal_curve_org, max_ratio=0.1, data_proportion=[-1, -1], DEPTH_USE=False):
    ExtremePointNum = int(data_normal_curve_org.shape[0] * max_ratio + 1)

    data_normal_curve_new = copy.deepcopy(data_normal_curve_org)
    if DEPTH_USE:
        data_normal_curve_new = copy.deepcopy(data_normal_curve_org[:, 1:])

    for i in range(data_normal_curve_new.shape[1]):
        bigTop = np.mean(np.sort(data_normal_curve_new[:, i].reshape(1, -1)[0])[-ExtremePointNum:])
        smallTop = np.mean(np.sort(data_normal_curve_new[:, i].reshape(1, -1)[0])[:ExtremePointNum])
        Step = 1 / (bigTop - smallTop)
        # print(bigTop, smallTop, Step)
        for j in range(data_normal_curve_new.shape[0]):
            d_temp = (data_normal_curve_new[j][i] - smallTop) * Step
            d_temp = max(min(d_temp, 1), 0)
            data_normal_curve_new[j][i] = d_temp

    # data_normal_curve_new[:, 0] = data_normal_curve[:, 0]

    for i in range(len(data_proportion)):
        if data_proportion[i] > 0:
            for j in range(data_normal_curve_new.shape[0]):
                data_normal_curve_new[j, i] *= data_proportion[i]

    if DEPTH_USE:
        data_normal_curve_org[:, 1:] = data_normal_curve_new
        data_normal_curve_new = data_normal_curve_org

    return data_normal_curve_new



def get_data_by_depth(data_normal_curve, depth, continue_index=0):
    step = (data_normal_curve[-1, 0] - data_normal_curve[0, 0]) / data_normal_curve.shape[0]
    data = []

    if (depth<data_normal_curve[0][0]) | (depth>data_normal_curve[-1][0]):
        return data, continue_index

    for i in range(data_normal_curve.shape[0] - continue_index):
        depth_temp = data_normal_curve[i + continue_index][0]
        if  (abs(depth-depth_temp) <= step/2) | (i==data_normal_curve.shape[0]-continue_index-1) | (depth<=depth_temp):
            continue_index -= 2
            break
    data = data_normal_curve[i, :]
    continue_index += i

    return data, max(continue_index, 0)


def get_data_by_depths(data_normal_curve, depth):
    if (depth[0] < data_normal_curve[0][0]) | (depth[1] > data_normal_curve[-1][0]):
        print('depths error :{} {} {}'.format(depth, data_normal_curve[0][0], data_normal_curve[-1][0]))
        exit(0)

    index_start = 0
    index_end = 0

    for i in range(data_normal_curve.shape[0]):
        if data_normal_curve[i][0] > depth[0]:
            index_start = i
            break

    for i in range(data_normal_curve.shape[0]):
        if data_normal_curve[i][0] > depth[1]:
            index_end = i
            break

    return data_normal_curve[index_start:index_end, :]

# 将三维的 dep_start dep_end class 转换成二维的 dep class
def layer_table_to_list(np_layer, step=0.0762, depth=[-1, -1]):
    if np_layer.shape[1] > 3:
        print('label if too larget, please give label as n*3 shape...')
        exit()
    dep_start = np_layer[0][0]
    dep_end = np_layer[-1][1]
    if (depth[0] > 0) & (depth[0] >= dep_start-step) & (depth[0] <= dep_end + step):
        dep_start = depth[0]
    if (depth[1] > 0) & (depth[1] >= dep_start-step) & (depth[0] <= dep_end + step):
        dep_end = depth[1]

    num_dep = int((dep_end - dep_start) / step)

    # print(num_dep)
    list_layer_depth = []
    list_layer_class = []
    layer_index = 0
    # print(list_layer_class.shape)
    for i in range(num_dep):
        dep_temp = dep_start + i * step

        while (layer_index < np_layer.shape[0]):
            if ((dep_temp >= np_layer[layer_index][0]-step) & (
                    dep_temp <= np_layer[layer_index][1]+step)):
                list_layer_depth.append(dep_temp)
                list_layer_class.append(np_layer[layer_index][2])
                # layer_index -= 1
                break
            else:
                layer_index += 1
    list_layer_depth = np.array(list_layer_depth).astype(float)
    list_layer_class = np.array(list_layer_class)

    return np.hstack((list_layer_depth.reshape((-1, 1)), list_layer_class.reshape((-1, 1))))



def data_analysis(data_normal_curve, list_layer_class):
    index_continue = 0
    data_box = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]

    num_cluster = int(max(list_layer_class[:, -1]))
    if num_cluster >= len(data_box):
        print('cluster is too many.......')
        exit(0)

    # 对不同类别的数据进行分别装盒
    for i in range(list_layer_class.shape[0]):
        layer_class = int(list_layer_class[i, -1])
        # print(list_layer_class[i, 0], layer_class)
        data, index_continue = get_data_by_depth(data_normal_curve, depth=list_layer_class[i, 0], continue_index=index_continue)
        if data != []:
            data_box[layer_class].append(data)
        # print(data, index_continue)

    # 分别对不同盒中的数据进行分析
    data_statistical = []
    for i in range(num_cluster+1):
        # 对输入的j列数据进行分析，j==0时，数据为深度列
        for j in range(data_normal_curve.shape[1]):
            if j == 0:
                continue
            if data_box[i] != []:
                data_temp = np.array(data_box[i])
                a = np.mean(data_temp[:, j])
                c = np.var(data_temp[:, j], ddof=1)
                d = np.sqrt(np.var(data_temp[:, j]))
                # print(a, c, d)
                data_statistical.append([a, c, d])
            # 如果盒子为空，代表没有此类数据
            else:
                data_statistical.append([9999, 99999, 99999])
    data_statistical_3D = np.array(data_statistical).reshape((num_cluster+1, data_normal_curve.shape[1]-1, 3))
    data_statistical = np.array(data_statistical)
    # print(data_statistical)

    return data_statistical, data_statistical_3D