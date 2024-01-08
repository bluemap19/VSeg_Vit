import argparse
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import torch
import torchvision.models as models
from tqdm import trange
from simsiam.sim_model import model_stage1
from simsiam.ELE_data import Ele_data_img
from simsiam.sim_model.model_stage_1_vit import vit_simsiam
from src_ele.dir_operation import traverseFolder_folder
from sklearn.cluster import AgglomerativeClustering

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--DIR_in', metavar='DIR', default=r'D:\Data\target5', type=str,
                    help='path to dataset')
# parser.add_argument('--Model_stage1_path', metavar='modle_path', default=r'D:\Data\target_answer\256-dyna-stat\checkpoint_res50_240_0100.pth.tar',
parser.add_argument('--Model_stage1_path', metavar='modle_path', default=r'D:\Data\target_answer\models\checkpoint_res50_256_0028.pth.tar',
# parser.add_argument('--Model_stage1_path', metavar='modle_path', default=r'D:\Data\target_answer\256-dyna-stat\checkpoint_vit_240_0051.pth.tar',
                    type=str, help='path to dataset')
parser.add_argument('--stat_use', metavar='STAT_USE', default=True, type=bool,
                    help='whether use stat img')
########################################################################################
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
###############################################################################
args = parser.parse_args()

# 该函数主要进行无监督聚类   输入数据的收集
# 输入数据来自于stage1模型的输出
def stage1_info_coll(i, charter):
    # ############### 如果特征已经被保存，直接读取既可以
    # list = os.listdir()
    # print(list)
    # for i in list:
    #     if i.__contains__(charter) & i.__contains__('stage1_feature'):
    #         a = np.loadtxt(i, skiprows=0, encoding='GBK')
    #         print(a.shape)
    #         depth_info, all_feature_stage1 = a[:, :2], a[:, 2:]
    #         return depth_info, all_feature_stage1

    model_temp = model_stage1.SimSiam()
    # model_temp = vit_simsiam()

    if os.path.isfile(args.Model_stage1_path):
        print("=> loading checkpoint '{}'".format(args.Model_stage1_path))
        DEVICE = torch.device("cpu")
        model_temp.to(DEVICE)
        checkpoint = torch.load(args.Model_stage1_path, map_location=DEVICE)
        model_temp.load_state_dict(checkpoint['model_dict'])    # model_dict

    all_feature_stage1 = []
    # all_stat_feature_stage1 = []
    depth_info = []
    # well_name_info = []


    model_temp.eval()
    # for i in path_list:
    # print(i)
    ELE_Temp = Ele_data_img(i)
    dep_start = ELE_Temp.dep_start
    dep_end = ELE_Temp.dep_end
    step = 0.3
    windows_length = 1.5
    pic_num = int((dep_end-dep_start-windows_length)/step+1)
    for j in trange(pic_num):
        depth_temp = dep_start + (j+1)*step
        pic_dyna_windows, pic_static_windows, depth_data_windows = ELE_Temp.get_pic_from_depth(depth_temp)
        # pic_dyna_windows = torch.reshape(torch.from_numpy(pic_dyna_windows).float(), (1, 1, 224, 224))
        # pic_static_windows = torch.reshape(torch.from_numpy(pic_static_windows).float(), (1, 1, 224, 224))

        pic_dyna = pic_dyna_windows.reshape((1, 224, 224))
        pic_stat = pic_static_windows.reshape((1, 224, 224))
        pic_all = np.append(pic_dyna, pic_stat, axis=0)

        pic_all = torch.reshape(torch.from_numpy(pic_all).float(), (1, 2, 224, 224))
        r_temp = model_temp.encoder(pic_all)
        # r_temp = model_temp.arch_model(pic_all)

        # print(r_temp['z1'].cpu().detach().numpy().shape)
        # print(pic_all.shape, r_temp.shape)

        all_feature_stage1.append(r_temp.cpu().detach().numpy().ravel())

        depth_info.append([depth_data_windows[0, 0], depth_data_windows[-1, 0]])
        # well_name_info.append(i.split('/')[-1].split('\\')[-1])

    all_feature_stage1 = np.array(all_feature_stage1)
    # all_stat_feature_stage1 = np.array(all_stat_feature_stage1)
    depth_info = np.array(depth_info)

    # 好不容易 计算出了特征，保存一下
    np.savetxt('stage1_feature_res50_{}.txt'.format(charter), np.hstack((depth_info, all_feature_stage1)),
                       delimiter='\t', comments='', fmt='%.4f')

    # np.savetxt('stage1_all_feature.txt', all_feature_stage1, delimiter='\t', comments='', fmt='%.4f')
    # # np.savetxt('stage1_stat_feature.txt', all_stat_feature_stage1, delimiter='\t', comments='', fmt='%.4f')
    # np.savetxt('stage1_depth_info.txt', depth_info, delimiter='\t', comments='', fmt='%.4f')
    # np.savetxt('stage1_well_name.txt', well_name_info, delimiter='\t', comments='', fmt='%s')

    # print(model_temp)
    return depth_info, all_feature_stage1


def DisScatter(KmeansPred, X_Input, ClusterNum = 4, pltTitle = ''):
    from matplotlib import pyplot as plt
    X_Scatter = PCA(n_components=2).fit(X_Input).transform(X_Input)
    print('Drawing Scatter..........')
    ClassNum = np.max(KmeansPred)+1
    ColorList = ['b', 'g', 'k', 'm', 'r', 'w', 'grey', 'y', 'c', 'plum', 'slategray']
    MarkerList = [',', '.', '1', '2', '3', '4', '8', 's', 'o', 'v', '+', 'x', '^', '<', '>', 'p']
    LableName = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6',
                 'Cluster 7', 'Cluster 8', 'Cluster 9', 'Cluster 10', 'Cluster 11', 'Noise']
    # 基于类绘制图形
    pltPic = [plt.scatter(0, 0), plt.scatter(0, 0), plt.scatter(0, 0),
              plt.scatter(0, 0), plt.scatter(0, 0), plt.scatter(0, 0),
              plt.scatter(0, 0), plt.scatter(0, 0), plt.scatter(0, 0),
              plt.scatter(0, 0), plt.scatter(0, 0), plt.scatter(0, 0)]
    for i in range(KmeansPred.shape[0]):
        for j in range(ClassNum):
            if KmeansPred[i] == j:
                pltPic[j] = plt.scatter(X_Scatter[i, 0], X_Scatter[i, 1], c=ColorList[j], marker=MarkerList[j], s=10)
    plt.legend(pltPic[:ClusterNum], LableName[:ClusterNum])
    plt.title(pltTitle)
    plt.savefig('D:/Data/pic/{}.png'.format(pltTitle))
    plt.show()


def combine_class_inf_to_table(result_class):
    process_index = [0]
    for i in range(result_class.shape[0]-1):
        if result_class[i+1, -1] != result_class[i, -1]:
            process_index.append(i)

    # print(process_index)
    # exit(0)
    processed_target = []
    for i in range(len(process_index)-1):
        item = [result_class[process_index[i], 0], result_class[process_index[i+1], 0], result_class[process_index[i], -1]]

        processed_target.append(item)



    # processed_target = []
    # for i in range(result_class.shape[0]-1):
    #     if jump_num != 1:
    #         jump_num -= 1
    #         continue
    #     else:
    #         start_dep = result_class[i][0]
    #         end_dep = result_class[i+1][0]
    #         inf_type = int(result_class[i][-1])
    #         while (i+jump_num < result_class.shape[0]-1):
    #             inf_type_end = int(result_class[i+jump_num][-1])
    #             if inf_type_end == inf_type:
    #                 jump_num += 1
    #                 continue
    #             else:
    #                 break
    #             end_dep = result_class[i+jump_num][0]
    #
    #         item = np.array([start_dep, end_dep, inf_type])
    #         processed_target.append(item)
    return np.array(processed_target)


def feature_cluster(depth_info, feature_all, charter):
    pca_n_components = 0.98
    pca = PCA(n_components=pca_n_components)
    print('Starting PCA........')
    pca.fit(feature_all)
    X_Input = pca.transform(feature_all)
    print('PCA_Answer: Input X’Shape：{}，PCA Size:{}'.format(X_Input.shape, pca_n_components))

    n_clusters = 6
    # Fit the model:
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(X_Input)
    all_predictions = clustering.labels_
    # n_clusters = clustering.n_clusters_
    print('all_predictions:{}'.format(all_predictions))

    # print('Starting KMeans.........')
    # # 声明模型
    # model = KMeans(n_clusters=n_clusters)
    # # 拟合模型
    # model.fit(X_Input)
    # # 用全部数据预测
    # all_predictions = model.predict(X_Input)

    DisScatter(all_predictions, X_Input, ClusterNum=n_clusters, pltTitle=charter)

    target_split = depth_info
    # depth_new[:, 0] = (depth_new[:, 0] + depth_new[:, 1])/2
    target_split[:, 1] = all_predictions
    # np.savetxt('all_pred.txt', depth_new, delimiter='\t', comments='', fmt='%.4f')

    del X_Input
    del feature_all

    # print(all_predictions.shape)

    # np.savetxt('{}_answer_split_{}.txt'.format(charter.split('\n')[0], n_clusters), depth_info_temp, delimiter='\t', comments='', fmt='%.4f')
    target = combine_class_inf_to_table(target_split)
    print(target)

    return target, target_split

    # np.savetxt('{}_answer_{}.txt'.format(charter.split('\n')[0], n_clusters), target, delimiter='\t', comments='', fmt='%.4f')

    # index_start = index_end
    # charter = well_name_info[index_start].split('\n')[0]


if __name__ == '__main__':
    path_list = traverseFolder_folder(args.DIR_in)
    for i in (path_list):
        charter = i.split('/')[-1].split('\\')[-1]

        if i.__contains__('LG7-12'):
            print(i)
            depth_info, all_feature_stage1 = stage1_info_coll(i, charter)
        else:
            continue

        print(depth_info.shape, all_feature_stage1.shape)
        target, target_split = feature_cluster(depth_info, all_feature_stage1, charter)
        np.savetxt('{}_answer_{}.txt'.format(charter.split('\n')[0], 10), target, delimiter='\t', comments='',
                   fmt='%.4f')
