import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def combine_class_inf_to_table(result_class):
    process_index = [0]
    for i in range(result_class.shape[0]-1):
        if result_class[i+1, -1] != result_class[i, -1]:
            process_index.append(i)

    processed_target = []
    for i in range(len(process_index)-1):
        item = [result_class[process_index[i], 0], result_class[process_index[i+1], 0], result_class[process_index[i], -1]]
        processed_target.append(item)

    return np.array(processed_target)

def DisScatter(KmeansPred, X_Input, ClusterNum = 4, pltTitle = ''):

    draw_pot = []
    KmeansPred_plot = []
    for i in range(X_Input.shape[0]):
        if i%10 == 0:
            draw_pot.append(X_Input[i, :])
            KmeansPred_plot.append(KmeansPred[i])

    draw_pot = np.array(draw_pot)
    KmeansPred_plot = np.array(KmeansPred_plot)


    from matplotlib import pyplot as plt
    X_Scatter = PCA(n_components=2).fit(draw_pot).transform(draw_pot)
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
    for i in range(KmeansPred_plot.shape[0]):
        for j in range(ClassNum):
            if KmeansPred[i] == j:
                pltPic[j] = plt.scatter(X_Scatter[i, 0], X_Scatter[i, 1], c=ColorList[j], marker=MarkerList[j], s=10)
    plt.legend(pltPic[:ClusterNum], LableName[:ClusterNum])
    plt.title(pltTitle)
    plt.savefig('D:/Data/pic/{}.png'.format(pltTitle))
    plt.show()

def feature_cluster():
    # feature_dyna = np.loadtxt('stage1_dyna_feature.txt', delimiter='\t', encoding='GBK')
    # feature_stat = np.loadtxt('stage1_stat_feature.txt', delimiter='\t', encoding='GBK')
    feature_all = np.loadtxt('stage1_all_feature.txt', delimiter='\t', encoding='GBK')
    depth_info = np.loadtxt('stage1_depth_info.txt', delimiter='\t', encoding='GBK')
    # well_name_info = np.loadtxt('stage1_well_name.txt', delimiter='\t', encoding='utf-8')
    # print(feature_dyna.shape)
    f = open("stage1_well_name.txt", 'r', encoding='GBK')
    well_name_info = f.readlines()
    well_name_info = np.array(well_name_info)
    f.close()
    # print(type(well_name_info))
    print(well_name_info.shape)

    # feature_all = np.hstack((feature_dyna, feature_stat))
    # print(feature_all.shape)

    pca_n_components = 0.98
    pca = PCA(n_components=pca_n_components)
    print('Starting PCA........')
    pca.fit(feature_all)
    X_Input = pca.transform(feature_all)
    print('PCA_Answer: Input X’Shape：{}，PCA Size:{}'.format(X_Input.shape, pca_n_components))

    n_clusters = 10
    print('Starting KMeans.........')
    # 声明模型
    model = KMeans(n_clusters=n_clusters)
    # 拟合模型
    model.fit(X_Input)
    # 用全部数据预测
    all_predictions = model.predict(X_Input)

    DisScatter(all_predictions, X_Input, ClusterNum=6, pltTitle='')

    depth_new = depth_info
    # depth_new[:, 0] = (depth_new[:, 0] + depth_new[:, 1])/2
    depth_new[:, 1] = all_predictions
    np.savetxt('all_pred.txt', depth_new, delimiter='\t', comments='', fmt='%.4f')

    del X_Input
    del feature_all
    # del feature_dyna
    # del feature_stat

    print(all_predictions.shape)

    charter = well_name_info[0].split('\n')[0]
    index_start = 0
    index_end = 0

    for i in range(depth_info.shape[0]):
        charter_temp = well_name_info[i].split('\n')[0]
        if (charter_temp == charter) & (i < depth_info.shape[0]-1):
            continue
        else:
            index_end = i
            print(index_end)

        # print(charter, (charter_temp==charter), charter_temp)
        if index_end == 0:
            print('error.....')
            exit(0)


        # print(index_start, index_end, all_predictions[index_start:index_end].shape)
        depth_info_temp = depth_info[index_start:index_end, :]
        # depth_info_temp[:, 0] = (depth_info_temp[:, 0]+depth_info_temp[:, 1])/2
        depth_info_temp[:, 1] = all_predictions[index_start:index_end]
        print(depth_info_temp.shape)
        # print(depth_info_temp[:5, :])

        # np.savetxt('{}_answer_split_2_{}.txt'.format(charter.split('\n')[0], n_clusters), depth_info_temp, delimiter='\t', comments='', fmt='%.4f')
        # target = combine_class_inf_to_table(depth_info_temp)
        # print(target)
        #
        # np.savetxt('{}_answer_2_{}.txt'.format(charter.split('\n')[0], n_clusters), target, delimiter='\t', comments='', fmt='%.4f')

        index_start = index_end
        charter = well_name_info[index_start].split('\n')[0]


if __name__ == '__main__':
    feature_cluster()