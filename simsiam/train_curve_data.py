import argparse
import numpy as np
from keras import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from simsiam.ELE_data import Ele_data_normal
from src_ele.dir_operation import traverseFolder_folder

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# metavar - 在 usage 说明中的参数名称，对于必选参数默认就是参数名称，对于可选参数默认是全大写的参数名称.


parser.add_argument('--DIR_in', metavar='DIR_FULL_IMAGE_IN', default=r'D:\Data\target5', type=str,
                    help='path to input full ele dyna or state dataset')

parser.add_argument('--windows_length', default=10, type=float, help='how long every windows is')
parser.add_argument('--target_folder_charter', default=['LG7-4', 'LG7-8', 'LG7-12', 'LG7-16',
                                                        'LG701', 'LG701-H1', 'LN11-4', 'LN11-5'],
                    type=list, help='which folder is going to process')
args = parser.parse_args()
print(args)


def start_normal_curve_classfy():
    # 遍历访问 输入文件夹下的所有文件夹
    folder_list = traverseFolder_folder(args.DIR_in)

    # 训练集文件生成
    # 对所有文件夹进行判断，判断是否是 目标文件夹
    target_folder = []
    for i in range(len(folder_list)):
        for j in range(len(args.target_folder_charter)):
            if folder_list[i].__contains__(args.target_folder_charter[j]):
                target_folder.append(folder_list[i])
                break
    print(target_folder, len(target_folder))

    # 把每一个文件夹，实例化成一个 dataset 对象, 并进行分割
    dataset_ele_normal = np.array([])
    for i in range(len(target_folder)):
        print('loading pic from path:{}'.format(target_folder[i].split('/')[-1]))
        a = Ele_data_normal(target_folder[i], VIEW_LENGTH=args.windows_length)
        # print(a.get_layer_class_and_curve_info_by_dep()[:10, :])
        if i == 0:
            dataset_ele_normal = a.get_layer_class_and_curve_info_by_dep()
        else:
            dataset_ele_normal = np.vstack((dataset_ele_normal, a.get_layer_class_and_curve_info_by_dep()))
    # print(dataset_ele_normal.shape)
    # print(dataset_ele_normal[:10, :])


    # 全连接神经网络
    model = Sequential()
    # input = X.shape[1]
    # 隐藏层128
    model.add(Dense(128, activation='relu', input_shape=(12,)))
    # Dropout层用于防止过拟合
    model.add(Dropout(0.2))
    # 隐藏层128
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    # 没有激活函数用于输出层，因为这是一个回归问题，我们希望直接预测数值，而不需要采用激活函数进行变换。
    model.add(Dense(1))
    # 使用高效的 ADAM 优化算法以及优化的最小均方误差损失函数
    model.compile(loss='mean_squared_error', optimizer='adam')
    # early stoppping
    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)

    X_train, X_test, y_train, y_test = train_test_split(dataset_ele_normal[:, 1:-1], dataset_ele_normal[:, -1:], test_size=0.2, random_state=100)
    print(X_train.shape)

    # 训练
    #  X_train, X_test, y_train, y_test
    history = model.fit(X_train, y_train, epochs=100, batch_size=20, validation_data=(X_test, y_test), verbose=2,
                        shuffle=False, callbacks=[early_stopping])

    # 预测
    y_pred = model.predict(X_train)
    print(mean_absolute_error(y_pred, y_train))
    print(mean_squared_error(y_pred, y_train))
    print(r2_score(y_pred, y_train))

if __name__ == '__main__':
    start_normal_curve_classfy()