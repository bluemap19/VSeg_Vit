
import numpy as np



# 计算 一维预测数据 的准确率，包括 二分类、多分类
def get_charter_acc(y_pred, labels, binary_use=False, num_multi_pred=7):
    y_pred_n = np.around(y_pred.ravel()).astype(int)
    labels_n = np.around(labels.ravel()).astype(int)
    # print(y_pred_n.shape, labels_n.shape)

    if binary_use:
        if len(y_pred) != len(labels):
            print('labels shape is not equal to pred shape:{},{}'.format(y_pred.shape, labels.shape))
            exit(0)

        TP_N, FN_N, FP_N, TN_N, ALL_N, TL_N, FL_N = 0, 0, 0, 0, 0, 0, 0


        for i in range(len(y_pred_n)):
            y_pred_t = 0
            label_t = 0
            # 二值化 让预测结果，标签结果 只有0与1
            if y_pred_n[i] > 0.5:
                y_pred_t = 1
            else:
                y_pred_t = 0

            if labels_n[i] > 0.5:
                label_t = 1
            else:
                label_t = 0

            # 计算TP、FN、FP、TN，ALL的数量
            ALL_N += 1
            if y_pred_t == label_t:
                # 正样本数量
                if label_t == 1:
                    TP_N += 1
                    TL_N += 1
                else:
                    TN_N += 1
                    FL_N += 1
            else:
                # 负样本数量
                if label_t == 1:
                    FN_N += 1
                    TL_N += 1
                else:
                    FP_N += 1
                    FL_N += 1

        return TP_N, TN_N, FN_N, FP_N, TL_N, FL_N, ALL_N
    else:
        labels_N = []
        labels_corr_N = []
        for i in range(num_multi_pred):
            labels_N.append(0)
            labels_corr_N.append(0)

        for i in range(labels.shape[0]):
            class_index = labels[i]
            labels_N[class_index] += 1

            if labels[i] == y_pred[i]:
                labels_corr_N[class_index] += 1

        count_labels = np.array(labels_N)
        count_labels_corr = np.array(labels_corr_N)

        return count_labels_corr, count_labels



# 计算 多维预测数据 的准确率，包括 二分类、多分类
def get_multi_charter_acc(y_pred, labels, binary_use=False, num_multi_pred=7):
    # TP_N_A, TN_N_A, FN_N_A, FP_N_A, TL_N_A, FL_N_A, ALL_N_A = 0, 0, 0, 0, 0, 0, 0
    ACC_FL = np.zeros((labels.shape[1], 7))

    for j in range(labels.shape[1]):
        TP_N, TN_N, FN_N, FP_N, TL_N, FL_N, ALL_N = get_charter_acc(y_pred[:, j], labels[:, j], binary_use=binary_use, num_multi_pred=num_multi_pred)
        flag_t = get_charter_acc(y_pred[:, j], labels[:, j], binary_use=True)

        # TP_N_A += TP_N
        # TN_N_A += TN_N
        # FN_N_A += FN_N
        # FP_N_A += FP_N
        # TL_N_A += TL_N
        # FL_N_A += FL_N
        # ALL_N_A += ALL_N
        ACC_FL[j, :] += np.array(flag_t)

    # ACC = (TP_N_A + TN_N_A) / ALL_N_A
    return ACC_FL


# y_pred_rand = np.random.randn(100, 2)
# labels_rand = np.around(np.random.randn(100, 2))
# labels_rand -= int(np.min(labels_rand))
#
# print(y_pred_rand[:10, 0])
# print(np.around(y_pred_rand[:10, 0]))
# print(labels_rand[:10, 0])
# y_pred_rand = np.array([1,2,3,4,5,6,7,6,5,4,3,2,1])
# labels_rand = np.array([1,2,3,4,3,2,1,2,3,4,3,2,1])
# print(get_charter_acc(y_pred_rand, labels_rand, binary_use=False, num_multi_pred=5))

#

def single_pic_seg_acc_cal(pic_pred=np.array([]), pic_org=np.array([])):
    if (len(pic_pred.shape) != 2) | (len(pic_pred.shape) != 2):
        print('single pic seg acc cal shape is error:{},{}'.format(pic_pred.shape, pic_org.shape))
        exit(0)
        pass
    if (pic_pred.shape[0]!=pic_org.shape[0]) | (pic_pred.shape[0]!=pic_org.shape[0]):
        print('pic shape error:{}*{}'.format(pic_pred.shape, pic_org.shape))
        exit(0)

    # pic_pred = np.around(pic_pred).astype(int)
    # pic_org = pic_org.astype(int)


    TP_N, FN_N, FP_N, TN_N, ALL_N, TL_N, FL_N = 0, 0, 0, 0, 0, 0, 0
    for i in range(pic_pred.shape[0]):
        for j in range(pic_pred.shape[1]):
            if pic_pred[i][j] > 0.5:
                pic_pred_t = 1
            else:
                pic_pred_t = 0

            if pic_org[i][j] > 0.5:
                pic_org_t = 1
            else:
                pic_org_t = 0

            # print('pred v:{},{}, org v:{},{}'.format(pic_pred[i][j], pic_pred_t, pic_org[i][j], pic_org_t))
            # 计算TP、FN、FP、TN，ALL的数量
            ALL_N += 1
            if pic_pred_t == pic_org_t:
                # 正样本数量
                if pic_org_t > 0.5:
                    TP_N += 1
                    TL_N += 1
                else:
                    TN_N += 1
                    FL_N += 1
            else:
                # 负样本数量
                if pic_org_t > 0.5:
                    FN_N += 1
                    TL_N += 1
                else:
                    FP_N += 1
                    FL_N += 1

    # correct_prediction = np.mean(np.equal(pic_pred, pic_org))
    correct_prediction = (TP_N + TN_N) / ALL_N
    MIOU = TP_N / (FN_N + FP_N + TP_N)
    F1 = 2*TP_N / (FN_N + FP_N + 2*TP_N)
    # print(TP_N, FN_N, FP_N, TN_N, ALL_N, TL_N, FL_N)
    return correct_prediction, MIOU, F1

# pic_pred = np.array([[1, 2, 3], [0, 1, 2], [-1, 0, 1]])
# pic_org = np.array([[0, 1, 1], [1, 1, 1], [1, 2, 0]])
# print(single_pic_seg_acc_cal(pic_pred, pic_org))


def multi_pic_seg_acc_cal(pic_pred=np.array([]), pic_org=np.array([])):
    if (len(pic_pred.shape) != 3) | (len(pic_pred.shape) != 3):
        print('single pic seg acc cal shape is error:{},{}'.format(pic_pred.shape, pic_org.shape))
        exit(0)

    acc_list = []
    miou_list = []
    F1_list = []
    for i in range(pic_pred.shape[0]):
        acc_t, miou_t, F1 = single_pic_seg_acc_cal(pic_pred[i,:,:], pic_org[i,:,:])
        acc_list.append(acc_t)
        miou_list.append(miou_t)
        F1_list.append(F1)

    acc_list = np.array(acc_list)
    miou_list = np.array(miou_list)
    F1_list = np.array(F1_list)
    return acc_list, miou_list, F1_list


# pic_pred_0 = np.array([[1, 2, 3], [0, 1, 2], [-1, 0, 1]])
# pic_org = np.array([[0, 1, 1], [1, 1, 1], [1, 2, 0]])
# pic_pred = np.array([pic_pred_0, pic_pred_0, pic_pred_0])
# pic_org = np.array([pic_org, pic_org, pic_pred_0])
# print(multi_pic_seg_acc_cal(pic_pred, pic_org))