import numpy as np
from sklearn.decomposition import PCA
import pickle

def get_pca_model_ele_info(path = r'D:\GitHubProj\dino\ele_unsup_cluster\stage1_all_feature_512_res50.txt',dim=6):
    # feature_all = np.loadtxt('stage1_all_feature.txt', delimiter='\t', encoding='GBK')
    feature_all = np.loadtxt(path, delimiter='\t', encoding='GBK')
    # feature_all = np.loadtxt(r'D:\GitHubProj\dino\ele_unsup_cluster\stage1_all_feature_20_res50.txt', delimiter='\t', encoding='GBK')

    pca_n_components = dim
    pca = PCA(n_components=pca_n_components)
    print('Starting PCA........')
    pca.fit(feature_all)
    X_Input = pca.transform(feature_all)
    print('PCA_Answer: Input X’Shape：{}，PCA Size:{}-->{}'.format(feature_all.shape, pca_n_components, X_Input.shape))

    # pickle.dumps(pca)

    return pca
# get_pca_model_ele_info()