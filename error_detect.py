import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


class k_means():    
    def k_choose(self ,data_list):
        SSE = []
        for k in range(1,9):
            k_tmp = KMeans(n_clusters=k)
            k_tmp.fit(np.array(data_list).reshape(-1, 1))
            SSE.append(k_tmp.inertia_) 
        return SSE
    

    def k_means_train(self, k, data_list):
        label_cnt = []
        error_index = []
        K_means_ = KMeans(n_clusters=k)
        K_means_.fit(np.array(data_list).reshape(-1, 1))
        # print(K_means_.labels_)
        data_labels = list(K_means_.labels_)
        # for i in range(len(data_labels)):
        #     if data_labels[i] == 1:
        #         print(i)
        for i in range(k):
            label_cnt.append(0)
        for i in range(len(data_labels)):
            label_cnt[data_labels[i]] = label_cnt[data_labels[i]] + 1
        # print(label_cnt)
        error_label = label_cnt.index(min(label_cnt))
        # print(error_label)
        for i in range(len(data_labels)):
            if data_labels[i] == error_label: error_index.append(i)
        return error_index


class three_sigma():
    def __init__(self):
        pass

    def three_sigma_(self, SLdata_list):
        print(len(SLdata_list))
        if np.std(SLdata_list) > np.mean(SLdata_list)/10:
            error_index = []
            for i in range(len(SLdata_list)):
                if abs(SLdata_list[i] - np.mean(SLdata_list)) > 1.5 * np.std(SLdata_list):
                    error_index.append(i)
        else:
            error_index = []
            for i in range(len(SLdata_list)):
                if abs(SLdata_list[i] - np.mean(SLdata_list)) > 3 * np.std(SLdata_list):
                    error_index.append(i)

        return error_index


if __name__ == "__main__":
    pass
