import numpy as np
# from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn import svm
from math import ceil, floor
from outlier_dict import outlier_dict
import datetime
from scipy.interpolate import interp1d



def detect_st_threshold(SLdata_list, sensor_id):
    error_index_list = []
    upper_limit = limit_dict[sensor_id][1]
    lower_limit = limit_dict[sensor_id][0]
    for i in range(len(SLdata_list)):
        if SLdata_list[i] > upper_limit or SLdata_list[i] < lower_limit:
            error_index_list.append(i)
    return error_index_list


def fill_point_B_spline(SLdata_list, error_index_list):
    new_list = []
    num_list, SLdata_list_new = throw_outlier(SLdata_list, error_index_list)
    fx = interp1d(num_list, SLdata_list_new, kind='cubic', fill_value="extrapolate") 
    # print(num_list)
    # xInterp = np.linspace(0,len(SLdata_list) - 1,(len(SLdata_list) - 1) * 100)
    # yInterp = fx(xInterp)
    # print(num_list[-1])
    # print(num_list)
    for i in range(len(SLdata_list)):
        if i in error_index_list:
            if i >= num_list[0] and i <= num_list[-1]:
                # print(i, fx(i))
                new_list.append(fx(i))
            else:
                new_list.append(3186)
        else:
            new_list.append(SLdata_list[i])
    return new_list



def throw_outlier(SLdata_list, error_index_list):
    new_SLdata_list = []
    num_LIST = []
    for i in range(len(SLdata_list)):
        if i in error_index_list:   pass
        else:   
            num_LIST.append(i)
            new_SLdata_list.append(SLdata_list[i])
    return num_LIST, new_SLdata_list


def detect_sb_2order(SLdata_list, num):
    error_index_list = []
    for i in range(num):
        diff_l = peak_find(SLdata_list, i * 100 - 1, 0)
        diff_r = peak_find(SLdata_list, i * 100 - 1, 1)
        if diff_l != None and diff_r != None:
            if diff_l * diff_r < 0 and abs(diff_r * diff_r) > 400:
                error_index_list.append(i * 100 - 1)
    return error_index_list


def peak_find(SLdata_list, point, direction):
    close_point = -1 if direction == 0 else 1
    cnt = -2 if direction == 0 else 2
    if point + close_point <= len(SLdata_list) - 1 and point + close_point >= 0:
        diff = SLdata_list[point + close_point] - SLdata_list[point]
        while(1):
            if point + cnt <= len(SLdata_list) - 1 and point + cnt >= 0:
                temp = SLdata_list[point + cnt] - SLdata_list[point + cnt - close_point]
                cnt += close_point
                if diff * temp > 0:
                    diff += temp
                else:
                    break
            else:
                break
        return diff
    return None


def detect_2n(SLdata_list):
    error_index = []
    for i in range(len(SLdata_list)):
        if i == 0 or i == len(SLdata_list) - 1: pass
        else:
            if SLdata_list[i + 1] < -7 and SLdata_list[i - 1] > 7: 
                error_index.append(i)
            if SLdata_list[i - 1] > 7 and SLdata_list[i - 1] < -7:
                error_index.append(i)
    return error_index


def one_order(SLdata_list):
    new_list = []
    for i in range(len(SLdata_list) - 3):
        # new_list.append(SLdata_list[i + 1] - SLdata_list[i])
        s1 = SLdata_list[i + 1] - SLdata_list[i]
        s2 = SLdata_list[i + 2] - SLdata_list[i]
        s3 = SLdata_list[i + 3] - SLdata_list[i]
        if s1 > 0 and s2 - s1 > 0 and s3 - s2 > 0:   new_list.append(max(s1, s2, s3))
        elif s1 < 0 and s2 - s1 < 0 and s3 - s2< 0: new_list.append(min(s1, s2, s3))
        else: new_list.append(s1)

    return new_list


# def one_order_2s(SLdata_list):
#     new_list = []
#     for i in range(len(SLdata_list) - 2):
#         new_list.append(SLdata_list[i + 2] - SLdata_list[i])
#     return new_list


def twice_order(SLdata_list):
    new_list = one_order(SLdata_list)
    newnew_list = []
    newnew_list = one_order(new_list)
    return newnew_list


class detect_error(object):
    def get_accuracy(self, SLdata_list):
        pass

    def get_false_alarm(self, SLdata_list, SLMdate_list, error_index_list, month, senor_id):
        false_alarm_cnt = 0
        true_cnt = 0
        new_outlier_list = []
        # new_SLdata_list = []
        for data in outlier_dict[month][senor_id]["SLdata"]:
            new_outlier_list.append(int(data))
        for error_index in error_index_list:
            if int(SLdata_list[error_index]) in new_outlier_list:
               dict_index = new_outlier_list.index(int(SLdata_list[error_index]))
               
               if SLMdate_list[error_index].month == outlier_dict[month][senor_id]["Mdate"][dict_index].month and\
                  SLMdate_list[error_index].day == outlier_dict[month][senor_id]["Mdate"][dict_index].day:
                    true_cnt += 1
                    # print(SLdata_list[error_index], SLMdate_list[error_index])
            else:
                false_alarm_cnt += 1
        print(outlier_dict[month][senor_id]["SLdata"], outlier_dict[month][senor_id]["Mdate"])
        # return false_alarm_cnt/len(outlier_dict[month][senor_id]["SLdata"])
        return true_cnt, false_alarm_cnt, len(outlier_dict[month][senor_id]["SLdata"]) - true_cnt



def fill_point(data_list, error_index_list):
    new_list = []
    for data in data_list:
        if data_list.index(data) in error_index_list:
            new_list.append(4605)
            if data_list.index(data) == 0:
                new_list.append(data_list[1])
            elif data_list.index(data) == len(data_list) - 1:
                new_list.append(data_list[len(data_list) - 2])
            elif data_list.index(data) - 1 and data_list.index(data) + 1 not in error_index_list:
                new_list.append((data_list[data_list.index(data) - 1] + data_list[data_list.index(data) + 1])/2)
            else:
                # new_list.append(4500)
                cnt = 1
                while(1):
                    if data_list.index(data) - 1 - cnt not in error_index_list:
                        new_list.append((data_list[data_list.index(data) - 1 - cnt]))
                        break
                    else:
                        cnt += 1
        else:
            new_list.append(data)
    return new_list

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
        # print(data_labels)
        # for i in range(len(data_labels)):
        #     if data_labels[i] == 1:
        #         print(i)
        for i in range(k):
            label_cnt.append(0)
        for i in range(len(data_labels)):
            label_cnt[data_labels[i]] = label_cnt[data_labels[i]] + 1
        error_label = label_cnt.index(min(label_cnt))
        if label_cnt[error_label] > label_cnt[1 - error_label] * 0.2:
            return []
        else:
            # print(label_cnt)
            error_label = label_cnt.index(min(label_cnt))
            # print(error_label)
            for i in range(len(data_labels)):
                if data_labels[i] == error_label: error_index.append(i)
            return error_index

limit_dict = {'SLS01': [5600, 6000], 'SLS02': [3450, 3700], 'SLS03': [3000, 3400], 'SLS04': [2850, 3300], 'SLS05': [2750, 3000], 'SLS06': [2800, 3200], 'SLS07': [3400, 3800], 'SLS08': [5600, 6000], 'SLS09': [6930, 6930], 'SLS10': [4100, 4400], 'SLS11': [3650, 3900], 'SLS12': [3350, 3650], 'SLS13': [3350, 3650], 'SLS14': [3500, 3900], 'SLS15': [4000, 4400], 'SLS16': [6500, 7000], 'SLS17': [6300, 6700], 'SLS18': [3650, 3950], 'SLS19': [3200, 3500], 'SLS20': [3100, 3500], 'SLS21': [3250, 3550], 'SLS22': [3300, 3600], 'SLS23': [3850, 4100], 'SLS24': [6050, 6500], 'SLX01': [5600, 6000], 'SLX02': [3400, 3700], 'SLX03': [3000, 3400], 'SLX04': [2900, 3300], 'SLX05': [2750, 3050], 'SLX06': [2850, 3150], 'SLX07': [3400, 3800], 'SLX08': [5600, 6000], 'SLX09': [6700, 7200], 'SLX10': [4100, 4400], 'SLX11': [3600, 3900], 'SLX12': [3300, 3700], 'SLX13': [3300, 3700], 'SLX14': [3500, 3900], 'SLX15': [4000, 4400], 'SLX16': [6500, 7000], 'SLX17': [6300, 6700], 'SLX18': [3650, 3900], 'SLX19': [3200, 3500], 'SLX20': [3200, 3450], 'SLX21': [3250, 3550], 'SLX22': [3300, 3600], 'SLX23': [3800, 4100], 'SLX24': [6100, 6600]}

def set_scale(func):
    def inner(*args, **kwargs):
        upper_limit = limit_dict[args[4]][1]
        lower_limit = limit_dict[args[4]][0]
        error_index_list = func(*args, **kwargs)
        new_error = []
        # print("测试前：")
        # for error_index in error_index_list:
            # print(args[1][error_index])
        for error_index in error_index_list:
            # print("将要测试")
            # print(args[1][error_index])
            if args[1][error_index] >= lower_limit and \
               args[1][error_index] <= upper_limit:
                # print(args[1][error_index])
                pass
            else: new_error.append(error_index)
        return new_error
    return inner


class three_sigma(object):
    def __init__(self):
        pass

    @set_scale
    def three_sigma_(self, SLdata_list, SLdata_mean, SLdata_std, sensor_id):
        # print(len(SLdata_list))
        error_index = []
        if SLdata_std > SLdata_mean/10:
            for i in range(len(SLdata_list)):
                if abs(SLdata_list[i] - SLdata_mean) > 1.5 * SLdata_std:
                    error_index.append(i)
        else:
            for i in range(len(SLdata_list)):
                if abs(SLdata_list[i] - SLdata_mean) > 3 * SLdata_std:
                    error_index.append(i)

        return error_index
    

    

def return_index(func):
    def wrapper(*args, **kwargs):
        label_ = func(*args, **kwargs)
        error_index = []
        for i in range(len(label_)):
            if label_[i] == -1 : error_index.append(i)
        return error_index
    return wrapper


def return_lrweights(func):
    def wrapper(self, number):
        close_num = func(self, number)
        rn = close_num if close_num > number else close_num + 1
        rw = 1 - rn + number
        lw = 1 - rw
        return lw, rw
    return wrapper


class iostation_forest():
    @return_index
    def ioslation_forest_train(self, SLdata_list):
        isof = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(0.05),max_features=1.0)
        isof.fit(np.array(SLdata_list).reshape(-1, 1))
        SLdata_label = isof.predict(np.array(SLdata_list).reshape(-1, 1))
        return SLdata_label


class box_plot():
    def box_plot_train(self, data_list):
        error_index_list = []
        sorted_list = sorted(data_list)
        num = len(sorted_list)
        q1_pos = (num + 1) / 4 - 1
        q2_pos = (num + 1) / 2 - 1
        q3_pos = 3 * (num + 1)/ 4 - 1
        q1 = self.get_q(q1_pos, sorted_list)
        q2 = self.get_q(q2_pos ,sorted_list)
        q3 = self.get_q(q3_pos ,sorted_list)
        # print(q1, q2, q3)
        IQR = q3 - q1
        bottom_line = q1 - 1.5 * IQR
        top_line = q3 + 1.5 * IQR
        for data in data_list:
            if data < bottom_line or data > top_line : error_index_list.append(data_list.index(data))
        return error_index_list
    

    def get_q(self, q_pos, data_list):

        q_lw, q_rw = self.left_or_right(q_pos)
        return q_lw * data_list[floor(q_pos)] + q_rw * data_list[ceil(q_pos)]


    @return_lrweights
    def left_or_right(self, number):
        left_diff = abs(number - ceil(number))
        right_diff = abs(floor(number) - number)
        return ceil(number) if left_diff <= right_diff else floor(number)
    
class svm_():
    @return_index
    def svm_train_(self, SLdata_list):
        clf = svm.OneClassSVM(nu=0.001, kernel="rbf", gamma=0.0001)
        clf.fit(np.array(SLdata_list).reshape(-1, 1))
        error_label = clf.predict(np.array(SLdata_list).reshape(-1, 1))
        return error_label
        # print(error_label)
        # dl.detect_error_pic(SLdata_list, error_label)

if __name__ == "__main__":
    # a = [1, 4.5, 6, 2, 5, 7]
    # print(peak_r(a, 4, 0))
    pass