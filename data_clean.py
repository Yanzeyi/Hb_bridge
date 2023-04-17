import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn import svm
import numpy as np
from error_detect import k_means, three_sigma, iostation_forest, box_plot, fill_point, svm_, return_index, detect_error
from sensor_id_list import SENSOR_ID_LIST
# from sklearn.ensemble import IsolationForest


class ds_load():
    def __init__(self, file_path):
        self.file_path = file_path
        self.ds = pd.read_csv(self.file_path, encoding='ANSI', parse_dates=['MDATE'], date_parser=lambda x : datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
        
        self.rows = len(self.ds)
        self.cols = len(self.ds.columns)

        self.SLData_dict = {}
        self.SLMdata_dict = {}


    def sort_by_time_order(self):
        """
        根据时间顺序进行排序
        """
        self.ds = self.ds.sort_values(by='MDATE')
        self.ds = self.ds.reset_index(drop=True)

    
    def get_SLdata_index(self, sensor_id):
        """
        arg:sensor_id type str
        return: index_list 某一个传感器的全部数据的索引列表
        """
        sensor_id_df = self.ds[['SENSOR_ID']]
        index_list = []
        for i in range(self.rows):
            if str(sensor_id_df.iloc[i, 0]) == sensor_id:
                index_list.append(i)
        return index_list


    def get_data_st_index(self, index_list, column_index):
        """
        arg:index_list type list 数据的行索引列表 colunm_index type int 需要取得的数据的列索引
        return: SLdata_list type list 根据索引值得到数据列表
        根据行索引列表以及列索引取得相应数据列表
        """
        SLdata_list = []
        for index in index_list:
            SLdata_list.append(self.ds.iloc[index, column_index])
        return SLdata_list
    
    
    def df2dict_SLData_Date(self, country_id):
        """
        arg:country_id type bool China 0 Russsia 1
        将索力数据转换成字典形式
        example: {"SLS01": [1, 3, 5, 6 ,7], "SLS02":[2, 5, 7 ,8], ...}
        """
        self.SLData_dict = {}
        self.SLMdata_dict = {}
        SENSOR_PRIFIX = "SLS" if country_id == 0 else "SLX"
        for i in range(1, 25, 1):
            if i < 10 : sensor_id = SENSOR_PRIFIX + "0" + str(i)
            elif 9 < i < 25 : sensor_id = SENSOR_PRIFIX + str(i)
            # print(sensor_id)
            index_list = self.get_SLdata_index(sensor_id)
            SLdata_list = self.get_data_st_index(index_list, 11)
            SLMDATE_list = self.get_data_st_index(index_list, 0)
            self.SLData_dict[str(sensor_id)] = SLdata_list
            self.SLMdata_dict[str(sensor_id)] = SLMDATE_list


class show_picture():
    def show_pic(self, SLdata_list):
        """
        arg:SLdata_list type list 数据列表
        根据数据列表生成折线图
        """
        x = range(len(SLdata_list))
        plt.plot(x, SLdata_list)
        plt.show()


    def detect_error_pic(self, SLdata_list, error_index_list):
        """
        arg:SLdata_list type list 数据列表
            error_index_list type list 异常值索引列表
        将数据画成折线图，并将异常值标成红点
        """
        x = range(len(SLdata_list))
        plt.plot(x, SLdata_list)
        for i in range(len(error_index_list)):
            plt.scatter(error_index_list[i], SLdata_list[error_index_list[i]], c = 'r')
        plt.show()


class get_value(object):
    def get_mean(self,SLdata_list, error_index_list):
        new_SLdata_list = []
        for i in range(len(SLdata_list)):
            if i in error_index_list:
                pass
            else:
                new_SLdata_list.append(SLdata_list[i])
        return np.mean(new_SLdata_list), np.std(new_SLdata_list)
    

    def get_center_distance(self, SLdata_list, error_index_list):
        new_SLdata_list = []
        for i in range(len(SLdata_list)):
            if i in error_index_list:
                pass
            else:
                new_SLdata_list.append(SLdata_list[i])
        SLdata_mean = np.mean(new_SLdata_list)
        # print(SLdata_mean)
        # print(len(new_SLdata_list))
        # print(np.sum(new_SLdata_list))
        # print(len(new_SLdata_list) * SLdata_mean)
        # print(np.max(new_SLdata_list) - SLdata_mean)
        return (np.sum(new_SLdata_list) - len(new_SLdata_list) * SLdata_mean) / len(new_SLdata_list)
    

def get_year_SLdata(sensor_id):
    file_path =  r"D:\Jilin_university\Harbin_bridge_pro\原始数据-按月\原始数据-按月"  #导入数据库路径
    
    SLS01data_list = []
    SLSMdate_list = []
    for i in range(1, 10):
        dataset_path = file_path + '\\' + "2022-0" + str(i) + ".csv" 

        dl = ds_load(dataset_path) #加载数据库
        dl.sort_by_time_order() #按时间进行排序

        index_list = dl.get_SLdata_index(sensor_id) #查找传感器数据的索引
        SLdata_list = dl.get_data_st_index(index_list, 11) #根据索引值得到数据
        SLSMdate_list = dl.get_data_st_index(index_list, 0)
        SLS01data_list += SLdata_list
        SLSMdate_list += SLSMdate_list
    return SLS01data_list, SLSMdate_list


if __name__ == "__main__":
    
  
    # for i in range(1, 13):
 
    #     file_path =  r"D:\Jilin_university\Harbin_bridge_pro\原始数据-按月\原始数据-按月"
    #     year_month = "2022-0" if i < 10 else "2022-"
    #     file_path += "\\" + year_month + str(i) + ".csv"
    #     dl = ds_load(file_path)
    #     # dl = dl.read_csv(file_path, encoding='ANSI', parse_dates=['MDATE'], date_parser=lambda x : datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    #     dl.sort_by_time_order()

    #     fg = plt.figure(figsize = (40, 50))

    #     for sensor_id in SENSOR_ID_LIST:


    #         index_list = dl.get_SLdata_index(sensor_id) #查找传感器数据的索引
    #         SLdata_list = dl.get_data_st_index(index_list, 11) #根据索引值得到数据
    #         SLMdate_list = dl.get_data_st_index(index_list, 0)
    #         # @return_index
    #         # def Dbscan_train(SLdata_list):
    #         #     db = DBSCAN(eps = 50, min_samples=7).fit(np.array(SLdata_list).reshape(-1, 1))
    #         #     return db.labels_

    #         def Z_score(SLdata_list):
    #             new_list = []
    #             if np.std(SLdata_list) != 0:
    #                 for data in SLdata_list:
    #                     new_list.append((data - np.mean(SLdata_list)) / np.std(SLdata_list))
    #             else: new_list = eval("[" + ",".join("0" * len(SLdata_list)) + "]")
    #             return new_list

    #         # def mm(SLdata_list):
    #         #     new_list = []
    #         #     for data in SLdata_list:
    #         #         new_list.append((data - min(SLdata_list)) / (max(SLdata_list) - min(SLdata_list)))
    #         #     return new_list

    #         @return_index
    #         def Dbscan_train(SLdata_list):
    #             db = DBSCAN(eps = 0.65, min_samples=25).fit(np.array(SLdata_list).reshape(-1, 1))
    #             return db.labels_

    #         new_SLdata_list = Z_score(SLdata_list)
    #         error_index_list = Dbscan_train(new_SLdata_list)
    #         # gv = get_value()
    #         # ts = three_sigma()
    #         # error_index_list = ts.three_sigma_(SLdata_list, np.mean(SLdata_list), np.std(SLdata_list))
    #         # new_mean, new_std = gv.get_mean(SLdata_list, error_index_list)
    #         # error_index_list = ts.three_sigma_(SLdata_list, new_mean, new_std)

    #         ax = fg.add_subplot(24,2,SENSOR_ID_LIST.index(sensor_id) + 1)
    #         ax.plot(SLMdate_list, SLdata_list)
    #         ax.set_xlabel("Date")
    #         ax.set_ylabel("SLDate")
    #         ax.set_title(sensor_id)
    #         for error_index in error_index_list:
    #             ax.scatter(SLMdate_list[error_index], SLdata_list[error_index], c = "r")
        
    #     fg.tight_layout()
    #     fg.savefig(str(i) + '.jpg')
    sensor_id = "SLX22"
    file_path =  r"D:\Jilin_university\Harbin_bridge_pro\原始数据-按月\原始数据-按月"
    file_path += "\\" + "2022-0" + str(7) + ".csv"
    dl = ds_load(file_path)
    dl.sort_by_time_order()
    index_list = dl.get_SLdata_index(sensor_id) #查找传感器数据的索引
    SLdata_list = dl.get_data_st_index(index_list, 11) #根据索引值得到数据
    SLMdate_list = dl.get_data_st_index(index_list, 0)

    km = k_means()
    error_index_list = km.k_means_train(2, SLdata_list)

    def Z_score(SLdata_list):
        new_list = []
        if np.std(SLdata_list) != 0:
            for data in SLdata_list:
                new_list.append((data - np.mean(SLdata_list)) / np.std(SLdata_list))
        else: new_list = eval("[" + ",".join("0" * len(SLdata_list)) + "]")
        return new_list

# def mm(SLdata_list):
#     new_list = []
#     for data in SLdata_list:
#         new_list.append((data - min(SLdata_list)) / (max(SLdata_list) - min(SLdata_list)))
#     return new_list

    @return_index
    def Dbscan_train(SLdata_list):
        db = DBSCAN(eps = 0.65, min_samples=25).fit(np.array(SLdata_list).reshape(-1, 1))
        return db.labels_
    
    new_SLdata_list = Z_score(SLdata_list)
    error_index_list = Dbscan_train(new_SLdata_list)
    # gv = get_value()
    # ts = three_sigma()
    # error_index_list = ts.three_sigma_(SLdata_list, np.mean(SLdata_list), np.std(SLdata_list))
    # new_mean, new_std = gv.get_mean(SLdata_list, error_index_list)
    # error_index_list = ts.three_sigma_(SLdata_list, new_mean, new_std)
    # bp = box_plot()
    # iosf = iostation_forest()
    # error_index_list = bp.box_plot_train(SLdata_list)
    # error_index_list = iosf.ioslation_forest_train(SLdata_list)

    fg = plt.figure(figsize=(30, 10))
    ax = fg.add_subplot(1,1,1)
    ax.plot(SLMdate_list, SLdata_list)
    ax.set_xlabel("Date")
    ax.set_ylabel("SLDate")
    ax.set_title(sensor_id)
    for error_index in error_index_list:
        ax.scatter(SLMdate_list[error_index], SLdata_list[error_index], c = "r")
    fg.savefig("Dbscan" + '.jpg')