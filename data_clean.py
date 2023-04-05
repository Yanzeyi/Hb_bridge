import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from error_detect import k_means, three_sigma


class ds_load():
    def __init__(self, file_path):
        self.file_path = file_path
        self.ds = pd.read_csv(self.file_path, encoding='ANSI', parse_dates=['MDATE'], date_parser=lambda x : datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
        
        self.rows = len(self.ds)
        self.cols = len(self.ds.columns)

        self.SLData_dict = {}


    def sort_by_time_order(self):
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


    def get_SLdata(self, index_list):
        SLdata_list = []
        for index in index_list:
            SLdata_list.append(self.ds.iloc[index, 11])
        return SLdata_list
    

    def show_pic(self, SLdata_list):
        x = range(len(SLdata_list))
        plt.plot(x, SLdata_list)
        plt.show()

    
    def df2dict_SLData(self):
        for i in range(1, 25, 1):
            if i < 10 : sensor_id = "SLS0" + str(i)
            elif 9 < i < 25 : sensor_id = "SLS" + str(i)
            print(sensor_id)
            index_list = self.get_SLdata_index(sensor_id)
            SLdata_list = self.get_SLdata(index_list)
            self.SLData_dict[str(sensor_id)] = SLdata_list


    def detect_error_pic(self, SLdata_list, error_index_list):
        x = range(len(SLdata_list))
        plt.plot(x, SLdata_list)
        for i in range(len(error_index_list)):
            plt.scatter(error_index_list[i], SLdata_list[error_index_list[i]], c = 'r')
        plt.show()


if __name__ == "__main__":

    file_path =  r"D:\Jilin_university\Harbin_bridge_pro\原始数据-按月\原始数据-按月\2022-01.csv"  #导入数据库路径


    # ds = pd.read_csv(file_path, encoding='ANSI', parse_dates=['MDATE'], date_parser=lambda x : datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    # ds = ds.sort_values(by='MDATE')

    dl = ds_load(file_path) #加载数据库


    dl.sort_by_time_order() #按时间进行排序
    index_list = dl.get_SLdata_index("SLS03") #查找传感器数据的索引
    SLdata_list = dl.get_SLdata(index_list) #根据索引值得到数据


    # dl.df2dict_SLData()
    # print(dl.SLData_dict)
    # dl.show_pic(SLdata_list)

    # detect_test = detect_error()
    # error_list = detect_test.three_sigma(SLdata_list)
    # print(error_list)

    km = k_means()  #kmeans检测
    error_index_list = km.k_means_train(2, SLdata_list) #得到异常值索引
    dl.detect_error_pic(SLdata_list, error_index_list)  #作图(异常值标红)


    ts = three_sigma() #3sigma检测
    error_index_list_0 = ts.three_sigma_(SLdata_list)
    dl.detect_error_pic(SLdata_list, error_index_list_0)

    # x = range(len(SLdata_list))
    # plt.plot(x, SLdata_list)
    # for i in range(len(error_index_list)):
    #     plt.scatter(error_index_list[i], SLdata_list[error_index_list[i]], c = 'r')
    # plt.show()










