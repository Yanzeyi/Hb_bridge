import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from error_detect import k_means, three_sigma, iostation_forest, box_plot, fill_point
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


if __name__ == "__main__":

    file_path =  r"D:\Jilin_university\Harbin_bridge_pro\原始数据-按月\原始数据-按月\2022-01.csv"  #导入数据库路径


    # ds = pd.read_csv(file_path, encoding='ANSI', parse_dates=['MDATE'], date_parser=lambda x : datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    # ds = ds.sort_values(by='MDATE')

    dl = ds_load(file_path) #加载数据库


    dl.sort_by_time_order() #按时间进行排序
    # dl.df2dict_SLData(0)
    # print(dl.SLData_dict)
    index_list = dl.get_SLdata_index("SLS03") #查找传感器数据的索引
    SLdata_list = dl.get_data_st_index(index_list, 11) #根据索引值得到数据


    # dl.df2dict_SLData()
    # print(dl.SLData_dict)
    # dl.show_pic(SLdata_list)

    # detect_test = detect_error()
    # error_list = detect_test.three_sigma(SLdata_list)
    # print(error_list)

    ################################### kmeans检测################################### 
    # km = k_means()  #kmeans检测
    # error_index_list = km.k_means_train(2, SLdata_list) #得到异常值索引
    # dl.detect_error_pic(SLdata_list, error_index_list)  #作图(异常值标红)

    ################################### 3sigma检测################################### 
    ts = three_sigma() #3sigma检测
    error_index_list_0 = ts.three_sigma_(SLdata_list)
    dl.show_pic(SLdata_list)
    print(error_index_list_0)
    newSL = fill_point(SLdata_list, error_index_list_0)
    dl.show_pic(newSL)
    # print(SLdata_list)
    # print(newSL)
    # dl.show_pic(SLdata_list)
    # dl.show_pic(newSL)
    # print(error_index_list_0)
    # dl.detect_error_pic(SLdata_list, error_index_list_0)

    ################################### isolation forest检测################################### 
    # # print(len(SLdata_list))
    # isofor = iostation_forest()
    # error_index_list = isofor.ioslation_forest_train(SLdata_list)
    # dl.detect_error_pic(SLdata_list, error_index_list)

    ################################### box_plot检测################################### 
    # bp = box_plot()
    # error_index_list1 = bp.box_plot_train(SLdata_list)
    # # print(error_index_list1)
    # dl.detect_error_pic(SLdata_list, error_index_list1)

    # # x = range(len(SLdata_list))
    # # plt.plot(x, SLdata_list)
    # # for i in range(len(error_index_list)):
    # #     plt.scatter(error_index_list[i], SLdata_list[error_index_list[i]], c = 'r')
    # # plt.show()












