import pandas as pd
import numpy as np
import os
from datetime import datetime


if __name__ == "__main__":
    outlier_dict = {}

    # file_path = r"D:\Jilin_university\Harbin_bridge_pro\1-9月(改好)"
    # file_list = os.listdir(file_path)
    # for file_ in file_list:
    #     dataset_path = file_path + "\\" + file_
        # print(dataset_path)
    
    # file_path = r"D:\Jilin_university\Harbin_bridge_pro\1-9月(改好)\一月.xlsx"
    def month_file2dict(dataset_path):

        dl = pd.read_excel(io=dataset_path, header=None)

        dl_columns = dl.columns
        dl_rows = dl.index

        dl_row_len = len(dl_rows)
        dl_col_len = len(dl_columns)

        # outlier_SLdata_list = []
        # outlier_Mdate_list = []

        outlier_senor_dict = {}
        # outlier_data_dict = {}

        for i in range(len(dl_columns)):
            for j in range(len(dl_rows)):
                if type(dl.iloc[j, i]) == str and dl.iloc[j, i][0] == "S" and len(dl.iloc[j, i]) == 5:
                    outlier_data_dict = {}
                    outlier_SLdata_list = []
                    outlier_Mdate_list = []
                    sensor_id = dl.iloc[j, i]
                    row_cnt = j + 1
                    if row_cnt <= dl_row_len - 1:
                        while type(dl.iloc[row_cnt, i]) == datetime:

                            outlier_Mdate_list.append(dl.iloc[row_cnt, i])
                            outlier_SLdata_list.append(dl.iloc[row_cnt, i + 1]) 
                            row_cnt += 1
                            if row_cnt > dl_row_len - 1:
                                break
                        # print(outlier_SLdata_list)
                        outlier_data_dict["SLdata"] = outlier_SLdata_list
                        outlier_data_dict["Mdate"] = outlier_Mdate_list
                        outlier_senor_dict[sensor_id] = outlier_data_dict
                            
        return outlier_senor_dict
        # outlier_dict[file_list.index(file_) + 1] = outlier_senor_dict
    # od = month_file2dict(file_path)
    # print(od)
    # print(outlier_dict)

    file_path = r"D:\Jilin_university\Harbin_bridge_pro\1-9月(改好)"
    file_list = os.listdir(file_path)
    for file_ in file_list:
        dataset_path = file_path + "\\" + file_
        outlier_senor_dict = month_file2dict(dataset_path)
        outlier_dict[file_list.index(file_) + 1] = outlier_senor_dict


    sensor_list = []
    def generate_list(prefix, sensor_list):
        for i in range(1, 25):
            if i < 10:
                sensor_id = prefix + "0" + str(i)
            if 9 < i < 25:
                sensor_id = prefix + str(i)
            sensor_list.append(sensor_id)
        return sensor_list
    sensor_list = generate_list("SLS", sensor_list)
    sensor_list = generate_list("SLX", sensor_list)
    # print(sensor_list)

    # print(outlier_dict[9])
    # for sensor_id in sensor_list:
    #     if sensor_id in outlier_dict[9].keys():
    #         print(sensor_id)

    for i in range(1, 10):
        for sensor_id in sensor_list:
            if sensor_id in outlier_dict[i].keys():
                # print(outlier_dict[i][sensor_id].items())
                pass
            else:
                
                outlier_data_dict = {}
                outlier_data_dict["SLdata"] = []
                outlier_data_dict["Mdate"] = []
                outlier_dict[i][sensor_id] = outlier_data_dict
                

    f = open(r"D:\Project\VS_code\harbin_bridge_pro\outlier_dict.txt", "w")
    f.write(str(outlier_dict))
    f.close()




