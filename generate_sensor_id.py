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

f = open(r"D:\Project\VS_code\harbin_bridge_pro\sensor_id_list.txt", "w")
f.write(str(sensor_list))
f.close()