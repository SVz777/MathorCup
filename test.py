import pandas as pd
import numpy as np

from model import station, trains, time_decode

# 分值统计
# info = data.groupby(od.in_time).size()

# a = pd.DataFrame(data=False,index=range(1,329),columns=range(1,329),dtype=np.bool)
# print(a)



stations = station.get_route(1)
s1, s2 = stations[2:4]

t1, t2 = trains.get_station_trains(s1), trains.get_station_trains(s2)
t1 = t1[[trains.station_id, trains.train_id, trains.end_time, trains.start_time]]
t2 = t2[[trains.station_id, trains.train_id, trains.end_time, trains.start_time]]
t1 = t1.set_index(trains.train_id)
t2 = t2.set_index(trains.train_id)

m = pd.merge(t1, t2, on=[trains.train_id], suffixes=['_s', '_e'])
# print(m)
dd = list(m[trains.start_time + '_e'] - m[trains.start_time + '_s'])
print(abs(dd[0]))

# 格式化输出
# df: pd.DataFrame = trains.get_route_trains(10)
# df.loc[:, trains.station_id] = df.loc[:,trains.station_id].apply(station.get_name)
# df.loc[:, trains.route_id] = df.loc[:,trains.route_id].apply(route_name.get_name)
# df.loc[:, trains.end_time] = df.loc[:,trains.end_time].apply(time_decode)
# df.loc[:, trains.start_time] = df.loc[:,trains.start_time].apply(time_decode)