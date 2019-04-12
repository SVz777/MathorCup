import pandas as pd
import numpy as np

from model import station, trains, route_name, time_decode

# 分值统计
# info = data.groupby(od.in_time).size()

# a = pd.DataFrame(data=False,index=range(1,329),columns=range(1,329),dtype=np.bool)
# print(a)

# 格式化输出
# df: pd.DataFrame = trains.get_route_trains(10)
# df.loc[:, trains.station_id] = df.loc[:,trains.station_id].apply(station.get_name)
# df.loc[:, trains.route_id] = df.loc[:,trains.route_id].apply(route_name.get_name)
# df.loc[:, trains.end_time] = df.loc[:,trains.end_time].apply(time_decode)
# df.loc[:, trains.start_time] = df.loc[:,trains.start_time].apply(time_decode)