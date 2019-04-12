import matplotlib.pyplot as plt

import pandas as pd

from model import Od, Trains, Route, Station, time_decode

tmp = 'tmp_'


def df_sum(dfs):
    df = dfs[0]
    for d in dfs[1:]:
        df.add(d, fill_value=0)
    return df


od = Od()
trains = Trains()
route_name = Route()
station = Station()

datas = od.get_data()
print(datas)
in_time_counts = []
out_time_counts = []
for data in [datas]:
    in_time_count = data.groupby(od.in_time).size()
    in_time_counts.append(in_time_count.rename(time_decode, axis=0))
    out_time_count = data.groupby(od.out_time).size()
    out_time_counts.append(out_time_count.rename(time_decode, axis=0))

in_time_count: pd.DataFrame = df_sum(in_time_counts)
out_time_count: pd.DataFrame = df_sum(out_time_counts)

print(in_time_count)
print()
print(out_time_count)
# plt.show()

# out_time_count.plot.line(x=out_time_count.index, y=od.out_time, color='Yellow', label=od.out_time)
# data.plot.scatter(x=out_time_count.index,y=od.out_time,color='Yellow',label=od.out_time)
# in_time_count.plot.line(x=in_time_count.index, y=od.out_time, color='LightGreen', label=od.in_time)
# data.plot.scatter(x=in_time_count.index,y=od.in_time,color='LightGreen',label=od.in_time)
# plt.show()
