import pandas as pd

from model import od, station, trains, route_name, time_decode, time_encode

od.data['cost'] = od.data[od.out_time]-od.data[od.in_time]
print(od.format_time(od.data,[od.in_time,od.out_time,'cost']))
