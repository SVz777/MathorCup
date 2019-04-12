import pandas as pd

from model import station, trains, route_name, time_decode

print(station.get_all_route())
station_ids = station.get_route_stations(10)
print(station_ids)

stations = list(map(station.get_name, station_ids))
print(stations)

all_route = station.get_all_route()
floyd = station.get_floyd()
print(floyd)

ss = floyd.loc[26]
print(ss[ss < float('inf')])

ss = floyd.loc[28]
print(ss[ss < float('inf')])
