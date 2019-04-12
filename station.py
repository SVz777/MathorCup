import pandas as pd

from model import station, trains, route_name, time_decode, get_min_map_floyd

station_ids = station.get_route_stations(14)

stations = list(map(station.get_name, station_ids))

all_route = station.get_all_route(trains)
floyd = get_min_map_floyd(all_route)
print(floyd)
#
# for index,data in all_route.iterrows():
#     print(index)
#     print(data[data == True].index)
#     exit()
