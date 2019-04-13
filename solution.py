import pandas as pd

from model import od, station, trains, route_name, time_decode, time_encode


# od.data['cost'] = od.data[od.out_time]-od.data[od.in_time]
# print(od.format_time(od.data,[od.in_time,od.out_time,'cost']))


def main():
    users = [int(i) for i in [2, 7, 19, 31, 41, 71, 83, 89, 101, 113, 2845, 124801, 140610, 164834, 193196, 223919, 275403, 286898, 314976, 315621]]
    for user in users:
        user_info = od.data[od.data[od.user_id] == user]
        src, dest = user_info[od.src_station].values[0], user_info[od.dest_station].values[0]
        in_time, out_time = time_decode(user_info[od.in_time].values[0]), time_decode(user_info[od.out_time].values[0])
        print(user, in_time, out_time, src, dest, [station.get_name(i) for i in station.get_path(src, dest)])


if __name__ == '__main__':
    floyd = station.get_floyd()
    main()
