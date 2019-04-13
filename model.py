import os

import pandas as pd
import pickle

transfer_time = 3  # 换乘时间
buffer_time = 20  # 可接受波动时间


def get_time_func(type='s'):
    if type == 'h':
        def encode(time):
            h, m, s = [int(i) for i in time.split(':')]
            return h * 3600
    elif type == 'm':
        def encode(time):
            h, m, s = [int(i) for i in time.split(':')]
            return h * 3600 + m * 60
    else:
        def encode(time):
            h, m, s = [int(i) for i in time.split(':')]
            return h * 3600 + m * 60 + s

    def decode(timenum):
        h = timenum // 3600
        m = timenum % 3600 // 60
        s = timenum % 60

        return '%02d:%02d:%02d' % (h, m, s)

    return encode, decode


time_encode, time_decode = get_time_func()


class Data:
    def _get_data(self, small=True):
        if small:
            self.base_data = pd.read_csv('./info/' + self.file_name, converters=self.converters, nrows=100000)
        else:
            self.base_data = pd.read_csv('./info/' + self.file_name, converters=self.converters)  # , chunksize=100000)
        self.data = self.base_data.copy()

    @staticmethod
    def format_time(df, columns):
        """
        格式化时间输出
        :param df: 需要格式化时间输出的dataframe
        :param columns: 需要格式化的列名列表
        :return: 原地修改，并且返回
        """
        for column in columns:
            if column not in df.columns:
                continue
            df.loc[:, column] = df.loc[:, column].apply(time_decode)
        return df


class Od(Data):
    def __init__(self):
        self.file_name = 'o_d.csv'

        self.user_id = 'user_id'
        self.src_station = 'src_station'
        self.dest_station = 'dest_station'
        self.in_time = 'in_time'
        self.out_time = 'out_time'

        self.converters = {
            self.user_id: int,
            self.src_station: int,
            self.dest_station: int,
            self.in_time: time_encode,
            self.out_time: time_encode
        }

        self._get_data()
        self._deal_data()

    def _deal_data(self):
        """
        处理同名车站不同id问题
        :return:
        """
        data = station.station_map

        for station_id, station_real_id in data.items():
            idx = []
            idx.extend(list(self.data[self.data[self.src_station] == station_id].index))

            for i in idx:
                self.data.at[i, self.src_station] = station_real_id

            idx = []
            idx.extend(list(self.data[self.data[self.dest_station] == station_id].index))

            for i in idx:
                self.data.at[i, self.dest_station] = station_real_id


class Trains(Data):
    def __init__(self):
        self.file_name = 'trains.csv'

        self.route_id = 'route_id'
        self.train_id = 'train_id'
        self.station_id = 'station_id'
        self.end_time = 'end_time'
        self.start_time = 'start_time'

        self.converters = {
            self.route_id: int,
            self.train_id: int,
            self.station_id: int,
            self.end_time: time_encode,
            self.start_time: time_encode,
        }

        self._get_data()
        self._deal_data()

    def _deal_data(self):
        """
        处理同名车站不同id问题
        :return:
        """
        data = station.station_map
        for station_id, station_real_id in data.items():
            idx = []
            idx.extend(list(self.data[self.data[self.station_id] == station_id].index))

            for i in idx:
                self.data.at[i, self.station_id] = station_real_id

    def get_route_trains(self, route_id):
        """
        获取一个路线上所有的车次
        :param route_id:
        :return:
        """
        df: pd.DataFrame = self.data[self.data[self.route_id] == route_id]
        if df.empty:
            return []
        return df

    def get_station_trains(self, station_id):
        """
        获取一个站点所有的车次
        :param station_id:
        :return:
        """
        df: pd.DataFrame = self.data[self.data[self.station_id] == station_id]
        if df.empty:
            return []
        return df

    def get_station_drive_trains(self, start_station_id, end_station_id):
        """
        获取 A 站点 开往 B 站点的所有车次
        :param start_station_id: A
        :param end_station_id: B
        :return:
        """
        t1, t2 = self.get_station_trains(start_station_id), self.get_station_trains(end_station_id)
        t1 = t1.set_index(self.train_id)
        t2 = t2.set_index(self.train_id)

        m: pd.DataFrame = pd.merge(t1, t2, on=[self.train_id], suffixes=['_s', '_e'])

        df = (m[self.end_time + '_e'] - m[self.start_time + '_s']).drop_duplicates()
        df = df[df > 0]
        return df

    def get_station_drive_time(self, start_station_id, end_station_id):
        """
        获取 A 站点 开往 B 站点的路程耗时
        :param start_station_id: A
        :param end_station_id: B
        :return:
        """
        dd = self.get_station_drive_trains(start_station_id, end_station_id)
        real_time = list(dd)

        ret_time = float('inf')
        if real_time:
            ret_time = abs(sum(real_time) / len(real_time))

        return ret_time

    def get_station_wait_time(self, start_station_id, end_station_id):
        """
        获取 A 站点 开往 B 站点的停站耗时
        :param start_station_id: A
        :param end_station_id: B
        :return:
        """
        ts = self.get_station_drive_trains(start_station_id, end_station_id)
        idx = list(ts.index)
        m = self.data[self.data[trains.train_id] == idx]
        m = m[m[trains.station_id] == start_station_id]
        dd = (m[self.start_time] - m[self.end_time]).drop_duplicates()
        real_time = list(dd[dd > 0])

        ret_time = float('inf')
        if real_time:
            ret_time = abs(sum(real_time) / len(real_time))

        return ret_time


class RouteName(Data):
    def __init__(self):
        self.file_name = 'route_name.csv'

        self.route_id = 'route_id'
        self.route_name = 'route_name'

        self.converters = {
            self.route_id: int,
            self.route_name: str,
        }

        self._get_data()

    def get_name(self, route_id):
        """
        获取路线的名称
        :param route_id:
        :return:
        """
        df: pd.DataFrame = self.data[self.data[self.route_id] == route_id]
        if df.empty:
            return ''
        return df[self.route_name].values[0]


class Station(Data):
    def __init__(self):
        self.file_name = 'station.csv'

        self.station_id = 'station_id'
        self.station_name = 'station_name'
        self.route_id = 'route_id'

        self.converters = {
            self.station_id: int,
            self.station_name: str,
            self.route_id: int,
        }

        self.station_map = {}
        self._route = {}
        self._all_route = None
        self._floyd = None

        self._get_data()
        self._deal_data()

    def _get_same_name_station(self):
        """
        获取同名车站所有id
        :return:
        """
        data = self.data.groupby(self.station_name)
        idx = {k: v + 1 for k, v in data.indices.items()}
        return idx

    def _deal_data(self):
        """
        处理同名车站不同id问题
        :return:
        """
        data = self._get_same_name_station()
        for station_name, station_ids in data.items():
            if len(station_ids) > 1:
                c_id = station_ids[0]
                for id in station_ids[1:]:
                    self.station_map[id] = c_id
                    self.data.at[id - 1, self.station_id] = c_id

    def get_real_station_id(self, station_id):
        """
        获取同名车站的唯一ID
        :param station_id:
        :return:
        """
        return self.station_map[station_id]

    def get_station_id(self, station_name, base=False):
        """
        根据车站名称获取车站id
        :param station_name:
        :param base: 是否用未处理过的数据
        :return:
        """
        if base:
            df: pd.DataFrame = self.base_data[self.base_data[self.station_name] == station_name]
            if df.empty:
                return ''
            return list(df[self.station_id])
        else:
            df: pd.DataFrame = self.data[self.data[self.station_name] == station_name]
            if df.empty:
                return ''
            return df[self.station_id].values[0]

    def get_name(self, station_id):
        """
        根据车站名称获取车站id，用的没有处理过同名车站的数据
        :param station_id:
        :return:
        """
        df: pd.DataFrame = self.base_data[self.base_data[self.station_id] == station_id]
        if df.empty:
            return ''
        return df[self.station_name].values[0]

    def get_station_routes(self, station_id):
        """
        获取站点所有的路线
        :param station_id:
        :return:
        """
        df: pd.DataFrame = self.data[self.data[self.station_id] == station_id]
        if df.empty:
            return []
        return list(df[self.route_id])

    def get_route_stations(self, route_id):
        """
        获取路线所有车站,根据文件里站点顺序
        :param route_id:
        :return:
        """
        if self._route.get(route_id):
            return self._route[route_id]

        df: pd.DataFrame = self.data[self.data[self.route_id] == route_id]
        if df.empty:
            return []
        stations = list(df[self.station_id])
        self._route[route_id] = stations

        return stations

    def get_all_route(self):
        """
        计算所有路线的相邻站点耗时
        :return:
        """
        if self._all_route is not None:
            print('cache')
            return self._all_route

        if os.path.exists('all_route.pds'):
            with open('all_route.pds', 'rb') as f:
                print('load')
                self._all_route = pickle.load(f)
                return self._all_route

        self._all_route = pd.DataFrame(data=float('inf'), index=range(1, len(self.data) + 1), columns=range(1, len(self.data) + 1), dtype=pd.np.float)

        for i in range(1, len(self.data) + 1):
            self._all_route[i][i] = 0

        s: pd.Series = self.data[self.route_id]
        route_ids = list(s.drop_duplicates())
        for route_id in route_ids:
            stations = self.get_route_stations(route_id)
            for i in range(len(stations) - 1):
                now = stations[i]
                next = stations[i + 1]
                self._all_route[now][next] = trains.get_station_drive_time(now, next)
                self._all_route[next][now] = trains.get_station_drive_time(next, now)

        with open('all_route.pds', 'wb') as f:
            pickle.dump(self._all_route, f)
        return self._all_route

    def get_floyd(self):
        """
        计算所有站点间耗时
        :return:
        """
        if self._floyd:
            return self._floyd
        if os.path.exists('floyd.pds'):
            with open('floyd.pds', 'rb') as f:
                print('load')
                self._floyd = pickle.load(f)
                return self._floyd

        l = self.data.shape[0]
        self._floyd = self.get_all_route().copy()

        processor = {
            'now': 0,
            'all': l ** 3
        }
        for k in range(1, l + 1):
            for i in range(1, l + 1):
                for j in range(1, l + 1):
                    processor['now'] += 1
                    if self._floyd[i][j] > self._floyd[i][k] + self._floyd[k][j]:
                        # todo 路径
                        self._floyd[i][j] = self._floyd[i][k] + self._floyd[k][j]
                    print(f"{processor['now']}/{processor['all']} -- {(processor['now']/processor['all'])*10000 //1 /100}%")

        with open('floyd.pds', 'wb') as f:
            pickle.dump(self._floyd, f)

        return self._floyd


station = Station()
od = Od()
trains = Trains()
route_name = RouteName()

del Data
del Od
del Trains
del RouteName
del Station
