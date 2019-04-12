import pandas as pd

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


def get_min_map_floyd(df: pd.DataFrame):
    l = df.shape[0]
    for i in range(1, l + 1):
        for j in range(1, l + 1):
            for k in range(1, l + 1):
                if df[i][j] > df[i][k] + df[k][j]:
                    # todo 路径
                    df[i][j] = df[i][k] + df[k][j]
    return df


class Data:
    def get_data(self, small=True):
        if small:
            self.data = pd.read_csv('./info/' + self.file_name, converters=self.converters, nrows=100000)
        else:
            self.data = pd.read_csv('./info/' + self.file_name, converters=self.converters)  # , chunksize=100000)


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

        self.get_data()

    def format_time(self, df):
        df.loc[:, self.in_time] = df.loc[:, self.in_time].apply(time_decode)
        df.loc[:, self.out_time] = df.loc[:, self.out_time].apply(time_decode)
        return df


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

        self.get_data()

    def format_time(self, df):
        df.loc[:, self.end_time] = df.loc[:, self.end_time].apply(time_decode)
        df.loc[:, self.start_time] = df.loc[:, self.start_time].apply(time_decode)
        return df

    def get_route_trains(self, route_id):
        df: pd.DataFrame = self.data[self.data[self.route_id] == route_id]
        if df.empty:
            return []
        return df

    def get_station_trains(self, station_id):
        df: pd.DataFrame = self.data[self.data[self.station_id] == station_id]
        if df.empty:
            return []
        return df

    def get_station_time(self, start_station_id, end_station_id):
        t1, t2 = trains.get_station_trains(start_station_id), trains.get_station_trains(end_station_id)
        t1 = t1[[trains.station_id, trains.train_id, trains.end_time, trains.start_time]]
        t2 = t2[[trains.station_id, trains.train_id, trains.end_time, trains.start_time]]
        t1 = t1.set_index(trains.train_id)
        t2 = t2.set_index(trains.train_id)

        m: pd.DataFrame = pd.merge(t1, t2, on=[trains.train_id], suffixes=['_s', '_e'])

        dd = (m[trains.start_time + '_e'] - m[trains.start_time + '_s']).drop_duplicates()

        front = list(dd[dd > 0])
        back = list(dd[dd < 0])

        ret_front, ret_back = float('inf'), float('inf')
        if front:
            ret_front = abs(sum(front) / len(front))

        if back:
            ret_back = abs(sum(back) / len(back))

        return ret_front, ret_back


class RouteName(Data):
    def __init__(self):
        self.file_name = 'route_name.csv'

        self.route_id = 'route_id'
        self.route_name = 'route_name'

        self.converters = {
            self.route_id: int,
            self.route_name: str,
        }

        self.get_data()

    def get_name(self, route_id):
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

        self.get_data()

        self._route = {}
        self._all_route = None

    def get_station_ids(self, station_name):
        df: pd.DataFrame = self.data[self.data[self.station_name] == station_name]
        if df.empty:
            return ''
        return list(df[self.station_id])

    def get_name(self, station_id):
        df: pd.DataFrame = self.data[self.data[self.station_id] == station_id]
        if df.empty:
            return ''
        return df[self.station_name].values[0]

    def get_station_routes(self, station_id):
        df: pd.DataFrame = self.data[self.data[self.station_id] == station_id]
        if df.empty:
            return []
        return list(df[self.route_id])

    def get_route_stations(self, route_id):
        df: pd.DataFrame = self.data[self.data[self.route_id] == route_id]
        if df.empty:
            return []
        return list(df[self.station_id])

    def get_route(self, route_id):
        if self._route.get(route_id):
            return self._route[route_id]

        stations = self.get_route_stations(route_id)

        self._route[route_id] = stations
        return stations

    def get_all_route(self, trains):
        if self._all_route is not None:
            return self._all_route
        self._all_route = pd.DataFrame(data=float('inf'), index=range(1, 329), columns=range(1, 329), dtype=pd.np.float)

        s: pd.Series = self.data[self.route_id]
        route_ids = list(s.drop_duplicates())
        for route_id in route_ids:
            stations = self.get_route(route_id)
            for i in range(len(stations) - 1):
                now = stations[i]
                next = stations[i + 1]

                front, back = trains.get_station_time(now, next)
                self._all_route[now][next] = front
                self._all_route[next][now] = back

        return self._all_route


od = Od()
trains = Trains()
route_name = RouteName()
station = Station()

del Data
del Od
del Trains
del RouteName
del Station
