import pandas as pd


def get_time_func(type):
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


time_encode, time_decode = get_time_func('h')


class Data:
    def get_data(self):
        self.data = pd.read_csv('./info/'+self.file_name, converters=self.converters, nrows=10000)  # ,chunksize=10000)


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


class Route(Data):
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

    def get_route_station(self, route_id):
        df: pd.DataFrame = self.data[self.data[self.route_id] == route_id]
        if df.empty:
            return []
        return df

    def get_route(self, route_id):
        if self._route.get(route_id):
            return self._route[route_id]

        df = self.get_route_station(route_id)
        stations = list(df['station_id'])

        if route_id == 2 or route_id == 4:
            stations.append(stations[0])
        self._route[route_id] = stations
        return stations

    def get_all_route(self):
        if self._all_route is not None:
            return self._all_route
        self._all_route = pd.DataFrame(data=False, index=range(1, 329), columns=range(1, 329), dtype=pd.np.bool)

        s: pd.Series = self.data[self.route_id]
        route_ids = list(s.drop_duplicates())
        for route_id in route_ids:
            stations = self.get_route(route_id)
            for i in range(len(stations) - 1):
                now = stations[i]
                next = stations[i + 1]
                self._all_route[now][next] = self._all_route[next][now] = True

        return self._all_route
