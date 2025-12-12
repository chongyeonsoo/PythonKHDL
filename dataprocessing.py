import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#store data
path1 = 'data/air_visit_data.csv'
air_visit = pd.read_csv(path1)

#xu li data
air_visit.index = pd.to_datetime(air_visit['visit_date'])
air_visit = air_visit.groupby('air_store_id').apply(lambda g: g['visitors'].resample('1d').sum()).reset_index()
air_visit['visit_date'] = air_visit['visit_date'].dt.strftime('%Y-%m-%d')
air_visit['check_null'] = air_visit['visitors'].isnull()
air_visit['visitors'].fillna(0, inplace = True)

path2 = 'data/date_info.csv'
date_info = pd.read_csv(path2)
date_info.rename(columns={'holiday_flg': 'is_holiday', 'calendar_date': 'visit_date'},inplace=True)
date_info['pre_holiday'] = date_info['is_holiday'].shift().fillna(0)
date_info['next_holiday'] = date_info['is_holiday'].shift(-1).fillna(0)

#air store info dung bo cua weather vi co tien xu li du lieu san

path3 = 'weather/air_store_info_with_nearest_active_station.csv'
air_store_info = pd.read_csv(path3)

submission = pd.read_csv('data/sample_submission.csv')
submission['air_store_id'] = submission['id'].str.slice(0,20)
submission['visit_date'] = submission['id'].str.slice(21)
submission['is_test'] = True
submission['visitors'] =np.nan #xem như một cột thiếu để bỏ qua
submission['number_test']= range(len(submission))

data = pd.concat((air_visit, submission.drop('id', axis ='columns')))
print(data.isnull().sum())
data['is_test'].fillna(False, inplace = True)
data = pd.merge(left = data, right = date_info, on = 'visit_date', how = "left")
data = pd.merge(left = data, right = air_store_info, on = 'air_store_id', how = "left")
data['visitors'] = data['visitors'].astype(float)
print(data.isnull().sum())

import glob2 as glob
weather = []
for path in glob.glob('weather/1-1-16_5-31-17_Weather/weather_data/*.csv'):
    weather_df = pd.read_csv(path)
    weather_df['station_id'] = path.split('\\')[-1].rstrip('.csv')
    weather.append(weather_df)
weather = pd.concat(weather, axis= 'rows')
weather.rename(columns = {'calendar_date' : 'visit_date'}, inplace = True)
print(weather.head())

#theo tác giả tham khảo thì chỉ cần nhiệt độ trung bình và lượng mưa là đủ
mean = weather.groupby('visit_date')[['avg_temperature', 'precipitation']].mean().reset_index()
mean.rename(columns = {'avg_temperature':'global_avg_temperature', 'precipitation': 'global_precipitation'}, inplace = True)
print(mean.head(10))
print(weather.isnull().sum())
weather = pd.merge(left = weather, right = mean, on = 'visit_date', how = 'left')
weather['avg_temperature'].fillna(weather['global_avg_temperature'], inplace = True)
weather['precipitation'].fillna(weather['global_precipitation'], inplace = True)
print(weather[['visit_date', 'avg_temperature', 'precipitation']].head()) 


#xử lí data 1 lần nữa
data['visit_date'] = pd.to_datetime(data['visit_date'])
data.sort_values(['visit_date','air_store_id'])
data = data.set_index(data['visit_date'], drop = False)
print(data.isnull().sum())
print(data.head(10))
#tien xu li
def find_oulier(data):
    q1 = np.quantile(data, 0.25)
    q3 = np.quantile(data, 0.75)
    iqr = q3 - q1
    return (data < q1 - iqr*1.5) | (data > q3 + 1.5*iqr)
def replace_oulier(data):
    outlier = find_oulier(data)
    max_val = data[~outlier].max()
    data[outlier] = max_val
    return data
data_outlier = find_oulier(data['visitors'])
