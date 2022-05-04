import googlemaps
import geopandas as gpd
import os
import shapely as shp
import pandas as pd
import datetime
import pickle
import locale
import polyline
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

roads = gpd.read_file("data/xn--Straennetz_-_Berlin-otb.geojson")
roads['geometry_wkt'] = [i.to_wkt() for i in roads['geometry']]
dir(roads['geometry'].iloc[0])


roads['geometry'].iloc[0].contains(shp.geometry.Point(52.63874819500006, 13.29229461800003))

print(roads['geometry'].iloc[0])

point = shp.geometry.Point(13.478324, 52.487493)

roads.head()

mini = min(roads['geometry'], key=point.distance)

print(mini)


[print(i) for i in roads if i['geometry'].equals(mini)]


years = ['2016', '2017', '2018', '2019', '2020']
dfs = [gpd.read_file('data/accidents_data_shape/Unfallorte' + i + '_EPSG25832_Shape/Shapefile/Unfallorte' + i + '_LinRef.shp') for i in years]
dfs = [i.drop(['OBJECTID'], axis=1) if 'OBJECTID' in i.columns else i for i in dfs]
dfs = [i.drop(['UIDENTSTLA'], axis=1) if 'UIDENTSTLA' in i.columns else i for i in dfs]

dfs[0].rename(columns={'IstStrasse': 'STRZUSTAND'}, inplace=True)
dfs[1].rename(columns={'LICHT':'ULICHTVERH'}, inplace=True)

all_accs = pd.concat(dfs)
#all_accs=all_accs.astype({col:pd.StringDtype() for col in list(all_accs.select_dtypes('int64').columns)})
bike_accs = all_accs[(all_accs['IstRad'] == '1') & (all_accs['ULAND'] == '11')]
bike_accs.to_csv('data/bike_accsidents1820.csv')
measure_locs = pd.read_csv("data/locs.csv", delimiter=";", decimal=",")
measured_traf = pd.read_csv("data/bike_frequency.csv", delimiter=";", decimal=",")

measured_traf = pd.melt(measured_traf, id_vars=['Date'], value_vars=measured_traf.columns[1:], var_name='Zählstelle')

measured_traf = measured_traf.merge(measure_locs, on ='Zählstelle')

locale.setlocale(locale.LC_TIME, "de_DE")
measured_traf['Date'] = pd.to_datetime(measured_traf['Date'], format = "%a, %d. %b %Y %H:%M")
measured_traf['Year'] = measured_traf['Date'].dt.year
measured_traf['Month'] = measured_traf['Date'].dt.month
measured_traf['Weekday'] = measured_traf['Date'].dt.weekday
measured_traf['Hour'] = measured_traf['Date'].dt.hour

measured_traf = measured_traf[~measured_traf.Zählstelle.isin(['05-FK-OBB-O', '05-FK-OBB-W', '03-MI-SAN-O', '06-FK-FRA-O', '18-TS-YOR-W', '18-TS-YOR-O'])]

unique_mlocs = measured_traf.groupby(['Zählstelle']).mean()

unique_mlocs['geometry_wkt'] = ''
for i, row in unique_mlocs.iterrows():
    x, y = row['Längengrad'], row['Breitengrad']
    point = shp.geometry.Point(x, y)   
    mini = min(roads['geometry'], key=point.distance)
    unique_mlocs['geometry_wkt'].loc[i] = mini.to_wkt()
    print(f'{i}')

unique_mlocs = unique_mlocs.merge(roads, on='geometry_wkt')



bike_accs[['UJAHR', 'UMONAT', 'UWOCHENTAG', 'USTUNDE']] = bike_accs[['UJAHR', 'UMONAT', 'UWOCHENTAG', 'USTUNDE']].apply(pd.to_numeric)

weekday_dic = {1:6, 2:0, 3:1, 4:2, 5:3, 6:4, 7:5}

bike_accs['UWOCHENTAG'] = [weekday_dic[i] for i in bike_accs['UWOCHENTAG']]



bike_accs['geometry_wkt'] = ''
bike_accs.reset_index(inplace=True)
for i, row in bike_accs.iterrows():
    x, y = row['XGCSWGS84'], row['YGCSWGS84']
    point = shp.geometry.Point(x, y)   
    mini = min(roads['geometry'], key=point.distance)
    bike_accs['geometry_wkt'].iloc[i] = mini.to_wkt()
    print(f'{i} of {len(bike_accs)}')


df = bike_accs.merge(roads, on='geometry_wkt', how='outer')

df.to_csv('bike_accs_time_outer.csv')




## nur mit ort
roads['geometry_wkt'] = [i.to_wkt() for i in roads['geometry']]
roads['n_accidents'] = ([0] * len(roads))
bike_accs.reset_index(inplace=True)
for i, row in bike_accs.iterrows():
    x, y = row['XGCSWGS84'], row['YGCSWGS84']
    point = shp.geometry.Point(x, y)   
    mini = min(roads['geometry'], key=point.distance)
    roads.loc[roads['geometry_wkt'] == mini.to_wkt(), ['n_accidents']] +=1
    print(roads.loc[roads['geometry_wkt'] == mini.to_wkt(), ['n_accidents']])
    print(f'accident {i} of {len(bike_accs)}')

roads.to_csv("roads_with_accs_and_traffic.csv")


bike_locs2 = pd.read_pickle('data/nextbike_3.pickle')

bike_locs = pd.DataFrame.from_records([{'id': i[0], 'time': i[1], 'lat': i[2], 'long': i[3]} for key in bike_locs2.keys() for i in bike_locs2[key]])


api_key="AIzaSyCahsuk_JzN_mvkD29RNTIcmfknKazfYwU"
gmaps = googlemaps.Client(key=api_key)

bike_locs = bike_locs['time' != ]
#### Add time!!!!
count = 0
bike_trips = {}
for id in set(bike_locs['id']):
    count += 1
    print(f"{count} of {len(set(bike_locs['id']))}")
    df = bike_locs[bike_locs['id'] == id].drop_duplicates(['lat', 'long'], keep='last').reset_index()
    df_len = len(df)
    if df_len > 1:
        for i in range(df_len-1):
            bike_trips[str(id) + '_' + str(i)] = {'directions': gmaps.directions((df['lat'].iloc[i], df['long'].iloc[i]), (df['lat'].iloc[i+1], df['long'].iloc[i+1]), mode='bicycling'), 'time': df['time'].iloc[i]}


with open('bike_trips2.pickle', 'wb') as handle:
        pickle.dump(bike_trips, handle, protocol=pickle.HIGHEST_PROTOCOL)
        


 # bike_trips = pd.read_pickle('bike_trips.pickle')


{'I': 'big_road',
 'II': 'super_road',
 'III': 'local road',
 'IV': 'compl_road',
 'V': 'no_StEP',
 '0': 'other'
 }




accs_16 = gpd.read_file("C:/Users/julia/Downloads/2016_accs/Unfaelle_2016_LinRef.shp")


for key in list(bike_trips.keys()):
    poly=polyline.decode(bike_trips[key]['directions'][0]['overview_polyline']['points'])
    time=bike_trips[key]['time']
    xyz_list=[]
    for i in poly: 
        point = shp.geometry.Point(i[0], i[1])
        mini = min(roads['geometry'], key=point.distance)
        if roads_accs.loc[roads['geometry_wkt'] == mini.to_wkt()]['ID'] not in xyz_list:
            xyz_list.append()
            roads_accs.loc[roads['geometry_wkt'] == mini.to_wkt(),['traffic']] +=1
        



list(bike_trips.keys())[4000]
### proper waypoints list!
### polyline!!
waypoints_l = {}

for i_key in list(bike_trips.keys()):
    i_trip = bike_trips[i_key]
    wp_list_i = []
    for step in range(len(i_trip[0]['legs'][0]['steps'])):
        wp_list_i.append(i_trip[0]['legs'][0]['steps'][step]['start_location'])
    wp_list_i.append(i_trip[0]['legs'][0]['steps'][len(i_trip[0]['legs'][0]['steps'])-1]['end_location'])
    waypoints_l[i_key] = wp_list_i


with open('trips.pickle', 'wb') as handle:
        pickle.dump(waypoints_l, handle, protocol=pickle.HIGHEST_PROTOCOL)



waypoints_l = pd.read_pickle('trips.pickle')





count=0
roads['traffic'] = ([0] * len(roads))
for trip in waypoints_l.keys():
    count += 1
    for waypoint in waypoints_l[trip]:
        x, y = waypoint['lng'], waypoint['lat']
        point = shp.geometry.Point(x, y)   
        mini = min(roads['geometry'], key=point.distance)
        roads.loc[roads['geometry_wkt'] == mini.to_wkt(),['traffic']] +=1
    print(f'waypoint {count} of {len(list(waypoints_l.keys()))}')

roads.to_csv("roads_with_accs_and_traffic.csv")

# check whether correct order of lat and long in point assignment - how was it done in the code before?

roads.loc[roads['geometry_wkt'] == mini.to_wkt()]['traffic']


###### compute yearly trend


list(set([(y, x) for x, y in zip(measure_locs['Breitengrad'], measure_locs['Längengrad'])]))


count=0
roads['traffic'] = ([0] * len(roads))
for trip in waypoints_l.keys():
    count += 1
    for waypoint in waypoints_l[trip]:
        x, y = waypoint['lng'], waypoint['lat']
        point = shp.geometry.Point(x, y)   
        mini = min(roads['geometry'], key=point.distance)
        roads.loc[roads['geometry_wkt'] == mini.to_wkt(),['traffic']] +=1
    print(f'waypoint {count} of {len(list(waypoints_l.keys()))}')


measured_traf.drop_duplicates(['Date', 'Zählstelle', ''])



####### data prep for bikes
bike_locs = {}
for i in range(7):
    bike_locs_i = pd.read_pickle('data/nextbike_data/nextbike_' + str(i) + '.pickle')
    bike_locs = {**bike_locs, **bike_locs_i}

bike_locs_df = pd.DataFrame.from_records([{'id': i[0], 'time': i[1], 'lat': i[2], 'long': i[3]} for key in bike_locs.keys() for i in bike_locs[key]])

bike_locs_df['day'] = bike_locs_df['time'].dt.day
bike_locs_df['weekday'] = bike_locs_df['time'].dt.weekday

bike_locs_df_final =  bike_locs_df[bike_locs_df['day'] !=  24]

count = 0
bike_trips = {}
for id in set(bike_locs_df_final['id']):
    count += 1
    print(f"{count} of {len(set(bike_locs_df_final['id']))}")
    df = bike_locs_df_final[bike_locs_df_final['id'] == id].drop_duplicates(['lat', 'long'], keep='last').reset_index()
    df_len = len(df)
    if df_len > 1:
        for i in range(df_len-1):
            bike_trips[str(id) + '_' + str(i)] = {'directions': gmaps.directions((df['lat'].iloc[i], df['long'].iloc[i]), (df['lat'].iloc[i+1], df['long'].iloc[i+1]), mode='bicycling'), 'time': df['time'].iloc[i]}


with open('trips_all2.pickle', 'wb') as handle:
        pickle.dump(bike_trips, handle, protocol=pickle.HIGHEST_PROTOCOL)
        


all_trips = sum([list(pd.read_pickle('trips_all.pickle').values()), list(pd.read_pickle('trips_all2.pickle').values()), list(pd.read_pickle('trips_24.pickle').values())], [])

roads = pd.get_dummies(roads, columns=['STRKLASSE', 'STRKLASSE1', 'STRKLASSE2', 'VRICHT'])

roads['y'] = roads['geometry'].centroid.y
roads['x'] = roads['geometry'].centroid.x


agg_dic1 = {col:'sum' for col in roads.columns[22:]}
agg_dic2= {    
    'SHAPE_Length': 'sum',
    'LAENGE': 'sum',
    'traffic': 'sum',
    'n_accidents': 'sum',
    'y': np.mean,
    'x': np.mean
}
agg_dic = {**agg_dic1, **agg_dic2}


roads = roads.groupby('STADTTEIL').agg(agg_dic)

df_list=[]
count=0
for trip in all_trips:
    count+=1
    time = trip['time']
    poly = polyline.decode(trip['directions'][0]['overview_polyline']['points'])
    poly = [t[::-1] for t in poly]
    if len(poly) > 1:
        ls = shp.geometry.LineString(poly)
        df_list.append({'time': time, 'directions': ls})
        print(f'{count} of {len(all_trips)}')


trips_df = gpd.GeoDataFrame(df_list, crs={'init':'epsg:4326'})
trips_df['Month'] = trips_df.time.dt.month
trips_df['Weekday'] = trips_df.time.dt.weekday
trips_df['hour'] = trips_df.time.dt.hour

b = [0,6,12,18,24]
l = ['Night', 'Morning', 'Afternoon','Evening']
trips_df['time_p'] = pd.cut(trips_df['hour'], bins=b, labels=l, include_lowest=True)
trips_df['wday'] = ['weekday' if i < 5 else 'weekend' for i in trips_df['Weekday']]

district_shp = gpd.read_file('lor_ortsteile.geojson')
trips_df['geometry'] = gpd.GeoSeries(trips_df['directions'])

trips_df = gpd.sjoin(trips_df, district_shp, how = 'left', op='intersects')

trips_df.to_csv('trips_by_dist.csv')


trips_df.groupby(['OTEIL', 'Month', 'time_p']).agg({'directions': 'count'})

trips_df




line_df = gpd.GeoDataFrame(crs={'init':'epsg:4326'})
line_df['geometry'] = gpd.GeoSeries(line_list)
line_df['traffic'] = [1] * len(line_list)

line_df['geometry'] = line_df['geometry'].to_crs({'init':'epsg:4326'})
df_join = gpd.sjoin(line_df, roads, how = 'left', op='intersects')



def count_accs(trip):
    poly = polyline.decode(trip['directions'][0]['overview_polyline']['points'])
    time = trip['time']
    xyz_list = []
    for i in poly: 
        point = shp.geometry.Point(i[1], i[0])
        mini = min(roads['geometry'], key=point.distance)
        if roads.loc[roads['geometry_wkt'] == mini.to_wkt()]['ELEM_NR'].iloc[0] not in xyz_list:
            xyz_list.append(roads.loc[roads['geometry_wkt'] == mini.to_wkt()]['ELEM_NR'].iloc[0])
            roads.loc[roads['geometry_wkt'] == mini.to_wkt(),['traffic']] +=1


test = roads[:10]
    
ls = shp.geometry.LineString(polyline.decode(all_trips[100]['directions'][0]['overview_polyline']['points']))


poly = polyline.decode(all_trips[100]['directions'][0]['overview_polyline']['points'])
poly = [t[::-1] for t in poly]
ls_bool = [shp.geometry.LineString(poly).buffer(10).intersects(i) for i in roads['geometry']]

count=0
for trip in all_trips:
    count +=1
    poly = polyline.decode(trip['directions'][0]['overview_polyline']['points'])
    poly = [t[::-1] for t in poly]
    ls_bool = [shp.geometry.LineString(poly).buffer(10).intersects(i) for i in roads['geometry']]
    roads.loc[pd.Series(ls_bool), 'trafic'] += 1 
    print(f'{count} of {len(all_trips)}')
    
for i, road in roads.iterrows():
    print(f'{i} of {len(roads)}')
    for trip in all_trips:
        poly = polyline.decode(trip['directions'][0]['overview_polyline']['points'])
        ls = shp.geometry.LineString(poly)
        x = True
        while x:
            for p in poly:
                point = shp.geometry.Point(p[1], p[0])
                if point.within(road['geometry']):
                    roads['traffic'].iloc[i] +=1
                    x = False



roads['traffic'] = [0] * len(roads)
count=0
for trip in all_trips:
    count += 1
    print(f'{count} of {len(all_trips)}')
    poly = polyline.decode(trip['directions'][0]['overview_polyline']['points'])
    time = trip['time']
    xyz_list = []
    for i in poly: 
        point = shp.geometry.Point(i[1], i[0])
        mini = min(roads['geometry'], key=point.distance)
        if roads.loc[roads['geometry_wkt'] == mini.to_wkt()]['ELEM_NR'].iloc[0] not in xyz_list:
            xyz_list.append(roads.loc[roads['geometry_wkt'] == mini.to_wkt()]['ELEM_NR'].iloc[0])
            roads.loc[roads['geometry_wkt'] == mini.to_wkt(),['traffic']] +=1
            
with open('roads_accs_24.pickle', 'wb') as handle:
        pickle.dump(roads, handle, protocol=pickle.HIGHEST_PROTOCOL)

roads = pd.read_pickle('roads_accs_24.pickle')
    
roads['n_accidents'] = ([0] * len(roads))
bike_accs.reset_index(inplace=True)
for i, row in bike_accs.iterrows():
    x, y = row['XGCSWGS84'], row['YGCSWGS84']
    point = shp.geometry.Point(x, y)   
    mini = min(roads['geometry'], key=point.distance)
    roads.loc[roads['geometry_wkt'] == mini.to_wkt(), ['n_accidents']] +=1
    print(roads.loc[roads['geometry_wkt'] == mini.to_wkt(), ['n_accidents']])
    print(f'accident {i} of {len(bike_accs)}')
    
    
df = roads.groupby(['BEZIRK']).sum()
d = pd.read_pickle("C:/Users/julia/Downloads/nextbike (4).pickle")
min(bike_trips, key=bike_trips.get)


tl=[]
[tl.append(bike_trips[i]['time']) for i in list(bike_trips.keys())]
bike_locs_df_final = bike_locs_df_final[(~bike_locs_df_final.time.isin(tl)) & (bike_locs_df_final.day != 24)]

max(tl)
