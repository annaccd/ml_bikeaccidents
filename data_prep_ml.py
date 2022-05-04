import pandas as pd
import datetime
import pickle
import locale
import shapely as shp
import geopandas as gpd
import os


roads = pd.read_csv("roads_with_accs_and_traffic.csv")
roads['geometry'] = roads['geometry'].apply(shp.wkt.loads)
roads = gpd.GeoDataFrame(roads, crs='epsg:4326')

roads['geometry_wkt'] = [i.to_wkt() for i in roads['geometry']]

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
measured_traf['Day'] = measured_traf['Date'].dt.day


b = [0,6,12,18,24]
l = ['Night', 'Morning', 'Afternoon','Evening']
measured_traf['time_p'] = pd.cut(measured_traf['Hour'], bins=b, labels=l, include_lowest=True)

measured_traf = measured_traf[~measured_traf.Zählstelle.isin(['05-FK-OBB-O', '05-FK-OBB-W', '03-MI-SAN-O', '06-FK-FRA-O', '18-TS-YOR-W', '18-TS-YOR-O'])]

unique_mlocs = measured_traf.groupby(['Zählstelle', 'Year', 'Month', 'Day', 'time_p']).agg(day_sum = ('value', sum)).reset_index().groupby(['Zählstelle', 'Month', 'time_p']).mean().reset_index()
sum_df = unique_mlocs.groupby(['Zählstelle', 'Month']).day_sum.sum()

unique_mlocs = unique_mlocs.merge(sum_df, on=['Zählstelle', 'Month'])

unique_mlocs['share'] = unique_mlocs.day_sum_x/unique_mlocs.day_sum_y


april = unique_mlocs[unique_mlocs['Month'] == 4]

months = unique_mlocs.Month.unique()
unique_mlocs['scale_m_tp'] = [0] * len(unique_mlocs)
for i in months:
    unique_mlocs.loc[unique_mlocs['Month'] == i, 'scale_m_tp'] = april['day_sum_x'].values/unique_mlocs[unique_mlocs['Month'] == i]['day_sum_x'].values

# create bin distributions for nextbike data

unique_mlocs = unique_mlocs.merge(measure_locs, on ='Zählstelle')


roads_non_zero = roads[(roads['n_accidents'] > 0) & (roads['traffic'] > 3)]
unique_mlocs['geometry_wkt'] = ''
for i, row in unique_mlocs.iterrows():
    x, y = row['Längengrad'], row['Breitengrad']
    point = shp.geometry.Point(x, y)   
    mini = min(roads_non_zero['geometry'], key=point.distance)
    unique_mlocs['geometry_wkt'].loc[i] = mini.to_wkt()
    print(f'{i}')

unique_mlocs = unique_mlocs.merge(roads, on='geometry_wkt')



unique_mlocs['scale_loc'] = unique_mlocs['day_sum_x']/unique_mlocs['traffic']*unique_mlocs['share']


trips_df = pd.read_csv("trips_by_dist.csv")
trips_df['geometry'] = trips_df['geometry'].apply(shp.wkt.loads)
trips_df = gpd.GeoDataFrame(trips_df, crs='epsg:4326')

trips_df['geometry_wkt'] = [i.to_wkt() for i in trips_df['geometry']]

trips_dist = trips_df.groupby(['OTEIL', 'Month', 'time_p']).agg({'directions': 'count'}).reset_index().rename(columns={'directions':'traffic'})

district_shp = gpd.read_file('lor_ortsteile.geojson')

trips_dist = trips_dist.merge(district_shp, on='OTEIL', how='left')

unique_mlocs['geometry'] = gpd.GeoSeries(unique_mlocs['geometry'], crs='epsg:4326')
unique_mlocs['center'] = [i.centroid for i in unique_mlocs['geometry']]

coord_m = unique_mlocs[['Zählstelle', 'center']].rename(columns={'center':'geometry'})
trips_dist = gpd.GeoDataFrame(trips_dist, crs='epsg:4326')
coord_m = gpd.GeoDataFrame(coord_m, crs='epsg:4326')
coord_m['geometry_wkt'] = [i.to_wkt() for i in coord_m['geometry']]

tri = gpd.sjoin(trips_dist, coord_m, op='intersects', how='left')

trips_dist['m_point'] = ''
for i, row in trips_dist.iterrows():
    mini = min(unique_mlocs['center'], key=row['geometry'].distance)
    trips_dist['m_point'].iloc[i] = unique_mlocs.loc[unique_mlocs['center'] == mini]['Zählstelle'].iloc[0]


### based on the assigned m_point now assign values

for d in trips_dist.OTEIL.unique():
    for z in trips_dist.Zählstelle.unique():
        nearest = min(coord_m['geometry'], key=)
        for m in unique_mlocs.Month.unique():
            for t in unique_mlocs.time_p.unique():
            [{'district': d, 'm_point': z, 'month': m, 'time_p' : t, 'traffic': }] 

for point in unique_mlocs['center']:
    x, y = waypoint['lng'], waypoint['lat']
    point = shp.geometry.Point(x, y)   
    mini = min(roads['geometry'], key=point.distance)
    roads.loc[roads['geometry_wkt'] == mini.to_wkt(),['traffic']] +=1
    print(f'waypoint {count} of {len(list(waypoints_l.keys()))}')


for i, row in trips_dist.iterrows():
    for m in 



