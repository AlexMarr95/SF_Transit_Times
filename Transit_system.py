# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 19:39:58 2019

@author: Python
"""

#Import Packages
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import geopandas as gpd
from shapely.geometry import LineString, Point,Polygon
import matplotlib.pyplot as plt
import igraph
from sklearn.neighbors import DistanceMetric
from haversine import haversine
from scipy.special import comb
from itertools import combinations
import itertools
from sklearn.neighbors import DistanceMetric
###############################################################################
#Read all files in the raw data file folder
mypath = "C:/Users/Python/Desktop/San Francisco Commutes/San_Francisco_Transit/"
routes = pd.read_csv(mypath + 'routes.txt', sep=",", header=0)
shapes = pd.read_csv(mypath + 'shapes.txt', sep=",", header=0)
stops = pd.read_csv(mypath + 'stops.txt', sep=",", header=0)
trips = pd.read_csv(mypath + 'trips.txt', sep=",", header=0)
times = pd.read_csv(mypath + 'stop_times.txt', sep=",", header=0)
calendar = pd.read_csv(mypath + 'calendar.txt',sep=",", header=0)

###############################################################################
#Keep only trip data from weekday service
calendar = calendar[(calendar.monday == 1) & (calendar.tuesday == 1) & (calendar.wednesday == 1) & (calendar.thursday == 1) & (calendar.friday == 1)]
trips = trips[trips.service_id.isin(calendar.service_id)]
trips = trips[trips['service_id'] == '1_merged_8846470']
#Merge trips and stop time information (inner merge so we drop non-weekday trips)
times = times.merge(trips[['trip_id','route_id','direction_id','service_id','trip_headsign']],"inner",['trip_id'])
times = times.merge(stops[['stop_id','stop_lat','stop_lon']],"inner",['stop_id'])


#Keep only trips that happen between 6 and 10 am
times['departure_time'] = pd.to_datetime(times['departure_time'] ,format= '%H:%M:%S',errors = 'coerce')
times['arrival_time'] = pd.to_datetime(times['arrival_time'] ,format= '%H:%M:%S',errors = 'coerce')

times = times[(times['departure_time'] >= pd.to_datetime('06:00:00',format = '%H:%M:%S')) &(times['departure_time'] <= pd.to_datetime('10:00:00',format = '%H:%M:%S')) ]



times['new_stop_id'] = times['stop_id'].astype(str) + ","
times.sort_values(['trip_id','stop_sequence'],inplace = True)
Specific_Routes = times.groupby(['trip_id','route_id'],as_index = False)['new_stop_id'].sum()
Route_Nums = Specific_Routes[['route_id','new_stop_id']].drop_duplicates()
Route_Nums['Route_Num'] = 1
Route_Nums['Route_Num'] = Route_Nums.groupby('route_id')['Route_Num'].cumsum().values


Route_Nums['combos'] = [list(combinations(x.split(",")[:-1],2)) for x in Route_Nums['new_stop_id']]
Combos = Route_Nums.apply(lambda x : [xs + (x['route_id'],x['Route_Num'],) for xs in x['combos']] , axis = 1)
Combos = list(itertools.chain.from_iterable(Combos))
Combos = pd.DataFrame(Combos)
Combos.columns = ['stop_1','stop_2','route_id','Route_Num']

Route_Nums = Route_Nums.merge(Specific_Routes[['trip_id','new_stop_id','route_id']],how = "left", on = ['new_stop_id','route_id'])
times = times.merge(Route_Nums[['trip_id','Route_Num']],how = "left", on = 'trip_id')


Combos['stop_id'] = Combos['stop_1']
Combos = Combos.merge(times, "left", ['route_id','Route_Num','stop_id'])
Combos['stop_id'] = Combos['stop_2']
Combos = Combos.merge(times, "left", ['route_id','Route_Num','stop_id','trip_id'])
Combos['travel_time'] = (Combos['arrival_time_y'] -Combos['departure_time_x']).dt.total_seconds()/60
Combos['new_route'] = Combos['stop_1'].astype(str) + "_" + Combos['stop_2'].astype(str)
Combos.sort_values(['new_route','departure_time_x'],inplace = True)
Combos['departure_time_z'] = Combos.groupby('new_route')['departure_time_x'].shift(1).values 
Combos['wait_time'] = (Combos['departure_time_x'] -Combos['departure_time_z']).dt.total_seconds()/60

Weights = Combos.groupby(['new_route','stop_1','stop_2','stop_lat_x','stop_lon_x','stop_lat_y','stop_lon_y'],as_index = False)[['wait_time','travel_time']].mean()
Weights['Weights'] = Weights['travel_time'] + (Weights['wait_time']/2)



Combos = Combos[['new_route','route_id','stop_1','stop_2']].drop_duplicates()
Combos = Combos.merge(routes[['route_id','route_short_name']],"left",'route_id')
Combos['route_short_name'] = Combos['route_short_name'] + ","

Stop_Names = Combos.groupby(['new_route','stop_1','stop_2'],as_index = False)[['route_short_name']].sum()
Stop_Names['stop_id'] = Stop_Names['stop_1']
Stop_Names = Stop_Names.merge(stops[['stop_id','stop_name']],"left","stop_id")
Stop_Names['stop_id'] = Stop_Names['stop_2']
Stop_Names = Stop_Names.merge(stops[['stop_id','stop_name']],"left","stop_id")

Weights.to_pickle("C:/Users/Python/Desktop/San Francisco Commutes/bus_map")
Stop_Names.to_pickle("C:/Users/Python/Desktop/San Francisco Commutes/stop_names")
###############################################################################
#Get walking times in minutes between each stop and each school
#Read in the San Francisco base map from the San Francisco website
df = gpd.read_file('C:/Users/Python/Desktop/San Francisco Commutes/SF_Base_Map/SF_Base_Map.shp')
df = df[df['layer'].isin(['streets','Streets_Pedestri','STREETS','streets_ybi','streets_ti','Streets_HuntersP','PARKS','Parks_NPS_Presidio'])]
df.plot()

df['len'] = [len(x.coords) for x in df['geometry']]
Test = df[df['len'] > 2]

dis = []
for i in range(len(Test)) : 
 print(i)
 this = Test['geometry'].iloc[i].coords
 for j in range(len(this)-1) :   
  dis.append(LineString(this[j:j+2]))
  
df = df[df['len'] == 2][['geometry']]
df = df.append(pd.DataFrame(dis,columns = ['geometry']))
df.reset_index(drop = True,inplace = True)

def to_grid(total_bounds,s):
    xlength =  np.arange(total_bounds[0],total_bounds[2] + .00001 ,(total_bounds[2] - total_bounds[0])/s)
    ylength = np.arange(total_bounds[1],total_bounds[3]+ .00001,(total_bounds[3] - total_bounds[1])/s)
    polygons = []
    for j in range((s)):
     for i in range((s)):
        polygons.append(Polygon([(xlength[0+i],ylength[0+j]),(xlength[1+i],ylength[0+j]),(xlength[1+i],ylength[1+j]),(xlength[0+i],ylength[1+j])]))
    return polygons


def check_in_polygon(data,polygons,suffix):
    temp_df = pd.DataFrame()
    for j in range(len(polygons)):
        print(j)
        temp = []
        for i in range(len(df)):
            this = polygons[j].contains(data['geometry'][i]) |data['geometry'][i].intersects(polygons[j])
            temp.append(this)
        temp_df['%s_%i'%(suffix,j)] = temp
    return temp_df
    
polygons = to_grid(df['geometry'].total_bounds,8)
   
new_df = check_in_polygon(df,polygons,'0')
new_df.columns = range(len(new_df.columns))
new_df['geometry'] = df['geometry']
del df

dis = []
for p in range(new_df.shape[1]-1): 
    df2 = new_df[new_df[p]==1]
    for l in range(len(df2)):
        print(p,l)
        for i in range((l+1),len(df2)):
            try: 
                this = df2['geometry'].iloc[l].intersection(df2['geometry'].iloc[i])
                if str(this) != 'GEOMETRYCOLLECTION EMPTY':
                    coords1 = df2['geometry'].iloc[l].coords
                    coords2 = df2['geometry'].iloc[i].coords
                    if this.type == 'LineString':
                        if df2['geometry'].iloc[l] == df2['geometry'].iloc[i]:
                            dis.append([coords1[0],coords1[1]])
                        else:
                            if df2['geometry'].iloc[l].length >= df2['geometry'].iloc[i].length:
                                dis.append([coords1[0],coords1[1]])
                            else:
                                dis.append([coords2[0],coords2[1]])
                    if this.type == 'Point':
                        if coords1[0] != this.coords[0]:
                            dis.append([coords1[0],this.coords[0]])
                        if coords1[1] != this.coords[0]:
                            dis.append([coords1[1],this.coords[0]])
                        if coords2[0] != this.coords[0]:
                            dis.append([coords2[0],this.coords[0]])
                        if coords2[1] != this.coords[0]:
                            dis.append([coords2[1],this.coords[0]])
                        
            except:
             pass

dis = pd.DataFrame(dis)

dis['Time']= (((dis.apply(lambda x: haversine(x[0], x[1]), axis = 1)) * 0.621371)/3)*60
dis['point1'] = [str(x[0]) + ',' + str(x[1]) for x in dis[0]]
dis['point2'] = [str(x[0]) + ',' + str(x[1]) for x in dis[1]]


dis.to_pickle("C:/Users/Python/Desktop/San Francisco Commutes/Walking_Map")

###############################################################################
#Make sure walking map is all reachable
Walking_Graph = pd.read_pickle("C:/Users/Python/Desktop/San Francisco Commutes/Walking_Map")
tuples = [tuple(x) for x in Walking_Graph[['point1','point2','Time']].values]
Gm = igraph.Graph.TupleList(tuples, directed = False,vertex_name_attr='name', edge_attrs = ['weight'])
names = pd.Series(Gm.vs["name"])


STime = pd.DataFrame(Gm.shortest_paths(names.sample(1).values[0],target =names.values.tolist(), weights = 'weight'))
STime.columns = names.values.tolist()
This = STime.max()[STime.max() == np.inf].index.tolist()
This = pd.DataFrame(This)
This['long'] = [x.split(",")[0] for x in This[0]]
This['lat'] = [x.split(",")[1] for x in This[0]]
This.sort_values('lat',inplace = True)
This1 = This[This['lat'].astype(float) < 37.808259288] 
This2 = This[This['lat'].astype(float) >= 37.808259288] 

Walking_Graph = Walking_Graph[~((Walking_Graph['point1'].isin(This1[0].tolist())) | (Walking_Graph['point2'].isin(This1[0].tolist())) )]
Walking_Graph.to_pickle("C:/Users/Python/Desktop/San Francisco Commutes/Walking_Map")

###############################################################################

Student_Addresses = pd.read_excel("C:/Users/Python/Desktop/San Francisco Commutes/Census_Tracts_Addresses.xlsx")
Student_Addresses['name'] = "home_" + Student_Addresses['UCI_ID'].astype(str)
Student_Addresses = Student_Addresses[['name','Latitude','Longitude']]
Home_Names = Student_Addresses['name'].values.tolist()

School_Addresses = pd.read_excel("C:/Users/Python/Desktop/San Francisco Commutes/School_Adresses.xlsx")
School_Addresses =School_Addresses[['Field1','Y','X']]
School_Names = School_Addresses['Field1'].values.tolist()

Bus_Stops = stops[['stop_id','stop_lat','stop_lon']]
Bus_Stops['stop_id'] = "Bus_Stop_" + Bus_Stops['stop_id'].astype(str)

Walking_Points_1 = pd.DataFrame(Student_Addresses.values.tolist() + School_Addresses.values.tolist() + Bus_Stops.values.tolist())
latlon_walk_1 = Walking_Points_1[[1,2]]
latlon_walk_1= np.radians(latlon_walk_1)

#Get distances between bus stops and walking points
Walking_points_2 = pd.read_pickle("C:/Users/Python/Desktop/San Francisco Commutes/Walking_Map")
Walking_points_2 = pd.DataFrame(list(set(Walking_points_2[0].values.tolist() + Walking_points_2[1].values.tolist())))
latlon_walk_2 = Walking_points_2[[1,0]]
latlon_walk_2= np.radians(latlon_walk_2)



hav = DistanceMetric.get_metric('haversine')
dists = pd.DataFrame()
Closest_Walking_Point = pd.DataFrame()
for i in np.arange(0,len(latlon_walk_1),1000):
 print(i)
 dists = pd.DataFrame(hav.pairwise(latlon_walk_1.iloc[i:i+1000,:],latlon_walk_2))
 temp =  pd.DataFrame(dists.astype(float).idxmin(1))
 temp['dist'] = (((dists.astype(float).min(1))*.621371)/3)*60
 Closest_Walking_Point = Closest_Walking_Point.append(temp)


Closest_Walking_Point.reset_index(drop = True, inplace = True) 
Closest_Walking_Point[['Walk_Long','Walk_Lat']] = Walking_points_2.ix[Closest_Walking_Point[0].values.tolist()].reset_index(drop = True)
Closest_Walking_Point['name'] = Walking_Points_1[0]
Closest_Walking_Point['walk_name'] = Closest_Walking_Point.apply(lambda x: str(x['Walk_Long']) + ',' + str(x['Walk_Lat']), axis = 1)
Closest_Walking_Point.to_pickle("C:/Users/Python/Desktop/San Francisco Commutes/Closest_Walking_Point")





###############################################################################
Walking_Graph = pd.read_pickle("C:/Users/Python/Desktop/San Francisco Commutes/Walking_Map")
Closest_Walking_Point = pd.read_pickle("C:/Users/Python/Desktop/San Francisco Commutes/Closest_Walking_Point")
Bus_Graph = pd.read_pickle("C:/Users/Python/Desktop/San Francisco Commutes/bus_map")
Bus_Graph['stop_1'] = "Bus_Stop_" + Bus_Graph['stop_1'].astype(str)
Bus_Graph['stop_2'] = "Bus_Stop_" + Bus_Graph['stop_2'].astype(str)
Bus_Graph = Bus_Graph[~Bus_Graph['Weights'].isna()]



tuples = [tuple(x) for x in Walking_Graph[['point1','point2','Time']].values]
tuples = tuples + [tuple(x) for x in Walking_Graph[['point2','point1','Time']].values]
tuples = tuples + [tuple(x) for x in Closest_Walking_Point[['name','walk_name','dist']].values]
tuples = tuples + [tuple(x) for x in Closest_Walking_Point[['walk_name','name','dist']].values]
tuples = tuples + [tuple(x) for x in Bus_Graph[['stop_1','stop_2','Weights']].values]

Gm = igraph.Graph.TupleList(tuples, directed = True,vertex_name_attr='name', edge_attrs = ['weight'])



###############################################################################
#Calculate Time
STime = pd.DataFrame(Gm.shortest_paths(Home_Names,target = School_Names, weights = 'weight'))
STime.columns = School_Names
STime.to_pickle("C:/Users/Python/Desktop/San Francisco Commutes/Transit_Times_2")
STime.index = Home_Names
STime.reset_index(drop = False,inplace = True)
STime['index'] = STime['index'].str.replace('home_','')
STime['index'] = STime['index'].astype(int)
STime.to_csv("C:/Users/Python/Desktop/San Francisco Commutes/Transit_Times_2.csv")
###############################################################################
