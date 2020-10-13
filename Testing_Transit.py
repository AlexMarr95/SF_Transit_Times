# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 19:41:37 2019

@author: Python
"""

STime = pd.read_csv("C:/Users/Python/Desktop/San Francisco Commutes/Transit_Times.csv")
STime['name'] = "home_" + STime['UCI ID'].astype(str)
STime = STime.merge(Student_Addresses,"left","name")

This = STime[STime['name'] == "home_2386"]
#This = STime.sample(1)
print(This[['Latitude','Longitude']])
This = This.transpose()

SPath = Gm.get_shortest_paths(This.ix['name'].values[0],to = School_Names,weights = 'weight')
SPath = pd.DataFrame(SPath)


names = pd.Series(Gm.vs["name"])
for i in range(SPath.shape[1]):
    print(i)
    SPath[i] = names[SPath[i]].values

SPath.index = School_Names
School = 'sunset es'
This2 = pd.DataFrame(SPath.ix[School][SPath.fillna("").ix[School].str.contains('Bus_Stop') ].str.replace("Bus_Stop_",""))
This2.reset_index(inplace = True,drop = True)
This2[0] = This2[School]
This2[1] = This2[School].shift(-1)
This2['new_route'] = This2[0].astype(str) + "_" + This2[1].astype(str)
This2 = This2[This2.index % 2 == 0]
This2 = This2.merge(Stop_Names,"left","new_route")
print(This2[['route_short_name','stop_name_x','stop_name_y']])
print(This.ix[['Latitude','Longitude']])