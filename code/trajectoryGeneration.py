import pandas as pd
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import psycopg2
import datetime
import plotly.io as pio
import osmnx as ox
import time
from shapely.geometry import Polygon
import os
import gc
from os import walk
import geopandas as gpd
import plotly


# trajectory generation
bbox_list = [-95.35305,-91.544,43.1759,45.891999999999996]
# output file folder
filepath = '../data/trajectorySampledEvery3s'
if not os.path.exists(filepath):
    os.makedirs(filepath)

trajectoryCnt = 0
for f, m, n in walk(r'../datasets/Baseline Data (Murphy)'):
    if not m and n: # obd file
        print(f,n)
        print(sorted(n))
        for i in range(0,len(n),2):
            fn_gps = n[i+1]
            filename_gps = os.path.join(f,fn_gps)
            print("filename_gps",filename_gps)
            raw_data_gps = pd.read_csv(filename_gps, header=0)
            raw_data_gps.drop(raw_data_gps.index[0],inplace=True)
            raw_data_gps= raw_data_gps.astype("float")
            raw_data_gps = raw_data_gps[(raw_data_gps.gps_Longitude>=-180) & (raw_data_gps.gps_Longitude<=180) & (raw_data_gps.gps_Latitude>=-90) & (raw_data_gps.gps_Latitude<=90)]
            if raw_data_gps.empty:
                print('empty gps file:'+ filename_gps)
                continue
            else:
                cnt = 0
                list_wkt = []
                fn_obd = n[i]
                filename_obd = os.path.join(f,fn_obd)
                print("filename_obd",filename_obd)
                assert filename_obd[-11:-4] == filename_gps[-15:-8]
                raw_data_obd = pd.read_csv(filename_obd, header=0,usecols=['File_index'])
                raw_data_obd.drop(raw_data_obd.index[0],inplace=True)
                raw_data_obd = raw_data_obd.loc[raw_data_gps.index].reset_index(drop = True)
                raw_data_obd= raw_data_obd.astype("int")
                raw_data_gps = raw_data_gps.reset_index(drop = True)
                raw_data_obd.iloc[0,0] = 1
                raw_data_obd.iloc[-1,0] = 1
                file_index = raw_data_obd[raw_data_obd.File_index==1].index
                file_a = file_index[:-1]
                file_b = file_index[1:]
                for j in range(len(file_a)):
                    beg = file_a[j]
                    end = file_b[j]
                    west,east,south,north = min(raw_data_gps[beg:end].gps_Longitude), max(raw_data_gps[beg:end].gps_Longitude), min(raw_data_gps[beg:end].gps_Latitude),max(raw_data_gps[beg:end].gps_Latitude)
                    if west >= bbox_list[0] and east <= bbox_list[1] and south >= bbox_list[2] and north <= bbox_list[3]:
                        trajectoryCnt += 1
                        #print('yes')
                        df_sub_sampled = raw_data_gps[beg:end:30]
                        #print(df_sub_sampled.head())
                        df_sub_sampled.drop_duplicates(subset=['gps_Latitude','gps_Longitude'],inplace=True)
                        if len(df_sub_sampled) >= 2:
                            cnt+= 1
                            wkt = "LINESTRING("
                            for k in df_sub_sampled.index:
                                if k == df_sub_sampled.index[-1]:
                                    wkt += str(df_sub_sampled.loc[k,'gps_Longitude']) + " " + str(df_sub_sampled.loc[k,'gps_Latitude']) + ")"
                                else:
                                    wkt += str(df_sub_sampled.loc[k,'gps_Longitude']) + " " + str(df_sub_sampled.loc[k,'gps_Latitude']) + ","
                            list_wkt.append([cnt,wkt,beg,end])
                del raw_data_obd
                gc.collect()
                df_wkt = pd.DataFrame(list_wkt,columns=['id','geom','beg','end'])
                df_wkt.to_csv(os.path.join(filepath, 'trips_sub_sampled_'+str(n[i])),sep=";",index=False,line_terminator='\n')
            del raw_data_gps
            gc.collect()

print("trajectory count:", trajectoryCnt)
# 3646 trajectories 3/17/2022
