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

# trip division & bounding box
bbox_list = []
for f, m, n in walk(r'D:/data/Baseline Data (Murphy)'):
    if not m and n: # obd file
        #print(f,n)
        for i in range(0,len(n),2):
            fn_gps = n[i+1]
            filename_gps = os.path.join(f,fn_gps)
            print(filename_gps)
            raw_data_gps = pd.read_csv(filename_gps, header=0)
            raw_data_gps.drop(raw_data_gps.index[0],inplace=True)
            raw_data_gps= raw_data_gps.astype("float")
            raw_data_gps = raw_data_gps[(raw_data_gps.gps_Longitude>=-180) & (raw_data_gps.gps_Longitude<=180) & (raw_data_gps.gps_Latitude>=-90) & (raw_data_gps.gps_Latitude<=90)]
            if raw_data_gps.empty:
                print('empty gps file:'+ filename_gps)
                continue
            else:
                fn_obd = n[i]
                filename_obd = os.path.join(f,fn_obd)
                print(filename_obd)
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
                    bbox_list.append([west,east,south,north])
                print(len(bbox_list))
                del raw_data_obd
                gc.collect()
            del raw_data_gps
            gc.collect()
bbox_df = pd.DataFrame(bbox_list)
bbox_df.columns = ['west','east','south','north']
bbox_df.to_csv("./bbox.csv",index=False)
bbox_df.describe()

bbox_df = pd.read_csv("./bbox.csv")
bbox_df.head()

west_bound,east_bound,south_bound,north_bound = bbox_list_old[0], bbox_list_old[1], bbox_list_old[2], bbox_list_old[3]
bbox_df['inside'] = bbox_df.apply(lambda x: x['west'] >= west_bound and x['south'] >= south_bound and x['east'] <= east_bound and x['north'] <= north_bound,axis = 1)

# trajectory generation
bbox_list = [-95.35305,-91.544,43.1759,45.891999999999996] # old map
filepath = './trajectory_old_sampled2s'
if not os.path.exists(filepath):
    os.makedirs(filepath)
for f, m, n in walk(r'D:/data/Baseline Data (Murphy)'):
    if not m and n: # obd file
        #print(f,n)
        for i in range(0,len(n),2):
            fn_gps = n[i+1]
            filename_gps = os.path.join(f,fn_gps)
            #print(filename_gps)
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
                print(filename_obd)
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
                        #print('yes')
                        df_sub_sampled = raw_data_gps[beg:end:20]
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

# plot map matching results
def plot_edge_list_dataframe(edge_tu_list,df,cnt_out,cnt_in):
    directory = './match_image_old/'+str(cnt_out)
    print(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    long_edge = []
    lat_edge = []
    lat_data = df['gps_Latitude'].tolist()
    long_data = df['gps_Longitude'].tolist()
    for i in edge_tu_list:
        data = network_gdf.loc[i]
        if 'geometry' in data:
            xs, ys = data['geometry'].xy
            z = list(zip(xs, ys))
            l1 = list(list(zip(*z))[0])
            l2 = list(list(zip(*z))[1])
            #long_point.append(l1[0])
            #lat_point.append(l2[0])
            for j in range(len(l1)):
                long_edge.append(l1[j])
                lat_edge.append(l2[j])
    #long_point.append(l1[-1])
    #lat_point.append(l2[-1])
    fig = go.Figure(go.Scattermapbox(
        name = "Raw Data",
        mode = "markers",
        lon = long_data,
        lat = lat_data,
        marker = {'size': 7, 'color':"black"}))
        #line = dict(width = 4.5, color = 'blue')))
        # adding source marker
    fig.add_trace(go.Scattermapbox(
        name = "Matched Path",
        mode = "lines",
        lon = long_edge,
        lat = lat_edge,
        marker = {'size': 5, 'color':"red"},
        line = dict(width = 3, color = 'red')))
    # getting center for plots:
    lat_center = np.mean(lat_edge)
    long_center = np.mean(long_edge)
    zoom = 9.5
    # defining the layout using mapbox_style
    fig.update_layout(mapbox_style="stamen-terrain",
        mapbox_center_lat = 30, mapbox_center_lon=-80)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0},
                    mapbox = {
                            'center': {'lat': lat_center,
                            'lon': long_center},
                            'zoom': zoom})
    plotly.offline.plot(fig,filename = directory+'/trajecoty'+str(cnt_in)+'.html',auto_open=False)
    #pio.write_image(fig,'./match_005_image/'+str(cnt)+'image'+str(zoom)+'.png')
    #fig.show()
    del fig
    gc.collect()

bbox_list = [-95.35305,-91.544,43.1759,45.891999999999996] # old map
filepath = './trajectory_old'
if not os.path.exists(filepath):
    os.makedirs(filepath)
cnt_file = -1
f_t = r'./trajectory_old_sampled2s'
f_m = r'./match_result_old'
for f, m, n in walk(r'D:/data/Baseline Data (Murphy)'):
    if not m and n: # obd file
        #print(f,n)
        for i in range(0,len(n),2):
            fn_gps = n[i+1]
            filename_gps = os.path.join(f,fn_gps)
            #print(filename_gps)
            raw_data_gps = pd.read_csv(filename_gps, header=0)
            raw_data_gps.drop(raw_data_gps.index[0],inplace=True)
            raw_data_gps= raw_data_gps.astype("float")
            raw_data_gps = raw_data_gps[(raw_data_gps.gps_Longitude>=-180) & (raw_data_gps.gps_Longitude<=180) & (raw_data_gps.gps_Latitude>=-90) & (raw_data_gps.gps_Latitude<=90)]
            if raw_data_gps.empty:
                print('empty gps file:'+ filename_gps)
                continue
            else:
                cnt_file += 1
                if cnt_file >= 0:
                    # read trajectory data
                    df_traj = read_trajectory(f_t,cnt_file)
                    # read match results
                    df_matched = read_matched(f_m,cnt_file)
                    df_matched.sort_values(by=['id'],inplace=True)
                    df_matched["opath_list"] = df_matched['opath'].apply(lambda x: extract_opath(x))
                    for j in range(len(df_traj)):
                        beg = df_traj.loc[j,'beg']
                        end = df_traj.loc[j,'end']
                        print(cnt_file,beg,end)
                        segment = list(df_matched.iloc[j,4])
                        seg_drop_dup = []
                        seg_drop_dup.append(segment[0])
                        for k in range(1,len(segment)):
                            if segment[k] != segment[k-1]:
                                seg_drop_dup.append(segment[k])
                        df_sub = raw_data_gps[beg:end:20]
                        plot_edge_list_dataframe(seg_drop_dup,df_sub,cnt_file,j)
            break
        break