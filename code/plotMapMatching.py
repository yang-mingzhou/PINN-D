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

# plot map matching results
def plot_edge_list_dataframe(network_gdf, edge_tu_list,df,cnt_out,cnt_in):
    directory = '../results/mapMatchingImages/'+str(cnt_out)
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