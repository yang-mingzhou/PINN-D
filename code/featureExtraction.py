import pandas as pd
import matplotlib.pyplot as plt
import osmnx as ox
# osmnx.version = 0.16.1
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
import math
from tqdm import tqdm
import copy
from PhysicsModel.Code.diesel_vehicleModel_MERL import *


def timeInterpolation(l: list):
    lSub = l[1:] - l[:-1]
    for i, v in enumerate(lSub):
        if v < 0.1:
            l[i + 1] = l[i] + 0.1
    return l


def read_data(filename, id):
    '''Read raw data.

    Read raw csv data. Remove unreasonable gps data points. Set the FileIndex of the beginning/end of the csv to 1.
    id: a range of index

    Args:
        filename of the raw csv data.

    Returns:
        A dataframe of a csv data.
    '''
    cols = ['Time_abs', 'Time', 'AmbAirTemp', 'BarPressure', 'EngineFuel', 'inclination', 'VehicleSpeed',
            'VehicleWeight', 'FileIndex']
    # df = pd.read_csv('test.csv', header=0, skiprows=2,usecols=['B,D,E,N,O,P,Q,T,U,W'],names=cols)0,2,3,8,12,15,16,18,-6,-5,-4,-3,-2,-1
    raw_data = pd.read_csv(filename, header=0, usecols=[0, 2, 3, 8, 12, 15, 16, 18, 19, 20, 21, 22, 23, 24],
                           parse_dates=[[8, 9, 10, 11, 12, 13]])
    raw_data.drop(raw_data.index[0], inplace=True)
    # df = df[::10].reset_index(drop = True)
    raw_data.columns = cols
    raw_data['Time_abs'] = raw_data['Time_abs'].apply(lambda x: x if '.' in x else x + '.0')
    raw_data['Time_abs'] = pd.to_datetime(raw_data['Time_abs'], format='%Y %m %d %H %M %S.%f')
    raw_data[['Time', 'AmbAirTemp', 'BarPressure', 'EngineFuel', 'inclination', 'VehicleSpeed', 'VehicleWeight',
              'FileIndex']] = raw_data[
        ['Time', 'AmbAirTemp', 'BarPressure', 'EngineFuel', 'inclination', 'VehicleSpeed', 'VehicleWeight',
         'FileIndex']].astype("float")
    raw_data['VehicleWeight'] = raw_data['VehicleWeight'].apply(
        lambda x: 12500 if x < 12500 else (36500 if x > 36500 else x))
    raw_data = raw_data.loc[id]
    raw_data.iloc[0, -1] = 1
    raw_data.iloc[-1, -1] = 1
    raw_data['Time'] = timeInterpolation(raw_data['Time'].values)
    # print(raw_data.shape)
    return raw_data


def read_trajectory(f, cnt):
    '''Read the No. cnt trajectory file in the folder f

    Args:
        f: File path of a folder
        cnt: Location of the file of interest

    Returns:
        A dataframe of the No. cnt file in the folder f
    '''
    path_list = os.listdir(f)
    fn = path_list[cnt]
    filename = f + "/" + fn
    print(filename)
    return pd.read_csv(filename, sep=";")


def readAndProcess_matched(f, cnt, edges):
    '''Read the No. cnt matched result file in the folder f

    Args:
        f: File path of a folder
        cnt: Location of the file of interest

    Returns:
        A dataframe of the No. cnt file in the folder f
    '''
    path_list = os.listdir(f)
    path_list.sort(key=lambda x: int(x[10:-4]))
    fn = path_list[cnt]
    filename = f + "/" + fn
    print(filename)
    df_matched = pd.read_csv(filename, sep=";")
    df_matched.sort_values(by=['id'], inplace=True)
    df_matched["opath_list"] = df_matched['opath'].apply(lambda x: extract_opath(x))
    df_matched["opath_list"] = df_matched['opath_list'].apply(lambda x: dropRedundant(x, edges))
    return df_matched


def extract_opath(opath):
    '''Divide the map-matched results connected by commas into a list.

    Args:
        opath: Map-matched results connected by commas

    Returns:
        A list of matched edges
    '''
    if (opath == ''):
        return []
    elif isinstance(opath, float):
        return int(opath)
    return [int(s) for s in opath.split(',')]


def dropRedundant(opList, edges):
    '''
    drop redundant result from the map matching results(e.g. a->b->a->b->c => a->b->c)
    '''
    opathList = copy.deepcopy(opList)
    length = len(opathList)
    i = 0
    flag = 0
    while i < length - 1:
        curU = edges.loc[opathList[i], 'u']
        curV = edges.loc[opathList[i], 'v']
        nextU = edges.loc[opathList[i + 1], 'u']
        nextV = edges.loc[opathList[i + 1], 'v']
        if nextU == curV:
            if nextV != curU:  # correct result
                flag = 1
            else:  # redundant result
                # print(i, opathList[i],opathList[i+1], curU, curV, nextU, nextV)
                if flag:  # current one is correct: replace next one as current one
                    opathList[i + 1] = opathList[i]
                else:  # donnot which one is correct
                    flag = 1
                    s = set([opathList[i], opathList[i + 1]])
                    for j in range(i + 1, length):  # find the next edge
                        if opathList[j] not in s:
                            break
                    if opathList[j] not in s:
                        jU = edges.loc[opathList[j], 'u']
                        if jU == nextV:  # reverse one is the correct one
                            valueI = opathList[i]
                            for k in range(i, -1, -1):
                                # print(opath_list[k])
                                if opathList[k] == valueI:
                                    opathList[k] = opathList[i + 1]
                                else:
                                    break
                        else:
                            opathList[i + 1] = opathList[i]
                    else:  # no next edge
                        opathList[i + 1] = opathList[i]
        elif opathList[i] != opathList[i + 1]:  # uncertain result (for equal results: just continue)
            flag = 0
        i += 1
    return opathList


def elevation_cal(df_sub, beg_i, end_i):
    '''Calcuate the elevation change of an edge

    Calcuate the elevation change of an edge, ignoring the data points with gps_Altitude >= 500 m. If there is not result, return None.

    Args:
        df_sub_gps: A dataframe of OBD data
        beg_i: Index of the origin of the edge in df_sub
        end_i: Index of the destination of the edge in df_sub

    Returns:
        Elevation change (m) of an edge
    '''
    flag_b = 0
    flag_e = 0
    for j in range(beg_i, end_i):
        if abs(df_sub.loc[j, 'gps_Altitude']) < 500:
            elevation_b = df_sub.loc[j, 'gps_Altitude']
            flag_b = 1
            break
    for j in range(end_i - 1, beg_i - 1, -1):
        if abs(df_sub.loc[j, 'gps_Altitude']) < 500:
            elevation_e = df_sub.loc[j, 'gps_Altitude']
            flag_e = 1
            break
    if flag_b and flag_e:
        elevation = elevation_e - elevation_b
        return float(elevation)
    else:
        return None


def highway_cal(network_seg):
    '''Calcuate the road type of an edge ('unclassified' in default)

    Args:
        Attributes of an edge

    Returns:
        Road type of the edge
    '''
    if 'highway' in network_seg and network_seg['highway']:
        if isinstance(network_seg['highway'], str):
            return network_seg['highway']
        elif isinstance(network_seg['highway'], list):
            return network_seg['highway'][0]
    else:
        return 'unclassified'


def length_cal(network_seg):
    '''Calcuate the length of an edge ('unknown' in default)

    Args:
        Attributes of an edge

    Returns:
        length (m) of the edge
    '''
    if 'length' in network_seg and network_seg['length']:
        if isinstance(network_seg['length'], float):
            return network_seg['length']
        elif isinstance(network_seg['length'], list):
            return network_seg['length'][0]
    else:
        return 'unknown'


def speedlimit_cal(network_seg, highway):
    '''Calcuate the speedlimit of an edge (30*1.609 km/h in default)

    Args:
        Attributes of an edge

    Returns:
        Speedlimit (km/h) of the edge
    '''
    if 'maxspeed' in network_seg and network_seg['maxspeed']:
        res = ''
        flag = 0
        for i in list(network_seg['maxspeed']):
            if i.isdigit():
                flag = 1
                res += i
            else:
                if flag == 1:
                    speed = int(res) * 1.609
                    return speed
    elif highway == "motorway":
        return 55 * 1.609
    elif highway == "motorway_link":
        return 50 * 1.609
    return 30 * 1.609


def average_speed_cal(df_sub, beg_i, end_i):
    '''Calcuate the average speed of an edge (None in default)

    Drop the data points with VehicleSpeed = 0.

    Args:
        df_sub_obd: A dataframe of OBD data
        beg_i: Index of the origin of the edge in df_sub
        end_i: Index of the destination of the edge in df_sub

    Returns:
        Average speed (km/h) of the edge
        A list of Speed
    '''
    speed = list()
    speed_sum = 0.0
    count = 0
    flag = 0
    time = 0
    # print(df_sub.head())
    for i in range(beg_i, end_i):
        '''
        if (i-beg_i)%10 == 0:
            speed.append(df_sub.loc[i,'VehicleSpeed'])
        '''
        # print(i)
        if df_sub.loc[i, 'VehicleSpeed'] == 0:
            continue
        else:
            speed_sum += df_sub.loc[i, 'VehicleSpeed']
            speed.append(df_sub.loc[i, 'VehicleSpeed'])
            time += 0.1
            count += 1
        if flag == 0 and df_sub.loc[i, 'VehicleSpeed'] >= 15:
            flag = 1
    if not flag:
        return None, speed, time
    else:
        return speed_sum / count, speed[::10], time


def energy_consumption_cal(df_sub, beg_i, end_i):
    '''Calcuate the fuel consumption of an edge (0 in default)

    Drop the data points with VehicleSpeed = 0.

    Args:
        df_sub_obd: A dataframe of OBD data
        beg_i: Index of the origin of the edge in df_sub
        end_i: Index of the destination of the edge in df_sub

    Returns:
        Fuel consumption (liter) of the edge
    '''
    energy_sum = 0.0
    count = 0
    for i in range(beg_i, end_i):
        if df_sub.loc[i, 'VehicleSpeed'] == 0:
            continue
        else:
            count += 1
            energy_sum += df_sub.loc[i, 'EngineFuel']
    return energy_sum, count


def ori_cal(coor_a, coor_b, coor_c):
    '''Calcuate the orientation change from vector ab to bc (0 in default, right turn > 0, left turn < 0)

    10 degree is the threshold of a turn.

    Args:
        coor_a: coordinate of point a
        coor_b: coordinate of point b
        coor_c: coordinate of point c

    Returns:
        's': straight
        'l': left-hand turn
        'r': right-hand turn
    '''
    a = np.array(coor_a)
    b = np.array(coor_b)
    c = np.array(coor_c)
    v_ab = b - a
    v_bc = c - b
    cosangle = v_ab.dot(v_bc) / (np.linalg.norm(v_bc) * np.linalg.norm(v_ab) + 1e-16)
    return math.acos(cosangle) * 180 / np.pi if np.cross(v_ab, v_bc) < 0 else -math.acos(cosangle) * 180 / np.pi


def simulateFuelConsumption(df, truckId):
    df = df.rename(columns={'Time_abs': 'Time_abs', 'Time': 't', 'AmbAirTemp': 'T', \
                            'BarPressure': 'p', 'EngineFuel': 'fr', \
                            'inclination': 'th', 'VehicleSpeed': 'v', 'VehicleWeight': 'm', 'FileIndex': 'FileIndex'})
    # print(df['th'])
    df['th'] = 0
    df['V'] = 0
    df['SOC'] = 0
    df['E'] = 0
    df['v'] = df['v'] / 3.6
    df = get_data(df)
    # fileFolder = 'PhysicsModel/Code'
    fileFolder = './PhysicsModel/Code'
    veh = get_veh(fileFolder, truckId)
    output = sim_drive(df, veh, plots=0)
    output = pd.DataFrame.from_dict(output).reset_index()
    output['fuel_est'] = output['fuel_est'] * 3.7854

    return output['fuel_est'].values


def post_process(raw_data_gps, raw_data_obd, beg, end, segment, cnt_out, cnt_in, df_edge, network_gdf, truckId):
    '''Calcuate the features of edges in a trip

    Args:
        raw_data_gps,raw_data_obd: GPS data and OBD data
        beg: Index of the origin of the trip in raw_data
        end: Index of the destination of the trip in raw_data
        segment: map matched result of the trip
        cnt_out: count of the csv file
        cnt_in: count of the trip in the csv file
        df_edge: result dataframe, containing the features of each edge
        network_gdf: geodataframe of the network


    Returns:
        A dataframe contains the features of each edge
    '''
    trip_id = (cnt_out, cnt_in)
    seg_drop_dup = []
    seg_drop_dup.append(segment[0])
    for j in range(1, len(segment)):
        if segment[j] != segment[j - 1]:
            seg_drop_dup.append(segment[j])
    df_sub_gps = raw_data_gps[beg:end]
    # print(len(df_sub_gps))
    df_sub_obd = raw_data_obd[beg:end]
    # add simulated function here
    df_sub_obd['fuel_est'] = simulateFuelConsumption(df_sub_obd, truckId)
    df_sub_obd['fuel_est'].fillna(method="ffill", inplace=True)
    # print(df_sub_obd['fuel_est'])
    # print(df_sub_obd)
    # print(len(df_sub_obd['fuel_est']))
    df_sub_sampled = df_sub_gps[::30]
    # plot_edge_list_dataframe(seg_drop_dup,df_sub,cnt_out,cnt_in)
    data_sampled_index = list(df_sub_sampled.index)
    # print(data_sampled_index)
    intersection_index = [data_sampled_index[0]]
    # print(len(segment))
    for i in range(1, len(segment)):
        if segment[i] != segment[i - 1]:
            intersection_index.append(data_sampled_index[i])
    intersection_index.append(df_sub_gps.index[-1])
    # print(intersection_index)
    inter_f = intersection_index[:-1]
    inter_l = intersection_index[1:]
    position = 0
    flag = 0
    for i in range(len(inter_f)):
        position += 1
        beg_i = inter_f[i]
        end_i = inter_l[i]
        # print(beg)
        # print(beg_i,end_i)
        average_speed, speed_list, time = average_speed_cal(df_sub_obd, beg_i, end_i)
        siumlatedEnergyConsumption = df_sub_obd.loc[end_i, 'fuel_est'] - df_sub_obd.loc[beg_i, 'fuel_est']
        # print(siumlatedEnergyConsumption)
        # print(beg_i,end_i)
        # print(average_speed==None)
        time_abs = (raw_data_obd.loc[end_i, 'Time_abs'] - raw_data_obd.loc[beg_i, 'Time_abs']).total_seconds()

        def time_s(x):
            if x.hour is None:
                return 0
            else:
                return x.hour // 4 + 1

        time_truth = raw_data_obd.loc[beg_i, 'Time_abs']
        time_stage = time_s(raw_data_obd.loc[beg_i, 'Time_abs'])
        if raw_data_obd.loc[beg_i, 'Time_abs'].weekday() is not None:
            week_day = raw_data_obd.loc[beg_i, 'Time_abs'].weekday() + 1
        else:
            week_day = 0
        elevation = elevation_cal(df_sub_gps, beg_i, end_i)
        # print(average_speed)
        if average_speed and isinstance(elevation, float):
            mass = df_sub_obd.loc[beg_i, 'VehicleWeight']
            network_id = segment[data_sampled_index.index(beg_i)]
            # print(network_id)
            # print(beg,end,network_id)
            network_seg = network_gdf.loc[network_id]
            # print(network_seg)
            osmid = network_seg['osmid']
            tag = (network_seg['u'], network_seg['v'], network_seg['key'])
            lanes = network_seg['lanes_normed']
            bridge = network_seg['bridge_normed']
            signal_u = network_seg['signal_u_d']
            signal_v = network_seg['signal_v_d']
            # print(tag)
            highway = highway_cal(network_seg)
            speed_limit = speedlimit_cal(network_seg, highway)
            length = length_cal(network_seg)
            energy_consumption, count = energy_consumption_cal(df_sub_obd, beg_i, end_i)
            energy_consumption_total = energy_consumption / 36000

            energy_consumption_per_hour = energy_consumption_total / count
            energy_consumption_per_100km = 100000 * energy_consumption_total / length
            latitude_o, longitude_o = raw_data_gps.loc[beg_i, 'gps_Latitude'], raw_data_gps.loc[beg_i, 'gps_Longitude']
            latitude_d, longitude_d = raw_data_gps.loc[end_i, 'gps_Latitude'], raw_data_gps.loc[end_i, 'gps_Longitude']
            direction = [latitude_d - latitude_o, longitude_d - longitude_o]
            direction_array = np.array(direction)
            cosangle = direction_array.dot(np.array([1, 0])) / (np.linalg.norm(direction_array))
            if np.cross(direction_array, np.array([1, 0])) < 0:
                direction_angle = math.acos(cosangle) * 180 / np.pi
            else:
                direction_angle = -math.acos(cosangle) * 180 / np.pi
            if flag == 0:
                flag = 1
                orientation = 0
            elif 'geometry' in network_seg and network_seg['geometry']:
                xs, ys = network_seg['geometry'].xy
                id_before = segment[data_sampled_index.index(inter_f[i - 1])]
                # print(id_before)
                xl, yl = network_gdf.loc[id_before, 'geometry'].xy
                coor_b = [xs[0], ys[0]]
                coor_c = [xs[1], ys[1]]
                coor_a = [xl[-2], yl[-2]]
                # print(beg,coor_a,coor_b,coor_c)
                orientation = ori_cal(coor_a, coor_b, coor_c)
            else:
                coor_b = [df_sub_gps.loc[beg_i, 'gps_Longitude'], df_sub_gps.loc[beg_i, 'gps_Latitude']]
                for j in range(beg_i - 10, inter_f[i - 1] - 1, -20):
                    coor_a = [df_sub_gps.loc[j, 'gps_Longitude'], df_sub_gps.loc[j, 'gps_Latitude']]
                    if not coor_a == coor_b:
                        break
                for j in range(beg_i + 10, end_i, 20):
                    coor_c = [df_sub_gps.loc[j, 'gps_Longitude'], df_sub_gps.loc[j, 'gps_Latitude']]
                    if not coor_c == coor_b:
                        break
                orientation = ori_cal(coor_a, coor_b, coor_c)
            # print(i, orientation)
            df_edge.loc[len(df_edge)] = [network_id, osmid, tag, (beg_i, end_i), trip_id, position, highway,
                                         average_speed, \
                                         speed_limit, mass, elevation, orientation, length, energy_consumption_total, \
                                         siumlatedEnergyConsumption, energy_consumption_per_hour, \
                                         energy_consumption_per_100km, min(time, time_abs), \
                                         speed_list, direction, direction_angle, time_stage, week_day, time_truth, \
                                         lanes, bridge, signal_u, signal_v]
        else:
            continue
    del raw_data_gps
    del raw_data_obd
    del df_sub_gps
    del df_sub_sampled
    del df_sub_obd
    gc.collect()
    return df_edge


def extract_opath(opath):
    '''Divide the map-matched results connected by commas into a list.

    Args:
        opath: Map-matched results connected by commas

    Returns:
        A list of matched edges
    '''
    if (opath == ''):
        return []
    elif isinstance(opath, float):
        return int(opath)
    return [int(s) for s in opath.split(',')]


def main():
    percentile = '005'
    filefold = r'D:/cygwin64/home/26075/workspace/'
    network_gdf = gpd.read_file(filefold + 'network_' + percentile + '/edges.shp')
    nodes_gdf = gpd.read_file(filefold + 'network_' + percentile + '/nodes.shp')
    nodes_gdf.index = nodes_gdf.osmid
    network_gdf['signal_u'] = network_gdf.u.apply(lambda x: nodes_gdf.loc[x, 'highway']).fillna("None")
    network_gdf['signal_v'] = network_gdf.v.apply(lambda x: nodes_gdf.loc[x, 'highway']).fillna("None")
    list_uv = list(network_gdf.signal_u.unique()) + list(network_gdf.signal_v.unique())
    endpoints_dictionary = dict()
    endpoints_dictionary['None'] = 0
    cnt_endpoint = 1
    for i in list_uv:
        if i not in endpoints_dictionary:
            endpoints_dictionary[i] = cnt_endpoint
            cnt_endpoint += 1
    np.save('endpoints_dictionary.npy', endpoints_dictionary)
    network_gdf['signal_u_d'] = network_gdf.signal_u.apply(lambda x: endpoints_dictionary[x])
    network_gdf['signal_v_d'] = network_gdf.signal_v.apply(lambda x: endpoints_dictionary[x])

    # endpoints_dictionary = np.load('endpoints_dictionary.npy', allow_pickle=True).item()

    def cal_lanes(array_like):
        if pd.isna(array_like['lanes']):
            return 0
        if array_like['lanes'].isalpha():
            return 0
        if array_like['lanes'].isalnum():
            return int(array_like['lanes']) if int(array_like['lanes']) > 0 else 0
        else:
            for i in array_like['lanes']:
                if i.isdecimal():
                    return int(i)

    network_gdf['lanes_normed'] = network_gdf.apply(lambda x: cal_lanes(x), axis=1)
    network_gdf['lanes_normed'] = network_gdf['lanes_normed'].apply(lambda x: x if x <= 8 else 8)

    def cal_bridge(array_like):
        if not array_like['bridge']:
            return 0
        s = array_like['bridge']
        if 'viaduct' in s:
            return 2
        if 'yes' in s:
            return 1
        return 0

    network_gdf['bridge_normed'] = network_gdf.apply(lambda x: cal_bridge(x), axis=1)

    G = ox.utils_graph.graph_from_gdfs(nodes_gdf, network_gdf)
    # ox.plot_graph(G)
    filefold_res = r'D:/STUDYYYYYYYY/ecoRouting/Murphy/pythonWorkspace'
    f_t = filefold_res + '/trajectory_old_sampled3s'
    f_m = filefold_res + '/match_result_old'

    output_filepath = r'D:/STUDYYYYYYYY/ecoRouting/Murphy/pythonWorkspace/resultNewOct/features_percentile' + percentile
    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)

    bbox_list = [-95.35305, -91.544, 43.1759, 45.891999999999996]  # old map
    cnt_file = -1
    for f, m, n in tqdm(walk(r'D:/data/Baseline Data (Murphy)')):
        if not m and n:  # obd file
            print(f, n)
            print(f[-3:])
            truckIdDict = {'218': 3, '226': 4, '227': 5, '231': 6}
            truckId = truckIdDict.get(f[-3:], 3)
            print(truckId)
            for i in range(0, len(n), 2):
                # print('i',i)
                fn_gps = n[i + 1]
                filename_gps = os.path.join(f, fn_gps)
                print(filename_gps)
                raw_data_gps = pd.read_csv(filename_gps, header=0)
                raw_data_gps.drop(raw_data_gps.index[0], inplace=True)
                raw_data_gps = raw_data_gps.astype("float")
                raw_data_gps = raw_data_gps[
                    (raw_data_gps.gps_Longitude >= -180) & (raw_data_gps.gps_Longitude <= 180) & (
                                raw_data_gps.gps_Latitude >= -90) & (raw_data_gps.gps_Latitude <= 90)]
                index = raw_data_gps.index
                raw_data_gps = raw_data_gps.reset_index(drop=True)
                if raw_data_gps.empty:
                    print('empty gps file:' + filename_gps)
                    continue
                else:

                    cnt_file += 1
                    if cnt_file >= 0:
                        # read trajectory data
                        df_traj = read_trajectory(f_t, cnt_file)
                        # read match results
                        df_matched = readAndProcess_matched(f_m, cnt_file, network_gdf)
                        df_edge = pd.DataFrame(
                            columns=['network_id', 'osmid', 'tags', 'index', 'trip_id', 'position', 'road_type',
                                     'average_speed', 'speed_limit', 'mass', 'elevation_change', 'previous_orientation',
                                     'length', 'energy_consumption_total', 'siumlatedEnergyConsumption',
                                     'energy_consumption_per_hour', 'energy_consumption_per_100km', 'time', 'speed',
                                     'direction', 'direction_angle', 'time_stage', 'week_day', "time_acc", 'lanes',
                                     'bridge', 'endpoint_u', 'endpoint_v'])
                        df_edge.loc[len(df_edge)] = ['-', '-', '(origin node id, destination node id, key)',
                                                     '(index begin , index end)', '(csv id, trip id)', '-',
                                                     'mortorway ~ tertiary', 'km/h', 'km/h', 'kg', 'm',
                                                     'degree(right>0)', 'm', 'l', 'l', 'l/h', 'l/100km', 's',
                                                     'list of km/h', 'vector of direction', 'degree', '-', '0-6',
                                                     "true time", 'number of lanes', 'bridge', 'property of endpoint u',
                                                     'property of endpoint v']
                        # int:id ; (int,int):index begin , index end ; (int,int,int):(origin node id, destination node id, key); str:mortorway ~ tertiary; float:km/h ; float:km/h ; int:kg ; float:m ; str:l(left),r(right),s(straight) ; #float:l/h; int: id in network ; length; divide a day into several hour group; day of the week
                        fn_obd = n[i]
                        filename_obd = os.path.join(f, fn_obd)
                        raw_data_obd = read_data(filename_obd, index)
                        raw_data_obd = raw_data_obd.reset_index(drop=True)
                        for j in range(len(df_traj)):
                            # print(len(df_traj))
                            beg = df_traj.loc[j, 'beg']
                            end = df_traj.loc[j, 'end']
                            # print(cnt_file,beg,end)
                            segment = list(df_matched.iloc[j, 4])

                            df_edge = post_process(raw_data_gps, raw_data_obd, beg, end, segment, cnt_file, j, df_edge,
                                                   network_gdf, truckId)
                        df_edge.to_csv(output_filepath + '/feature_of_file' + str(cnt_file) + '.csv', index=False)
                    del raw_data_gps
                    del raw_data_obd
                    del df_traj
                    del df_matched
                    del df_edge
                    gc.collect()


if __name__ == "__main__":
    main()
