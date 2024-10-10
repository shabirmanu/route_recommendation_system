import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import googlemaps as gm
from math import *
import sys


gmaps = gm.Client(key='AIzaSyCLykzskHmIhimZdRNSQC3HTug3UNUY_SA')

def predictMM():

    df = pd.read_csv("C:/Users/Shabir/PycharmProjects/OptimizationProject/tasks_v.txt")

    distance_df = pd.read_csv('C:/Users/Shabir/PycharmProjects/OptimizationProject/springdistance.txt')
#    print(distance_df)
#    print(distance_df)

    features = df.loc[:, ['dt_quarterly', 'move_path', 'total_path_count']]

    #features.to_csv('features.csv', sep=',')


    spring_criteria = df['dt_quarterly'] == 1
    #summer_criteria = df['dt_quarterly'] == 2
    #fall_criteria = df['dt_quarterly'] == 3
    #winter_criteria = df['dt_quarterly'] == 4

    spring_data = features[spring_criteria]
    #summer_data = features[summer_criteria]
    #fall_data = features[fall_criteria]
    #winter_data = features[winter_criteria]




    #unique_locations = _getUniqueLocationsPerSeason(features)


    spring_locations = _getUniqueLocationsPerSeason(spring_data)
    #fall_locations = _getUniqueLocationsPerSeason(fall_data)
    #summer_locations = _getUniqueLocationsPerSeason(summer_data)
    #winter_locations = _getUniqueLocationsPerSeason(winter_data)

    # coordinate_data = _findCoordinate(spring_locations)

    spring_df = _getTransitionMatrix(spring_locations, features)
    print(spring_df.index)
    spring_df.to_csv('sp_season_matrix.csv', sep=',')


    #spring_distance_df = _getDistanceTransitionMatrix(unique_locations, distance_df)
    #spring_distance_df.to_csv('spring_distance_matrix', sep=',')
    #_getTaggedLocations(spring_locations)

    spring_v = np.zeros(len(spring_df.index))
    spring_v[0] = 1
    spring_states = _getLongTermStates(spring_df, spring_v)

    print(spring_states)
    np.savetxt("longstates.csv", spring_states, delimiter=",")


    #
    # print ('------------For Spring----------')
    # #print (_showPlot(spring_states.index, spring_states.values, 'Spring'))
    # print (spring_states.sort_values(ascending=False))
    # print ('------------End------------------')

    # fall_df = _getTransitionMatrix(fall_locations, fall_data)
    # v = np.zeros(len(fall_df.index))
    # v[0] = 1
    # fall_states = _getLongTermStates(fall_df, v)
    #
    # print ('------------For Fall----------')
    # print (fall_states.sort_values(ascending=False))
    # print ('------------End------------------')

    # winter_df = _getTransitionMatrix(winter_locations, winter_data)
    # v = np.zeros(len(winter_df.index))
    # v[0] = 1
    # winter_states = _getLongTermStates(winter_df, v)

    # print ('------------For Winter----------')
    # #print (_showPlot(winter_states.index, winter_states.values, 'Winter'))
    # print (winter_states.sort_values(ascending=False))
    # print ('------------End------------------')

    # summer_df = _getTransitionMatrix(summer_locations, summer_data)
    # v = np.zeros(len(summer_df.index))
    # v[0] = 1
    # summer_states = _getLongTermStates(summer_df, v)
    #
    # print ('------------For Fall----------')
    # #print (_showPlot(summer_states.index, summer_states.values, 'Summer'))
    # print (summer_states.sort_values(ascending=False))
    # print ('------------End------------------')

    # x = []
    # y = []
    # for i in df2.index:
    #     x.append(i)
    #
    # for j in df2.columns:
    #     y.append(j)


    #print(spring_data.as_matrix())


    # _showPlot(data['x'], data['y'], "Spring")
    #
   # data = _unpackData3d(df2)
   # _showPlot3d(data)

    # data = _unpackData(fall_locations)
    # _showPlot(data['x'], data['y'], "Fall")
    #
    # data = _unpackData(winter_locations)
    # _showPlot(data['x'], data['y'], "Winter")





 #   transition_matrix = _getTransitionMatrix()
  #  print (winter_locations)


    return ""


# def _findCoordinate(season):
#     data = _unpackData(season)
#     locations = data['x']
#     print(locations)
#     loc = gmaps.geocode(locations[0])
#     print(loc)

# def _getTaggedLocations(season):
#     data = {"location": [], "code": []}
#     for loc, total_count in season.items():
#         data["location"].append(loc)
#         data["code"].append(location[5:]+random(3))
#     return data





def _getTransitionMatrix(season, season_data):
    data = _unpackData(season)
    df2 = pd.DataFrame(index=data['x'], columns=data['x'])

    for i in range(len(df2.index)):
        for j in range(len(df2.columns)):
            if (df2.index[i] == df2.columns[j]):
                df2.loc[df2.index[i], df2.columns[j]] = 0
            else:
                si = season_data[season_data['move_path'].str.contains(df2.index[i] + "->" + df2.columns[j])]
                total = si['total_path_count'].sum()
                df2.loc[df2.index[i], df2.columns[j]] = total

    df2['sum'] = df2.sum(axis=1)



    for i in range(len(df2.index)):
        for j in range(len(df2.columns)):
            if (df2.index[i] == df2.columns[j]):
                df2.loc[df2.index[i], df2.columns[j]] = 0
            else:
                if(df2['sum'][i] != 0):
                    total = df2.loc[df2.index[i], df2.columns[j]] / df2['sum'][i]
                    df2.loc[df2.index[i], df2.columns[j]] = total
                else:
                    df2.loc[df2.index[i], df2.columns[j]] = 0

    df2 = df2.drop(labels='sum', axis=1)
    dd = tabulate([list(row) for row in df2.values], headers=list(df2.columns))
    print(dd)
    return df2


def _getUniqueLocationsPerSeason(season_data):
    location_pair = []

    for index, row in season_data.iterrows():
        move_pathArr = row['move_path'].split('->')
        for arr_item in move_pathArr:
            location_pair.append((arr_item, row['total_path_count']))
    results = {}
    for k, v in location_pair:
        results[k] = results.get(k, 0) + v

    return results;

def _showPlot(x, y, season):
    plt.plot(x, y, 'ro-')
    plt.rcParams["figure.figsize"] = [4,2]
    plt.xticks(rotation=270)
    plt.ylabel("Long Term Steady State Probabilities")
    plt.title("Long Term Steady state probabilities of Jeju tourism spots in " + season+ " Season")
    plt.show()

def _unpackData(season):
    data = {"x": [], "y": []}
    for loc, total_count in season.items():
        data["x"].append(loc)
        data["y"].append(total_count)
    return data

def _unpackData3d(df2):
    data = {"x": [], "y": [], "z":[]}
    for i in range(len(df2.index)):
        data['x'].append(df2.index[i])
        data['y'].append(df2.columns[i])
        inner_list = []
        for j in range(len(df2.columns)):
            inner_list.append(df2.loc[df2.index[i],df2.columns[j]])
        data['z'].append(inner_list)


    return data

def _getDistanceTransitionMatrix(season, all_data):

    data = _unpackData(season)
    df2 = pd.DataFrame(index=data['x'], columns=data['x'])
    #data = _unpackData(season)
    # print(data['x'])

    for i in range(len(df2.index)):
        #print(season.Latitude[i])
        for j in range(len(df2.columns)):
            if (df2.index[i] == df2.columns[j]):
                df2.loc[df2.index[i], df2.columns[j]] = 0
            else:
                R = 6373.0
                x_lat = radians(all_data.Latitude[df2.columns[j]])

                x_long = radians(season.Longitude[i])
                y_lat = radians(season.Latitude[j])
                y_long = radians(season.Longitude[j])

                #print("asa", x_lat)
                #print('asaaaaa', x_long)
                #print('asadf', y_lat)
                #print('asasasasasasa', y_long)

                dlon = y_long - x_long
                dlat = y_lat - x_lat
                a = sin(dlat / 2) ** 2 + cos(x_lat) * cos(y_lat) * sin(dlon / 2) ** 2
                c = 2 * atan2(sqrt(a), sqrt(1 - a))
                distan = R * c

                # for item in season:
                #     for items in season:
                #         lat1 = radians(season['Latitude'][item])
                #         lon1 = radians(season['Longitude'][item])
                #         lat2 = radians(season['Latitude'][items])
                #         lon2 = radians(season['Longitude'][items])
                #
                #

                df2.loc[df2.index[i], df2.columns[j]] = distan

    dd = tabulate([list(row) for row in df2.values], headers=list(df2.columns))
    print(dd)
# def _showPlot3d(data):
#     ax = plt.axes(projection='3d')
#     xline = [1,2,3,4]
#     yline = [1,2,3,4]
#     zline = [1,2,3,4]
#
#
#     print(xline)
#     ax.plot3D(xline, yline, zline, 'gray')
#     plt.show()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # print(data['z'])
    # ax.scatter(data['x'], data['y'], data['z'])
    # plt.show()

def _getLongTermStates(season_df, steady_state_v):
    transposed = season_df.transpose()
    ##transposed.apply(pd.to_numeric)
    #newdf = pd.to_numeric(transposed)

    for i in range(100):
        steady_state_v = transposed.as_matrix().dot(steady_state_v)

    return steady_state_v

def OptimalRoute():

    transition_matrix = pd.read_csv("C:/Users/Shabir/PycharmProjects/OptimizationProject/spring_matrix.csv", index_col=0)
    distance_matrix = pd.read_csv("C:/Users/Shabir/PycharmProjects/OptimizationProject/distance_matrix.csv", index_col=0)


    #longterm_v = np.zeros(len(transition_matrix.index))
    #longterm_v[0] = 1
    #longterm_states = _getLongTermStates(transition_matrix, longterm_v)
    current_location = 'Bijarim'
    total_locations = 4
    total_distance = 70
    total_time = 120

    sw = 0.8
    lw = 0.2

    tw = 0.4
    dw = 0.6

    current_location_prob = transition_matrix.loc[current_location]
    current_location_dist = distance_matrix.loc[current_location]

    candidate_locations = current_location_dist[current_location_dist < total_distance]

    arr_indecis = candidate_locations.index
    candidate_loc_prob = current_location_prob.loc[current_location_prob.index.isin(arr_indecis)]

    optimization_index_series = pd.Series(index = candidate_loc_prob.index)

    #print(optimization_index_series)

    for i in range(len(candidate_loc_prob.index)):
        key = candidate_loc_prob.index[i]
        prob_val = candidate_loc_prob[i]
        dist_val = candidate_locations[key]
        if(dist_val == 0):
            optimization_index_series[key] = 0
            continue
        computed_value = (sw*prob_val + lw*prob_val)/(dw*dist_val + tw*(dist_val/60))
        optimization_index_series[key] = computed_value
        #print(dist_val,prob_val, key, computed_value)

    top_optimal_location = optimization_index_series.sort_values(ascending=False)[:total_locations]
    top_distance = candidate_locations.loc[candidate_locations.index.isin(top_optimal_location.index)]
    print(top_distance.sort_values(ascending=True), top_optimal_location)




    #print(longterm_states)

if __name__ == '__main__':
    # maxDistance = sys.argv[1]
    # maxTime = sys.argv[2]
    # numberOfSpots = sys.argv[3]
    # current_location = sys.argv[4]


    #predictMM()
    OptimalRoute()
