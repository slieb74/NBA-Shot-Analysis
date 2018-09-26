import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_columns',50)

import plotly
import plotly.plotly as py
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=True)

import time, itertools, re, warnings, math
warnings.filterwarnings('ignore')

from court import court_shapes

import ipywidgets as widgets
from ipywidgets import interact

#LOAD NBA SAVANT DATA
if True:
    oct_nov_ = pd.read_csv('./data/nba_savant/oct-nov-14-15.csv')
    dec_ = pd.read_csv('./data/nba_savant/dec-14-15.csv')
    jan_ = pd.read_csv('./data/nba_savant/jan-14-15.csv')
    feb_ = pd.read_csv('./data/nba_savant/feb-14-15.csv')
    mar_ = pd.read_csv('./data/nba_savant/mar-14-15.csv')
    apr_ = pd.read_csv('./data/nba_savant/apr-14-15.csv')

    #MERGE MONTHLY DFs
    df = pd.concat([oct_nov_,dec_,jan_,feb_,mar_,apr_])
    #reverse x values to plot correctly
    df.x = -df.x
    df.game_date = pd.to_datetime(df.game_date)
    df = df.reset_index(drop=True)

#Load adv stats and clean up discrepancies btwn names
if False:
    #Load Advanced stats
    stats = pd.read_excel('./data/adv-stats-14-15.xlsx')
    # Clean up name discrepancies between two dfs
    stats.Player = stats.Player.apply(lambda x: re.sub(r'([^\s\w]|_)+', '', x))
    df.name[df.name=='Jose Juan Barea'] = 'JJ Barea'
    df.name[df.name=='Tim Hardaway Jr'] = 'Tim Hardaway'
    df.name[df.name=='Charles Hayes'] = 'Chuck Hayes'
    df.name[df.name=='Glen Rice Jr'] = 'Glen Rice'
    df.name[df.name=='Louis Williams'] = 'Lou Williams'

    stats.Player[stats.Player=='Nene Hilario'] = 'Nene'
    stats.Player[stats.Player=='Jeffery Taylor'] = 'Jeff Taylor'
    stats.Player[stats.Player== 'Luigi Datome'] = 'Gigi Datome'


    #convert defender name to first name last name format
    df.defender_name[df.defender_name.isnull()] = 'None'

    def convert_defender_names(player):
        if player =='None':
            return 'None'
        elif player=='Nene':
            return 'Nene'
        else:
            name = player.split(', ')
            full_name = ' '.join((name[1],name[0]))
            return re.sub(r'([^\s\w]|_)+', '', full_name)

    df.defender_name = df.defender_name.apply(convert_defender_names)

    # Clean up name discrepancies between two dfs
    df.defender_name[df.defender_name=='Jose Juan Barea'] = 'JJ Barea'
    df.defender_name[df.defender_name=='Tim Hardaway Jr'] = 'Tim Hardaway'
    df.defender_name[df.defender_name=='Charles Hayes'] = 'Chuck Hayes'
    df.defender_name[df.defender_name=='Glen Rice Jr'] = 'Glen Rice'
    df.defender_name[df.defender_name=='Louis Williams'] = 'Lou Williams'

    #map player ids to new df column matching to defender name
    player_ids_df = df[['name','player_id']].rename(columns={'name':'defender_name','player_id':'defender_id'})
    player_ids_df = player_ids_df.groupby('defender_name').max()

    #merge two dataframes with defensive stats
    df = df.merge(player_ids_df, on='defender_name')

#helper function to calcuate distance for each shot to determine zone and area
df.shot_type = np.where(df.shot_type=='2PT Field Goal', 2, 3)
def get_shot_distance(x,y):
    x_squared=x**2
    y_squared=y**2
    shot_distance = math.sqrt(x_squared + y_squared) / 10  # unit for distance is off by factor of 10, divide by 10 to convert to feet
    return round(shot_distance, 1)
#function to determine shot zone and region
def get_shot_zone(row):
    x = row.x
    y = row.y

    shot_zone = ''
    shot_area = ''

    #restricted area, shots within 4ft of hoop
    if get_shot_distance(x,y)<=4:
        shot_zone = 'Restricted Area'

    #abov break 3 pointers
    elif (get_shot_distance(x,y)>=23.9) & (y>=92.5):
        shot_zone = 'Above Break 3'
    #corner 3s
    elif (y<92.5) & ((x<=-220) | (x>=220)):
        shot_zone = 'Corner 3'
    #in the paint shots excluding restricted area
    elif (-80<=x<=80) & (-47.5<=y<=143.5) & (get_shot_distance(x,y)>4):
        shot_zone = 'Paint'
    #mid range shots, left and right side
    elif (get_shot_distance(x,y)<23.9) & ((-220<x<-80) | (80<x<220)):
        shot_zone = 'Mid Range'
    #mid range shots, center (above foul line)
    else:
        shot_zone = 'Mid Range'

    #heaves (defined as shots 35+ feet from basket)
    if get_shot_distance(x,y)>35:
        shot_zone = 'Heave'

    #Get area of court (left, right, or center)
    if shot_zone !='Paint':
        if (x <= 80) & (x>=-80):
            shot_area = 'C'
        elif (x>80):
            shot_area = 'L'
        else:
            shot_area = 'R'
    #for shots in paint, they have special designation for left, right, and center
    else:
        if x>40:
            shot_area = 'L'
        elif x<-40:
            shot_area = 'R'
        else:
            shot_area = 'C'
    return shot_zone, shot_area

#add shot zones and areas to df
def add_shot_zones_area_to_df(df):
    shot_zones = []
    shot_areas = []

    for index, row in df.iterrows():
        shot_zones.append(get_shot_zone(row)[0])
        shot_areas.append(get_shot_zone(row)[1])

    df['shot_zone'] = shot_zones
    df['shot_area'] = shot_areas
if False:
    add_shot_zones_area_to_df(df)

#gets league average fg% for each shot zone and area
def get_lg_avgs(shot_zone_area_tup, df):
    sz = shot_zone_area_tup[0]
    sa = shot_zone_area_tup[1]
    shots_made = len(df[(df.shot_zone==sz) & (df.shot_area==sa) & (df.shot_made_flag==1)])
    total_shots = len(df[(df.shot_zone==sz) & (df.shot_area==sa)])
    if total_shots ==0:
        make_pct = 0
    else:
        make_pct = round((shots_made / total_shots),4)
    return make_pct
if False:
    #get unique shot zones and areas and store them in tuples
    sz = set(shot_zones)
    sa = set(shot_areas)
    sza_tups = list(itertools.product(sz,sa))
    #create dict with shot zone area tuples and corresponding league avg fg%
    sza_dict = {}
    for sza in sza_tups:
        sza_dict[sza] = get_lg_avgs(sza, df)
    #function to add lg_avg to corresponding df rows
    def add_lg_avg_to_df(df):
    df['lg_avg']=0
    for k,v in sza_dict.items():
        df['lg_avg'] = np.where((df.shot_zone==k[0]) & (df.shot_area==k[1]), v, df['lg_avg'])

    add_lg_avg_to_df(df)

#generate unique team ids from 1-30 and add to df
def create_team_ids(df):
    team_id_dict = {}
    for id_, team in enumerate(list(set(df.team_name))):
        team_id_dict[team]=id_+1

    df['opp_id']=0
    #get team ids from 1-30
    for k,v in team_id_dict.items():
        df['team_id'] = np.where(df.team_name==k, v, df['team_id'])
        df['opp_id'] = np.where(df.opponent==k, v, df['opp_id'])
    return team_id_dict
if False:
    create_team_ids(df)

#LOAD stats.nba.com SCRAPED DATA
if True:
    nba_shots = pd.read_csv('./data/shots_1415.csv',index_col=0)
    nba_shots.GAME_DATE = nba_shots.GAME_DATE.astype('str')

    #Adds dashes to date string so it can be converted to datetime format
    def add_dashes(string):
        date = string[:4] + '-' + string[4:6] + '-' + string[-2:]
        return date

    nba_shots.GAME_DATE = nba_shots.GAME_DATE.apply(lambda x: add_dashes(x))
    nba_shots.GAME_DATE = pd.to_datetime(nba_shots.GAME_DATE)
    nba_shots.LOC_X = -nba_shots.LOC_X

#MERGE DATAFRAMES, DROP EXTRANEOUS COLUMNS
if True:
    merged_df = df.merge(nba_shots, left_on=['team_name','game_date','period','minutes_remaining','seconds_remaining','x','y'],
              right_on=['TEAM_NAME','GAME_DATE','PERIOD','MINUTES_REMAINING','SECONDS_REMAINING','LOC_X','LOC_Y'])

    merged_df = merged_df.drop(columns=['GRID_TYPE','PLAYER_NAME', 'TEAM_ID', 'TEAM_NAME', 'PERIOD', 'MINUTES_REMAINING',
           'SECONDS_REMAINING','SHOT_DISTANCE','LOC_X', 'LOC_Y', 'SHOT_ATTEMPTED_FLAG', 'SHOT_MADE_FLAG', 'GAME_DATE',
           'espn_player_id', 'espn_game_id', 'EVENT_TYPE','ACTION_TYPE', 'SHOT_TYPE','SHOT_ZONE_BASIC',
           'SHOT_ZONE_AREA', 'SHOT_ZONE_RANGE'])

#Function to determing if the shooter is playing at home
if False:
    #get dictionary matching team names to home and away team acronyms
    def create_home_acronym_dict():
        team_acronyms = sorted(list(merged_df.HTM.unique()))
        team_names = sorted(list(merged_df.team_name.unique()))

        team_name_ac_dict = dict(zip(team_names,team_acronyms))
        team_name_ac_dict['Boston Celtics'] = 'BOS'
        team_name_ac_dict['Brooklyn Nets'] = 'BKN'
        return team_name_ac_dict

    #function to add column for whether the shooter is playing at home
    def get_home_team():
        start = time.time()
        is_home_arr = []

        team_name_ac_dict=create_home_acronym_dict()

        for index, row in merged_df.iterrows():
            if team_name_ac_dict[row.team_name]==row.HTM:
                is_home_arr.append(1)
            else:
                is_home_arr.append(0)
            if index%50000==0:
                print('Runtime: {} seconds. {} iterations to go.'.format(round(time.time()-start,2), len(merged_df)-index))
        return is_home_arr

    merged_df['is_home'] = get_home_team()

#SORT DATA FRAME BY DATE, GAME_ID, NAME, EVENT_ID
sorted_df = merged_df.copy().sort_values(by=['game_date','GAME_ID','name','GAME_EVENT_ID']).reset_index(drop=True)

#Function to calculate whether player is hot, i.e. whether they have hit 1, 2, or 3 previous shots
def is_player_hot(dataframe):
    start=time.time()
    df = dataframe
    #create array that stores whether previous 1, 2, or 3 shots were made, respectively
    heat_check_array=np.zeros((len(df),3))

    for index, row in df.iterrows():
        if index==0:
            heat_check_array[index,:]+=[0,0,0]
        elif index==1:
            if (df.name[index]==df.name[index-1]) & (row.GAME_ID==df.GAME_ID[index-1]) & (df.shot_made_flag[index-1]==1):
                heat_check_array[index,:]+=[1,0,0]
            else:
                heat_check_array[index,:]+=[0,0,0]
        elif index==2:
            if (df.name[index]==df.name[index-1]) & (df.name[index]==df.name[index-2]) & (row.GAME_ID==df.GAME_ID[index-1]) & (df.shot_made_flag[index-1]==1) & (df.shot_made_flag[index-2]==1):
                heat_check_array[index,:]+=[1,1,0]
            elif (df.name[index]==df.name[index-1]) & (row.GAME_ID==df.GAME_ID[index-1]) & (df.shot_made_flag[index-1]==1) & (df.shot_made_flag[index-2]==0):
                heat_check_array[index,:]+=[1,0,0]
            else:
                heat_check_array[index,:]+=[0,0,0]
        else:
            if (df.name[index]==df.name[index-1]) & (df.name[index]==df.name[index-2]) & (df.name[index]==df.name[index-2]) & (row.GAME_ID==df.GAME_ID[index-1]) & (df.shot_made_flag[index-1]==1) & (df.shot_made_flag[index-2]==1) & (df.shot_made_flag[index-3]==1):
                heat_check_array[index,:]+=[1,1,1]
            elif (df.name[index]==df.name[index-1]) & (df.name[index]==df.name[index-2]) & (row.GAME_ID==df.GAME_ID[index-1]) & (df.shot_made_flag[index-1]==1) & (df.shot_made_flag[index-2]==1) & (df.shot_made_flag[index-3]==0):
                heat_check_array[index,:]+=[1,1,0]
            elif (df.name[index]==df.name[index-1]) & (row.GAME_ID==df.GAME_ID[index-1]) & (df.shot_made_flag[index-1]==1):
                heat_check_array[index,:]+=[1,0,0]
            else:
                heat_check_array[index,:]+=[0,0,0]

        if index%50000==0:
            print('Runtime: {} seconds. {} iterations remaining.'.format(round(time.time()-start,2),len(df)-index))
    return heat_check_array
if False:
    heat_check_array = is_player_hot(sorted_df)
    #add heat check stats to dataframe
    sorted_df['prev_shot_made'] = heat_check_array[:,0]
    sorted_df['prev_2_made'] = heat_check_array[:,1]
    sorted_df['prev_3_made'] = heat_check_array[:,2]

#Calculate FG% in each zone for player at given time
def get_fg_pct_by_player_for_each_zone(df):
    start = time.time()
    player_names = list(df.name.unique())
    player_df_list = []

    for c, player in enumerate(player_names):
        df_ = df[df.name==player].reset_index(drop=True)
        shot_arr = np.zeros((len(df_),26))

        if (c+1)%100==0:
            print('Runtime: {} seconds. {} of {} players completed.'.format(round(time.time()-start,2), c+1, len(player_names)))

        for index, row in df_.iterrows():
            if index != 0:
                shot_arr[index,:] = shot_arr[index-1,:]
            if row.shot_zone=='Mid Range':
                if row.shot_area=='R':
                    if row.shot_made_flag==1:
                        shot_arr[index,0:2]+=[1,1]
                    else:
                        shot_arr[index,0:2]+=[0,1]
                elif row.shot_area=='C':
                    if row.shot_made_flag==1:
                        shot_arr[index,2:4]+=[1,1]
                    else:
                        shot_arr[index,2:4]+=[0,1]
                else:
                    if row.shot_made_flag==1:
                        shot_arr[index,4:6]+=[1,1]
                    else:
                        shot_arr[index,4:6]+=[0,1]
            elif row.shot_zone=='Restricted Area':
                if row.shot_made_flag==1:
                    shot_arr[index,6:8]+=[1,1]
                else:
                    shot_arr[index,6:8]+=[0,1]
            elif row.shot_zone=='Heave':
                if row.shot_made_flag==1:
                    shot_arr[index,8:10]+=[1,1]
                else:
                    shot_arr[index,8:10]+=[0,1]
            elif row.shot_zone=='Above Break 3':
                if row.shot_area=='R':
                    if row.shot_made_flag==1:
                        shot_arr[index,10:12]+=[1,1]
                    else:
                        shot_arr[index,10:12]+=[0,1]
                elif row.shot_area=='C':
                    if row.shot_made_flag==1:
                        shot_arr[index,12:14]+=[1,1]
                    else:
                        shot_arr[index,12:14]+=[0,1]
                else:
                    if row.shot_made_flag==1:
                        shot_arr[index,14:16]+=[1,1]
                    else:
                        shot_arr[index,14:16]+=[0,1]
            elif row.shot_zone=='Paint':
                if row.shot_area=='R':
                    if row.shot_made_flag==1:
                        shot_arr[index,16:18]+=[1,1]
                    else:
                        shot_arr[index,16:18]+=[0,1]
                elif row.shot_area=='C':
                    if row.shot_made_flag==1:
                        shot_arr[index,18:20]+=[1,1]
                    else:
                        shot_arr[index,18:20]+=[0,1]
                else:
                    if row.shot_made_flag==1:
                        shot_arr[index,20:22]+=[1,1]
                    else:
                        shot_arr[index,20:22]+=[0,1]
            elif row.shot_zone=='Corner 3':
                if row.shot_area=='R':
                    if row.shot_made_flag==1:
                        shot_arr[index,22:24]+=[1,1]
                    else:
                        shot_arr[index,22:24]+=[0,1]
                else:
                    if row.shot_made_flag==1:
                        shot_arr[index,24:26]+=[1,1]
                    else:
                        shot_arr[index,24:26]+=[0,1]
        player_df_list.append(pd.DataFrame(shot_arr,index=df_.name))

    print('Total Runtime: {} seconds.'.format(round(time.time()-start,2),
                                              c, len(player_names)))
    return player_df_list

#Add zone breakdown FG% to dataframe
def add_zone_fg_pct_to_df(df):
    df_list = get_fg_pct_by_player_for_each_zone(df)
    zone_df = pd.concat([df_ for df_ in df_list])
    zone_df = zone_df.groupby(zone_df.index).max()

    column_names = ['mid_R_pct', 'mid_C_pct', 'mid_L_pct', 'restricted_pct', 'heave_pct', 'ab_3_R_pct', 'ab_3_C_pct',
                'ab_3_L_pct', 'paint_R_pct', 'paint_C_pct', 'paint_L_pct', 'corner_3_R_pct', 'corner_3_L_pct',]

    counter = 0
    for col in column_names:
        zone_df[col] = np.round(zone_df[counter]/zone_df[counter+1],4)
        counter+=2

    zone_df = zone_df.drop(columns=list(range(0,26))).reset_index()

    zone_fg_df = pd.concat((sorted_df,zone_df),axis=1)

    return zone_fg_df
