import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
pd.set_option('display.max_columns',100)

import warnings
warnings.filterwarnings('ignore')

import itertools, math, time, re

############################--LOAD DATA--#############################
def load_data_to_df():
    oct_nov_ = pd.read_csv('./data/nba_savant/oct-nov-14-15.csv')
    dec_ = pd.read_csv('./data/nba_savant/dec-14-15.csv')
    jan_ = pd.read_csv('./data/nba_savant/jan-14-15.csv')
    feb_ = pd.read_csv('./data/nba_savant/feb-14-15.csv')
    mar_ = pd.read_csv('./data/nba_savant/mar-14-15.csv')
    apr_ = pd.read_csv('./data/nba_savant/apr-14-15.csv')

    df = pd.concat([oct_nov_,dec_,jan_,feb_,mar_,apr_])
    #reverse x values to plot correctly
    df.x = -df.x
    df.game_date = pd.to_datetime(df.game_date)
    df = df.reset_index(drop=True)
    return df

df = load_data_to_df()
######################################################################

###########################--BASIC CLEANING--#########################
df.shot_type = np.where(df.shot_type=='2PT Field Goal', 2, 3)

def create_team_ids(df):
    team_id_dict = {}
    for id_, team in enumerate(list(set(df.team_name))):
        team_id_dict[team]=id_+1

    df['opp_id']=0
    #get team ids from 1-30
    for k,v in team_id_dict.items():
        df['team_id'] = np.where(df.team_name==k, v, df['team_id'])
        df['opp_id'] = np.where(df.opponent==k, v, df['opp_id'])
create_team_ids(df)
######################################################################


####################--LOAD NBA SCRAPED DATA--######################
nba_shots = pd.read_csv('./data/shots_1415.csv',index_col=0)
nba_shots.GAME_DATE = nba_shots.GAME_DATE.astype('str')

#Adds dashes to date string so it can be converted to datetime format
def add_dashes(string):
    date = string[:4] + '-' + string[4:6] + '-' + string[-2:]
    return date

def clean_scraped_nba_data():
    nba_shots.GAME_DATE = nba_shots.GAME_DATE.apply(lambda x: add_dashes(x))
    nba_shots.GAME_DATE = pd.to_datetime(nba_shots.GAME_DATE)
    nba_shots.LOC_X = -nba_shots.LOC_X
clean_scraped_nba_data()
######################################################################

########################--MERGE NBA AND SAVANT--######################
def merge_nba_and_savant_data(df,nba_shots):
    merged_df = df.merge(nba_shots, left_on=['team_name','game_date','period', 'minutes_remaining','seconds_remaining','x','y'], right_on=['TEAM_NAME','GAME_DATE','PERIOD','MINUTES_REMAINING', 'SECONDS_REMAINING','LOC_X','LOC_Y'])

    merged_df = merged_df.drop(columns=['GRID_TYPE','PLAYER_NAME', 'TEAM_ID', 'TEAM_NAME', 'PERIOD', 'MINUTES_REMAINING', 'SECONDS_REMAINING','SHOT_DISTANCE','LOC_X', 'LOC_Y', 'SHOT_ATTEMPTED_FLAG', 'SHOT_MADE_FLAG', 'GAME_DATE', 'espn_player_id', 'espn_game_id', 'EVENT_TYPE', 'SHOT_TYPE', 'ACTION_TYPE'])

    return merged_df
merged_df = merge_nba_and_savant_data(df,nba_shots)
######################################################################

########################--FEATURE ENGINEERING--######################
#helper function to get dictionary matching team names to home and away team acronyms
def create_home_acronym_dict():
    team_acronyms = sorted(list(merged_df.HTM.unique()))
    team_names = sorted(list(merged_df.team_name.unique()))

    team_name_ac_dict = dict(zip(team_names,team_acronyms))
    team_name_ac_dict['Boston Celtics'] = 'BOS'
    team_name_ac_dict['Brooklyn Nets'] = 'BKN'
    return team_name_ac_dict

#Function to determing if the shooter is playing at home
def get_home_team():
    start = time.time()
    is_home_arr = []

    team_name_ac_dict=create_home_acronym_dict()

    for index, row in merged_df.iterrows():
        if team_name_ac_dict[row.team_name]==row.HTM:
            is_home_arr.append(1)
        else:
            is_home_arr.append(0)
        if index%100000==0:
            print('Runtime: {} seconds. {} iterations to go.'.format(round(time.time()-start,2), len(merged_df)-index))
    return is_home_arr
merged_df['is_home'] = get_home_team()

#sort the dataframe by date, game_id, player_name, and game_event_id
sorted_df = merged_df.copy().sort_values(by=['game_date','GAME_ID','name','GAME_EVENT_ID']).reset_index(drop=True)

#Function to calculate whether player is hot, i.e. whether they have hit 1, 2, or 3 previous shots
def is_player_hot(df):
    start=time.time()

    #create array that stores whether previous 1, 2, or 3 shots were made, respectively
    heat_check_array=np.zeros((len(df),3))

    for index, row in df.iterrows():
        #If index < 3, cant check prior three shots
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
        #If index >=3
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
            print('Runtime: {} seconds. {} iterations remaining.'.format(round(time.time()-start,2), len(df)-index))

    return heat_check_array

def add_heat_check_to_df(df):
    heat_check_array = is_player_hot(df)
    df['prev_shot_made'] = heat_check_array[:,0]
    df['prev_2_made'] = heat_check_array[:,1]
    df['prev_3_made'] = heat_check_array[:,2]
add_heat_check_to_df(sorted_df)
######################################################################


####################--LOAD ADVANCED STATS--######################
stats = pd.read_excel('./data/adv-stats-14-15.xlsx',index_col=0)
stats['DWS/48'] = round(stats.DWS/stats.MP*48,3)

# Clean up name discrepancies between two dfs
def clean_name_discrepancies(df,stats):
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
clean_name_discrepancies(sorted_df, stats)

#convert defender names from last,first to first,last
def convert_defender_names(player):
    if player =='None':
        return 'None'
    elif player=='Nene':
        return 'Nene'
    else:
        name = player.split(', ')
        full_name = ' '.join((name[1],name[0]))
        return re.sub(r'([^\s\w]|_)+', '', full_name)
sorted_df.defender_name = sorted_df.defender_name.apply(convert_defender_names)

# Clean up name discrepancies between two dfs
def clean_defender_names(df):
    df.defender_name[df.defender_name=='Jose Juan Barea'] = 'JJ Barea'
    df.defender_name[df.defender_name=='Tim Hardaway Jr'] = 'Tim Hardaway'
    df.defender_name[df.defender_name=='Charles Hayes'] = 'Chuck Hayes'
    df.defender_name[df.defender_name=='Glen Rice Jr'] = 'Glen Rice'
    df.defender_name[df.defender_name=='Louis Williams'] = 'Lou Williams'
clean_defender_names(sorted_df)

############# OFFENSE ###########
def merge_off_stats(df,stats):
    off_stats = stats[['Player','Pos','Age','TS%','3PAr','USG%']]
    df = df.merge(off_stats, left_on='name', right_on='Player').drop(columns=['Player'])
    df.columns = map(str.lower, df.columns)
    return df
sorted_df = merge_off_stats(sorted_df,stats)

############ DEFENSE ###########
#map player ids to new df column matching to defender name
def add_defender_ids(df):
    player_ids_df = df[['name','player_id']].rename(columns={'name': 'defender_name', 'player_id':'defender_id'})
    player_ids_df = player_ids_df.groupby('defender_name').max()

    none_id = pd.DataFrame(data=[('None',0)],
                           columns=['defender_name', 'defender_id']).set_index('defender_name')
    player_ids_df = pd.concat((player_ids_df,none_id))

    #merge two dataframes with defender ids
    df = df.merge(player_ids_df, on='defender_name')
    return df
sorted_df = add_defender_ids(sorted_df)

def merge_def_stats(df,stats):
    def_stats = stats[['Player', 'BLK%', 'DWS/48', 'DBPM']].rename(columns={'Player':'defender_name', 'BLK%':'blk_pct', 'DWS/48':'dws/48', 'DBPM':'dbpm'})

    #add dummy stats for no defender (id=0) and append to defense stats
    none_stats = pd.DataFrame(data = [('None', 0, 0, 0)], columns=['defender_name', 'blk_pct', 'dws/48', 'dbpm'])

    #add player advanced def stats
    def_stats = pd.concat((def_stats, none_stats)).reset_index(drop= True)
    df = df.merge(def_stats, on='defender_name')

    #add team defensive rating
    d_rating_14 = pd.read_excel('./data/drating_2014.xlsx')
    df = df.merge(d_rating_14, left_on='team_name', right_on='Team').drop(columns='Team')

    return df
sorted_df = merge_def_stats(sorted_df,stats)

######################################################################


########################--ADDITIONAL CLEANING--#######################
def clean_positions(df):
    df.pos[df.name=='Giannis Antetokounmpo'] = 'SF'
    df.pos[df.pos=='PG-SG'] = 'SG'
    df.pos[df.pos=='SF-SG'] = 'SF'
    df.pos[df.pos=='SG-PG'] = 'PG'
    df.pos[df.pos=='PF-SF'] = 'SF'
    df.pos[df.pos=='SF-PF'] = 'PF'
    df.pos[df.pos=='SG-SF'] = 'SF'
clean_positions(sorted_df)

def clean_shot_zones(df):
    df.shot_zone_basic[df.shot_zone_basic=='In The Paint (Non-RA)'] = 'Paint'
    #change shots misclassified as above_break_3 to backcourt
    df.shot_zone_basic[(df.shot_zone_area=='Back Court(BC)') &  (df.shot_zone_basic=='Above the Break 3')] = 'Backcourt'
clean_shot_zones(sorted_df)
######################################################################


########################--GET FG % FOR EACH ZONE--####################
def get_zone_fg_pct(df, date=None, event=None):
    fg_pct_list = []
    column_names = []

    # if date:
    #     df = df[df.game_date<date]
    #     if date<='2014-10-28':
    #         return 'Invalid Date'
    #     if event:
    #         df = df[df.game_event_id<event]

    #create crosstab with number of makes and misses in each of the 15 zones
    tab = pd.crosstab(df.name, [df.shot_zone_area, df.shot_zone_basic, df.shot_made_flag])

    ## format of col - ('Back Court(BC)', 'Backcourt', 0)
    for col in tab.columns:
        #if it is the shot made column
        if col[2]==1:
            #calculate percentages
            pct = round(tab[col]/(tab[col]+tab[col[0]][col[1]][0]),3)
            fg_pct_list.append(pct)
            column_names.append('_'.join(col[:2]).replace(' ','_').replace(')','').split('(')[1])

    #concatenate each player's percentages into one df
    zone_pct_df = pd.concat([fg_df for fg_df in fg_pct_list],axis=1).fillna(0)
    #add column names
    zone_pct_df.columns=column_names

    return zone_pct_df.reset_index()
zone_fg_pct = get_zone_fg_pct(sorted_df)

def create_zone_ids_df(df):
    #create table matching shot_zones to unique ids
    zone_ids = []

    id_=0
    for zone_ in df.shot_zone_basic.unique():
        for area_ in df.shot_zone_area.unique():
            #if combo exists (i.e. there is no possibility to shoot a corner 3 from the center)
            if len(df[(df.shot_zone_basic==zone_) & (df.shot_zone_area==area_)]) > 0:
                zone_ids.append((id_, zone_, area_))
                id_+=1

    zone_id_df = pd.DataFrame.from_records(zone_ids, columns=['zone_id', 'shot_zone_basic', 'shot_zone_area'])
    return zone_id_df
zone_ids = create_zone_ids_df(sorted_df)

sorted_df = sorted_df.merge(zone_ids, on=['shot_zone_basic', 'shot_zone_area'])
######################################################################
#rearrange columns for better visability
# clean = sorted_df[['name','pos','age','player_id', 'team_name', 'team_id', 'game_date',
#        'game_id', 'game_event_id','season', 'period',
#        'minutes_remaining', 'seconds_remaining', 'shot_made_flag',
#        'action_type', 'shot_zone_basic', 'shot_zone_area', 'shot_zone_range',
#        'shot_type', 'shot_distance', 'x', 'y', 'dribbles', 'touch_time',
#        'opponent', 'opp_id', 'defender_name', 'defender_distance', 'shot_clock', 'htm', 'vtm',
#        'is_home', 'prev_shot_made', 'prev_2_made', 'prev_3_made', 'ts%', '3par', 'usg%']]
