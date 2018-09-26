import requests
import pandas as pd
import numpy as np
from data.all_players_list import players_list
import time
from court import court_shapes

vets = [player[0:5] for player in players_list if (player[3] >1990) & (player[4] >2014)]

vets_df = pd.DataFrame(vets, columns=['ID', 'Name', 'Active', 'RookieYear', 'LastSeasonPlayed'])
vets_df = vets_df.drop(columns=['Active', 'RookieYear', 'LastSeasonPlayed'])

player_ids = [player[0] for player in vets]

#MULTIPLE YEARS
sc_url_1 = 'https://stats.nba.com/stats/shotchartdetail?AheadBehind=&CFID=33&CFPARAMS='
sc_url_2 = '&ClutchTime=&Conference=&ContextFilter=&ContextMeasure=FGA&DateFrom=&DateTo=&Division=&EndPeriod=10&EndRange=28800&GROUP_ID=&GameEventID=&GameID=&GameSegment=&GroupID=&GroupMode=&GroupQuantity=5&LastNGames=0&LeagueID=00&Location=&Month=0&OnOff=&OpponentTeamID=0&Outcome=&PORound=0&Period=0&PlayerID='
sc_url_3 = '&PlayerID1=&PlayerID2=&PlayerID3=&PlayerID4=&PlayerID5=&PlayerPosition=&PointDiff=&Position=&RangeType=0&RookieYear=&Season='
sc_url_4 = '&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StartPeriod=1&StartRange=0&StarterBench=&TeamID=0&VsConference=&VsDivision=&VsPlayerID1=&VsPlayerID2=&VsPlayerID3=&VsPlayerID4=&VsPlayerID5=&VsTeamID='

headers = requests.utils.default_headers()
headers.update({
    "user-agent": 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
    })

#year in yyyy-yy format (i.e. '2017-18')
def get_all_players_shot_data(player_ids, year):
    all_shots = []
    c=0
    start=time.time()
    for player_id in player_ids:
        full_url = sc_url_1 + str(year) + sc_url_2 + str(player_id) + sc_url_3 + str(year) + sc_url_4
        shots = requests.get(full_url, headers=headers).json()
        all_shots.append(shots)
        time.sleep(1)
        c+=1
        if c%50==0:
            print('Runtime: {} seconds. {} players completed'.format(time.time()-start, c))
    return all_shots

def convert_dict_to_df(all_shot_data):
    start=time.time()

    league_avgs = all_shot_data[0]['resultSets'][1]['rowSet']
    league_avg_columns = all_shot_data[0]['resultSets'][1]['headers']
    league_avgs_df = pd.DataFrame.from_records(league_avgs, columns=league_avg_columns)

    columns = all_shot_data[0]['resultSets'][0]['headers']

    df_list=[]

    for player in all_shot_data:
        data = player['resultSets'][0]['rowSet']
        player_df = pd.DataFrame.from_records(data, columns=columns)
        df_list.append(player_df)

    df = pd.concat(df_list, ignore_index=True)
    print('Total Runtime: {} seconds.'.format(time.time()-start))

    return df, league_avgs_df

all_shots_1415 = get_all_players_shot_data(player_ids, '2014-15')
shots_1415_df, lg_avgs_1415 = convert_dict_to_df(all_shots_1415)

shots_1415_df.to_csv('data/shots_1415.csv')
