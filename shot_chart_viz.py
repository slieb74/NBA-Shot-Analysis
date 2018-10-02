import requests, time, itertools, math, shutil, matplotlib
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import numpy as np

from court import court_shapes

pd.set_option('display.max_columns',40)
import warnings
warnings.filterwarnings('ignore')

import ipywidgets as widgets
from ipywidgets import interact

import plotly
import plotly.plotly as py
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=True)

#####READ DATAFRAME#####
df = pd.read_csv('data/sorted_df_14_15.csv',index_col=0)

#####DRAW PLAYER SHOT CHART (PLOTLY)#####
def draw_shot_chart(name):
    player = df[df.name==name]

    missed_shot_trace = go.Scattergl(
        x = player[player.shot_made_flag == 0]['x'],
        y = player[player.shot_made_flag == 0]['y'],
        mode = 'markers',
        name = 'Miss',
        marker={'color':'blue', 'size':5}
    )
    made_shot_trace = go.Scattergl(
        x = player[player.shot_made_flag == 1]['x'],
        y = player[player.shot_made_flag == 1]['y'],
        mode = 'markers',
        name='Make',
        marker={'color':'red', 'size':5}
    )

    data = [missed_shot_trace, made_shot_trace]
    layout = go.Layout(
        title= name + ' Shot Chart 2014-2015',
        showlegend =True,
        xaxis={'showgrid':False, 'range':[-300,300]},
        yaxis={'showgrid':False, 'range':[-100,500]},
        height = 600,
        width = 650,
        shapes=court_shapes)

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.iplot(fig, filename = name + ' Shot Chart')

#####DRAW TEAM SHOT CHART (PLOTLY)#####
def draw_team_sc(team):
    team_df = df[df.team_name==team]

    missed_shot_trace = go.Scattergl(
        x = team_df[team_df['shot_made_flag'] == 0]['x'],
        y = team_df[team_df['shot_made_flag'] == 0]['y'],
        mode = 'markers',
        name = 'Miss',
        marker={'color':'blue', 'size':5}
    )
    made_shot_trace = go.Scattergl(
        x = team_df[team_df['shot_made_flag'] == 1]['x'],
        y = team_df[team_df['shot_made_flag'] == 1]['y'],
        mode = 'markers',
        name='Make',
        marker={'color':'red', 'size':5}
    )

    data = [missed_shot_trace, made_shot_trace]
    layout = go.Layout(
        title= team + ' Shot Chart 2014-2015',
        showlegend =True,
        xaxis={'showgrid':False, 'range':[-300,300]},
        yaxis={'showgrid':False, 'range':[-100,500]},
        height = 600,
        width = 650,
        shapes=court_shapes)

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.iplot(fig, filename = team + ' Shot Chart')

#####DROPDOWNS#####
if False:
    # team_dropdown = widgets.Dropdown(
    #     options = sorted(list(set(df.team_name))),
    #     value='New York Knicks',
    #     description='Team:',
    #     disabled=False,
    # )
    #
    # interact(draw_team_sc, team=team_dropdown);

    player_dropdown = widgets.Dropdown(
    options = sorted(list(set(df.name))),
    value='James Harden',
    description='Player:',
    disabled=False
    )

    grid_slider = widgets.IntSlider(
    value=15,
    min=5, max=60,
    step=5,
    description='Bubble Size:',
    disabled=False,
    )

    interact(freq_shooting_plot, player_name=player_dropdown, gridNum=grid_slider);

#####DRAW COURT MATPLOTLIB#####
def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    from matplotlib.patches import Circle, Rectangle, Arc
    if ax is None:
        ax = plt.gca()
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]
    if outer_lines:
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    for element in court_elements:
        ax.add_patch(element)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

#####FIND PLAYER FG% FOR EACH HEX#####
def find_shootingPcts(shot_df, gridNum):
    x = shot_df.x[shot_df['y']<425.1]
    y = shot_df.y[shot_df['y']<425.1]

    x_made = shot_df.x[(shot_df['shot_made_flag']==1) & (shot_df['y']<425.1)]
    y_made = shot_df.y[(shot_df['shot_made_flag']==1) & (shot_df['y']<425.1)]

    #compute number of shots made and taken from each hexbin location
    hb_shot = plt.hexbin(x, y, gridsize=gridNum, extent=(-250,250,425,-50));
    plt.close()
    hb_made = plt.hexbin(x_made, y_made, gridsize=gridNum, extent=(-250,250,425,-50),cmap=plt.cm.Reds);
    plt.close()

    #compute shooting percentage
    ShootingPctLocs = hb_made.get_array() / hb_shot.get_array()
    ShootingPctLocs[np.isnan(ShootingPctLocs)] = 0 #makes 0/0s=0
    return (ShootingPctLocs, hb_shot)

#####SCRAPE PLAYER IMAGE#####
def acquire_playerPic(player_id, zoom, offset=(-165,400)):
    from matplotlib import offsetbox as osb
    ID = str(player_id.unique()[0])

    url = "http://stats.nba.com/media/players/230x185/"+ ID +".png"
    pic = requests.get(url,stream=True)

    with open(ID + '.png', 'wb') as out_file:
        shutil.copyfileobj(pic.raw, out_file)

    player_pic = plt.imread(ID + '.png')
    img = osb.OffsetImage(player_pic, zoom)
    img = osb.AnnotationBbox(img, offset,xycoords='data',pad=0.0, box_alignment=(1,0), frameon=False)

    return img

#####SCRAPE TEAM LOGO#####
def get_team_logo(team_acronym, zoom, offset=(-185,400)):
    from matplotlib import offsetbox as osb

    URL = 'https://www.nba.com/assets/logos/teams/primary/web/' + team_acronym + '.png'

    pic = requests.get(URL,stream=True)

    with open('scraped_images/team_images/' + str(team_acronym) + '.png', 'wb') as out_file:
        shutil.copyfileobj(pic.raw, out_file)

    team_pic = plt.imread('scraped_images/team_images/' + str(team_acronym) + '.png')
    img = osb.OffsetImage(team_pic, zoom)
    img = osb.AnnotationBbox(img, offset,xycoords='data',pad=0.0, box_alignment=(1,0), frameon=False)

    return img

#####COLOR MAP DICTIONARY#####
cdict = {
    'blue': [(0.0, 0.6313725709915161, 0.6313725709915161), (0.25, 0.4470588266849518, 0.4470588266849518), (0.5, 0.29019609093666077, 0.29019609093666077), (0.75, 0.11372549086809158, 0.11372549086809158), (1.0, 0.05098039284348488, 0.05098039284348488)],
    'green': [(0.0, 0.7333333492279053, 0.7333333492279053), (0.25, 0.572549045085907, 0.572549045085907), (0.5, 0.4156862795352936, 0.4156862795352936), (0.75, 0.0941176488995552, 0.0941176488995552), (1.0, 0.0, 0.0)],
    'red': [(0.0, 0.9882352948188782, 0.9882352948188782), (0.25, 0.9882352948188782, 0.9882352948188782), (0.5, 0.9843137264251709, 0.9843137264251709), (0.75, 0.7960784435272217, 0.7960784435272217), (1.0, 0.40392157435417175, 0.40392157435417175)]}
mymap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)
mymap = mymap.from_list('Color Map',[(0,'#ff0000'),(.45,'#ffff00'),(1,'#00ff00')])


####################CALCULATE SEASON STATS TO ADD TO CHART####################
def get_season_stats(player_name):
    player = df[df.name==player_name]

    stats = {}

    stats['NUM_GAMES'] = len(player.game_date.unique())
    stats['FG_PCT'] = player.groupby(by=['season']).mean().shot_made_flag.sum()
    stats['THREE_PT_PCT'] = player[player.shot_type==3].groupby(by=['season']).mean().shot_made_flag.sum()

    twos = player.groupby(['shot_type']).sum().iloc[0].shot_made_flag
    threes = player.groupby(['shot_type']).sum().iloc[1].shot_made_flag * 1.5
    stats['EFFECTIVE_FG_PCT'] = (twos+threes)/player.shape[0]

    stats['AVG_SHOT_DISTANCE'] = round(player.shot_distance.mean())
    stats['MOST_FGM'] = player.groupby('game_date').sum().shot_made_flag.max()
    stats['MOST_THREES_MADE'] = player[player.shot_type==3].groupby(by=['game_date']).sum().shot_made_flag.max()

    printout = """Games: {}\nFG: {:4.1%}\n3PT: {:4.1%}\nEFG: {:4.1%}\nAvg Shot Distance: {} ft.\nGame High: FGM - {}, 3PM - {}""".format(*[stats.get(k) for k in stats.keys()])

    return stats, printout

##################CALCULATE TEAM STATS TO ADD TO CHART########################
def get_team_stats(team):
    team_df = df[df.team_name==team]
    stats = {}

    stats['FG_PCT'] = team_df.groupby(by=['season']).mean().shot_made_flag.sum()
    stats['THREE_PT_PCT'] = team_df[team_df.shot_type==3].groupby(by=['season']).mean().shot_made_flag.sum()

    twos = team_df.groupby(['shot_type']).sum().iloc[0].shot_made_flag
    threes = team_df.groupby(['shot_type']).sum().iloc[1].shot_made_flag * 1.5
    stats['EFFECTIVE_FG_PCT'] = (twos+threes)/team_df.shape[0]

    stats['AVG_SHOT_DISTANCE'] = round(team_df.shot_distance.mean())

    printout = """FG: {:4.1%}\n3PT: {:4.1%}\nEFG: {:4.1%}\nAvg Shot Distance: {} ft.""".format(*[stats.get(k) for k in stats.keys()])

    return stats, printout

#################PLOT PLAYER FREQUENCY SHOT CHART (MATPLOTLIB)################
def freq_shooting_plot(player_name,gridNum=25):
    plot_size=(12,8)
    shot_df = df[df.name==player_name]

    from matplotlib.patches import Circle
    x = shot_df.x[shot_df['y']<425.1]
    y = shot_df.y[shot_df['y']<425.1]

    #compute shooting percentage and # of shots
    (ShootingPctLocs, shotNumber) = find_shootingPcts(shot_df, gridNum)

    #draw figure and court
    fig = plt.figure(figsize=plot_size)#(12,7)
    cmap = mymap #my modified colormap
    ax = plt.axes([0.1, 0.1, 0.8, 0.8]) #where to place the plot within the figure
    draw_court(outer_lines=False)
    plt.xlim(-250,250)
    plt.ylim(400, -25)

    #draw player image
    zoom = np.float(plot_size[0])/(12.0*2) #how much to zoom the player's pic. I have this hackily dependent on figure size
    img = acquire_playerPic(shot_df.player_id, zoom)
    ax.add_artist(img)

    #draw circles
    for i, shots in enumerate(ShootingPctLocs):
        restricted = Circle(shotNumber.get_offsets()[i], radius=shotNumber.get_array()[i],
                            color=cmap(shots),alpha=1, fill=True)
        if restricted.radius > 240/gridNum: restricted.radius=240/gridNum
        ax.add_patch(restricted)

    #draw color bar
    ax2 = fig.add_axes([0.95, 0.1, 0.02, 0.8])
    cb = matplotlib.colorbar.ColorbarBase(ax2,cmap=cmap, orientation='vertical')
    cb.set_label('Field Goal %')
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(['0%','25%', '50%','75%', '100%'])

    ax.set_title(shot_df.name.unique()[0] +' - Shot Chart 2014-15')
    #plot season stats
    ax.text(135,395,get_season_stats(player_name)[1])
    plt.show()
    return ax

#################PLOT TEAM FREQUENCY SHOT CHART (MATPLOTLIB)#################
def team_freq_plot(team, gridNum=25):
    plot_size=(8,8)
    team_df = df[df.team_name==team]

    from matplotlib.patches import Circle

    #compute shooting percentage and # of shots
    (ShootingPctLocs, shotNumber) = find_shootingPcts(team_df, gridNum)

    #draw figure and court
    fig = plt.figure(figsize=plot_size)
    cmap = mymap #my modified colormap
    ax = plt.axes([0.1, 0.1, 0.8, 0.8]) #where to place the plot within the figure
    draw_court(outer_lines=False)
    plt.xlim(-250,250)
    plt.ylim(400, -25)

    #draw team image
    team_ac = team_df.htm[team_df.is_home==1].unique()[0]
    zoom = 1 #np.float(plot_size[0])/(8.0)
    img = get_team_logo(team_ac, zoom)
    ax.add_artist(img)

    #draw circles
    for i, shots in enumerate(ShootingPctLocs):
        restricted = Circle(shotNumber.get_offsets()[i], radius=shotNumber.get_array()[i],
                            color=cmap(shots),alpha=.95, fill=True)
        if restricted.radius > 240/gridNum: restricted.radius=240/gridNum
        ax.add_patch(restricted)

    #draw color bar
    ax2 = fig.add_axes([0.95, 0.1, 0.02, 0.8])
    cb = matplotlib.colorbar.ColorbarBase(ax2,cmap=cmap, orientation='vertical')
    cb.set_label('Field Goal %')
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(['0%','25%', '50%','75%', '100%'])

    ax.set_title(team_df.team_name.unique()[0] +' - Shot Chart 2014-15')
    #plot season stats
    ax.text(150,395,get_team_stats(team)[1])
    plt.show()
    return ax
