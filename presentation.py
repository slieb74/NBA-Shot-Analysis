############################### IMPORTS ###############################
if True:
    import itertools, math, time, re, pickle

    import plotly
    import plotly.plotly as py
    import plotly.graph_objs as go
    plotly.offline.init_notebook_mode(connected=True)

    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    pd.set_option('display.max_columns',100)

    import ipywidgets as widgets
    from ipywidgets import interact

    import warnings
    warnings.filterwarnings('ignore')

    from court import court_shapes

    from shot_chart_viz import acquire_playerPic, get_team_logo, get_season_stats, get_team_stats, draw_court

    cdict = {
        'blue': [(0.0, 0.6313725709915161, 0.6313725709915161), (0.25, 0.4470588266849518, 0.4470588266849518), (0.5, 0.29019609093666077, 0.29019609093666077), (0.75, 0.11372549086809158, 0.11372549086809158), (1.0, 0.05098039284348488, 0.05098039284348488)],
        'green': [(0.0, 0.7333333492279053, 0.7333333492279053), (0.25, 0.572549045085907, 0.572549045085907), (0.5, 0.4156862795352936, 0.4156862795352936), (0.75, 0.0941176488995552, 0.0941176488995552), (1.0, 0.0, 0.0)],
        'red': [(0.0, 0.9882352948188782, 0.9882352948188782), (0.25, 0.9882352948188782, 0.9882352948188782), (0.5, 0.9843137264251709, 0.9843137264251709), (0.75, 0.7960784435272217, 0.7960784435272217), (1.0, 0.40392157435417175, 0.40392157435417175)]}
    mymap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)
############################## LOAD DATA ##############################
df = pd.read_csv('final_df_1415.csv', index_col=0)

######################################################################
###########################--SHOT CHARTS--############################
######################################################################

########################--BUBBLE SHOT CHARTS--########################
def find_shootingPcts(shot_df, gridNum):
    x2 = shot_df.x[(shot_df['y']<425.1) & (shot_df.shot_type==2)]
    y2 = shot_df.y[(shot_df['y']<425.1) & (shot_df.shot_type==2)]

    x2_made = shot_df.x[(shot_df['shot_made_flag']==1) & (shot_df['y']<425.1) & (shot_df.shot_type==2)]
    y2_made = shot_df.y[(shot_df['shot_made_flag']==1) & (shot_df['y']<425.1) & (shot_df.shot_type==2)]

    #compute number of shots made and taken from each hexbin location
    hb_shot2 = plt.hexbin(x2, y2, gridsize=gridNum, extent=(-250,250,425,-50));
    plt.close()
    hb_made2 = plt.hexbin(x2_made, y2_made, gridsize=gridNum, extent=(-250,250,425,-50),cmap=plt.cm.Reds);
    plt.close()

    #compute shooting percentage
    ShootingPctLocs2 = hb_made2.get_array() / hb_shot2.get_array()
    ShootingPctLocs2[np.isnan(ShootingPctLocs2)] = 0 #makes 0/0s=0

    #############################################################################################################
    #############################################################################################################
    ###########################################  THREE POINTERS  ################################################
    #############################################################################################################
    #############################################################################################################

    x3 = shot_df.x[(shot_df['y']<425.1) & (shot_df.shot_type==3)]
    y3 = shot_df.y[(shot_df['y']<425.1) & (shot_df.shot_type==3)]

    x3_made = shot_df.x[(shot_df['shot_made_flag']==1) & (shot_df['y']<425.1) & (shot_df.shot_type==3)]
    y3_made = shot_df.y[(shot_df['shot_made_flag']==1) & (shot_df['y']<425.1) & (shot_df.shot_type==3)]

    #compute number of shots made and taken from each hexbin location
    hb_shot3 = plt.hexbin(x3, y3, gridsize=gridNum, extent=(-250,250,425,-50));
    plt.close()
    hb_made3 = plt.hexbin(x3_made, y3_made, gridsize=gridNum, extent=(-250,250,425,-50),cmap=plt.cm.Reds);
    plt.close()

    #compute shooting percentage
    ShootingPctLocs3 = hb_made3.get_array() / hb_shot3.get_array()
    ShootingPctLocs3[np.isnan(ShootingPctLocs3)] = 0 #makes 0/0s=0

    return (ShootingPctLocs2, hb_shot2, ShootingPctLocs3, hb_shot3)

def freq_shooting_plot(player_name, gridNum=25):
    plot_size=(10,8)
    shot_df = df[df.name==player_name]

    from matplotlib.patches import Circle
    #compute shooting percentage and # of shots
    (ShootingPctLocs2, shotNumber2) = find_shootingPcts(shot_df, gridNum)[0:2]
    (ShootingPctLocs3, shotNumber3) = find_shootingPcts(shot_df, gridNum)[2:]

    #draw figure and court
    fig = plt.figure(figsize=plot_size)#(12,7)
    ax = plt.axes([0.1, 0.1, 0.8, 0.8]) #where to place the plot within the figure
    draw_court(outer_lines=False)
    plt.xlim(-250,250)
    plt.ylim(400, -25)

    #draw player image
    zoom = np.float(plot_size[0])/(12.0*2) #how much to zoom the player's pic. I have this hackily dependent on figure size
    img = acquire_playerPic(shot_df.player_id, zoom)
    ax.add_artist(img)

    ############################################  TWO POINTERS  #################################################
    cmap = mymap.from_list('Color Map',[(0,'#ff0000'),(.45,'#ffff00'),(.6,'#00ff00'), (1,'#004d00')])
    #draw circles
    for i, shots in enumerate(ShootingPctLocs2):
        restricted2 = Circle(shotNumber2.get_offsets()[i], radius=shotNumber2.get_array()[i],
                            color=cmap(shots),alpha=1, fill=True)
        if restricted2.radius > 240/gridNum: restricted2.radius=240/gridNum
        ax.add_patch(restricted2)

    #draw color bar
    ax2 = fig.add_axes([0.95, 0.1, 0.02, 0.8])
    cb = matplotlib.colorbar.ColorbarBase(ax2,cmap=cmap, orientation='vertical')
    cb.set_label('Field Goal %', labelpad=20)
    cb.set_ticks([0.0, 0.25, .485, 0.75, 1.0])
    cb.set_ticklabels(['0%','25%','48.5%\nLg Avg', '75%', '100%'])

    ###########################################  THREE POINTERS  ################################################
    #plotting 3 pointers separately to account for expected lower fg% from deep
    cmap3 = mymap.from_list('Color Map',[(0,'#ff0000'),(.35,'#ffff00'),(.6,'#00ff00'),(1,'#004d00')])
    #draw circles
    for i, shots in enumerate(ShootingPctLocs3):
        restricted3 = Circle(shotNumber3.get_offsets()[i], radius=shotNumber3.get_array()[i],
                            color=cmap3(shots),alpha=1, fill=True)
        if restricted3.radius > 240/gridNum: restricted3.radius=240/gridNum
        ax.add_patch(restricted3)

    #draw color bar
    ax3 = fig.add_axes([1.1, 0.1, 0.02, 0.8])
    cb3 = matplotlib.colorbar.ColorbarBase(ax3,cmap=cmap3, orientation='vertical')
    cb3.set_label('Three Point %',labelpad=-8)
    cb3.set_ticks([0.0, 0.25,.35, 0.5, 0.75, 1.0])
    cb3.set_ticklabels(['0%','25%','35% - Lg Avg', '50%','75%', '100%'])

    ax.set_title(shot_df.name.unique()[0] +' - Shot Chart 2014-15')
    #plot season stats
    ax.text(135,395,get_season_stats(player_name)[1])

    plt.show()

#################PLOT TEAM FREQUENCY SHOT CHART (MATPLOTLIB)#################
def team_freq_plot(team, gridNum=25):
    plot_size=(10,8)
    team_df = df[df.team_name==team]

    from matplotlib.patches import Circle
    #compute shooting percentage and # of shots
    (ShootingPctLocs2, shotNumber2) = find_shootingPcts(team_df, gridNum)[0:2]
    (ShootingPctLocs3, shotNumber3) = find_shootingPcts(team_df, gridNum)[2:]

    #draw figure and court
    fig = plt.figure(figsize=plot_size)
    ax = plt.axes([0.1, 0.1, 0.8, 0.8]) #where to place the plot within the figure
    draw_court(outer_lines=False)
    plt.xlim(-250,250)
    plt.ylim(400, -25)

    #draw team image
    team_ac = team_df.htm[team_df.is_home==1].unique()[0]
    zoom = 1 #np.float(plot_size[0])/(8.0)
    img = get_team_logo(team_ac, zoom)
    ax.add_artist(img)

    ############################################  TWO POINTERS  #################################################
    cmap = mymap.from_list('Color Map',[(0,'#ff0000'),(.45,'#ffff00'),(.6,'#00ff00'), (1,'#004d00')])
    #draw circles
    for i, shots in enumerate(ShootingPctLocs2):
        restricted2 = Circle(shotNumber2.get_offsets()[i], radius=shotNumber2.get_array()[i],
                            color=cmap(shots),alpha=1, fill=True)
        if restricted2.radius > 240/gridNum: restricted2.radius=240/gridNum
        ax.add_patch(restricted2)

    #draw color bar
    ax2 = fig.add_axes([0.95, 0.1, 0.02, 0.8])
    cb = matplotlib.colorbar.ColorbarBase(ax2,cmap=cmap, orientation='vertical')
    cb.set_label('Field Goal %', labelpad=20)
    cb.set_ticks([0.0, 0.25, .485, 0.75, 1.0])
    cb.set_ticklabels(['0%','25%','48.5%\nLg Avg', '75%', '100%'])

    ###########################################  THREE POINTERS  ################################################
    #plotting 3 pointers separately to account for expected lower fg% from deep
    cmap3 = mymap.from_list('Color Map',[(0,'#ff0000'),(.35,'#ffff00'),(.6,'#00ff00'),(1,'#004d00')])
    #draw circles
    for i, shots in enumerate(ShootingPctLocs3):
        restricted3 = Circle(shotNumber3.get_offsets()[i], radius=shotNumber3.get_array()[i],
                            color=cmap3(shots),alpha=1, fill=True)
        if restricted3.radius > 240/gridNum: restricted3.radius=240/gridNum
        ax.add_patch(restricted3)

    #draw color bar
    ax3 = fig.add_axes([1.1, 0.1, 0.02, 0.8])
    cb3 = matplotlib.colorbar.ColorbarBase(ax3,cmap=cmap3, orientation='vertical')
    cb3.set_label('Three Point %',labelpad=-8)
    cb3.set_ticks([0.0, 0.25,.35, 0.5, 0.75, 1.0])
    cb3.set_ticklabels(['0%','25%','35% - Lg Avg', '50%','75%', '100%'])


    ax.set_title(team_df.team_name.unique()[0] +' - Shot Chart 2014-15')
    #plot season stats
    ax.text(150,395,get_team_stats(team)[1])
    plt.show()

########################--GROUPED SHOT CHART--########################
def grouped_plot(feature):
    groups = df.groupby(feature)
    colors = np.linspace(0,1,len(groups))

    color_list = ['aliceblue', 'aqua', 'steelblue','violet', 'blue',
              'blueviolet', 'brown', 'cadetblue',
              'chartreuse', 'darkgreen', 'darkmagenta', 'tomato',
             'gold', 'red', 'slategray']
    counter=0
    data = []
    for g, c in zip(groups, colors):
        data.append(go.Scattergl(
            x = g[1].x,
            y = g[1].y,
            mode = 'markers',
            name = g[0],
            marker= dict(symbol='circle', size=7,
                         line={'width':1}, opacity=0.7, color=color_list[counter]),
            text = g[0],
            hoverinfo = 'text')
        )
        counter+=1

    layout = go.Layout(
        title='Shot Distribution by ' + feature.title(),
        showlegend =True,
        xaxis={'showgrid':False, 'range':[-250,250]},
        yaxis={'showgrid':False, 'range':[-47.5,500]},
        height = 600,
        width = 750,
        hovermode='closest',
        shapes=court_shapes)

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.iplot(fig, filename = 'Shot Zone Breakdown')

##########################--SHOT FREQ HEATMAP--#########################
def shot_freq_heatmap(name):
    if name in df.name.unique():
        df_ = df[df.name==name]
        z_max=40
        z_min=0
    else:
        df_ = df[df.team_name==name]
        z_max=250
        z_min=5

    x_make = df_[df_.shot_made_flag == 1]['x']
    y_make = df_[df_.shot_made_flag == 1]['y']
    x_miss = df_[df_.shot_made_flag == 0]['x']
    y_miss = df_[df_.shot_made_flag == 0]['y']

    x = np.concatenate([x_make, x_miss])
    y = np.concatenate([y_make, y_miss])

    makes = go.Scatter(
        x=x_make,
        y=y_make,
        mode='markers',
        name='Make',
        showlegend=True,
        marker=dict(
            symbol='circle',
            opacity=0.7,
            color='green',
            size=4,
            line=dict(width=1),
        )
    )
    misses = go.Scatter(
        x=x_miss,
        y=y_miss,
        mode='markers',
        name='Miss',
        showlegend=True,
        marker=dict(
            symbol='x',
            opacity=0.7,
            color='yellow',
            size=4,
            line=dict(width=1),
        )
    )
    trace3 = go.Histogram2d(
        x=x,
        y=y,
        zmax=40,
        zmin=0,
    #     nbinsx=20,
    #     nbinsy=20,
        zsmooth='best',
        autobinx=True,
        autobiny=True,
        reversescale=False,
        opacity=.75,
        #zauto=True,
        #autocolorscale=True,
    )

    layout = go.Layout(
        xaxis=dict( ticks='', showgrid=False, zeroline=False, nticks=20, range=[-250,250]),
        yaxis=dict( ticks='', showgrid=False, zeroline=False, nticks=20, range=[-47.5,450]),
        autosize=False,
        height=600,
        width=750,
        hovermode='closest',
        shapes= court_shapes,
        title= name + ' - Shot Frequency Heatmap',
        showlegend=True,
        legend=dict(x=1.2, y=1),
    )

    data = [trace3]#, makes, misses]
    fig = go.Figure(data=data, layout=layout)

    plotly.offline.iplot(fig)

############################--PPS HEATMAP--#############################
#FIX FUNCTION - CHANGE ZONE TO FEATURE
def pps_heatmap(feature):
    pps_tab=pd.crosstab(df.team_name, df[feature], values=df.pps, aggfunc='mean',margins=False).fillna(0)

    team_heatmap = go.Heatmap(z=[np.array((pps_tab[pps_tab.index==pps_tab.index[i]])) for i in range(len(pps_tab.index))],
                       x=pps_tab.columns,
                       y= [team.split(' ')[-1] for team in pps_tab.index]
                      )

    layout = go.Layout(
        title='Points Per Shot Heatmap',
        xaxis = dict(ticks='', nticks=len(pps_tab.columns), automargin=True),
        yaxis = dict(ticks='', nticks=len(pps_tab.index), automargin=True),
    )

    fig = go.Figure(data=[team_heatmap], layout=layout)
    plotly.offline.iplot(fig, filename='labelled-heatmap')

########################--FREQUENCY BAR PLOT--########################
def freq_bar_plots(feature, round_=False):
    df_ = df.copy()
    if round_==True:
        df_[feature] = round(df_[feature])

    feat_tab = pd.crosstab(df_[feature], df_.shot_made_flag, margins=True)
    feat_tab['fg_pct'] = round(feat_tab[1]/feat_tab['All'],3)

    tab=feat_tab.drop(columns='All')[:-1]
    make_text= [str(round(t*100,1)) + '%' for t in tab.fg_pct]
    miss_text= [str(round((1-t)*100,1)) + '%' for t in tab.fg_pct]

    trace1 = go.Bar(
        x=tab.index,
        y=tab[1],
        name='Makes',
        text= make_text ,
        textposition = 'inside',
        textfont=dict(
            family='sans serif', size=12, color='white'),
        marker=dict(
            color='red'),
        opacity=0.75
    )
    trace2 = go.Bar(
        x=tab.index,
        y=tab[0],
        name='Misses',
        text= miss_text,
        textposition = 'inside',
        textfont=dict(
            family='sans serif', size=10, color='white'),
        marker=dict(
            color='blue'),
        opacity=0.75
    )

    line = go.Scatter(
        x=tab.index,
        y=tab[1],
        mode='markers+lines',
        name='# Makes',
        hoverinfo='skip',
        line=dict(
        color='black', width=.75)
    )

    data = [trace1, trace2]#, line]
    layout = go.Layout(
        barmode='stack',
        title='FG% by ' + feature.title().replace('_',' '),
        showlegend =True,
        xaxis=dict(
            automargin=True,
            autorange=True,
            ticks='',
            showticklabels=True,
            #tickangle=25,
            title=feature.replace('_',' ').title()
        ),
        yaxis=dict(
            automargin=True,
            ticks='',
            showticklabels=True,
            title='# of Shots'
        )
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.iplot(fig, filename='stacked-bar')

########################--PERCENTAGE BAR CHART--########################
def pct_bar_plots(feature, round_=False, player=None, team=None):
    if round_==True:
        df_ = df.copy()
        df_[feature] = round(df_[feature])
    else:
        df_ = df

    if player:
        df_ = df[df.name==player.title()]
        title= player.title() + ' - FG% by ' + feature.title().replace('_',' ')
    elif team:
        df_ = df[df.team_name==team.title()]
        title= team.title() + ' - FG% by ' + feature.title().replace('_',' ')
    else:
        df_ = df
        title= 'FG% by ' + feature.title().replace('_',' ')


    c_tab=pd.crosstab(df_[feature], df_.shot_made_flag, margins=True)
    c_tab['pct_made'] = c_tab[1]/c_tab.All
    c_tab['pct_missed'] = 1-c_tab.pct_made

    made_text= [str(round(t*100,1)) + '%' for t in c_tab.pct_made]
    missed_text= [str(round(t*100,1)) + '%' for t in c_tab.pct_missed]

    trace1 = go.Bar(
        x=c_tab.index,
        y=c_tab.pct_made,
        name='Makes',
        text= made_text,
        textposition = 'auto',
        textfont=dict(
            family='sans serif',
            size=12, color='white'),
        marker=dict(
            color='red'),
        opacity=0.75
    )
    trace2 = go.Bar(
        x=c_tab.index,
        y=c_tab.pct_missed,
        name='Misses',
        text= missed_text,
        textposition = 'auto',
        textfont=dict(
            family='sans serif',
            size=12, color='white'),
        marker=dict(
            color='blue'),
        opacity=0.75,
    )

    data = [trace1, trace2]
    layout = go.Layout(
        barmode='stack',
        title= title,
        showlegend =True,
        xaxis=dict(
            automargin=True,
            autorange=True,
            ticks='',
            showticklabels=True,
            title=feature.replace('_',' ').title()
        ),
        yaxis=dict(
            automargin=True,
            ticks='',
            showticklabels=True,
            title='FG %'
        )
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.iplot(fig, filename='stacked-bar')
