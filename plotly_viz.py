############################### IMPORTS ###############################
if True:
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

    from court import court_shapes

    import warnings
    warnings.filterwarnings('ignore')

    import itertools, math, time, re, pickle

############################## LOAD DATA ##############################
df = pd.read_csv('data/clean_df_1415.csv',index_col=0)
zone_ids = pd.read_csv('data/zone_ids.csv',index_col=0)
zone_fg_pct = pd.read_csv('data/zone_fg_pct.csv',index_col=0)

############################## CLEANING DATA ############################
def basic_cleaning(df):
    df.period[df.period>5]=5
    df.touch_time[df.touch_time<0]=0
    df.touch_time[df.touch_time>24]=24
    #df.touch_time=round(df.touch_time*4)/4
    df.defender_distance[df.defender_distance>10]=10
    #df.shot_clock[df.shot_clock>3] = round(df.shot_clock[df.shot_clock>3]*4)/4
    df.shot_distance[df.shot_distance>40]=40
    df.blk_pct[df.blk_pct>10]=10
    df.dbpm[df.dbpm>5.5]=5.5
    df['pps'] = df.shot_type*df.shot_made_flag
#basic_cleaning(df)

######################################################################
######################################################################
###########################--SHOT CHARTS--############################
######################################################################
######################################################################

######################--DRAW PLAYER SHOT CHART--######################
def draw_shot_chart(name):
    player = df[df.name==name]

    missed_shot_trace = go.Scattergl(
        x = player[player.shot_made_flag == 0]['x'],
        y = player[player.shot_made_flag == 0]['y'],
        mode = 'markers',
        name = 'Make',
        marker= dict(color='blue', symbol='x', size=8, line={'width':1}, opacity=0.7),
        text = [str(sd) for sd in player[player.shot_made_flag == 0]['action_type']],
        hoverinfo = 'text'
    )
    made_shot_trace = go.Scattergl(
        x = player[player.shot_made_flag == 1]['x'],
        y = player[player.shot_made_flag == 1]['y'],
        mode = 'markers',
        name='Make',
        marker= dict(color='red', symbol='circle', size=8, line={'width':1}, opacity=0.7),
        text = [str(sd) for sd in player[player.shot_made_flag == 1]['action_type']],
        hoverinfo = 'text'
    )

    data = [missed_shot_trace, made_shot_trace]
    layout = go.Layout(
        title= name + ' Shot Chart 2014-2015',
        showlegend =True,
        xaxis={'showgrid':False, 'range':[-250,250]},
        yaxis={'showgrid':False, 'range':[-47.5,500]},
        height = 600,
        width = 650,
        shapes=court_shapes)

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.iplot(fig, filename = name + ' Shot Chart')

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
        shapes=court_shapes)

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.iplot(fig, filename = 'Shot Zone Breakdown')

########################--FREQUENCY BAR PLOT--########################
def freq_bar_plots(df, feature, round_=False):
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

    data = [trace1, trace2, line]
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
def pct_bar_plots(feature, dataframe, round_=False, player=None, team=None):
    if round_==True:
        df_ = dataframe.copy()
        df_[feature] = round(df_[feature])
    else:
        df_ = dataframe

    if player:
        df = df_[df_.name==player.title()]
        title= player.title() + ' - FG% by ' + feature.title().replace('_',' ')
    elif team:
        df = df_[df_.team_name==team.title()]
        title= team.title() + ' - FG% by ' + feature.title().replace('_',' ')
    else:
        df = df_
        title= 'FG% by ' + feature.title().replace('_',' ')


    test=pd.crosstab(df[feature], df.shot_made_flag, margins=True)
    test['pct_made'] = test[1]/test.All
    test['pct_missed'] = 1-test.pct_made

    made_text= [str(round(t*100,1)) + '%' for t in test.pct_made]
    missed_text= [str(round(t*100,1)) + '%' for t in test.pct_missed]

    trace1 = go.Bar(
        x=test.index,
        y=test.pct_made,
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
        x=test.index,
        y=test.pct_missed,
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
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.iplot(fig, filename='stacked-bar')

############################--PPS HEATMAP--#############################
#FIX FUNCTION - CHANGE ZONE TO FEATURE
def pps_heatmap(df, feature):
    pps_tab=pd.crosstab(df.team_name, df[feature], values=df.pps, aggfunc='mean',margins=False).fillna(0)

    team_heatmap = go.Heatmap(z=[np.array((pps_tab[pps_tab.index==pps_tab.index[i]])) for i in range(len(pps_tab.index))],
                       x=pps_tab.columns,
                       y= [team.split(' ')[-1] for team in pps_tab.index]
                      )

    layout = go.Layout(
        title='Points Per Shot Heatmap',
        xaxis = dict(ticks='', nticks=len(pps_tab.columns)),
        yaxis = dict(ticks='', nticks=len(pps_tab.index)),
    )

    fig = go.Figure(data=[team_heatmap], layout=layout)
    plotly.offline.iplot(fig, filename='labelled-heatmap')

#############################--PIE CHART--#############################
def feature_pie_charts(feature):
    labels = df[feature].unique()
    values = df[feature].value_counts()
    colors = ['#FEBFB3', '#E1396C', '#005eff', '#D0F9B1']

    trace = go.Pie(labels=labels, values=values,
                   hoverinfo='label+percent', textinfo='value+percent',
                   textfont=dict(size=20),
                   marker=dict(colors=colors,
                               line=dict(color='#000000', width=1)))

    plotly.offline.iplot([trace], filename='styled_pie_chart')

##########################--SHOT FREQ HEATMAP--#########################
def shot_freq_heatmap(name):
    player = df[df.name==name]

    x_make = player[player.shot_made_flag == 1]['x']
    y_make = player[player.shot_made_flag == 1]['y']
    x_miss = player[player.shot_made_flag == 0]['x']
    y_miss = player[player.shot_made_flag == 0]['y']

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
        title= name + ' - Shot Frequency',
        showlegend=True,
        legend=dict(x=1.2, y=1),
    )

    data = [trace3, makes, misses]
    fig = go.Figure(data=data, layout=layout)

    plotly.offline.iplot(fig)
