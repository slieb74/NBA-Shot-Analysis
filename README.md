# NBA Shot Analysis

## Goal
Build a classification model to predict whether and NBA shot will go in or not, and create visualizations to help general managers, coaches, and players identify shooting patterns, eliminate bad shots, and optimize their strategy to increase shooting efficiency.

## ETL
I gathered my data from three sources:
 - Shot location data scraped from stats.nba.com (see my <a href="https://towardsdatascience.com/using-python-pandas-and-plotly-to-generate-nba-shot-charts-e28f873a99cb">blog post</a> for more detail)
 - Player tracking data from nbasavant.com
 - Defensive stats from basketball-reference.com
 
Since the NBA stopped providing tracking data such as the number of dribbles, and defender distance in the middle of the 2016 season, I focused my project on the 2014-15 season. I gathered data on over 200,000 shots, with features including, but not limited to:
 - Shot distance, (x,y) coordinates, and shot zone
 - Touch time and number of dribbles
 - Name and distance of the closest defender
 - Game context stats such as shot clock remaining, period, game clock
 - Shot type (jump shot, dunk, etc.)

I wanted to add more context to each shot, so I added advanced defensive stats for each defender (Block %, Defensive Win Shares/48, Defensive Box Score Plus Minus) and team (Defensive Rating). 

The data I gathered had two different zone breakdowns, one which detailed the directional area of the court (left, right or center) and the other which detailed a more precise location (paint, corner 3, etc.). I combined these into 15 zones, as seen below, and for every player I calculated their Field Goal % (FG%) in each zone so that my model would have a better understanding of the quality of the shot. 

<img src="https://github.com/slieb74/NBA-Shot-Analysis/blob/master/images/shot_zones.png">

I have never been a fan of the argument that momentum impacts basketball games, and have often argued against the concept of a "hot hand" which posits that a player is more likely to hit a shot if they have hit consecutive prior shots. In an attempt to disprove this hypothesis, I engineered new features that detailed whether the shooter has scored their previous 1, 2, and 3 shots. My models found that hitting prior shots did not have a significant impact on whether a player will score their next shot.

## Visualizations
I wanted to create a wide range of visualizations that would show the frequency and efficiency of player's and team's shots.

#### Binned Shot Chart
The first visualization I made is a binned shot chart that breaks the court down into equally sized hexes and groups nearby shots into bubbles, with the size determined by frequency and color by FG%. The color scale differed for two's and three's to account for the point value of each shot. I also added the player's image and some additional stats to the chart. In my dashboard, there is a dropdown where you can select any player, and there is also an option to change the bubble size depending on if you want to see a more precise or broad shot chart. 

<img src="https://github.com/slieb74/NBA-Shot-Analysis/blob/master/images/sc_shot_chart.png">

I made similar charts for each team, where you can get a strong sense of their shooting efficiency and frequency distribution.

<img src="https://github.com/slieb74/NBA-Shot-Analysis/blob/master/images/rockets.png">

#### Frequency Shot Heatmap
In order to get a better sense of where players and teams are shooting from, disregarding efficiency, I designed a heatmap to show the locations where they most frequently shoot from, complete with a dropdown that allows you to select any player or team.

<img src="https://github.com/slieb74/NBA-Shot-Analysis/blob/master/images/harden%20heatmap.png">

#### FG Frequency Bar Plot
To visualize how the league distributes its shots, I added an interactive bar plot to my dashboard that shows FG% and the number of shots for a given feature that can be selected from a dropdown.

<img src="https://github.com/slieb74/NBA-Shot-Analysis/blob/master/images/shot_dist.png">

#### FG Percentage Bar Plot
To visualize FG% without focusing on frequency, I built an interactive bar plot that shows leaguewide FG% and the number of shots for a range of features that can be selected from a dropdown.

<img src="https://github.com/slieb74/NBA-Shot-Analysis/blob/master/images/fg%20by%20zone.png">

#### Team Points Per Shot Heatmap Matrix
I wanted to compare how teams perform in different contexts, so created a heatmap matrix that helps visualize which teams under- and overperform in certain aspects. The color of each box is determined by the team's points per shot (PPS) provided the selected feature/context. This gives teams a better sense of where they need to improve and how they stack up among the rest of the league.

<img src="https://github.com/slieb74/NBA-Shot-Analysis/blob/master/images/team%20heatmap.png">

## Machine Learning Models
I trained 6 different machine learning classification models to predict whether a given shot would go in. The models I used were the following:
 - Logistic Regression
 - Random Forest
 - Gradient Boosting
 - AdaBoost
 - XGBoost
 - Neural Network
 
For each model, I went through a cross-validation process to help narrow down my feature set into only the most important ones that did not show signs of multicollinearity with other included features. I ultimately narrowed down my initial set of over 20 features to the following 6:
 - Shot Distance
 - Zone FG%
 - Defensive Win Shares per 48 Minutes
 - Defender Distance
 - Touch Time
 - Shot Clock Remaining

###### Feature Importances (Gradient Boosting Classifier)
<img src="https://github.com/slieb74/NBA-Shot-Analysis/blob/master/images/gb%20feats.png">

Due to the inconsistency in scale of my numeric features (FG% is a decimal but shot distance is measured in feet), I used Scikit-Learn's MinMaxScaler to normalize and vectorize my data. My cross-validation process included hyperparameter tuning for each of my models by running a grid search with Stratified Kfold splits to ensure that the class balance remained consistent across all splits. 
For the Neural Network, I used one hidden layer that contained 50 nodes, 'relu' activation due to the lack of negative values, and the 'adam' optimizer to obtain my best results.

###### ROC curves
<p align="center">
  <img src="https://github.com/slieb74/NBA-Shot-Analysis/blob/master/images/all_roc_curves.png" height="500" width="600">
</p>

###### Confusion Matrix Comparisons (left: Logistic Regression, center: Gradient Boosting, right: Neural Network)
<img src="https://github.com/slieb74/NBA-Shot-Analysis/blob/master/images/cm%20logreg.png" height="250" width="270"/> <img src="https://github.com/slieb74/NBA-Shot-Analysis/blob/master/images/gb%20cm.png" height="250" width="270"/> <img src="https://github.com/slieb74/NBA-Shot-Analysis/blob/master/images/nn%20cm.png" height="250" width="270"/>

My best performing model depends on how a team values the bias/variance tradeoff and whether they would prefer to minimize false negatives (predicting a miss when its actually a make) or false positives (predicting a make when its in fact a miss). A more aggressive team would prefer the Neural Network, which only recommended not to shoot when it was extremely confident the shot would miss, but often recommended the player should shoot, albeit with less than a 40% accuracy. An aggressive team would be fine with this model because it limited false negatives and gave the team more chances to score.

On the other hand, a more conservative team might prefer the Gradient Boosting model, which correctly classified makes with a much higher accuracy, yet only recommended a shot ~30% of the time. It would likely lead to a higher FG%, but limits the potential scoring opportunities by recommending a team take fewer shots. The Logistic Regression model is far more balanced, sacrificing a lower overall accuracy for better precision and recall.

###### Model Results
<img src="https://github.com/slieb74/NBA-Shot-Analysis/blob/master/images/model%20results.png" height="300" width="750">

In addition to my individual models, I built a stacked ensemble model that trained the XGBoost, Random Forest, and AdaBoost classifiers, and then trained a Gradient Boosting model on output. This would, in theory, give less biased predictions by weighing multiple models; however, its results were unfortunately worse than my single layer models, so I discarded it.

## Shot Recommender
For each player, I built a recommender system that outputs certain zones where the player should shoot more or less frequently from. The concept is based on the player's PPS relative to the league average in each zone. A player who has a high expected PPS relative to the league average in a zone would be recommended to shoot there more frequently. Conversely, a player who shoots poorly in a zone would be recommended to shoot less. In the future, I want to tune this recommender by accounting for the player's frequency of shots in each zone, so that it does not recommend a player shoot more in a zone that already contains a high percentage of their total shots.
###### Recommender Output
<img src="https://github.com/slieb74/NBA-Shot-Analysis/blob/master/images/harden%20recs.png">

## Next Steps 
- Adjust the color scale of binned plots to display efficiency relative to the league average, either in terms of FG% or PPS
- Tune the shot recommender to provide ideal shot distributions
- Classify 2s and 3s differently in my models to see if certain models predict one shot type with higher accuracy than others
- Cluster similarly skilled shooters and recommend an optimal shooting lineup that covers each shot zone
- Host the project online using Dash and Flask instead of the Jupyter Notebook dashboard

## Credits
* <a href="https://grantland.com/contributors/kirk-goldsberry/">Kirk Goldsberry</a> for inspiring me to work on this project
* <a href="http://savvastjortjoglou.com/nba-shot-sharts.html">Savvas Tjortjoglou</a> for his court dimensions 
