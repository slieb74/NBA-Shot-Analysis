############################### IMPORTS ###############################
if True:
    import pandas as pd
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    import itertools, math, time, re, pickle

    import warnings
    warnings.filterwarnings('ignore')

    import plotly
    import plotly.plotly as py
    import plotly.graph_objs as go
    plotly.offline.init_notebook_mode(connected=True)

    from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, ShuffleSplit
    from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, auc, confusion_matrix, precision_score, recall_score, roc_curve, f1_score
    from sklearn.preprocessing import MinMaxScaler

    from xgboost import XGBClassifier

    from pactools.grid_search import GridSearchCVProgressBar

############################## LOAD DATA ##############################
if False:
    df = pd.read_csv('data/final_df.csv', index_col=0)

    X = df.drop(columns=['name', 'age', 'pos','player_id','team_id', 'opp_id', 'team_name', 'game_date', 'opponent', 'defender_name', 'game_id', 'action_type', 'season', 'htm', 'vtm', 'game_event_id',  'minutes_remaining', 'seconds_remaining',
    'defender_id', 'shot_type', 'Heave', 'heave_pct', 'is_home', 'prev_shot_made', 'prev_2_made', 'prev_3_made', 'Above Break 3', 'Corner 3', 'Mid Range', 'Paint', 'Restricted Area', 'C', 'L', 'R', 'dribbles', 'shot_distance', 'shot_made_flag'])
    y = np.array(df.shot_made_flag)

    X_col_names = X.columns
    with open('./X_y_arrays/X_column_names', 'wb') as x_col:
        pickle.dump(X_col_names, x_col)

    minmax_scale = MinMaxScaler()
    X = minmax_scale.fit_transform(X)

    np.save('./X_y_arrays/X_shallow', X)
    np.save('./X_y_arrays/y_shallow', y)

#new data
if True:
    df = pd.read_csv('final_df_1415.csv', index_col=0)
    df[['zone_id', 'period']] = df[['zone_id', 'period']].astype('category')

    X = df.drop(columns=['name', 'team_name', 'game_date', 'season', 'team_id','minutes_remaining', 'seconds_remaining', 'shot_made_flag', 'shot_type', 'opponent', 'x', 'y', 'defender_name', 'opp_id', 'game_id', 'game_event_id',
    'player_id', 'shot_zone_basic', 'shot_zone_area', 'shot_zone_range', 'htm', 'vtm', 'pos', 'age', 'defender_id', 'zone', 'pps', 'zone_id', 'zone_minus_lg_avg', 'lg_zone_avg',
    'is_home', 'prev_shot_made', 'prev_2_made', 'prev_3_made', 'dribbles', 'period', 'action_type', 'ts%', 'dbpm', '3par', 'usg%', 'blk_pct', 'def_rating'])
    y = np.array(df.shot_made_flag)

    X_col_names = X.columns
    with open('./X_y_arrays/X_column_names', 'wb') as x_col:
        pickle.dump(X_col_names, x_col)

    minmax_scale = MinMaxScaler()
    X = minmax_scale.fit_transform(X)

    np.save('./X_y_arrays/X_shallow', X)
    np.save('./X_y_arrays/y_shallow', y)
################### SPLIT DATA INTO TRAIN/TEST SETS ###################
if True:
    with open ('./X_y_arrays/X_column_names', 'rb') as fp:
        X_col_names = pickle.load(fp)

    X = np.load('./X_y_arrays/X_shallow.npy')
    y = np.load('./X_y_arrays/y_shallow.npy')

    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=23,test_size=.2)

########################## HELPER FUNCTIONS ##########################
def build_model(model, path, X_train, X_test, y_train, y_test, decision_function=True):
    start = time.time()

    clf = model
    clf.fit(X_train,y_train)
    y_hat_test = clf.predict(X_test)

    if decision_function==True:
        y_score = clf.decision_function(X_test)
    else:
        y_score = clf.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_score)

    #Save model
    with open('./models/'+ path + '/' + str(path) + '_' + time.asctime().replace(' ', '_'), 'wb') as f:
        pickle.dump(clf, f)

    print('Total Runtime: {} seconds'.format(time.time()-start))
    return clf, y_hat_test, y_score, fpr, tpr

def plot_feature_importances(model, path):
    matplotlib.style.use('fivethirtyeight')
    n_features = X.shape[1]
    plt.figure(figsize=(10,6))
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X_col_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Features")
    #Save output
    plt.savefig('./models/'+ path + '/feature_importances/' + time.asctime().replace(' ', '_') + '.png')
    plt.show()

def plot_confusion_matrix(cm, path, title='Confusion matrix', cmap=plt.cm.Blues):
    #Create the basic matrix.
    plt.imshow(cm, cmap)

    #Add title and Axis Labels
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    #Add appropriate Axis Scales
    class_names = set(y)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    #Add Labels to Each Cell
    thresh = cm.max()*.75

    #Add a Side Bar Legend Showing Colors
    plt.colorbar()
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    #Save output
    plt.savefig('./models/'+ path + '/cm/' + time.asctime().replace(' ', '_') + '.png', bbox_inches='tight', dpi=480)
    plt.show()

def print_model_metrics(y_pred, y_score, path):
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, path, title='Confusion matrix', cmap=plt.cm.Blues)

    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    auc_ = auc(fpr, tpr)

    print('Accuracy:   {}'.format(round(accuracy,4)))
    print('Precision:  {}'.format(round(precision,4)))
    print('Recall:     {}'.format(round(recall,4)))
    print('F1          {}'.format(round(f1,4)))
    print('AUC:        {}'.format(round(auc_,4)))

    #Save output
    metrics = np.array([accuracy, precision, recall, f1, auc_])
    np.save('./models/'+ path + '/metrics/' + time.asctime().replace(' ', '_'), metrics)

def plot_roc_curve(fpr, tpr, path):
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    plt.figure(figsize=(10,6))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i/20.0 for i in range(21)])
    plt.xticks([i/20.0 for i in range(21)])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    #Save output
    plt.savefig('./models/'+ path + '/roc_curves/' + time.asctime().replace(' ', '_') + '.png', bbox_inches='tight', dpi=480)
    plt.show()
######################################################################

############################# GRID SEARCH ############################
def run_grid_search(model, path, param_grid, X, y, cv=3):
    start = time.time()

    search = GridSearchCVProgressBar(model, param_grid, scoring='roc_auc', cv=cv, n_jobs=-1, verbose=2)
    search.fit(X,y)

    print("Total Runtime for Grid Search: {:.4} seconds".format(round(time.time() - start, 2)))

    best_score = search.best_score_
    best_params = search.best_params_

    print("Testing Accuracy: {:.4}%".format(best_score * 100))
    print("\nOptimal Parameters: {}".format(best_params))

    search_results = pd.DataFrame.from_dict(search.cv_results_)

    search_results.to_csv('./grid_search_results/'+ path + '_' + str(round(best_score,4)).replace('.','') + '_' + time.asctime().replace(' ', '_'))

    return search_results, best_score, best_params
######################################################################

########################## PARAMETER GRIDS ###########################
if True:
    log_reg_param_grid = {'penalty':['l1','l2'],
                        'C': np.logspace(0, 4, 10)
                    }

    rf_param_grid = {'n_estimators':[100,250],
                    'criterion':['gini', 'entropy'],
                    'min_samples_leaf':[2,5,10],
                    'min_samples_split':[2,5,10],
                    'n_jobs':[-1]
                    }

    gb_param_grid = {'n_estimators':[50, 100, 250],
                    'learning_rate':[.01, .05, .1, 1],
                    'min_samples_leaf':[2, 5, 10],
                    'min_samples_split':[2, 5, 10],
                    'max_depth':[2, 5, 10]
                    }

    xgb_param_grid = {'learning_rate':[.01, .05, .1, 1],
                     'n_estimators':[100, 250],
                     'max_depth':[2, 5, 10],
                     'min_child_weight': [1, 5, 10],
                     'gamma': [0.5, 1, 2],
                     }

######################################################################


######################## LOGISTIC REGRESSION #########################
if True:
    # log_reg, log_y_preds, log_y_score, log_fpr, log_tpr = build_model(LogisticRegression(C=1, class_weight='balanced'),
    # 'logreg', X_train, X_test, y_train, y_test)
    #
    # print_model_metrics(log_y_preds, log_y_score, 'logreg')
    # plot_roc_curve(log_fpr, log_tpr, 'logreg')

    log_reg_search_results, log_reg_best_score, log_reg_best_params = run_grid_search(LogisticRegression(random_state=23),'logreg', log_reg_param_grid, X, y, cv=10)
######################################################################


###################### RANDOM FOREST CLASSIFIER ######################
if False:
    rf, rf_y_preds, rf_y_score, rf_fpr, rf_tpr = build_model(RandomForestClassifier(n_estimators=500, criterion='gini', min_samples_leaf=10, min_samples_split=10, verbose=.5, class_weight='balanced', n_jobs=-1, random_state=23),
    'rf', X_train, X_test, y_train, y_test, decision_function=False)

    print_model_metrics(rf_y_preds, rf_y_score, 'rf')
    plot_roc_curve(rf_fpr, rf_tpr, 'rf')
    plot_feature_importances(rf, 'rf')

    # rf_search_results, rf_best_score, rf_best_params = run_grid_search(RandomForestClassifier(random_state=23),'rf', rf_param_grid, X, y, cv=3)

    # [ParallelProgressBar(n_jobs=-1)]: Done 108 out of 108 | elapsed: 67.1min finished
    # Total Runtime for Grid Search: 4.095e+03 seconds
    # Testing Accuracy: 70.82%
    #
    # Optimal Parameters: {'criterion': 'gini', 'min_samples_leaf': 10, 'min_samples_split': 2, 'n_estimators': 250, 'n_jobs': -1}
######################################################################


#################### GRADIENT BOOSTING CLASSIFIER ####################
if False:
    gb, gb_y_preds, gb_y_score, gb_fpr, gb_tpr = build_model(GradientBoostingClassifier(learning_rate=0.05, n_estimators=500, max_depth=5, min_samples_leaf=7, min_samples_split=7, verbose=1, random_state=23),
    'gb', X_train, X_test, y_train, y_test)

    print_model_metrics(gb_y_preds, gb_y_score, 'gb')
    plot_roc_curve(gb_fpr, gb_tpr, 'gb')
    plot_feature_importances(gb, 'gb')
######################################################################


######################### ADABOOST CLASSIFIER #########################
if False:
    ada, ada_y_preds, ada_y_score, ada_fpr, ada_tpr = build_model(AdaBoostClassifier(learning_rate=.01, n_estimators=500, algorithm='SAMME.R', random_state=23),
    'ada', X_train, X_test, y_train, y_test)

    print_model_metrics(ada_y_preds, ada_y_score, 'ada')
    plot_roc_curve(ada_fpr, ada_tpr, 'ada')
    plot_feature_importances(ada, 'ada')
######################################################################


######################### XGBOOST CLASSIFIER #########################
if False:
    xgb, xgb_y_preds, xgb_y_score, xgb_fpr, xgb_tpr = build_model(XGBClassifier(learning_rate=0.1, n_estimators=500, max_depth=5, min_child_weight=1, gamma=1, algorithm='SAMME.R', objective='binary:logistic', reg_alpha=0, reg_lambda=0, n_jobs=-1, random_state=23),
    'xgb', X_train, X_test, y_train, y_test, decision_function=False)

    print_model_metrics(xgb_y_preds, xgb_y_score, 'xgb')
    plot_roc_curve(xgb_fpr, xgb_tpr, 'xgb')
    plot_feature_importances(xgb, 'xgb')

    # xgb_search_results, xgb_best_score, xgb_best_params = run_grid_search(XGBClassifier(random_state=23),'xgb', xgb_param_grid, X_train, y_train)

    # Testing Accuracy: 72.23%
    #
    # Optimal Parameters: {'gamma': 1, 'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 1, 'n_estimators': 250}
######################################################################


######################## STACKED ENSEMBLE MODEL ######################
def create_ensemble_model(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=23, test_size=.2)

    rf = RandomForestClassifier(n_estimators=500, criterion='gini',  max_features='sqrt', min_samples_leaf=10, min_samples_split=2, verbose=1, class_weight='balanced', n_jobs=-1, random_state=23)

    xgb = XGBClassifier(learning_rate=0.1, n_estimators=250, max_depth=5, min_child_weight=1, gamma=1, algorithm='SAMME.R', objective='binary:logistic', n_jobs=-1, random_state=23)

    ada = AdaBoostClassifier(learning_rate=.75, n_estimators=500, algorithm='SAMME.R', random_state=23)

    rf.fit(X_train, y_train)
    rf_train_preds = pd.DataFrame(rf.predict_proba(X_train))
    rf_test_preds = pd.DataFrame(rf.predict_proba(X_test))

    xgb.fit(X_train, y_train)
    xgb_train_preds = pd.DataFrame(xgb.predict_proba(X_train))
    xgb_test_preds = pd.DataFrame(xgb.predict_proba(X_test))

    ada.fit(X_train, y_train)
    ada_train_preds = pd.DataFrame(ada.predict_proba(X_train))
    ada_test_preds = pd.DataFrame(ada.predict_proba(X_test))

    train_df = pd.concat([rf_train_preds, xgb_train_preds, ada_train_preds], names=['rf','xgb','ada'], axis=1)
    test_df = pd.concat([rf_test_preds, xgb_test_preds, ada_test_preds], names=['rf','xgb','ada'], axis=1)

    model = LogisticRegression(random_state=1)
    model.fit(train_df,y_train)
    y_preds = model.predict(test_df)
    # y_score = model.score(y_preds, y_test)

    return train_df, test_df, y_preds
######################################################################
