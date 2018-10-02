import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, itertools

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import keras
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, TensorBoard

#####LOAD DATA#####
if False:
    df = pd.read_csv('data/final_df.csv', index_col=0)

    X = df.drop(columns=['name', 'pos', 'age', 'player_id', 'team_name', 'team_id', 'game_date', 'game_id', 'game_event_id', 'season', 'minutes_remaining', 'seconds_remaining', 'action_type', 'shot_type', 'opponent','opp_id',
    'defender_name', 'htm', 'vtm', 'defender_id', 'prev_shot_made', 'prev_2_made', 'prev_3_made',  'Heave', 'dribbles','shot_distance', 'shot_made_flag'])
    y = np.array(df.shot_made_flag)

    minmax_scale = MinMaxScaler()
    X = minmax_scale.fit_transform(X)

    np.save('X_y_arrays/X_', X)
    np.save('X_y_arrays/y_', y)
#####SPLIT DATA INTO TRAIN/TEST SETS#####
if True:
    X = np.load('X_y_arrays/X_.npy')
    y = np.load('X_y_arrays/y_.npy')

    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=23,test_size=.2)

#####HELPER FUNCTION TO PLOT CM#####
def plot_confusion_matrix(cm, name, cmap=plt.cm.Blues):
    #Create the basic matrix.
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(cm, cmap)

    #Add title and Axis Labels
    plt.title(name + ' - ' 'Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    #Add appropriate Axis Scales
    tick_marks = np.arange(0,2)
    plt.xticks(tick_marks, ['Miss', 'Make'])
    plt.yticks(tick_marks, ['Miss', 'Make'])

    #Add Labels to Each Cell
    thresh = 0.75 * cm.max()

    #Add a Side Bar Legend Showing Colors
    plt.colorbar()
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="black" if cm[i, j] <= thresh else "white")

    plt.tight_layout()
    fig.savefig('./models/nn/cm/' + name + '.png', bbox_inches='tight', dpi=480)
    plt.show()

def plot_val_loss_acc(model, name):
    model_val_dict = model.history.history
    loss_values = model_val_dict['loss']
    val_loss_values = model_val_dict['val_loss']
    acc_values = model_val_dict['acc']
    val_acc_values = model_val_dict['val_acc']

    epochs_ = range(1, len(loss_values) + 1)
    plt.plot(epochs_, loss_values, 'g', label='Training loss')
    plt.plot(epochs_, val_loss_values, 'g.', label='Validation loss')
    plt.plot(epochs_, acc_values, 'r', label='Training acc')
    plt.plot(epochs_, val_acc_values, 'r.', label='Validation acc')

    plt.title(name + ' - Training & validation loss / accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('models/nn/val_loss_acc/' + name + '.png', bbox_inches='tight')
    plt.show()

#####NEURAL NETWORK GENERATOR#####
def build_nn__(X_train, X_test, y_train, y_test, activation, epochs, batch_size, name, nodes, dropout):

    adam = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    nn_ = Sequential()

    #First layer
    nn_.add(Dense(X_train.shape[1], input_shape=(X_train.shape[1],), activation=activation))
    #Iterate through number of nodes and add hidden layers
    for i, node in enumerate(nodes):
        nn_.add(Dense(node, activation=activation))
        if dropout[i]==True:
            nn_.add(Dropout(0.2))
    #Output layer, use 'sigmoid' activation for binary classfication
    nn_.add(Dense(1, activation='sigmoid'))

    #Show NN summary
    nn_.summary()
    #Compile model
    nn_.compile(loss='binary_crossentropy', optimizer=adam, metrics=['acc'])

    #Add early stopping and tensorboard callbacks
    early_stopping = EarlyStopping(monitor='val_loss', min_delta = 0.001, patience = 15, verbose=1, mode='auto', baseline=None)
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)

    #Fit model
    nn_.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_split=0.1, callbacks = [early_stopping, tensorboard])

    plot_val_loss_acc(nn_, activation + '_' + name)

    nn_.save('./models/nn/' + name + '_' + activation +'_' + str(epochs) + '_' + str(batch_size) + '_' + str(len(nodes)) + '_' + '_'.join([str(i) for i in nodes]) + '.h5')

    print(nn_.evaluate(X_test, y_test))

    cm = confusion_matrix(nn_.predict_classes(X_test), y_test)
    print(cm)
    plot_confusion_matrix(cm, activation + '_' + name)

    print('Test Set Classification Report')
    print(classification_report(nn_.predict_classes(X_test), y_test, target_names=['Miss','Make']))
    return nn_

nn = build_nn__(X_train, X_test, y_train, y_test, activation='relu', epochs=50, batch_size=32, name='16th_run_101', nodes=[128,128,64,64,32,32,16,8], dropout=[False, False, False, False, False, False, False, False])
