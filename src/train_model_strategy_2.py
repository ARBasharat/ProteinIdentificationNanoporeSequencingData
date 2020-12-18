# -*- coding: utf-8 -*-

import os
import numpy as np
from process_data import read_data_strategy_2
from protein_feature_signal import discretize_strategy_2 
from sklearn.ensemble import RandomForestRegressor
from protein_feature_preparation_linear import ProteinFeaturePreparationLinear

## Read Data
project_directory = os.path.dirname(os.getcwd())
file_data = read_data_strategy_2(project_directory)
protein_data = ['DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA'] * len(file_data)

print("********* Model: Random Forest ")
for window_size in range(2, 6):
  print("#############################")
  print("********* Using Window Size:", window_size)
  params = ProteinFeaturePreparationLinear(window_size)

  #### Train Feature //////
  train_signal = []
  train_features = []
  for i in range(0, len(protein_data)):
    prot_seq = protein_data[i]
    signal = file_data[i]
    features = params.get_feature(prot_seq)
    discrete_signal = discretize_strategy_2(signal, len(prot_seq), window_size)
    train_signal.extend(discrete_signal)
    train_features.extend(features)
  train_features = np.array(train_features)
  train_signal = np.array(train_signal)
  print("Training Data Size:", train_features.shape, train_signal.shape)

  #### Train Model /////
  model_rf = RandomForestRegressor(n_estimators=1000, random_state = 42)
  model_rf.fit(train_features, train_signal)
  print("Training Performance, R-Square Score:", model_rf.score(train_features, train_signal))

  #### Test Feature //////
  test_signal = np.array([-1.105734161, 0.708892805, -1.105734161, 0.387283627, 
                          -1.694959734, -0.33105512, -0.92732323, 0.673680121, 
                          0.072716986, 0.687765195, 0.767580611, 0.387283627, 
                          -1.694959734, -1.105734161, 1.525827066, 0.75584305, 
                          0.387283627, 0.708892805, 1.521132042, -0.33105512, 
                          0.072716986, 0.528134362, -1.694959734, 0.673680121, 
                          0.072716986, 1.521132042, -1.694959734, -1.694959734, 
                          0.767580611, 0.708892805, 1.521132042, 1.490614383, 
                          0.072716986, 0.072716986, -0.33105512, 0.687765195, 
                          -1.694959734, -0.92732323, 0.673680121, -0.079871309, 
                          0.072716986, -1.105734161])
  test_prot = 'AIAEGDSHVLKEGAYMEIFDVQGHVFGGKIFRVVDLGSHNVA'
  test_features = params.get_feature(test_prot)
  test_features = test_features[window_size-1:len(test_features)]
  
  ## Evaluate on Test Data
  output_signal_rf = model_rf.predict(test_features)
  errors = abs(output_signal_rf - test_signal)
  mape = 100 * (errors / test_signal)
  accuracy = 100 - np.mean(mape)
  
  print("Testing Performance, R-Square Score:", model_rf.score(test_features, test_signal))
  print('Mean Absolute Error:', round(np.mean(errors), 2))
  print('Mean Average Percentage Error:', mape)
  print('Accuracy:', round(accuracy, 2), '%.')
