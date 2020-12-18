# -*- coding: utf-8 -*-

import os
from matplotlib import pyplot as plt

def process_data(data):
  processed_data = []
  for data_elem in data:
    splitted_data = [float(e) for e in data_elem.split(",")] 
    processed_data.append(splitted_data)
  return processed_data

def plot_data(file_data, project_directory):
  plot_directory = os.path.join(project_directory, "plots")
  for f_idx in range(0, len(file_data)):
    data = file_data[f_idx]
    for idx in range(0, len(data)):
      data_elem = data[idx]
      plot_name = "Data_" + str(f_idx) + "_" + str(idx)
      file_name = os.path.join(plot_directory, plot_name)
      plt.figure()
      plt.plot(data_elem)
      plt.title(plot_name)
      plt.savefig(file_name, dpi=250)
      plt.show(block=True)
      plt.close()

def read_data(project_directory):
  data_directory = os.path.join(project_directory, "data")
  # data_directory = os.path.join(project_directory, "data_new")
  event_files = os.listdir(data_directory)
  file_data = []
  for file in event_files:
     if "event" in file:
       event_file = os.path.join(data_directory, file)
       f = open(event_file)
       data = process_data(f.readlines())
       f.close()
       file_data.extend(data)
  return file_data


def read_data_strategy_2(project_directory):
  data_directory = os.path.join(project_directory, "data")
  # data_directory = os.path.join(project_directory, "data_new")
  event_files = os.listdir(data_directory)
  file_data = []
  for file in event_files:
     if "event" in file:
       event_file = os.path.join(data_directory, file)
       f = open(event_file)
       data = process_data(f.readlines())
       f.close()
       file_data.append(data)
  return file_data
