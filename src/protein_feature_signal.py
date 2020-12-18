# -*- coding: utf-8 -*-

import numpy as np
    
def _normalize(signal):
  return (signal - np.mean(signal)) / np.std(signal)

def discretize(original_signal, protein_length, window_size):
  """ Discretizes the signal assuming the given protein length """
  signal = _normalize(original_signal)
  WINDOW = window_size
  num_peaks = protein_length + WINDOW - 1
  discrete = []
  peak_shift = len(signal) / (num_peaks - 1)
  for i in range(0, num_peaks):
    signal_pos = i * (peak_shift - 1)
    left = max(0, int(signal_pos - peak_shift / 2))
    right = min(len(signal), int(signal_pos + peak_shift / 2))
    discrete.append(np.mean(signal[left:right]))
  return discrete 

def discretize_strategy_2(original_signal, protein_length, window_size):
  """ Discretizes the signal assuming the given protein length """
  signals = _normalize(original_signal)
  WINDOW = window_size
  num_peaks = protein_length + WINDOW - 1
  signal = _get_consensus(signals)
  discrete = []
  peak_shift = len(signal) / (num_peaks - 1)
  for i in range(0, num_peaks):
    signal_pos = i * (peak_shift - 1)
    left = max(0, int(signal_pos - peak_shift / 2))
    right = min(len(signal), int(signal_pos + peak_shift / 2))
    discrete.append(np.mean(signal[left:right]))
  return discrete 

def _get_consensus(signals):
  """ Calculates consensus of multiple signals """
  matrix = np.array([e for e in signals])
  medians = np.mean(matrix, axis=0)
  return medians
