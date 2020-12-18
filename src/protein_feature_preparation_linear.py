# -*- coding: utf-8 -*-

import numpy as np

class ProteinFeaturePreparationLinear():
  def __init__(self, window_size = 4):
    self.Amino_Acids = "GASCUTDPNVBEQZHLIMKXRFYW"
    self.AA_SIZE_TRANS = str.maketrans("GASCUTDPNVBEQZHLIMKXRFYW-",
                                       "MMMMMSSSSSSIIIIIIIIILLLL-")
    
    #amino acid volumes from Perkins, 1986
    self.volumes = {"I": 1688, "F": 2034, "V": 1417, "L": 1679,
                    "W": 2376, "M": 1708, "A": 915, "G": 664,
                    "C": 1056, "Y": 2036, "P": 1293, "T": 1221,
                    "S": 991, "H": 1673, "E": 1551, "N": 1352,
                    "Q": 1611, "D": 1245, "K": 1713, "R": 2021,
                    "X": 1500, "U": 1056, "Z": 1580, "B": 1300,
                    "-": 0}
    
    #hydrophilicity index from Janin, 1979
    self.hydro = {"L": 2.4, "I": 3.1, "F": 2.2, "V": 2.9, 
                  "W": 1.6, "M": 1.9, "A": 1.7, "G": 1.8,
                  "C": 4.6, "Y": 0.5, "P": 0.6, "T": 0.7,
                  "S": 0.8, "H": 0.8, "E": 0.3, "N": 0.4,
                  "Q": 0.3, "D": 0.4, "K": 0.05, "R": 0.1,
                  "X": 0.5, "U": 4.6, "Z": 0.6, "B": 0.4, "-": 0}
    
    ## 5 groups
    ## K: Non-polar aliphatic, S: Polar uncharged
    ## P: +ve charge, N:-ve charge, A: Non-polar aromatic
    self.AA_POLARITY_TRANS = str.maketrans("GAVMLISTCPRQRHKDEFWT-",
                                           "KKKKKKSSSSSSPPPNNAAA-")
    #pI from Lide, 1991
    self.pI =  {"A": 6.00, "R": 10.76, "N": 5.41, "D": 2.77,
                "C": 5.07, "E": 3.22, "Q": 5.65, "G": 5.97,
                "H": 7.59, "I": 6.02, "L": 5.98, "K": 9.74, 
                "M": 5.74, "F": 5.48, "P": 6.30, "U": 5.68,
                "S": 5.68, "T": 5.60, "W": 5.89, "Y": 5.66, 
                "V": 5.96, "-": 0}
    
    ## Kenedey et al. blockade current related to quadromer 
    self.window = window_size

  def _get_normlized_values(self, dic):
    factor = max(dic.values())
    normalised_d = {k: v/factor for k, v in dic.items()}
    return normalised_d
  
  
  ## get distribution -- size
  def _get_size_distribution(self, kmer):
      miniscule = kmer.count("M")
      small = kmer.count("S")
      intermediate = kmer.count("I")
      large = kmer.count("L")
      dimension_matcher = 0
      return (large, intermediate, small, miniscule, dimension_matcher)

  def _get_polarity_distribution(self, kmer):
      non_polar_aliphatic = kmer.count("K")
      polar = kmer.count("S")
      positive_charge = kmer.count("P")
      negtaive_charge = kmer.count("N")
      non_polar_aromatic = kmer.count("A")
      return (non_polar_aliphatic, polar, positive_charge, negtaive_charge, non_polar_aromatic)

  def get_feature(self, prot_seq):
    normalized_volumes = self._get_normlized_values(self.volumes)
    normalized_hydro = self._get_normlized_values(self.hydro)
    normalized_pI = self._get_normlized_values(self.pI)
    volumes = list(map(normalized_volumes.get, prot_seq))
    hydro = list(map(normalized_hydro.get, prot_seq))
    pI = list(map(normalized_pI.get, prot_seq))
    size = prot_seq.translate(self.AA_SIZE_TRANS)
    polarity = prot_seq.translate(self.AA_POLARITY_TRANS)
    ## add "window-1" elements at the start and end to consider the boundary cases
    flanked_volumes = ([0] * (self.window - 1) + volumes + [0] * (self.window - 1))
    flanked_hydro = ([0] * (self.window - 1) + hydro + [0] * (self.window - 1))
    flanked_pI = ([0] * (self.window - 1) + pI + [0] * (self.window - 1))
    flanked_size = ("-" * (self.window - 1) + size + "-" * (self.window - 1))
    flanked_polarity = ("-" * (self.window - 1) + polarity + "-" * (self.window - 1))

    num_peaks = len(prot_seq) + self.window -1 
    features = []
    for i in range(0, num_peaks):
        v = flanked_volumes[i : i + self.window]
        h = flanked_hydro[i : i + self.window]
        pi = flanked_pI[i : i + self.window]
        
        ## append zero at the end to make it len 5 -- match the polarity 
        while len(v) != 5:
          v.append(0)
          h.append(0)
          pi.append(0)
                
        kmer = flanked_size[i : i + self.window]
        s = self._get_size_distribution(kmer)        
        kmer = flanked_polarity[i : i + self.window]
        p = self._get_polarity_distribution(kmer)
        
        feature = np.array([np.array(s), np.array(p), np.array(v), np.array(h), np.array(pi)])
        # feature = np.array([np.array(v), np.array(p), np.array(h), np.array(pi)])
        features.append(feature.flatten())
    return np.array(features)

