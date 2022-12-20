#import corresponding functions from other files
!wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
!mkdir ./data
!tar -xf ./speech_commands_v0.02.tar.gz -C ./data

! git clone https://github.com/tigran-serobyan/keyword_spotting.git

from keyword_spotting.spectrogram import *
from keyword_spotting.loader import *
from keyword_spotting.word_spotting import *
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy
from tqdm import tqdm

def read_file(file_name):
  samplerate, signal = wavfile.read(file_name)
  signal = np.array(signal, dtype=np.float32)
  return signal, samplerate

def process(data):
  parent_list = os.listdir("./data")
  dict_data = {}
  for folder in parent_list:
    if os.path.isdir(os.path.join("./data",folder)):
      count = 0
      for audio in tqdm(os.listdir(os.path.join("./data",folder))):
        f, sr = read_file(os.path.join(os.path.join(os.path.join("./data",folder),audio)))
        is_word_full = get_word_timestamps(f,sr)
        if is_word_full is not None:
          spectogram_audio = spectogram(f,sr)
          if count == 0 :
            spec_array =  spectogram_audio
            spec_array = spec_array[np.newaxis,:]
            count +=1
          else:
            spec_array = np.concatenate((spec_array, spectogram_audio[np.newaxis,:]), axis=0)

      dict_data[folder] = spec_array
  return dict_data
