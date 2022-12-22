import subprocess
import wave
import os
import matplotlib.pyplot as plt
import numpy as np 
import scipy.io.wavfile as wav
import tensorflow as tf
from random import shuffle
import glob
import sys
import cv2
import numpy as np
import tensorflow as tf
import sys
import tensorflow_datasets as tfds


class Loader():
  '''
  Loader class allows to download audio files from a github repo
  and proccesses them, by creating their spectograms, saves them 
  in tf data record files
  '''

  def __init__(self, link, folder = './data'):
    '''
    Initializing function for Loader class

    :param link: string, link to the github repo
    :param folder: string, path to the folder where data will be processed and saved
    :return: None
    '''

    self.path = ''
    self.link = link
    self.folder = folder

  def git_download(self):
    '''
    Function which downloads a certain folder from github repo

    :param link: string, link to the repo itself
    :return: None
    '''

    old_ls = os.listdir()

    try:
      # download the github repo's folder 
      subprocess.run(["git", "clone", self.link], universal_newlines = True, stdout = subprocess.PIPE)

      new_ls = os.listdir()

      for dir in new_ls:
        if dir not in old_ls:
          self.path = './' + dir

    except Exception as e:
      print(str(e))


  def get_audios(self, num = 5):
    '''
    Function which plots some number of audios 

    :param path_to_files: string, indicates to the path of the saved audios
    :param num: integer, refers to the number of audios you want to plot
    :return: None
    '''

    parent_list = os.listdir(self.folder)
    for i in range(num): 
      signal_wave = wave.open(os.path.join(self.folder, parent_list[i]), 'r')
      sample_rate = 16000
      sig = np.frombuffer(signal_wave.readframes(sample_rate), dtype=np.int16)

      plt.figure(figsize=(12,12))
      plot_a = plt.subplot(211)
      plot_a.set_title(parent_list[i])
      plot_a.plot(sig)
      plot_a.set_xlabel('sample rate * time')
      plot_a.set_ylabel('energy')

      plot_b = plt.subplot(212)
      plot_b.specgram(sig, NFFT=1024, Fs=sample_rate, noverlap=900)
      plot_b.set_xlabel('Time')
      plot_b.set_ylabel('Frequency')

    plt.show()


  def classify_audios_by_spectogram(self):
    '''
    Function which copies audio files spectogram to the 'folder' and then it sets them to 
    their coresponding classes e.g file '3_george_38.wav' will be copied to the 
    folder 'class_3'

    :param path_to_files: string, indicates to the path of the saved audios
    :param folder: refers to the folder where you want to copy the audio files
    :return: None
    '''

    # Create the folder 'folder' if it doesn't exist
    if not os.path.exists(self.folder):
      os.mkdir(self.folder)


    for root, dirs, files in os.walk(self.path):
      for filename in files:

        if ".wav" in filename:
          #append the file name to the list
          file_path = os.path.join(root,filename)
          file_dist_path = os.path.join(self.folder, filename)

          if not os.path.exists(file_dist_path + '.png'):
            if not os.path.exists(self.folder):
              os.mkdir(self.folder)   

            frame_rate, data = wav.read(file_path)
            signal_wave = wave.open(file_path)
            sig = np.frombuffer(signal_wave.readframes(frame_rate), dtype=np.int16)
            fig = plt.figure()
            plt.specgram(sig, NFFT=1024, Fs=frame_rate, noverlap=900)
            plt.axis('off')
            fig.savefig(f'{file_dist_path}.png', dpi=fig.dpi)
            plt.close()

  def delete_folder(self):
    '''
    Function which deletes a certain folder

    :param link: string, path to the folder
    :return: None
    '''

    try:
      # delete the folder
      subprocess.run(["rm", "-rf", self.path], universal_newlines = True, stdout = subprocess.PIPE)

    except Exception as e:
      print(str(e))

  def load(self, split_train = 0.6, split_test = 0.2, split_val = 0.2):
    '''
    Function which loads the github repo and processes it, then it creates the 
    data records for the future models

    :param split_train: spliting percent for training
    :param split_test: spliting percent for testing
    :param split_val: spliting percent for validation
    :return: None
    '''

    self.git_download()
    self.classify_audios_by_spectogram()
    self.delete_folder()
    self.createDataRecords(split_train, split_test, split_val)

  def _int64_feature(self, value):
    '''
    Function which converts to int64 format
    :param value: any number
    :return: int64
    '''

    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  def _bytes_feature(self, value):
    '''
    Function which converts to int64 format
    :param value: string
    :return: bytes
    '''

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  def load_image(self, addr):
    '''
    Function which loads image and resizes to (224, 224)
    :param addr: address of the image
    :return: returns the image (cv2)
    '''

    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
  
  def createDataRecord(self, out_filename, addrs, labels):
    '''
    Function which creates a Data Record for a dataset
    :param out_filename: string, where to save the data record
    :param addrs: list of addresses, where the image files are saved
    :param labels: list of strings, which indicates to the classes to whcich the 
                  images belong
    :return: None
    '''

    print("START")
    # open the TFRecords file
    writer = tf.io.TFRecordWriter(out_filename)
    for i in range(len(addrs)):

      # print how many images are saved every 1000 images
      if not i % 1000:
        print('Train data: {}/{}'.format(i, len(addrs)))
        sys.stdout.flush()

      # Load the image
      img = self.load_image(addrs[i])
      label = labels[i]

      if img is None:
        continue

      # Create a feature
      feature = {
        'image_raw': self._bytes_feature(img.tostring()),
        'label': self._int64_feature(label)
      }

      # Create an example protocol buffer
      example = tf.train.Example(features=tf.train.Features(feature=feature))
      
      # Serialize to string and write on the file
      writer.write(example.SerializeToString())
        
    writer.close()
    sys.stdout.flush()

  def get_index(self, addr):
    '''
    Function which gets the first letter of the filename
    :param addr: address of the file
    :return: string, the first letter of filename
    '''

    idx = len(addr) - addr[::-1].index('/')
    return addr[idx]

  def parser(self, record):
    '''
    Function which parses a single data record to image and its corresponding label 
    :param record: name of the data record 
    :return: tensor which represents the image, string which indicates the label
    '''

    keys_to_features = {
        "image_raw": tf.io.FixedLenFeature([], tf.string),
        "label":     tf.io.FixedLenFeature([], tf.int64)
    }
    parsed = tf.io.parse_single_example(record, keys_to_features)
    image = tf.io.decode_raw(parsed["image_raw"], tf.uint8)
    image = tf.cast(image, tf.float32)
    label = tf.cast(parsed["label"], tf.int32)

    return {'image' : image}, label


  def input_fn(self, filenames):
    '''
    Function which reads the data records and creates inputs for the training model
    :param filename: name/s of the data record/s 
    :return: returns the input dataset
    '''

    dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=40)
    dataset = dataset.apply(
        tf.data.experimental.shuffle_and_repeat(1024, 1)
    )
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(self.parser, 32)
    ) 
    dataset = dataset.prefetch(buffer_size=2)
    return dataset


  def train_input_fn(self):
    '''
    Function which reads the data records returns the datarecord for training and testing
    :return: dataset, the dataset for training and testing
    '''

    return self.input_fn(filenames=["train.tfrecords", "test.tfrecords"])

  def val_input_fn(self):
    '''
    Function which reads the data records returns the datarecord for validation
    :return: dataset, the dataset for validation
    '''

    return self.input_fn(filenames=["val.tfrecords"])

  def createDataRecords(self, split_train = 0.6, split_test = 0.2, split_val = 0.2):
    '''
    Function which creates data records for the model (train, test, val) and saves them
    :param path_to_data: path to the data
    :param split_train: spliting percent for training
    :param split_test: spliting percent for testing
    :param split_val: spliting percent for validation
    :return: None
    '''

    addrs = [os.path.join(self.folder, filename) for filename in os.listdir(self.folder)] # read addresses and labels from the 'self.folder' folder
    labels = [int(self.get_index(addr)) for addr in addrs]
    print(labels)

    # to shuffle data
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)
        
    # Divide the data into (split_train * 100) % train, (split_val * 100) % validation, and (split_test * 100)% test
    train_addrs = addrs[0:int(split_train*len(addrs))]
    train_labels = labels[0:int(split_train*len(labels))]
    val_addrs = addrs[int(split_train*len(addrs)):int( (split_train + split_test) *len(addrs))]
    val_labels = labels[int(split_train*len(addrs)):int( (split_train + split_test) *len(addrs))]
    test_addrs = addrs[int((split_train + split_test)*len(addrs)):]
    test_labels = labels[int((split_train + split_test)*len(labels)):]

    self.createDataRecord('train.tfrecords', train_addrs, train_labels)
    self.createDataRecord('val.tfrecords', val_addrs, val_labels)
    self.createDataRecord('test.tfrecords', test_addrs, test_labels)
    

if __name__ == "__main__":
  loader = Loader("https://github.com/Jakobovski/free-spoken-digit-dataset.git")
  loader.load(0.6, 0.2, 0.2)
