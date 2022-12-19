import subprocess
import wave
import os
import matplotlib.pyplot as plt
import numpy as np 
import scipy.io.wavfile as wav


def git_loader(link, folder = './data'):
  '''
  Function which downloads a certain folder from github repo

  :param link: string, link to the repo itself
  :return: None
  '''

  try:
    # download the github repo's folder 
    subprocess.run(["git", "clone", link], universal_newlines = True, stdout = subprocess.PIPE)

  except Exception as e:
    print(str(e))


def get_audios(path_to_files, num = 5):
  '''
  Function which plots some number of audios 

  :param path_to_files: string, indicates to the path of the saved audios
  :param num: integer, refers to the number of audios you want to plot
  :return: None
  '''

  parent_list = os.listdir(path_to_files)
  for i in range(num): 
    signal_wave = wave.open(os.path.join(path_to_files, parent_list[i]), 'r')
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


def classify_audios_by_spectogram(path_to_files, folder = 'data'):
  '''
  Function which copies audio files spectogram to the 'folder' and then it sets them to 
  their coresponding classes e.g file '3_george_38.wav' will be copied to the 
  folder 'class_3'

  :param path_to_files: string, indicates to the path of the saved audios
  :param folder: refers to the folder where you want to copy the audio files
  :return: None
  '''

  # Create the folder 'folder' if it doesn't exist
  if not os.path.exists(folder):
      os.mkdir(folder)

  # Copy every audio files spectogram to the coresponding folder     
  for filename in os.listdir(path_to_files):
      if "wav" in filename:
          file_path = os.path.join(path_to_files, filename)
          target_dir = f'class_{filename[0]}'             
          dist_dir = os.path.join(folder, target_dir)
          file_dist_path = os.path.join(dist_dir, filename)

          if not os.path.exists(file_dist_path + '.png'):
              if not os.path.exists(dist_dir):
                  os.mkdir(dist_dir)   

              frame_rate, data = wav.read(file_path)
              signal_wave = wave.open(file_path)
              sig = np.frombuffer(signal_wave.readframes(frame_rate), dtype=np.int16)
              fig = plt.figure()
              plt.specgram(sig, NFFT=1024, Fs=frame_rate, noverlap=900)
              plt.axis('off')
              fig.savefig(f'{file_dist_path}.png', dpi=fig.dpi)
              plt.close()

def delete_folder(path_to_folder):
  '''
  Function which deletes a certain folder

  :param link: string, path to the folder
  :return: None
  '''

  try:
    # delete the folder
    subprocess.run(["rm", "-rf", path_to_folder], universal_newlines = True, stdout = subprocess.PIPE)

  except Exception as e:
    print(str(e))

# The dataset used in this current situation is called audio mnist you can find it
# with the this link
# https://github.com/Jakobovski/free-spoken-digit-dataset.git

if __name__ == "__main__":
  git_loader("https://github.com/Jakobovski/free-spoken-digit-dataset.git")
  get_audios("free-spoken-digit-dataset/recordings")
  classify_audios_by_spectogram("free-spoken-digit-dataset/recordings")
  delete_folder("free-spoken-digit-dataset")  