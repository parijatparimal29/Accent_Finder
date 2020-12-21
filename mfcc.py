import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import scipy.io.wavfile as wav
from python_speech_features import mfcc, logfbank
import pandas as pd
import sys

# Read the input audio file
if __name__ == '__main__':
    csv_file = sys.argv[1]
    df = pd.read_csv(csv_file)
    file_location = '/Users/Rahul/Desktop/MSIS/NLP/Accent Finder/wav/'
    metadata_path = '/Users/Rahul/Desktop/MSIS/NLP/Accent Finder/mfcc_features.csv'
    flag = 0
    for index, row in df.iterrows():
        file = file_location + row['filename'] + '.wav'
    #for f in glob(r'/Users/Rahul/Desktop/MSIS/NLP/Accent Finder/wav/*.wav', recursive=True):
        (rate, sig) = wav.read(file)
        # Take the first 10,000 samples for analysis
        #sig = sig[:10000]
        features_mfcc = mfcc(sig, rate)

        #Storing 12 MFCC features per audio into a csv file
        data = [row['country']]
        for j in range(len(features_mfcc[len(features_mfcc)-1]) - 1):
            data.append(list(features_mfcc[len(features_mfcc)-1])[j])
        ndf = pd.DataFrame([data], columns=['language', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12'])
        if flag == 0:
            ndf.to_csv(metadata_path, index=False, header=True)
            flag = 1
        else:
            ndf.to_csv(metadata_path, mode='a', index=False, header=False)


        #print(len(features_mfcc))

        # Print the parameters for MFCC
        print('\nMFCC:\nNumber of windows =', features_mfcc.shape[0])
        print('Length of each feature =', features_mfcc.shape[1])

        # Plot the features
        features_mfcc = features_mfcc.T
        plt.matshow(features_mfcc)
        plt.title('MFCC')

        # Extract the Filter Bank features
        features_fb = logfbank(sig, rate)

        # Print the parameters for Filter Bank
        print('\nFilter bank:\nNumber of windows =', features_fb.shape[0])
        print('Length of each feature =', features_fb.shape[1])

        # Plot the features
        features_fb = features_fb.T
        plt.matshow(features_fb)
        plt.title('Filter bank')

        plt.show()
