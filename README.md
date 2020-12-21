# Accent_Finder
Natural Language Processing Project on Accent Finder of English speakers.

For Dataset with audio files, the audio files are read and converted to extract MFCC as features.
These features are then used into various classification algorithms to classify the audio with the accent of the speaker.

MFCC_stages shows different phases of converting audio into MFCC.

mfcc_full.csv contains mfcc features from https://archive.ics.uci.edu/ml/datasets/Speaker+Accent+Recognition# dataset, which contains 329 speakers with 6 different accents. 

SVM.py takes in this file and divided into training and test data which is then trained in SVM model. The model then predicts the accent of the test data set. To get the mean accuracy, this is done 100 times by random choice tuples chosen for test an train.

CNN.py uses the dataset link to extract the data for training and testing. The data is then run on CNN and the predictions are used to calculate accuracy which improves with more iterations. For current implementation, 3000 iterations is chosen as the number of iterations.

Similarly, KNN, MCP etc are other algorithms that we have tried that did not perform as well as SVM and CNN. The other files are either test data sets, test audios and experimentation with different algorithms and different ways of transforming the input audio file to get better features.

CNN_Model.py details:
Once downloaded, the Kaggle dataset will have to be converted to individual .wav files. You can make use of 'mp3 to wav.py' file.
'speakers_all.csv' is a metadata file that contains information about each speaker. Both CNN_Model.py and mfcc.py make use of this as command line input.
CNN_Model.py parses through all wav files in the parent directory and splits the data by making a call to split_dataset.py where corresponding dataframes are returned.
Finally, the results are printed via the predictions.py program.

mfcc.py details:
This is a standalone python file to extract MFCC features from all .wav files and aggregates the [label, features] list to a single .csv file. The program also plots inidividual MFCC features using matplotlib.
