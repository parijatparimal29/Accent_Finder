# Accent_Finder
Natural Language Processing Project on Accent Finder of English speakers.

For Dataset with audio files, the audio files are read and converted to extract MFCC as features.
These features are then used into various classification algorithms to classify the audio with the accent of the speaker.

MFCC_stages shows different phases of converting audio into MFCC.

CNN_Model.py details:
Once downloaded, the Kaggle dataset will have to be converted to individual .wav files. You can make use of 'mp3 to wav.py' file.
'speakers_all.csv' is a metadata file that contains information about each speaker. Both CNN_Model.py and mfcc.py make use of this as command line input.
CNN_Model.py parses through all wav files in the parent directory and splits the data by making a call to split_dataset.py where corresponding dataframes are returned.
Finally, the results are printed via the predictions.py program.

mfcc.py details:
This is a standalone python file to extract MFCC features from all .wav files and aggregates the [label, features] list to a single .csv file. The program also plots inidividual MFCC features using matplotlib.
