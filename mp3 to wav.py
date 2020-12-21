"""from pydub import AudioSegment
import os


if __name__ == '__main__':
    #print(os.getcwd())
    source = '/Users/Rahul/Desktop/MSIS/NLP/Accent Finder/Audio files/recordings/'
    destination = '/Users/Rahul/Desktop/MSIS/NLP/Accent Finder/Audio files/wav files/'
    for filename in os.listdir(source):
        sound = AudioSegment.from_mp3(source+filename)
        sound.export(destination, format='wav')
        #print(filename)
    #sound = AudioSegment.from_mp3(source)
    #sound.export(destination, format="wav")"""

import os
import glob
from pydub import AudioSegment

audio_dir = '/audio'  # Path where the videos are located

os.chdir(audio_dir)

for audio in glob.glob('*.mp3'):
    sound = AudioSegment.from_mp3(audio)
    sound.export('../' + "{}.wav".format(lang_num), format="wav")

