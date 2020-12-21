import pandas as pd
from collections import Counter
import sys
import split_dataset
from tensorflow.keras import utils
import predictions
import multiprocessing
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import MaxPooling2D, Conv2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

DEBUG = True
SILENCE_THRESHOLD = .01
RATE = 24000
N_MFCC = 13
COL_SIZE = 30
EPOCHS = 10#35#250

def silence_removal(wav, thresh=0.04, chunk=5000):
    '''
    Searches wav form for segments of silence. If wav form values are lower than 'thresh' for 'chunk' samples, the values will be removed
    :param wav (np array): Wav array to be filtered
    :return (np array): Wav array with silence removed
    '''

    tf_list = []
    for x in range(int(len(wav) / chunk)):
        if (np.any(wav[chunk * x:chunk * (x + 1)] >= thresh) or np.any(wav[chunk * x:chunk * (x + 1)] <= -thresh)):
            tf_list.extend([True] * chunk)
        else:
            tf_list.extend([False] * chunk)

    tf_list.extend((len(wav) - len(tf_list)) * [False])
    return(wav[tf_list])



def to_mfcc(wav):
    '''
    Converts wav file to Mel Frequency Ceptral Coefficients
    :param wav (numpy array): Wav form
    :return (2d numpy array: MFCC
    '''
    return(librosa.feature.mfcc(y=wav, sr=RATE, n_mfcc=N_MFCC))

def categorical(y):
    '''
    Converts list of languages into a binary class matrix
    :param y (list): list of languages
    :return (numpy array): binary class matrix
    '''
    lang_dict = {}
    for index,language in enumerate(set(y)):
        lang_dict[language] = index
    y = map(lambda x: lang_dict[x],y)
    return utils.to_categorical(list(y), len(lang_dict))

def segment_mfccs(X_train):
    '''
    Creates segmented MFCCs from X_train
    :param X_train: list of MFCCs
    :return: segmented mfccs
    '''
    segmented_mfccs = []
    for mfcc in X_train:
        segmented_mfccs.append(segment(mfcc))
    return(segmented_mfccs)

def normalize(mfcc):
    '''
    Normalize mfcc
    :param mfcc:
    :return:
    '''
    mms = MinMaxScaler()
    return(mms.fit_transform(np.abs(mfcc)))

def segmentation(mfccs,labels):
    '''
    Makes segments of mfccs and attaches them to the labels
    :param mfccs: list of mfccs
    :param labels: list of labels
    :return (tuple): Segments with labels
    '''
    segments = []
    seg_labels = []
    for mfcc,label in zip(mfccs,labels):
        for start in range(0, int(mfcc.shape[1] / COL_SIZE)):
            segments.append(mfcc[:, start * COL_SIZE:(start + 1) * COL_SIZE])
            seg_labels.append(label)
    return(segments, seg_labels)

def segment(mfcc):
    '''
    Creates segments from on mfcc image. If last segments is not long enough to be length of columns divided by COL_SIZE
    :param mfcc (numpy array): MFCC array
    :return (numpy array): Segmented MFCC array
    '''
    segments = []
    for start in range(0, int(mfcc.shape[1] / COL_SIZE)):
        segments.append(mfcc[:, start * COL_SIZE:(start + 1) * COL_SIZE])
    return(np.array(segments))


def get_wav(language_num):
    '''
    Load wav file from disk and down-samples to RATE
    :param language_num (list): list of file names
    :return (numpy array): Down-sampled wav file
    '''
    y, sr = librosa.load('/Users/Rahul/Desktop/MSIS/NLP/Accent Finder/wav/{}.wav'.format(language_num))
    return(librosa.core.resample(y=y,orig_sr=sr,target_sr=RATE, scale=True))

def model_commit(model, model_filename):
    '''
    Save model to file
    :param model: Trained model to be saved
    :param model_filename: Filename
    :return: None
    '''
    model.save('../models/{}.h5'.format(model_filename))  # creates a HDF5 file 'my_model.h5'

def training(X_train,y_train,X_validation,y_validation, batch_size=128): #64
    '''
    Trains 2D convolutional neural network
    :param X_train: Numpy array of mfccs
    :param y_train: Binary matrix based on labels
    :return: Trained model
    '''

    # Get row, column, and class sizes
    rows = X_train[0].shape[0]
    cols = X_train[0].shape[1]
    val_rows = X_validation[0].shape[0]
    val_cols = X_validation[0].shape[1]
    num_classes = len(y_train[0])

    # input image dimensions to feed into 2D ConvNet Input layer
    input_shape = (rows, cols, 1)
    X_train = X_train.reshape(X_train.shape[0], rows, cols, 1 )
    X_validation = X_validation.reshape(X_validation.shape[0],val_rows,val_cols,1)


    print('x_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3,3), activation='relu',
                     data_format="channels_last",
                     input_shape=input_shape))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64,kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    # Stops training if accuracy does not change at least 0.005 over 10 epochs
    es = EarlyStopping(monitor='acc', min_delta=.005, patience=10, verbose=1, mode='auto')

    # Creates log file for graphical interpretation using TensorBoard
    tb = TensorBoard(log_dir='../logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True,
                     write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                     embeddings_metadata=None)

    # Image shifting
    datagen = ImageDataGenerator(width_shift_range=0.05)

    # Fit model using ImageDataGenerator
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(X_train) / 32
                        , epochs=EPOCHS,
                        callbacks=[es,tb], validation_data=(X_validation,y_validation))

    return (model)



if __name__ == '__main__':
    '''
        Console command example:
        python trainmodel.py bio_metadata.csv model50
        '''

    # Load arguments
    file = sys.argv[1]
    model_filename = sys.argv[2]

    # Load metadata
    df = pd.read_csv(file)

    # Filter metadata to retrieve only files desired
    df_filtered = split_dataset.filter_df(df)

    # Train test split
    Train_X, Test_X, Train_Y, Test_Y = split_dataset.split_people(df_filtered)

    # Get statistics
    train_count = Counter(Train_Y)
    test_count =  Counter(Test_Y)

    base_acc = test_count.most_common(1)[0][1] / float(np.sum(np.array(list(test_count.values())).astype(float)))
    #print(acc_to_beat)

    # To categorical
    Train_Y = categorical(Train_Y)
    Test_Y = categorical(Test_Y)

    # Get resampled wav files using multiprocessing
    if DEBUG:
        print('loading wav files')
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    Train_X = pool.map(get_wav, Train_X)
    Test_X = pool.map(get_wav, Test_X)

    # Convert to MFCC
    if DEBUG:
        print('converting to mfcc')
    Train_X = pool.map(to_mfcc, Train_X)
    Test_X = pool.map(to_mfcc, Test_X)

    # Create segments from MFCCs
    Train_X, Train_Y = segmentation(Train_X, Train_Y)
    Validate_X, Validate_Y = segmentation(Test_X, Test_Y)

    # Randomize training segments
    Train_X, _, Train_Y, _ = train_test_split(Train_X, Train_Y, test_size=0.8)

    # Train model
    model = training(np.array(Train_X), np.array(Train_Y), np.array(Validate_X),np.array(Validate_Y))

    # Make predictions on full X_test MFCCs
    Predicted_Y = predictions.predict_all(segment_mfccs(Test_X), model)

    # Print statistics
    print('The total number of training samples is ', train_count)
    print('The total number of testing samples is ', test_count)
    print('The baseline accuracy derived by the most common class occurrence is ', base_acc)
    print('The confusion matrix and count: ', np.sum(predictions.con_mat(Predicted_Y, Test_Y),axis=1))
    print(predictions.con_mat(Predicted_Y, Test_Y))
    print('The accuracy of the model is ', predictions.cal_acc(Predicted_Y,Test_Y))

    # Save model
    model_commit(model, model_filename)