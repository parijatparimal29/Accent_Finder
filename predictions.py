from collections import Counter
import numpy as np

def prob_class_audio(MFCCs, model):
    '''
    Predict class based on MFCC samples' probabilities
    :param MFCCs: Numpy array of MFCCs
    :param model: Trained model
    :return: Predicted class of MFCC segment group
    '''
    MFCCs = MFCCs.reshape(MFCCs.shape[0],MFCCs.shape[1],MFCCs.shape[2],1)
    y_predicted = model.predict_proba(MFCCs,verbose=0)
    return(np.argmax(np.sum(y_predicted,axis=0)))



def con_mat(y_predicted,y_test):
    '''
    Create confusion matrix
    :param y_predicted: list of predictions
    :param y_test: numpy array of shape (len(y_test), number of classes). 1.'s at index of actual, otherwise 0.
    :return: numpy array. confusion matrix
    '''
    confusion_matrix = np.zeros((len(y_test[0]),len(y_test[0])),dtype=int )
    for index, predicted in enumerate(y_predicted):
        confusion_matrix[np.argmax(y_test[index])][predicted] += 1
    return(confusion_matrix)

def cal_acc(y_predicted,y_test):
    '''
    Get accuracy
    :param y_predicted: numpy array of predictions
    :param y_test: numpy array of actual
    :return: accuracy
    '''
    matrix = con_mat(y_predicted,y_test)
    return( np.sum(matrix.diagonal()) / float(np.sum(matrix)))

def predict_all(X_train, model):
    '''
    :param X_train: List of segmented mfccs
    :param model: trained model
    :return: list of predictions
    '''
    predictions = []
    for mfcc in X_train:
        predictions.append(predict_audio_class(mfcc, model))
        # predictions.append(predict_prob_class_audio(mfcc, model))
    return predictions

def predict_audio_class(MFCCs, model):
    '''
    Predict class based on MFCC samples
    :param MFCCs: Numpy array of MFCCs
    :param model: Trained model
    :return: Predicted class of MFCC segment group
    '''
    MFCCs = MFCCs.reshape(MFCCs.shape[0],MFCCs.shape[1],MFCCs.shape[2],1)
    y_predicted = model.predict_classes(MFCCs,verbose=0)
    return(Counter(list(y_predicted)).most_common(1)[0][0])

if __name__ == '__main__':
    pass
