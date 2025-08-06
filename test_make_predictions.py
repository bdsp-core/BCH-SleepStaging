import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Use GPU 0 only
import numpy as np
import pandas as pd
import h5py
from tensorflow.keras.utils import Sequence
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import KFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
import time
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import logging
tf.get_logger().setLevel(logging.ERROR)
import mne
mne.set_log_level('WARNING')
import concurrent.futures

start_time = time.time()



def robust_normalization(signal):
    """Apply median-centered normalization with IQR scaling and outlier clipping."""
    
    median = np.median(signal, axis=0)  # Compute median per channel
    iqr = np.percentile(signal, 75, axis=0) - np.percentile(signal, 25, axis=0)  # Compute IQR per channel
    # Avoid division by zero by setting IQR to 1 for channels with zero IQR
    if iqr==0:
        iqr=1
    

    # Normalize
    signal = (signal - median) / iqr

    # Clip outliers beyond 20*IQR
    signal = np.clip(signal, -20, 20)

    return signal


# Define annotation mapping
VALID_CLASSES = {"N1", "N2", "N3", "REM", "WAKE"}
ANNOTATION_MAPPING = {"N1": 0, "N2": 1, "N3": 2, "REM": 3, "WAKE": 4}
INVERSE_ANNOTATION_MAPPING = {0: "N1", 1: "N2", 2: "N3", 3: "REM", 4: "WAKE"}

def preprocess_signal(h5_file):

    
    with h5py.File(h5_file, 'r') as f:
        # Load all channels
        channels = ['c3-m2', 'c4-m1', 'f3-m2', 'f4-m1', 
                    'o1-m2', 'o2-m1', 'e1-m2', 'e2-m1', 
                    'chin1-chin2']
        signals = np.stack([f['signals'][channel][:] for channel in channels], axis=1).squeeze(-1)   # Shape: (T * 3840, 9)
        
        stage_raw = f['annotations']['stage'][:].squeeze()

    
    signals = signals.T
    
    sfreq_in = 200         
    ch_types = ['eeg']*6 + ['eog']*2 + ['emg']   
    channel_names = channels
    info = mne.create_info(channel_names, sfreq=sfreq_in, ch_types=ch_types)
    raw = mne.io.RawArray(signals, info)
    
    raw.notch_filter(freqs=[60], picks='all', method='spectrum_fit', filter_length='auto', phase='zero')   
    raw.filter(l_freq=0.3, h_freq=35, picks=['eeg', 'eog'], fir_design='firwin', phase='zero-double')   
    raw.filter(l_freq=10,  h_freq=99.9, picks='emg', fir_design='firwin', phase='zero-double')  
    
    raw.resample(sfreq=128, npad='auto') 
    
    signal = raw.get_data().T 

    eeg_indices = list(range(8))
    chin_index = 8   
    for idx in eeg_indices:
        signal[:, idx] = robust_normalization(signal[:, idx])

    signal[:,chin_index] = robust_normalization(np.abs(hilbert(signal[:, chin_index])))

    signal = signal.astype(np.float32)  
    
    
    downsample_factor = 6000
    stage_down = stage_raw[::downsample_factor]
    annotations = [INVERSE_ANNOTATION_MAPPING.get(x, "UNKNOWN") for x in stage_down]


    # Compute total epochs
    T = signal.shape[0] // 3840  # Number of 30-sec epochs
    #print(T)

    # If T is not divisible by 35, repeat last epoch
    remainder = T % 35
    if remainder > 0:
        extra_epochs = 35 - remainder
        last_epoch = signal[-3840:]  # Last 30-sec epoch
        signal = np.concatenate([signal] + [last_epoch] * extra_epochs, axis=0)
        T_extra = T + extra_epochs  # Adjust total epochs
    else:
        T_extra = T

    # Reshape into (num_windows, 134400, 8)
    num_windows = T_extra // 35
    signal_windows = np.array([
        signal[i * 3840 * 35:(i + 1) * 3840 * 35].reshape(134400, 9)
        for i in range(num_windows)
    ])

    return signal_windows, T, annotations  # Return segmented signals & original number of epochs

def predict_sleep_stages_5class(h5_file,model_5class):

    # Preprocess signal
    signal_windows, T, annotations = preprocess_signal(h5_file)

    # Predict using model
    predictions = model_5class.predict(signal_windows,verbose = 0)  # Shape: (num_windows, 35, 5)

    # Convert softmax probabilities to class labels
    predicted_classes = np.argmax(predictions, axis=-1).flatten()  # Shape: (T,)

    # Remove extra predictions (if padded)
    predicted_classes = predicted_classes[0:T]

    # Convert to sleep stage labels
    predicted_labels = [INVERSE_ANNOTATION_MAPPING[cls] for cls in predicted_classes]

    return predicted_labels, annotations 

def evaluate_predictions_5class(h5file,model_5class):
    
    h5_file = os.path.join('/media/ayush/Elements1/BCH_h5_arranged', h5file)
    
    prediction_path = os.path.join( '/media/ayush/Elements1/predictions_normalized' ,h5file.replace('.h5','_pred.csv'))
    ground_path = os.path.join( '/media/ayush/Elements1/predictions_normalized' ,h5file.replace('.h5','_ground.csv'))
    
    predicted_labels, ground_truth_labels = predict_sleep_stages_5class(h5_file,model_5class)
    
    df_ground = pd.DataFrame({"Ground Stage": ground_truth_labels})
    df_ground.to_csv(ground_path, index=False)
    
    df_pred = pd.DataFrame({"Predicted Stage": predicted_labels})
    df_pred.to_csv(prediction_path, index=False)
    
    
    valid_indices = [i for i, stage in enumerate(ground_truth_labels) if stage in VALID_CLASSES]
    
    if not valid_indices:
        print(f"No valid sleep stages found in {h5file}. Skipping evaluation.")
        return np.nan, np.nan, np.array([]), np.array([])
        
    filtered_ground_truth = [ground_truth_labels[i] for i in valid_indices]
    filtered_predictions = [predicted_labels[i] for i in valid_indices]
    
    # Convert ground truth labels to integers
    ground_truth_classes = [ANNOTATION_MAPPING[label] for label in filtered_ground_truth]

    # Convert predicted labels to integers
    predicted_classes = [ANNOTATION_MAPPING[label] for label in filtered_predictions]

    # Compute accuracy
    accuracy = accuracy_score(ground_truth_classes, predicted_classes)

    # Compute Cohen's Kappa
    kappa = cohen_kappa_score(ground_truth_classes, predicted_classes)

    # Print results
    print(f"Results for {h5file}" + '  Accuracy: ' + str(accuracy) + ' Kappa:' + str(kappa))

    return accuracy,kappa,np.array(filtered_ground_truth),np.array(filtered_predictions)
    



    

# Load trained model
model_above1y = keras.models.load_model("/media/ayush/Elements1/trained_model_usleep_3models_normalized_above1y_EEGEOGEMG.h5")
model_6mo1y = keras.models.load_model("/media/ayush/Elements1/trained_model_usleep_3models_normalized_6mo1y_EEGEOGEMG.h5")
model_below6mo = keras.models.load_model("/media/ayush/Elements1/trained_model_usleep_3models_normalized_below6mo_EEGEOGEMG.h5")





root_dir = '/media/ayush/Elements1/BCH_h5_arranged'
all_files = []
for fo in os.listdir(root_dir):
    all_files.append(fo)
    
df_test = pd.read_csv('/media/ayush/Ayush/Documents/BCH_data_analysis/USleep_expt/CommonIDs/test_set.csv')
subID_test = df_test['subID'].to_numpy()
sess_test = df_test['Session'].to_numpy()
age_test = df_test['AgeDays'].to_numpy()
test_files = []
test_files_6mo = []
test_files_6mo1y = []
accuracy_vector = []
kappa_vector = []

for i in range(0,len(subID_test)):
    subID_sess_test = subID_test[i] + '_ses-' + str(sess_test[i])
    for fo in all_files:
        if fo.startswith(subID_sess_test):
            if age_test[i]<=180:
                test_files_6mo.append(fo)
            elif age_test[i]>180 and age_test[i]<=365:
                test_files_6mo1y.append(fo)
            elif age_test[i]>365:
                test_files.append(fo)


print(len(test_files_6mo))
print(len(test_files_6mo1y))
print(len(test_files))


for i in range(0,len(test_files)):
    accuracy,kappa,ground_truth_classes,predicted_classes = evaluate_predictions_5class(test_files[i],model_above1y)
    accuracy_vector.append(accuracy)
    kappa_vector.append(kappa)           
            

print(f"Mean Accuracy: {np.nanmean(accuracy_vector):.4f}")
print(f"Mean Cohen's Kappa: {np.nanmean(kappa_vector):.4f}")


accuracy_vector6mo1y = []
kappa_vector6mo1y = []
for i in range(0,len(test_files_6mo1y)):
    accuracy,kappa,ground_truth_classes,predicted_classes = evaluate_predictions_5class(test_files_6mo1y[i],model_6mo1y)
    accuracy_vector6mo1y.append(accuracy)
    kappa_vector6mo1y.append(kappa)  
            
     
print(f"Mean Accuracy: {np.nanmean(accuracy_vector6mo1y):.4f}")
print(f"Mean Cohen's Kappa: {np.nanmean(kappa_vector6mo1y):.4f}")


accuracy_vectorbelow6mo = []
kappa_vectorbelow6mo = []
for i in range(0,len(test_files_6mo)):
    accuracy,kappa,ground_truth_classes,predicted_classes = evaluate_predictions_5class(test_files_6mo[i],model_below6mo)
    accuracy_vectorbelow6mo.append(accuracy)
    kappa_vectorbelow6mo.append(kappa)  
    

print(f"Mean Accuracy: {np.nanmean(accuracy_vectorbelow6mo):.4f}")
print(f"Mean Cohen's Kappa: {np.nanmean(kappa_vectorbelow6mo):.4f}")






'''
Above 1y
Mean Accuracy: 0.7875
Mean Cohen's Kappa: 0.7060

6mo-1y
Mean Accuracy: 0.7208
Mean Cohen's Kappa: 0.6351

Below 6 mo 
Mean Accuracy: 0.6070
Mean Cohen's Kappa: 0.4878
'''