import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0 only
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
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from scipy.signal import hilbert
import mne
mne.set_log_level('WARNING')
import sys
from pathlib import Path
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm 

def build_usleep_model(input_shape=(134400, 9), alpha=1.67, num_classes=5, l2_lambda=None, dropout_rate=None):
    def encoder_block(x, filters, kernel_size=9):
        kernel_regularizer = tf.keras.regularizers.l2(l2_lambda) if l2_lambda else None
        x = layers.Conv1D(filters, kernel_size, padding='same', kernel_regularizer=kernel_regularizer)(x)
        x = layers.ELU()(x)
        x = layers.BatchNormalization()(x)
        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)  # Apply dropout only if specified
        res = x
        x = layers.ZeroPadding1D((0, 1))(x) if x.shape[1] % 2 != 0 else x
        x = layers.MaxPooling1D(2)(x)
        return x, res

    def decoder_block(x, res, filters, kernel_size=9):
        kernel_regularizer = tf.keras.regularizers.l2(l2_lambda) if l2_lambda else None
        x = layers.UpSampling1D(2)(x)
        x = layers.Conv1D(filters, kernel_size, padding='same', kernel_regularizer=kernel_regularizer)(x)
        x = layers.ELU()(x)
        x = layers.BatchNormalization()(x)
        
        # Crop or pad the residual connection to match x's shape
        diff = res.shape[1] - x.shape[1]
        if diff > 0:
            res = layers.Cropping1D((diff // 2, diff - diff // 2))(res)
        elif diff < 0:
            x = layers.Cropping1D((-diff // 2, -diff - (-diff // 2)))(x)
        
        x = layers.Concatenate()([x, res])
        x = layers.Conv1D(filters, kernel_size, padding='same', kernel_regularizer=kernel_regularizer)(x)
        x = layers.ELU()(x)
        x = layers.BatchNormalization()(x)
        return x

    inputs = keras.Input(shape=input_shape)
    x = inputs

    encoder_residuals = []
    filter_sizes = np.array([6, 9, 11, 15, 20, 28, 40, 55, 77, 108, 152, 214])

    for filters in filter_sizes:
        x, res = encoder_block(x, filters)
        encoder_residuals.append(res)

    x = layers.Conv1D(int(306 * np.sqrt(alpha)), 9, padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(l2_lambda) if l2_lambda else None)(x)
    x = layers.ELU()(x)
    x = layers.BatchNormalization()(x)
    if dropout_rate:
        x = layers.Dropout(dropout_rate)(x)

    for res, filters in zip(reversed(encoder_residuals), reversed(filter_sizes)):
        x = decoder_block(x, res, filters)

    x = layers.Conv1D(6, 1, padding='same', activation='tanh')(x)
    x = layers.AveragePooling1D(pool_size=3840)(x)
    x = layers.Conv1D(5, 1, padding='same', activation='elu')(x)
    if dropout_rate:
        x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Conv1D(num_classes, 1, padding='same', activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model

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



# Define valid sleep stage classes
VALID_CLASSES = {"N1", "N2", "N3", "REM", "WAKE"}
ANNOTATION_MAPPING = {"N1": 0, "N2": 1, "N3": 2, "REM": 3, "WAKE": 4}
INVERSE_ANNOTATION_MAPPING = {0: "N1", 1: "N2", 2: "N3", 3: "REM", 4: "WAKE"}


def save_all_segments_to_tfrecord(h5file):
    """
    Save all possible 35-segment groups from a folder to a TFRecord file.

    Args:
        folder (str): Path to the folder containing the h5 and csv files.
        tfrecord_path (str): Path to save the TFRecord file.
    """
    h5_file = os.path.join('/media/ayush/Elements1/BCH_h5_arranged', h5file)
    #print(h5_file)
    tfrecord_path = os.path.join( '/home/ayush/Documents/tfrecords_normalized' ,h5file.replace('.h5','.tfrecord'))
    #print(tfrecord_path)
    

    # Load signals
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

    
    # Filter valid indices and convert annotations to integers
    valid_indices = [i for i, ann in enumerate(annotations) if ann in ANNOTATION_MAPPING]
    valid_signals = [
        signal[i * 3840:(i + 1) * 3840, :] for i in valid_indices
    ]
    valid_annotations = [
        ANNOTATION_MAPPING[annotations[i]] for i in valid_indices
    ]
    

    # Ensure there are enough segments for at least one group
    if len(valid_signals) < 35:
        print(f"Skipping folder {h5file}: Not enough valid segments")
        return
    
    var = 0
    # Write all possible 35-segment groups to the TFRecord
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for i in range(0, len(valid_signals) - 35 + 1,35):
            var = var + 1
            # Take a group of 35 segments
            signals = np.concatenate(valid_signals[i:i + 35], axis=0)  # Shape: (134400, 12)
            annotations = valid_annotations[i:i + 35]  # Shape: (35,)

            # Save the group to TFRecord
            feature = {
                'signals': tf.train.Feature(bytes_list=tf.train.BytesList(value=[signals.tobytes()])),
                'annotations': tf.train.Feature(int64_list=tf.train.Int64List(value=annotations)),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
    
def parse_tfrecord(example_proto):
    """
    Parse a single TFRecord example into signals and annotations.

    Args:
        example_proto: Serialized TFRecord example.

    Returns:
        signals: Tensor of shape (134400, 8).
        annotations: Tensor of shape (35,).
    """
    feature_description = {
        'signals': tf.io.FixedLenFeature([], tf.string),
        'annotations': tf.io.FixedLenFeature([35], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    # Decode signals
    signals = tf.io.decode_raw(parsed_example['signals'], tf.float32)
    signals = tf.reshape(signals, (134400, 9))

    # Extract annotations
    annotations = tf.cast(parsed_example['annotations'], tf.int32)

    annotations = tf.one_hot(annotations, depth=5)  # 5 classes for the 5-class model

    return signals, annotations


def create_dataset(tfrecord_files, batch_size=32, shuffle_buffer_size=1000, prefetch_buffer_size=tf.data.AUTOTUNE):
    """
    Create a tf.data.Dataset pipeline for TFRecords.

    Args:
        tfrecord_files (list): List of TFRecord file paths.
        batch_size (int): Batch size for training.
        shuffle_buffer_size (int): Buffer size for shuffling.
        prefetch_buffer_size: Buffer size for prefetching.

    Returns:
        A tf.data.Dataset object.
    """
    dataset = tf.data.TFRecordDataset(tfrecord_files)  # Load TFRecords
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)  # Parse each record
    dataset = dataset.shuffle(shuffle_buffer_size)  # Shuffle dataset
    dataset = dataset.batch(batch_size)  # Batch the data
    dataset = dataset.prefetch(prefetch_buffer_size)  # Prefetch for performance
    return dataset




def compute_class_weights(dataset, num_classes=5):
    """
    Compute class weights efficiently from a large tf.data.Dataset.
    
    Args:
        dataset (tf.data.Dataset): The dataset containing (features, labels).
        num_classes (int): Number of classes (default: 5 for sleep staging).

    Returns:
        class_weights (dict): A dictionary mapping class indices to their computed weight.
    """
    def accumulate_counts(accumulated_counts, batch):
        _, labels = batch  # Extract labels
        labels = tf.reshape(labels, [-1])  # Flatten labels
        new_counts = tf.math.bincount(labels, minlength=num_classes, maxlength=num_classes)
        return accumulated_counts + new_counts

    # Initialize class counts to zeros
    initial_counts = tf.zeros([num_classes], dtype=tf.int32)

    # Accumulate label counts across the dataset
    class_counts = dataset.map(lambda x, y: (x, tf.cast(y, tf.int32))).reduce(initial_counts, accumulate_counts)

    # Convert class counts to numpy
    class_counts = class_counts.numpy()
    print(class_counts)
    
    # Compute class weights using sklearn's compute_class_weight()
    class_labels = np.arange(num_classes)
    class_weights = compute_class_weight(class_weight="balanced", classes=class_labels, y=np.repeat(class_labels, class_counts))

    # Convert to dictionary format
    class_weights_dict = {c: w for c, w in zip(class_labels, class_weights)}

    print("Computed Class Weights:", class_weights_dict)
    return class_weights_dict



root_dir = '/media/ayush/Elements1/BCH_h5_arranged'
all_files = []
for fo in os.listdir(root_dir):
	all_files.append(fo)

print(len(all_files))


df_train = pd.read_csv('/media/ayush/Ayush/Documents/BCH_data_analysis/USleep_expt/CommonIDs/train_set.csv')
subID_train = df_train['subID'].to_numpy()
sess_train = df_train['Session'].to_numpy()
age_train = df_train['AgeDays'].to_numpy()
train_files = []
train_files_6mo = []
train_files_6mo1y = []
all_train_files = []


for i in range(0,len(subID_train)):
    subID_sess_train = subID_train[i] + '_ses-' + str(sess_train[i])
    for fo in all_files:
        if fo.startswith(subID_sess_train):
            all_train_files.append(fo)
            if age_train[i]<=180:
                train_files_6mo.append(fo)
            elif age_train[i]>180 and age_train[i]<=365:
                train_files_6mo1y.append(fo)
            else:
                train_files.append(fo)

df_val = pd.read_csv('/media/ayush/Ayush/Documents/BCH_data_analysis/USleep_expt/CommonIDs/val_set.csv')
subID_val = df_val['subID'].to_numpy()
sess_val = df_val['Session'].to_numpy()
age_val = df_val['AgeDays'].to_numpy()
val_files = []
val_files_6mo = []
val_files_6mo1y = []
all_val_files = []

for i in range(0,len(subID_val)):
    subID_sess_val = subID_val[i] + '_ses-' + str(sess_val[i])
    for fo in all_files:
        if fo.startswith(subID_sess_val):
            all_val_files.append(fo)
            if age_val[i]<=180:
                val_files_6mo.append(fo)
            elif age_val[i]>180 and age_val[i]<=365:
                val_files_6mo1y.append(fo)
            else:
                val_files.append(fo)


print(len(val_files))
print(len(train_files))
print(len(val_files_6mo))
print(len(train_files_6mo))
print(len(val_files_6mo1y))
print(len(train_files_6mo1y))

N_WORKERS = min(5, cpu_count() - 1)  

root_tfr = Path('/home/ayush/Documents/tfrecords_normalized')

train_todo = [f for f in all_train_files
              if not (root_tfr / f.replace('.h5', '.tfrecord')).exists()]

val_todo   = [f for f in all_val_files
              if not (root_tfr / f.replace('.h5', '.tfrecord')).exists()]

def _worker(h5file):
    """
    Thin wrapper so that exceptions don’t crash the whole pool;
    they are bubbled up and can be logged.
    """
    try:
        save_all_segments_to_tfrecord(h5file=h5file)
        return (h5file, None)      
    except Exception as e:
        return (h5file, e)         

def run_parallel(file_list, label):
    total = len(file_list)
    if total == 0:
        print(f"[{label}] Nothing to do — all .tfrecord files exist.")
        return

    print(f"[{label}] Converting {total:,} files with {N_WORKERS} workers…")
    errors = []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_worker, h5): h5 for h5 in file_list}
        for fut in tqdm(as_completed(futures), total=total, unit="file"):
            h5file, err = fut.result()
            if err is not None:
                errors.append((h5file, err))

    # --- optional: report failures cleanly ------------------------
    if errors:
        print(f"[{label}] {len(errors)} file(s) failed:")
        for h5, err in errors:
            print(f"  {h5}: {err}")
    else:
        print(f"[{label}] All files processed successfully.")


run_parallel(train_todo, "TRAIN")
run_parallel(val_todo,   "VAL")


train_tfrecords = []
val_tfrecords = []

for files in train_files:
    if os.path.exists(os.path.join( '/home/ayush/Documents/tfrecords_normalized' ,files.replace('.h5','.tfrecord')) ): 
        train_tfrecords.append( os.path.join( '/home/ayush/Documents/tfrecords_normalized' ,files.replace('.h5','.tfrecord') ))
        
for files in val_files:
    if os.path.exists(os.path.join( '/home/ayush/Documents/tfrecords_normalized' ,files.replace('.h5','.tfrecord')) ): 
        val_tfrecords.append( os.path.join( '/home/ayush/Documents/tfrecords_normalized' ,files.replace('.h5','.tfrecord') ))
    
    

# Create datasets
train_dataset = create_dataset(train_tfrecords, batch_size=64, shuffle_buffer_size=500)
val_dataset = create_dataset(val_tfrecords, batch_size=64, shuffle_buffer_size=500)

#print('Computing class weights for >1year')
#class_weights_5class = compute_class_weights(train_dataset, num_classes=5)
class_weights_above1y = {0: 3.967282580844779, 1: 0.6087578526948138, 2: 0.7233979814826732, 3: 1.2021629896574961, 4: 1.1222713186563715}


train_tfrecords_6mo = []
val_tfrecords_6mo = []

for files in train_files_6mo:
    if os.path.exists(os.path.join( '/home/ayush/Documents/tfrecords_normalized' ,files.replace('.h5','.tfrecord'))): 
        train_tfrecords_6mo.append( os.path.join( '/home/ayush/Documents/tfrecords_normalized' ,files.replace('.h5','.tfrecord') ) )
    
for files in val_files_6mo:
    if os.path.exists(os.path.join( '/home/ayush/Documents/tfrecords_normalized' ,files.replace('.h5','.tfrecord'))): 
        val_tfrecords_6mo.append( os.path.join( '/home/ayush/Documents/tfrecords_normalized' ,files.replace('.h5','.tfrecord') ) )

train_dataset_6mo = create_dataset(train_tfrecords_6mo, batch_size=64, shuffle_buffer_size=500)
val_dataset_6mo= create_dataset(val_tfrecords_6mo, batch_size=64, shuffle_buffer_size=500)

#print('Computing class weights for <6mo')
#class_weights_5class = compute_class_weights(train_dataset_6mo, num_classes=5)
class_weights_below6mo = {0: 3.9489751708048657, 1: 1.3347865269798356, 2: 0.7301560892043949, 3: 0.6193398322138881, 4: 0.9867833705319964}


train_tfrecords_6mo1y = []
val_tfrecords_6mo1y = []

for files in train_files_6mo1y:
    if os.path.exists(os.path.join( '/home/ayush/Documents/tfrecords_normalized' ,files.replace('.h5','.tfrecord'))): 
        train_tfrecords_6mo1y.append( os.path.join( '/home/ayush/Documents/tfrecords_normalized' ,files.replace('.h5','.tfrecord') ) )
    
for files in val_files_6mo1y:
    if os.path.exists(os.path.join( '/home/ayush/Documents/tfrecords_normalized' ,files.replace('.h5','.tfrecord'))): 
        val_tfrecords_6mo1y.append( os.path.join( '/home/ayush/Documents/tfrecords_normalized' ,files.replace('.h5','.tfrecord') ) )
    

train_dataset_6mo1y = create_dataset(train_tfrecords_6mo1y, batch_size=64, shuffle_buffer_size=500)
val_dataset_6mo1y = create_dataset(val_tfrecords_6mo1y, batch_size=64, shuffle_buffer_size=500)

#print('Computing class weights for 6mo-1y')
#class_weights_5class = compute_class_weights(train_dataset_6mo1y, num_classes=5)
class_weights_6mo1y = {0: 3.08849508970209, 1: 0.8489134860434605, 2: 0.7077712259703032, 3: 0.8330965488663944, 4: 1.1299251319724615}



# Build the model
model = build_usleep_model(input_shape=(134400, 9), num_classes=5, l2_lambda=None, dropout_rate=None)
print(model.summary())


# Compile the model
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=50, restore_best_weights=True)

model.fit(train_dataset, validation_data=val_dataset, epochs=10000, callbacks=[early_stopping],verbose=1,class_weight=class_weights_above1y)
model.save("trained_model_usleep_3models_normalized_above1y_EEGEOGEMG.h5")




################# Below 6 months #####################################

# Load the trained 5-class model on age>1year
tf.keras.backend.clear_session()
model_below6mo = load_model("trained_model_usleep_3models_normalized_above1y_EEGEOGEMG.h5")

# Compile the model
model_below6mo.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping_below6mo = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=50, restore_best_weights=True)
print(model_below6mo.summary())

model_below6mo.fit(train_dataset_6mo, validation_data=val_dataset_6mo, epochs=10000, callbacks=[early_stopping_below6mo], verbose=1,class_weight=class_weights_below6mo)
model_below6mo.save("trained_model_usleep_3models_normalized_below6mo_EEGEOGEMG.h5")


################# 6mo - 1y #####################################

# Load the trained 5-class model
tf.keras.backend.clear_session()
model_6mo1y = load_model("trained_model_usleep_3models_normalized_above1y_EEGEOGEMG.h5")

# Compile the model
model_6mo1y.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping_6mo1y = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=50, restore_best_weights=True)
print(model_6mo1y.summary())

model_6mo1y.fit(train_dataset_6mo1y, validation_data=val_dataset_6mo1y, epochs=10000, callbacks=[early_stopping_6mo1y], verbose=1,class_weight=class_weights_6mo1y)
model_6mo1y.save("trained_model_usleep_3models_normalized_6mo1y_EEGEOGEMG.h5")
