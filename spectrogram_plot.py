import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import os 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.stats import wilcoxon
from scipy.stats import ttest_rel
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr
import numpy as np
import mne
from mne.filter import filter_data, notch_filter
from mne.time_frequency import psd_array_multitaper
from scipy.signal import detrend
from collections import Counter
from scipy.interpolate import interp2d
import h5py
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
mne.set_log_level('WARNING')
from concurrent.futures import ProcessPoolExecutor, as_completed

def evaluate_predictions(h5_file):
    """
    Compare predicted sleep stages with ground truth and compute accuracy and Cohen's Kappa.

    Args:
        h5_file (str): Path to the test file.

    Returns:
        None (prints accuracy and Cohen's Kappa)
    """

    VALID_CLASSES = {"N1", "N2", "N3", "REM", "WAKE"}
    ANNOTATION_MAPPING = {"N1": 1, "N2": 2, "N3": 3, "REM": 4, "WAKE": 0}

    gt_file = h5_file.replace(".h5", "_ground.csv")
    df_gt = pd.read_csv(os.path.join('predictions_normalized',gt_file))
    ground_truth_labels = df_gt['Ground Stage'].tolist()

    pred_file = h5_file.replace(".h5", "_pred.csv")
    df_pred = pd.read_csv(os.path.join('predictions_normalized',pred_file))
    predicted_labels = df_pred['Predicted Stage'].tolist()


    # Filter only valid sleep stages (ignore junk labels)
    valid_indices = [i for i, stage in enumerate(ground_truth_labels) if stage in VALID_CLASSES]

    if not valid_indices:
        #print(f"No valid sleep stages found in {csv_file}. Skipping evaluation.")
        return np.nan, np.nan, np.nan, np.nan, np.array([]), np.array([]), np.array([]), np.array([])

    filtered_ground_truth = [ground_truth_labels[i] for i in valid_indices]
    filtered_predictions = [predicted_labels[i] for i in valid_indices]

    # Convert ground truth and predicted labels to integers
    ground_truth_classes = [ANNOTATION_MAPPING[label] for label in filtered_ground_truth]
    predicted_classes = [ANNOTATION_MAPPING[label] for label in filtered_predictions]


    # Compute accuracy and cohen kappa for 5 classes
    accuracy = accuracy_score(ground_truth_classes, predicted_classes)
    kappa = cohen_kappa_score(ground_truth_classes, predicted_classes)

    return accuracy, kappa, ground_truth_labels, predicted_labels
    
def safe_spectrogram(raw_signal, Fs, **kwargs):
    if raw_signal.ndim == 1:
        raw_signal = raw_signal[np.newaxis, :]
    elif raw_signal.shape[1] == 1:  # if shape is (time, 1), transpose
        raw_signal = raw_signal.T
    return spectrogram(raw_signal, Fs=Fs, **kwargs)

def spectrogram(signal, Fs, signaltype=None, epoch_time=30, epoch_step_time=30, decibel=True, fmin=0.02, fmax=60, bandwidth=None, adaptive=True, n_jobs=1):
    """
    Inputs:
    signal: 1d numpy array of signal (time domain)
    Fs: sampling frequency
    signaltype: keywords/shortcuts (see code below, selects bandwith based on keyword)
    epoch_time: window-length in seconds
    epoch_step_time: stepsize in seconds
    decibel: boolean, if result shall be return in decibel (default True)
    fmin: minimum frequency of interest
    fmax: maximum frequency of interest
    bandwidth: multi-taper bandwidth parameter
    adaptive: (see MNE description. True=more accurate but slow)
    n_jobs: parallel jobs.
    Returns:
    # specs.shape = (#epoch, #channel, #freq)
    # freq.shape = (#freq,)
    """
    
    # segment
    epoch_size = int(round(epoch_time*Fs))
    epoch_step = int(round(epoch_step_time*Fs))
    start_ids = np.arange(0, signal.shape[1]-epoch_size+1, epoch_step)
    seg_ids = list(map(lambda x:np.arange(x,x+epoch_size), start_ids))
    signal_segs = signal[:,seg_ids].transpose(1,0,2)  # signal_segs.shape=(#epoch, #channel, Tepoch)
    if 0:
        print(signal_segs.shape)
    # compute spectrogram

    if bandwidth is None:
        if signaltype == 'eeg':       
            NW = 10.
            bandwidth = NW*2./epoch_time
        elif signaltype == 'resp_effort':
            NW = 1
            bandwidth = NW/epoch_time
        else:
            raise ValueError("Unexpected signaltype! ")

    # experimenting values with toy data:
    # bandwidth = 1
    # half_nbw = 0.55
    # bandwidth = half_nbw / (epoch_time  * Fs / (2. * Fs))
    # print(bandwidth)

    # this is how half nbw is computed in code:
    # n_times = signal_segs.shape[-1]
    # half_nbw = float(bandwidth) * n_times / (2. * sfreq)
    # n_tapers_max = int(2 * half_nbw)

    specs, freq = psd_array_multitaper(signal_segs, Fs, fmin=fmin, fmax=fmax, adaptive=adaptive, low_bias=True, verbose='ERROR', bandwidth=bandwidth, normalization='full', n_jobs=n_jobs)

    if decibel:
        specs = 10*np.log10(specs)
    
    return specs, freq, signal_segs


def save_spect_img(test_single_file,age_days):
	accuracy, kappa, ground_truth_labels, predicted_labels = evaluate_predictions(test_single_file)
	output_dir = 'results_normalized/spectrograms'
	png_fi = test_single_file.replace(".h5", "")
	if not np.isnan(kappa):
		h5_file = os.path.join(root_dir,test_single_file) 
		with h5py.File(h5_file, 'r') as f:
			channels = ['c3-m2', 'c4-m1', 'f3-m2', 'f4-m1','o1-m2', 'o2-m1']                    
			signals = np.stack([f['signals'][channel][:] for channel in channels], axis=1).squeeze(-1)   # Shape: (T * 3840, 9)
		signals = signals.T

		sfreq_in = 200         
		ch_types = ['eeg']*6 
		channel_names = channels
		info = mne.create_info(channel_names, sfreq=sfreq_in, ch_types=ch_types)
		raw = mne.io.RawArray(signals, info)

		raw.notch_filter(freqs=[60], picks='all', method='spectrum_fit', filter_length='auto', phase='zero')   
		raw.filter(l_freq=0.3, h_freq=35, picks=['eeg'], fir_design='firwin', phase='zero-double')   
		raw.resample(sfreq=128, npad='auto') 

		signal = raw.get_data().T 
		signal = signal.astype(np.float32)  

		specs_c3,freq, signal_segs = safe_spectrogram(signal[:,0], Fs = 128, signaltype='eeg', epoch_time=2, epoch_step_time=1, bandwidth=2, fmin=0, fmax=20)
		specs_c4,freq, signal_segs = safe_spectrogram(signal[:,1], Fs = 128, signaltype='eeg', epoch_time=2, epoch_step_time=1, bandwidth=2, fmin=0, fmax=20)
		specs_central = (specs_c3 + specs_c4)/2

		specs_f3,freq, signal_segs = safe_spectrogram(signal[:,2], Fs = 128, signaltype='eeg', epoch_time=2, epoch_step_time=1, bandwidth=2, fmin=0, fmax=20)
		specs_f4,freq, signal_segs = safe_spectrogram(signal[:,3], Fs = 128, signaltype='eeg', epoch_time=2, epoch_step_time=1, bandwidth=2, fmin=0, fmax=20)
		specs_frontal = (specs_f3 + specs_f4)/2

		specs_o1,freq, signal_segs = safe_spectrogram(signal[:,4], Fs = 128, signaltype='eeg', epoch_time=2, epoch_step_time=1, bandwidth=2, fmin=0, fmax=20)
		specs_o2,freq, signal_segs = safe_spectrogram(signal[:,5], Fs = 128, signaltype='eeg', epoch_time=2, epoch_step_time=1, bandwidth=2, fmin=0, fmax=20)
		specs_occipital = (specs_o1 + specs_o2)/2

			            
		# Time axis in hours (30s epochs)
		x = np.arange(specs_central.shape[0]) / 3600  # shape: (epochs,)

		# Convert to 2D arrays for plotting (freq x time)
		spec_f = specs_frontal.squeeze().T  # shape: (freq, time)
		spec_c = specs_central.squeeze().T
		spec_o = specs_occipital.squeeze().T
		# Create figure with 5 stacked axes
		fig, axs = plt.subplots(5, 1, figsize=(10, 8), 
		                        gridspec_kw={'height_ratios': [1, 1, 1, 0.1, 0.1]}, 
		                        sharex=True)

		# Plot frontal
		im = axs[0].imshow(spec_f, cmap='turbo', origin='lower', aspect='auto',
		              extent=(0, x[-1], freq.min(), freq.max()), vmin=0, vmax=20)
		axs[0].set_ylim([0,20])
		axs[0].set_ylabel('Frontal')

		# Plot central
		axs[1].imshow(spec_c, cmap='turbo', origin='lower', aspect='auto',
		              extent=(0, x[-1], freq.min(), freq.max()), vmin=0, vmax=20)
		axs[1].set_ylim([0,20])
		axs[1].set_ylabel('Central')

		# Plot occipital
		axs[2].imshow(spec_o, cmap='turbo', origin='lower', aspect='auto',
		              extent=(0, x[-1], freq.min(), freq.max()), vmin=0, vmax=20)
		axs[2].set_ylim([0,20])
		axs[2].set_ylabel('Occipital')
		                
		# Stage color mapping
		STAGE_COLOR = {
		    'WAKE': 'gold',
		    'N1': 'lightblue',
		    'N2': 'blue',
		    'N3': 'darkblue',
		    'REM': 'purple'
		}
		def stage_to_rgb(stage_list):
		    return [to_rgb(STAGE_COLOR.get(stage, 'white')) for stage in stage_list]

		# Convert GT and Pred to RGB
		gt_rgb = np.array([stage_to_rgb(ground_truth_labels)])  # shape: (1, time, 3)
		pred_rgb = np.array([stage_to_rgb(predicted_labels)])

		# Plot ground truth colorbar
		axs[3].imshow(gt_rgb, aspect='auto', extent=(0, x[-1], 0, 1))
		axs[3].set_yticks([])
		axs[3].set_ylabel('Tech' ,rotation=0, labelpad=30, y=0.2)

		# Plot prediction colorbar
		axs[4].imshow(pred_rgb, aspect='auto', extent=(0, x[-1], 0, 1))
		axs[4].set_yticks([])
		axs[4].set_ylabel('pedi-SS',rotation=0, labelpad=30, y=0.2)

		# Title

		age_years = age_days / 365.25
		axs[0].set_title(f"{png_fi} | Age: {age_days} days ({age_years:.2f} yrs) | Kappa: {kappa:.4f}", fontsize=10)

		# X-axis label
		axs[-1].set_xlabel("Time (hours)")

		# Define sleep stage patches
		stage_legend_handles = [
		    Patch(color='gold', label='WAKE'),
		    Patch(color='lightblue', label='N1'),
		    Patch(color='blue', label='N2'),
		    Patch(color='darkblue', label='N3'),
		    Patch(color='purple', label='REM')
		]

		# 1. Manually place sleep stage legend (left side)
		# [left, bottom, width, height] ? width/height don't matter for legend
		legend_ax = fig.add_axes([0.02, 0.93, 0.6, 0.05])
		legend_ax.axis('off')  # hide the box
		legend_ax.legend(handles=stage_legend_handles,
		                 loc='center left', ncol=len(stage_legend_handles),
		                 frameon=False, handlelength=1.5, handletextpad=0.5)

		# 2. Place spectrogram colorbar (right side)
		cbar_ax = fig.add_axes([0.72, 0.93, 0.25, 0.015])
		cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
		cbar.ax.tick_params(labelsize=8)

		# 3. Adjust layout to make room at the top
		plt.tight_layout(rect=[0, 0, 1, 0.91])


		# Final layout and save
		plt.savefig(os.path.join(output_dir, png_fi))
		plt.close()
    
 

   
root_dir = 'BCH_h5_arranged'
df_test = pd.read_csv('/media/ayush/Ayush/Documents/BCH_data_analysis/USleep_expt/CommonIDs/test_set.csv')
subID_test = df_test['subID'].to_numpy()
sess_test = df_test['Session'].to_numpy()
ageindays = df_test['AgeDays'].to_numpy()
test_files = []

for i in range(0,len(subID_test)):
    subID_sess_test = subID_test[i] + '_ses-' + str(sess_test[i])
    for fo in os.listdir(root_dir):
        if fo.startswith(subID_sess_test):
            test_files.append(fo)

print(len(test_files))

'''
for i in range(0,len(test_files)):
	print(i)
	save_spect_img(test_files[i],ageindays[i])
'''
	


def process_file(args):
    file, age = args
    try:
        save_spect_img(file, age)
    except Exception as e:
        print(f"Error processing {file}: {e}")

# Prepare arguments as tuples
args_list = list(zip(test_files, ageindays))

# Run with a pool of 20 processes
with ProcessPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(process_file, arg) for arg in args_list]

    for i, future in enumerate(as_completed(futures), 1):
        try:
            future.result()  # will raise exception if one occurred
        except Exception as e:
            print(f"Exception in task {i}: {e}")
        else:
            print(f"Completed task {i}/{len(futures)}")

