import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import os 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.stats import wilcoxon
from scipy.stats import ttest_rel
import warnings
warnings.filterwarnings("ignore")
import matplotlib.ticker as ticker
from scipy import stats


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
        return np.nan, np.nan

    filtered_ground_truth = [ground_truth_labels[i] for i in valid_indices]
    filtered_predictions = [predicted_labels[i] for i in valid_indices]

    # Convert ground truth and predicted labels to integers
    ground_truth_classes = [ANNOTATION_MAPPING[label] for label in filtered_ground_truth]
    predicted_classes = [ANNOTATION_MAPPING[label] for label in filtered_predictions]


    # Compute accuracy and cohen kappa for 5 classes
    accuracy = accuracy_score(ground_truth_classes, predicted_classes)
    kappa = cohen_kappa_score(ground_truth_classes, predicted_classes)

    return accuracy, kappa

def convert_tsv_to_epoch_array(file_path):
    # Load the TSV file
    data = pd.read_csv(file_path, sep='\t')

    data['stage'] = data['stage'].replace('Wake', 'WAKE')
    
    # Determine the total duration in seconds based on the last entry
    total_duration = int(data['init_sec'].iloc[-1] + data['duration_sec'].iloc[-1])
    
    # Create an array for 30-second epochs
    num_epochs = total_duration // 30
    annotations_array = np.full(num_epochs, 'Unknown', dtype=object)
    
    # Fill the array with the appropriate sleep stages
    for _, row in data.iterrows():
        start_epoch = int(row['init_sec'] // 30)
        end_epoch = int((row['init_sec'] + row['duration_sec']) // 30)
        annotations_array[start_epoch:end_epoch] = row['stage']
    
    # Print the number of 'Unknown' entries
    #unknown_count = np.sum(annotations_array == 'Unknown')
    #print(f"Number of 'Unknown' in the array: {unknown_count}")
    
    return annotations_array

def calculate_accuracy(predicted_array, ground_truth_array, exclude_values=['UNSCORED', 'UNSCORABLE', 'LEG MOVEMENT']):
    # Ensure the arrays have the same length
    if len(predicted_array) != len(ground_truth_array):
        raise ValueError("The predicted and ground truth arrays must have the same length.")
    
    # Create a mask to exclude certain values from the ground truth
    valid_mask = ~np.isin(ground_truth_array, exclude_values)
    
    # Filter the predicted and ground truth arrays based on the valid mask
    filtered_predicted = predicted_array[valid_mask]
    filtered_ground_truth = ground_truth_array[valid_mask]

    #print(filtered_ground_truth.shape)
    
    # Calculate the accuracy

    accuracy = accuracy_score(filtered_ground_truth,filtered_predicted)
    kappa = cohen_kappa_score(filtered_ground_truth, filtered_predicted)

    
    #print(f"Accuracy excluding specified values: {accuracy:.2%}")
    return accuracy, kappa
    

def evaluate_predictions_CAISR(h5_file):
    VALID_CLASSES = {"N1", "N2", "N3", "REM", "WAKE"}
    ANNOTATION_MAPPING_CAISR = {"N3": 1.0, "N2": 2.0, "N1": 3.0, "REM": 4.0, "WAKE": 5.0}

    # Load ground truth
    gt_file = h5_file.replace(".h5", "_ground.csv")
    df_gt = pd.read_csv(os.path.join('predictions_normalized',gt_file))
    ground_truth_labels = df_gt['Ground Stage'].tolist()


    # Load predicted labels
    df_pred = pd.read_csv(os.path.join('/media/ayush/Ayush/BCH_SleepStaging/CAISR_docker_BCH_test/caisr_output/intermediate/stage',os.path.basename(h5_file.replace(".h5", "_stage.csv"))))
    selected_stages = df_pred['stage'].iloc[::30].to_numpy()
    predicted_labels = [] 
    for i in range(0,len(selected_stages)):
        if selected_stages[i] == 1:
            predicted_labels.append('N3')
        elif selected_stages[i] == 2:
            predicted_labels.append('N2')
        elif selected_stages[i] == 3:
            predicted_labels.append('N1')
        elif selected_stages[i] == 4:
            predicted_labels.append('REM')
        elif selected_stages[i] == 5:
            predicted_labels.append('WAKE')
        else:
            predicted_labels.append('UNSCORED')
            
    if len(predicted_labels) != len(ground_truth_labels):
        print( os.path.join('/media/ayush/Ayush/BCH_SleepStaging/CAISR_docker_BCH_test/caisr_output/intermediate/stage',os.path.basename(h5_file.replace(".h5", "_stage.csv"))) )


    # Filter both for valid ground truth and prediction
    final_gt = []
    final_pred = []

    for gt, pred in zip(ground_truth_labels, predicted_labels):
        if gt in VALID_CLASSES and pred != 'UNSCORED':
            final_gt.append(gt)
            final_pred.append(pred)

    if not final_gt:
        return np.nan, np.nan

    accuracy = accuracy_score(final_gt, final_pred)
    kappa = cohen_kappa_score(final_gt, final_pred)

    return accuracy, kappa

def bootstrap_ci(data, n_bootstrap=1000, ci=95):
    if len(data) < 2:
        return np.nan
    medians = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        medians.append(np.median(sample))
    lower = np.percentile(medians, (100 - ci) / 2)
    upper = np.percentile(medians, 100 - (100 - ci) / 2)
    return (upper - lower) / 2  # symmetric CI margin


def plot_age_spline(accuracy_vector, age_vector, pred_model, metric):
	accuracy_vector = np.array(accuracy_vector)
	age_vector = np.array(age_vector)

	accuracy_clean = accuracy_vector[~np.isnan(accuracy_vector)]
	age_clean = age_vector[~np.isnan(accuracy_vector)]

	age_years = age_clean / 365.25
	max_age = 18
	interval = 0.5

	mask = age_years <= max_age
	age_filtered = age_years[mask]
	accuracy_filtered = accuracy_clean[mask]

	# Define age bins and compute the mean age and mean accuracy in each bin
	age_bins = np.arange(0, max(age_filtered) + interval, interval)
	bin_indices = np.digitize(age_filtered, age_bins)
 
	mean_age = []
	mean_accuracy = []
	median_accuracy = []
	ci_band = []
	q25_arr = []
	q75_arr = []

	for i in range(1, len(age_bins)):
		bin_ages = age_filtered[bin_indices == i]
		bin_accs = accuracy_filtered[bin_indices == i]
		if len(bin_accs) > 0:
			mean_age.append(np.median(bin_ages))
			median_accuracy.append(np.median(bin_accs))
			mean_accuracy.append(np.mean(bin_accs))
			q25_arr.append(np.percentile(np.mean(bin_accs),25))
			q75_arr.append(np.percentile(np.mean(bin_accs),75))
			
			'''
			if len(bin_accs) > 1:
				sem = np.std(bin_accs, ddof=1) / np.sqrt(len(bin_accs))
				ci = 1.96 * sem  # 95% CI
				ci_band.append(ci)
			else:
				ci_band.append(np.nan)
			'''
			if len(bin_accs) > 1:
				ci_band.append(bootstrap_ci(bin_accs))
				print(f"Age bin: {age_bins[i-1]:.1f}-{age_bins[i]:.1f} yrs, n={len(bin_accs)}, CI={ci_band[-1]:.3f}")
			else:
				ci_band.append(np.nan)
      
      
			
  
	mean_age = np.array(mean_age)
	median_accuracy = np.array(median_accuracy)
	mean_accuracy = np.array(mean_accuracy)
	ci_band = np.array(ci_band)
	q25_arr = np.array(q25_arr)
	q75_arr = np.array(q75_arr)


	# Remove NaNs for spline
	valid = ~np.isnan(median_accuracy)
	mean_age = mean_age[valid]
	median_accuracy = median_accuracy[valid]
	mean_accuracy = mean_accuracy[valid]
	ci_band = ci_band[valid]
	q25_arr = q25_arr[valid]
	q75_arr = q75_arr[valid]


	# Fit a smoothing spline
	smoothing_param = len(mean_age) * 0.001
	spline = UnivariateSpline(mean_age, median_accuracy, s=smoothing_param)
	smooth_x = np.linspace(0, max_age, 300)
	smooth_y = spline(smooth_x)
 
	ci_interp = UnivariateSpline(mean_age, ci_band, s=smoothing_param)
	ci_smooth = ci_interp(smooth_x)



	# Plotting
	fig, ax = plt.subplots(figsize=(10, 6))

	# Colors
	colors = {
	  'scatter': '#0072B2',
	  'spline': '#D55E00',
	  'ci_fill': '#0072B2'
	}

	# Raw data scatter
	ax.scatter(age_filtered, accuracy_filtered, color=colors['scatter'], alpha=0.2, s=10, label='Raw')

	# Spline curve
	ax.plot(smooth_x, smooth_y, color=colors['spline'], linewidth=3, label='Median')

	# CI shading
	#ax.fill_between(mean_age,mean_accuracy - ci_band,mean_accuracy + ci_band,color=colors['ci_fill'], alpha=0.4,label='95% CI')
	#ax.fill_between(mean_age,q25_arr,q75_arr,color=colors['ci_fill'], alpha=0.4,label='IQR')
	ax.fill_between(smooth_x, spline(smooth_x) - ci_smooth, spline(smooth_x) + ci_smooth, color=colors['ci_fill'], alpha=0.4, label='95% CI')

	# Axes and title
	ax.set_xlim(-1, 20)
	ax.set_ylim(0, 1)
	ax.set_xlabel('Age (years)', fontsize=24)
	if metric == 'Kappa':
		ax.set_ylabel("Cohen's Kappa", fontsize=24)
	else:
		ax.set_ylabel('Accuracy', fontsize=24)

	ax.set_title(f"Cohen's Kappa Across Age", fontsize=28)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)

	ax.tick_params(labelsize=14)
	#ax.grid(True, linestyle='--', alpha=0.5)
	ax.legend(fontsize=8, loc='upper left', frameon=True, framealpha=0.8, facecolor='white')

	plt.tight_layout()
	plt.savefig(f'results_normalized/{metric}_{pred_model}.png', dpi=600)
	plt.close()

	return smooth_x, smooth_y


root_dir = 'BCH_h5_arranged'
df_test = pd.read_csv('/media/ayush/Ayush/Documents/BCH_data_analysis/USleep_expt/CommonIDs/test_set.csv')
U_sleepfolder = '/media/ayush/Ayush/Documents/BCH_data_analysis/USleep_expt/USleep_output'
subID_test = df_test['subID'].to_numpy()
sess_test = df_test['Session'].to_numpy()
ageindays = df_test['AgeDays'].to_numpy()

test_files = []
kappa_USleep_vector = []
kappa_our_vector = []
kappa_CAISR_vector = []
accuracy_USleep_vector = []
accuracy_our_vector = []
accuracy_CAISR_vector = []
age_vector = []


for i in range(0,len(subID_test)):
    subID_sess_test = subID_test[i] + '_ses-' + str(sess_test[i])
    for fo in os.listdir(root_dir):
        if fo.startswith(subID_sess_test):
            test_files.append(fo)

print(len(test_files))

for i in range(0,len(test_files)):
    print(str(i+1) + '       ' + test_files[i])
    age_vector.append(ageindays[i])

    accuracy,kappa = evaluate_predictions(test_files[i])
    kappa_our_vector.append(kappa)
    accuracy_our_vector.append(accuracy)

    accuracy_CAISR,kappa_CAISR = evaluate_predictions_CAISR(test_files[i])
    kappa_CAISR_vector.append(kappa_CAISR)
    accuracy_CAISR_vector.append(accuracy_CAISR)

    h5_file = test_files[i]
    tsv_file = os.path.join(U_sleepfolder,h5_file.replace(".h5", ".tsv"))     
    predicted_array = convert_tsv_to_epoch_array(tsv_file)
    gt_file = h5_file.replace(".h5", "_ground.csv")
    df_gt = pd.read_csv(os.path.join('predictions_normalized',gt_file))
    ground_truth_labels = df_gt['Ground Stage'].to_numpy()
    accuracy, kappa = calculate_accuracy(predicted_array, ground_truth_labels)
    accuracy_USleep_vector.append(accuracy)
    kappa_USleep_vector.append(kappa)


            




print(len(age_vector))
print(len(accuracy_our_vector))
print(len(kappa_our_vector))
print(len(accuracy_USleep_vector))
print(len(kappa_USleep_vector))
print(len(accuracy_CAISR_vector))
print(len(kappa_CAISR_vector))

print(f"Mean Accuracy USleep: {np.nanmean(accuracy_USleep_vector):.4f}")
print(f"Mean Cohen's Kappa USleep: {np.nanmean(kappa_USleep_vector):.4f}")

print(f"Mean Accuracy CAISR: {np.nanmean(accuracy_CAISR_vector):.4f}")
print(f"Mean Cohen's Kappa CAISR: {np.nanmean(kappa_CAISR_vector):.4f}")

print(f"Mean Accuracy Our: {np.nanmean(accuracy_our_vector):.4f}")
print(f"Mean Cohen's Kappa Our: {np.nanmean(kappa_our_vector):.4f}")


age_acc_USleep, acc_USleep = plot_age_spline(accuracy_USleep_vector, age_vector, 'Pre-trained_USleep', 'Accuracy')
age_kappa_USleep, kappa_USleep = plot_age_spline(kappa_USleep_vector, age_vector, 'Pre-trained_USleep', 'Kappa')
age_acc_our, acc_our = plot_age_spline(accuracy_our_vector, age_vector, 'pedi-SS', 'Accuracy')
age_kappa_our, kappa_our = plot_age_spline(kappa_our_vector, age_vector, 'pedi-SS', 'Kappa')
age_acc_CAISR, acc_CAISR = plot_age_spline(accuracy_CAISR_vector, age_vector, 'Pre-trained_CAISR', 'Accuracy')
age_kappa_CAISR, kappa_CAISR = plot_age_spline(kappa_CAISR_vector, age_vector, 'Pre-trained_CAISR', 'Kappa')



age_bins = [
    (0, 180, '0-6 months'), 
    (181, 365, '6-12 months'),
]

for y in range(1, 18):
    start = y * 365 + 1
    end = (y + 1) * 365
    label = f'{y}-{y+1} years'
    age_bins.append((start, end, label))
    
age_bins.append((18*365+1, np.inf, 'over 18 years'))
    
results = []


def ci95(data, n_bootstrap=1000, ci=95):
    data = np.array(data)
    data = data[~np.isnan(data)]
    n = len(data)
    if n < 2:
        return np.nan
    stats_boot = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        stats_boot.append(np.mean(sample))  
    lower = np.percentile(stats_boot, (100 - ci) / 2)
    upper = np.percentile(stats_boot, 100 - (100 - ci) / 2)
    return (upper - lower) / 2  # CI margin
    
 
            
for start,end,title in age_bins:

    kappa_vector_our = []
    kappa_vector_USleep = []
    kappa_vector_CAISR = []

    for i in range(0,len(test_files)):
        if ageindays[i]>=start and ageindays[i]<=end:
            accuracy,kappa = evaluate_predictions(test_files[i])
            kappa_vector_our.append(kappa)

            accuracy_CAISR,kappa_CAISR = evaluate_predictions_CAISR(test_files[i])
            kappa_vector_CAISR.append(kappa_CAISR)

            h5_file = test_files[i]
            tsv_file = os.path.join(U_sleepfolder,h5_file.replace(".h5", ".tsv"))     
            predicted_array = convert_tsv_to_epoch_array(tsv_file)
            gt_file = h5_file.replace(".h5", "_ground.csv")
            df_gt = pd.read_csv(os.path.join('predictions_normalized',gt_file))
            ground_truth_labels = df_gt['Ground Stage'].to_numpy()
            accuracy, kappa = calculate_accuracy(predicted_array, ground_truth_labels)
            kappa_vector_USleep.append(kappa)
                    
    # Convert to numpy arrays if not already
    kappa_vector_our = np.array(kappa_vector_our)
    kappa_vector_USleep = np.array(kappa_vector_USleep)
    kappa_vector_CAISR = np.array(kappa_vector_CAISR)
    
    # Get valid indices where both are not nan
    valid_indices = ~np.isnan(kappa_vector_our) & ~np.isnan(kappa_vector_USleep) & ~np.isnan(kappa_vector_CAISR)
    
    # Filter both vectors
    kappa_our_clean = kappa_vector_our[valid_indices]
    kappa_USleep_clean = kappa_vector_USleep[valid_indices]
    kappa_CAISR_clean = kappa_vector_CAISR[valid_indices]
    
    if len(kappa_our_clean) > 1:
        t_stat, p_ttest = ttest_rel(kappa_our_clean, kappa_USleep_clean)
        
        t_stat_CAISR, p_ttest_CAISR = ttest_rel(kappa_our_clean, kappa_CAISR_clean)
        
    else:
        p_ttest = np.nan
        p_ttest_CAISR = np.nan
        
    print(title)
    print(p_ttest)
    print(p_ttest_CAISR)

    # Save results
    results.append({
        'Age Range': title,
        '#PSGs' : len(kappa_our_clean),
        'Mean Kappa (Our)': np.nanmean(kappa_our_clean),
        'Mean Kappa (U-Sleep)': np.nanmean(kappa_USleep_clean),
        'Mean Kappa (CAISR)': np.nanmean(kappa_CAISR_clean),
        'p-value (t-test) our-USleep': p_ttest,
        'p-value (t-test) our-CAISR': p_ttest_CAISR,
        'Std Kappa (Our)': np.nanstd(kappa_our_clean)/np.sqrt(len(kappa_our_clean)),
        'Std Kappa (U-Sleep)': np.nanstd(kappa_USleep_clean)/np.sqrt(len(kappa_USleep_clean)),
        'Std Kappa (CAISR)': np.nanstd(kappa_CAISR_clean)/np.sqrt(len(kappa_CAISR_clean)),
        'CI Kappa (Our)': ci95(kappa_our_clean),
        'CI Kappa (U-Sleep)': ci95(kappa_USleep_clean),
        'CI Kappa (CAISR)': ci95(kappa_CAISR_clean),
    })

# Convert to DataFrame and save as CSV
results_df = pd.DataFrame(results)
results_df.to_csv('results_normalized/kappa_comparison_by_age.csv', index=False)

# Extract relevant columns
age_ranges = results_df['Age Range']
x = range(len(age_ranges))

kappa_our = results_df['Mean Kappa (Our)']
kappa_usleep = results_df['Mean Kappa (U-Sleep)']
kappa_caisr = results_df['Mean Kappa (CAISR)']

#Error bars (std dev)
#std_our = results_df['Std Kappa (Our)']
#std_usleep = results_df['Std Kappa (U-Sleep)']
#std_caisr = results_df['Std Kappa (CAISR)']

#Error bars 95% CI
std_our = results_df['CI Kappa (Our)']
std_usleep = results_df['CI Kappa (U-Sleep)']
std_caisr = results_df['CI Kappa (CAISR)']


# Colors from Color Universal Design palette (colorblind-safe)
colors = {
    'pedi-SS': '#0072B2',
    'USleep': '#D55E00',
    'CAISR': '#009E73'
}

# Create the plot
fig, ax = plt.subplots(figsize=(14, 6))

# Plot with error bars
ax.errorbar(x, kappa_our, yerr=std_our, label='pedi-SS', color=colors['pedi-SS'],
            fmt='-o', capsize=3, linewidth=2)
ax.errorbar(x, kappa_usleep, yerr=std_usleep, label='USleep', color=colors['USleep'],
            fmt='-s', capsize=3, linewidth=2)
ax.errorbar(x, kappa_caisr, yerr=std_caisr, label='CAISR', color=colors['CAISR'],
            fmt='-^', capsize=3, linewidth=2)

# Axes formatting
ax.set_title("Mean Cohen’s Kappa Across Age Groups", fontsize=30, pad=15)
ax.set_xlabel("Age Group", fontsize=25)
ax.set_ylabel("Mean Cohen’s Kappa", fontsize=25)
ax.set_xticks(x)
ax.set_xticklabels(age_ranges, rotation=45, ha='right', fontsize=20)
ax.set_ylim(0.25, 0.8)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Grid and legend
#ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(loc='upper left', fontsize=12, frameon=False)

# Layout and save
plt.tight_layout()
plt.savefig("results_normalized/kappa_by_age_final_pubready.png", dpi=600)


'''
# Extract relevant columns
age_ranges = results_df['Age Range']
kappa_our = results_df['Mean Kappa (Our)']
kappa_usleep = results_df['Mean Kappa (U-Sleep)']
kappa_caisr = results_df['Mean Kappa (CAISR)']

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(age_ranges, kappa_our, marker='o', color='blue', label='pedi-SS')
plt.plot(age_ranges, kappa_usleep, marker='s', color='red', label='USleep')
plt.plot(age_ranges, kappa_caisr, marker='^', color='green', label='CAISR')

# Customize the plot
plt.title("Mean Cohen's Kappa by Age Group", fontsize=16)
plt.xlabel("Age Range", fontsize=14)
plt.ylabel("Mean Kappa", fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()

# Save and/or show the plot
plt.savefig("results_normalized/kappa_by_age_lineplot_colored.png", dpi=300)  # Save as PNG
'''

 
