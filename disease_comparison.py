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
import mne
from mne.filter import filter_data, notch_filter
from mne.time_frequency import psd_array_multitaper
from scipy.signal import detrend
from collections import Counter
from scipy.interpolate import interp2d
import h5py
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
from scipy import stats

def ci95(data, n_bootstrap=1000, ci=95):
    data = np.array(data)
    data = data[~np.isnan(data)]
    n = len(data)
    if n < 2:
        return np.nan
    stats_boot = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        stats_boot.append(np.mean(sample))  # Use np.median(sample) if you're plotting medians
    lower = np.percentile(stats_boot, (100 - ci) / 2)
    upper = np.percentile(stats_boot, 100 - (100 - ci) / 2)
    return (upper - lower) / 2  # CI margin
    
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
        return np.nan, np.nan, np.array([]), np.array([])

    filtered_ground_truth = [ground_truth_labels[i] for i in valid_indices]
    filtered_predictions = [predicted_labels[i] for i in valid_indices]

    # Convert ground truth and predicted labels to integers
    ground_truth_classes = [ANNOTATION_MAPPING[label] for label in filtered_ground_truth]
    predicted_classes = [ANNOTATION_MAPPING[label] for label in filtered_predictions]


    # Compute accuracy and cohen kappa for 5 classes
    accuracy = accuracy_score(ground_truth_classes, predicted_classes)
    kappa = cohen_kappa_score(ground_truth_classes, predicted_classes)

    return accuracy, kappa, ground_truth_labels, predicted_labels
    
    
def get_kappa_specific(df_test,min_age,max_age,specific_icd):

    subID_test = df_test['subID'].to_numpy()
    sess_test = df_test['Session'].to_numpy()
    ageindays = df_test['AgeDays'].to_numpy()

    mapping_df = pd.read_csv('/media/ayush/Ayush/Documents/BCH_data_analysis/Disease/Mapping_list.csv')
    diagnosis_folder = '/media/ayush/Ayush/Documents/BCH_dataset/Disease_info'
    root_dir = 'BCH_h5_arranged'
    
    test_files_disease = []
    kappa_vector_disease = []
    test_files_nondisease = []
    kappa_vector_nondisease = []
    
    for i in range(0,len(subID_test)):
    
        if ageindays[i]>=min_age and ageindays[i]<=max_age:
    
            diagnosis_df = pd.read_csv(os.path.join(diagnosis_folder, subID_test[i]+'_diagnosis.csv'))
            if diagnosis_df['Diagnosis'].dropna().empty:
            	found_icd = False
            else:
            
	            found_icd = diagnosis_df['Diagnosis'].str.contains(specific_icd, na=False).any()
	        
	            if not found_icd:
	                for term in diagnosis_df['Diagnosis'].dropna():
	                    mapped_icd = mapping_df.loc[mapping_df['Term'].str.contains(term, na=False, case=False), 'ICD_clean']
	                    if not mapped_icd.empty and specific_icd in mapped_icd.values:
	                        found_icd = True
	                        break
    
            if found_icd:
                subID_sess_test = subID_test[i] + '_ses-' + str(sess_test[i])
                for fo in os.listdir(root_dir):
                    if fo.startswith(subID_sess_test):
                        test_files_disease.append(fo)
                        
            else:
                subID_sess_test = subID_test[i] + '_ses-' + str(sess_test[i])
                for fo in os.listdir(root_dir):
                    if fo.startswith(subID_sess_test):
                        test_files_nondisease.append(fo)
                
            
    
                
    for i in range(0,len(test_files_disease)):
        accuracy, kappa, ground_truth_labels, predicted_labels = evaluate_predictions(test_files_disease[i])
        kappa_vector_disease.append(kappa)

                
    for i in range(0,len(test_files_nondisease)):
        accuracy, kappa, ground_truth_labels, predicted_labels = evaluate_predictions(test_files_nondisease[i])
        kappa_vector_nondisease.append(kappa)

    print('Disease files: ' + str(len([k for k in kappa_vector_disease if not np.isnan(k)])) + ' Non-disease files: ' + str(len([k for k in kappa_vector_nondisease if not np.isnan(k)])) )
    
    if kappa_vector_disease:
        mean_kappa = np.nanmean(kappa_vector_disease)
        ci95_kappa = ci95(kappa_vector_disease)
        
        mean_kappa_nondisease = np.nanmean(kappa_vector_nondisease)
        ci95_kappa_nondisease = ci95(kappa_vector_nondisease)
        
        stat, p_val = ttest_ind([k for k in kappa_vector_disease if not np.isnan(k)], [k for k in kappa_vector_nondisease if not np.isnan(k)], equal_var=False)
        #u_stat, p_val = mannwhitneyu([k for k in kappa_vector_disease if not np.isnan(k)], [k for k in kappa_vector_nondisease if not np.isnan(k)], alternative='two-sided')
        print('Disease kappa: ' + str(mean_kappa) + ' Non-disease kappa: ' + str(np.nanmean(kappa_vector_nondisease)) )
        print('Disease kappa CI: ' + str(ci95_kappa) + ' Non-disease kappa CI: ' + str(ci95_kappa_nondisease) )
        #print(f"Mann–Whitney U statistic = {u_stat}, p-value = {p_val}")
        print(f"Welch t-test statistic = {stat}, p-value = {p_val}")
    else:
        mean_kappa = np.nan
        p_val = np.nan
        ci95_kappa = np.nan
        ci95_kappa_nondisease = np.nan
       
    
    
                
    return mean_kappa, ci95_kappa, mean_kappa_nondisease, ci95_kappa_nondisease, len([k for k in kappa_vector_disease if not np.isnan(k)]), p_val
    
    

dataframe_test = pd.read_csv('/media/ayush/Ayush/Documents/BCH_data_analysis/USleep_expt/CommonIDs/test_set.csv')


age_bins = [
    (0, 180, '0-6 months'),
    (181, 365, '6-12 months'),
    (366, 2*365, '1-2 years'),
    (2*365+1, 3*365, '2-3 years'),
    (3*365+1, 4*365, '3-4 years'),
    (4*365+1, 5*365, '4-5 years'),
    (5*365+1, 6*365, '5-6 years'),
    (6*365+1, 12*365, '6-12 years'),
    (12*365+1, np.inf, 'over 12 years'),
    (0, np.inf, 'Test Set')
]



icd_codes = {
    'G40': 'Epilepsy',
    'Q07': 'Other congenital malformations of the nervous system',
    'G91': 'Hydrocephalus',
    'J45': 'Asthma',
    'Q90': "Down's syndrome",
    'F84': 'Pervasive developmental disorders',
    'F90': 'Hyperkinetic disorders',
}

results = []

for min_age, max_age, age_label in age_bins:
    for icd_code, icd_label in icd_codes.items():
        print(f"\nRunning for Age: {age_label}, ICD-10: {icd_code} ({icd_label})")
        
        mean_kappa, ci95_kappa, mean_kappa_nondisease, ci95_kappa_nondisease, num_subjects, p_value = get_kappa_specific(dataframe_test, min_age, max_age, icd_code)
        
        results.append({
            'Age Group': age_label,
            'ICD10 Code': icd_code,
            'ICD10 Description': icd_label,
            'Mean Kappa': mean_kappa,
            'CI Kappa': ci95_kappa,
            'Mean Kappa Non Disease': mean_kappa_nondisease,
            'CI Kappa Non Disease': ci95_kappa_nondisease,
            'Number of PSGs': num_subjects,
            'p-value': p_value
        })


df_results = pd.DataFrame(results)

df_results.to_csv('results_normalized/kappa_by_age_icd10.csv', index=False)

