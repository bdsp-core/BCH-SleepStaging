import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import os 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.stats import ttest_rel,ttest_ind
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
    
'''
def ci95(data):
    data = np.array(data)
    data = data[~np.isnan(data)]
    n = len(data)
    if n < 2:
        return np.nan
    sem = np.std(data, ddof=1) / np.sqrt(n)
    return sem * stats.t.ppf(0.975, df=n-1)  # 0.975 for 95% CI two-tailed
'''

def get_test_files(minage, maxage, whichsex=None):
    test_files = []
    root_dir = 'BCH_h5_arranged'
    df_test = pd.read_csv('/media/ayush/Ayush/Documents/BCH_data_analysis/USleep_expt/CommonIDs/test_set.csv')
    subID_test = df_test['subID'].to_numpy()
    sess_test = df_test['Session'].to_numpy()
    agedays_test = df_test['AgeDays'].to_numpy()
    sex_test = df_test['Sex']

    for i in range(len(subID_test)):
        subID_sess_test = f"{subID_test[i]}_ses-{sess_test[i]}"
        for fo in os.listdir(root_dir):
        	if fo.startswith(subID_sess_test) and minage <= agedays_test[i] <= maxage and sex_test[i] == whichsex:
        		test_files.append(fo)
                
        
    return np.array(test_files)

def get_metrics(test_files_metrics, num_classes=5):
	VALID_CLASSES = {"N1", "N2", "N3", "REM", "WAKE"}
	ANNOTATION_MAPPING = {"N1": 1, "N2": 2, "N3": 3, "REM": 4, "WAKE": 0}
	root_dir = 'predictions_normalized'
	kappa_vector = []

	for j in range(0,len(test_files_metrics)):
		h5_file = test_files_metrics[j]

		gt_file = h5_file.replace(".h5", "_ground.csv")
		df_gt = pd.read_csv(os.path.join('predictions_normalized',gt_file))
		ground_truth_labels = df_gt['Ground Stage'].tolist()

		pred_file = h5_file.replace(".h5", "_pred.csv")
		df_pred = pd.read_csv(os.path.join('predictions_normalized',pred_file))
		predicted_labels = df_pred['Predicted Stage'].tolist()
		

		valid_indices = [i for i, stage in enumerate(ground_truth_labels) if stage in VALID_CLASSES]

		filtered_ground_truth = [ground_truth_labels[k] for k in valid_indices]
		filtered_predictions = [predicted_labels[k] for k in valid_indices]

		ground_truth_classes = [ANNOTATION_MAPPING[label] for label in filtered_ground_truth]
		predicted_classes = [ANNOTATION_MAPPING[label] for label in filtered_predictions]

		accuracy = accuracy_score(ground_truth_classes, predicted_classes)
		kappa = cohen_kappa_score(ground_truth_classes, predicted_classes)
		#print(kappa)
		kappa_vector.append(kappa)
		
	return np.nanmean(kappa_vector), kappa_vector



# Define the list of age range, sex, and titles
age_combinations = [
    (0, 180, "0-6 months"),
    (181, 365, "6-12 months"),
    (366, 730, "1-2 years"),
    (731, 1095, "2-3 years"),
    (1096, 1460, "3-4 years"),
    (1461, 1825, "4-5 years"),
    (1826, 2190, "5-6 years"),
    (2191, 4380, "6-12 years"),
    (4381, np.inf, "over 12 years"),
    (0, np.inf, "Test Set"),
]


results = []

for minage, maxage, title in age_combinations:
    

    test_files_male = get_test_files(minage, maxage, 'M')
    kappa_male, kappa_vector_male = get_metrics(test_files_male, num_classes=5)    
    valid_indices_male = ~np.isnan(kappa_vector_male)
    kappa_vector_male_clean = np.array(kappa_vector_male)[valid_indices_male]
    ci95_male = ci95(kappa_vector_male_clean)
    
    test_files_female = get_test_files(minage, maxage, 'F')
    kappa_female, kappa_vector_female = get_metrics(test_files_female, num_classes=5)
    valid_indices_female = ~np.isnan(kappa_vector_female)
    kappa_vector_female_clean = np.array(kappa_vector_female)[valid_indices_female]
    ci95_female = ci95(kappa_vector_female_clean)
    
    t_stat, p_ttest = ttest_ind(kappa_vector_male_clean, kappa_vector_female_clean, equal_var=False)


    results.append([title,  kappa_male, len(kappa_vector_male_clean), ci95_male, kappa_female, len(kappa_vector_female_clean), ci95_female, p_ttest])
    


df_results = pd.DataFrame(results, columns=["Age Range", "Kappa Male", "#Male", "CI Male", "Kappa Female", "#Female", "CI Female", "p-value"])


csv_filename = "results_normalized/age_sex_metrics_results.csv"
df_results.to_csv(csv_filename, index=False)


df_results = pd.read_csv("results_normalized/age_sex_metrics_results.csv")

# Plot settings for high-quality grayscale figure
plt.rcParams.update({
    'font.size': 10,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'axes.edgecolor': 'black',
    'axes.linewidth': 0.8
})

# Prepare data
age_ranges = df_results["Age Range"]
kappa_male = df_results["Kappa Male"]
ci_male = df_results["CI Male"]
kappa_female = df_results["Kappa Female"]
ci_female = df_results["CI Female"]
p_values = df_results["p-value"]

x = np.arange(len(age_ranges))
width = 0.35

# Create the plot
fig, ax = plt.subplots(figsize=(10, 5))

bar_male = ax.bar(x - width/2, kappa_male, width, yerr=ci_male, capsize=4, label='Male', color='dimgrey', edgecolor='black')
bar_female = ax.bar(x + width/2, kappa_female, width, yerr=ci_female, capsize=4, label='Female', hatch='//',color='whitesmoke', edgecolor='black')

# Add significance stars
for i, p in enumerate(p_values):
    if p < 0.05:
        y_max = max(kappa_male[i] + ci_male.iloc[i], kappa_female[i] + ci_female.iloc[i])
        ax.text(x[i], y_max + 0.02, '*', ha='center', va='bottom', fontsize=12, color='black')

# Final formatting
ax.set_title("Cohen's Kappa Across Age and Sex", fontsize=30)
ax.set_ylabel("Cohen's Kappa", fontsize=20)
ax.set_xlabel("Age Group", fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(age_ranges, rotation=45, ha='right')
ax.set_ylim([0, max(max(kappa_male + ci_male), max(kappa_female + ci_female)) + 0.1])
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()

# Save the figure
plt.savefig("results_normalized/kappa_by_age_and_sex.png", format='png', dpi=600)


