import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import os 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline


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
	THREE_CLASS_MAPPING = {"N1": 1, "N2": 1, "N3": 1, "REM": 2, "WAKE": 0}

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

	# Convert ground truth and predicted labels to integers for 3 classes
	ground_truth_three = [THREE_CLASS_MAPPING[label] for label in filtered_ground_truth]
	predicted_three = [THREE_CLASS_MAPPING[label] for label in filtered_predictions]

	# Compute accuracy and cohen kappa for 5 classes
	accuracy = accuracy_score(ground_truth_classes, predicted_classes)
	kappa = cohen_kappa_score(ground_truth_classes, predicted_classes)

	accuracy_3 = accuracy_score(ground_truth_three, predicted_three)
	kappa_3 = cohen_kappa_score(ground_truth_three, predicted_three)

	return accuracy, kappa,accuracy_3, kappa_3, np.array(ground_truth_classes), np.array(predicted_classes), np.array(ground_truth_three), np.array(predicted_three)


root_dir = 'BCH_h5_arranged'
df_test = pd.read_csv('/media/ayush/Ayush/Documents/BCH_data_analysis/USleep_expt/CommonIDs/test_set.csv')
subID_test = df_test['subID'].to_numpy()
sess_test = df_test['Session'].to_numpy()
test_files = []


for i in range(0,len(subID_test)):
    subID_sess_test = subID_test[i] + '_ses-' + str(sess_test[i])
    for fo in os.listdir(root_dir):
        if fo.startswith(subID_sess_test):
            test_files.append(fo)

print(len(test_files))



age_bins = [(0, 180, '0-6 months'), (181, 365, '6-12 months'),
            (366, 2*365, '1-2 years'), (2*365+1, 3*365, '2-3 years'),
            (3*365+1, 4*365, '3-4 years'), (4*365+1, 5*365, '4-5 years'),
            (5*365+1, 6*365, '5-6 years'), (6*365+1, 12*365, '6-12 years'),
            (12*365+1, np.inf, 'over 12 years'), (0, np.inf, 'Test Set')]



demographics = pd.read_csv('/media/ayush/Ayush/Documents/BCH_data_analysis/USleep_expt/CommonIDs/test_set.csv')
subID = demographics['subID'].to_numpy()
sessID = demographics['Session'].to_numpy()
ageindays = demographics['AgeDays'].to_numpy()

output_folder = "results_normalized"
os.makedirs(output_folder, exist_ok=True)
results = []


for start,end,title in age_bins:

	kappa_vector = []
	accuracy_vector = []
	kappa3_vector = []
	accuracy3_vector = []
	all_predicted = []
	all_ground = []
	all_predicted3 = []
	all_ground3 = []
	agectr = 0

	for i in range(0,len(test_files)):
		ID,sess,_,_ = test_files[i].split('_')
		sess = int(sess[-1])
		age = ageindays[np.where((subID == ID) & (sessID == sess))[0][0]]

		if age>=start and age<=end:

			h5_file = test_files[i]

			accuracy, kappa, accuracy_3, kappa_3, ground, pred, ground_3, pred_3 = evaluate_predictions(h5_file)
			kappa_vector.append(kappa)
			accuracy_vector.append(accuracy)
			kappa3_vector.append(kappa_3)
			accuracy3_vector.append(accuracy_3)
			all_predicted.extend(pred)
			all_ground.extend(ground)
			all_predicted3.extend(pred_3)
			all_ground3.extend(ground_3)
			agectr = agectr + 1

	print(title)
	print(agectr)
	all_predicted = np.array(all_predicted)
	all_ground = np.array(all_ground)
	all_predicted3 = np.array(all_predicted3)
	all_ground3 = np.array(all_ground3)



	print(f"Mean Accuracy 5 class: {np.nanmean(accuracy_vector):.4f}")
	print(f"Mean Cohen's Kappa 5 class: {np.nanmean(kappa_vector):.4f}")

	print(f"Mean Accuracy 3 class: {np.nanmean(accuracy3_vector):.4f}")
	print(f"Mean Cohen's Kappa 3 class: {np.nanmean(kappa3_vector):.4f}")

	results.append([title, np.nanmean(accuracy_vector), np.nanmean(kappa_vector), 
	            np.nanmean(accuracy3_vector), np.nanmean(kappa3_vector)])

	# Create a normalized confusion matrix (as percentages) for 5 classes
	conf_matrix = confusion_matrix(all_ground, all_predicted, labels=[0, 1, 2, 3, 4], normalize='true')
	print(conf_matrix)
	plt.figure(figsize=(8,8))
	disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['WAKE', 'N1', 'N2', 'N3', 'REM'])
	disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), values_format=".1%")
	plt.title('Confusion Matrix for age ' + title, fontsize=25)
	plt.xlabel('')
	plt.ylabel('')
	#plt.xlabel('Predicted Labels', fontsize=30)
	#plt.ylabel('True Labels', fontsize=30)
	for text in disp.text_.ravel():
	    text.set_fontsize(25) 

	plt.gca().tick_params(axis='both', labelsize=25)
	disp.im_.colorbar.remove()
	plt.tight_layout()
	title_save = title.replace(" ", "_")
	plt.savefig(f"{output_folder}/{title_save}_confusion_5class.png")


	# Create a normalized confusion matrix (as percentages) for 3 classes
	conf_matrix = confusion_matrix(all_ground3, all_predicted3, labels=[0, 1, 2], normalize='true')
	plt.figure(figsize=(8,8))
	disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['WAKE', 'N', 'REM'])
	disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), values_format=".1%")
	plt.title('Confusion Matrix for age ' + title, fontsize=25)
	plt.xlabel('')
	plt.ylabel('')
	#plt.xlabel('Predicted Labels', fontsize=30)
	#plt.ylabel('True Labels', fontsize=30)
	for text in disp.text_.ravel():
	    text.set_fontsize(35) 

	plt.gca().tick_params(axis='both', labelsize=25)
	disp.im_.colorbar.remove()
	plt.tight_layout()
	title_save = title.replace(" ", "_")
	plt.savefig(f"{output_folder}/{title_save}_confusion_3class.png")

    

pd.DataFrame(results, columns=["Age Group", "Accuracy 5-class", "Kappa 5-class",
                               "Accuracy 3-class", "Kappa 3-class"]).to_csv(f"{output_folder}/evaluation_results.csv", index=False)