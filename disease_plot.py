import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("results_normalized/kappa_by_age_icd10.csv")

# Define disease list and age groups (including hypothetical 'Test Set')
diseases = df["ICD10 Description"].unique()
age_groups = list(df["Age Group"].unique()) + ["Test Set"]

# Set up the plot
fig, axes = plt.subplots(nrows=len(diseases), ncols=1, figsize=(10, 22), sharex=True)
plt.style.use("grayscale")

# Loop through diseases
for i, disease in enumerate(diseases):
    ax = axes[i]

    # Filter and align data for current disease
    data = df[df["ICD10 Description"] == disease]
    data = data.set_index("Age Group").reindex(age_groups[:-1]).reset_index()

    x = np.arange(len(data))
    width = 0.35

    # Bar plots
    bars1 = ax.bar(x - width/2, data["Mean Kappa"], width,
                   yerr=data["CI Kappa"], capsize=3,
                   label="Disease", edgecolor='black', color='dimgrey')
    bars2 = ax.bar(x + width/2, data["Mean Kappa Non Disease"], width,
                   yerr=data["CI Kappa Non Disease"], capsize=3,
                   label="Non-Disease", edgecolor='black', hatch='//', color='whitesmoke')

    # Annotate statistically significant differences with a star
    for j, p in enumerate(data["p-value"]):
        if pd.notnull(p) and p < 0.05:
            # Determine the height above both bars
            y1 = data["Mean Kappa"].iloc[j] + data["CI Kappa"].iloc[j]
            y2 = data["Mean Kappa Non Disease"].iloc[j] + data["CI Kappa Non Disease"].iloc[j]
            y_star = max(y1, y2) - 0.05
            ax.text(x[j], y_star, '*', ha='center', va='bottom', fontsize=30, color='black',fontweight='bold')

    # Title and axis labels
    ax.set_title(disease, fontsize=20)
    if i ==3:
        ax.set_ylabel("Cohen's Kappa", fontsize=40)
    ax.set_ylim(0, 1) 

# Add legend to the first subplot
axes[0].legend(loc="upper left")

# Common X-axis labels
plt.xticks(ticks=np.arange(len(age_groups)-1), labels=age_groups[:-1], rotation=45, ha='right',fontsize = 20)
axes[-1].set_xlabel("Age Group", fontsize=40)

# Final formatting
plt.tight_layout()

# Save high-resolution figure
plt.savefig("results_normalized/kappa_disease.png", dpi=300)
