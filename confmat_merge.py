from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

def confmat_5class():
    image_paths = [
        "results_normalized/0-6_months_confusion_5class.png", "results_normalized/6-12_months_confusion_5class.png", "results_normalized/1-2_years_confusion_5class.png",
        "results_normalized/2-3_years_confusion_5class.png", "results_normalized/3-4_years_confusion_5class.png", "results_normalized/4-5_years_confusion_5class.png",
        "results_normalized/5-6_years_confusion_5class.png", "results_normalized/6-12_years_confusion_5class.png", "results_normalized/over_12_years_confusion_5class.png"
    ]
    
    
    images = [Image.open(p) for p in image_paths]
    
    # Ensure uniform size
    img_width, img_height = images[0].size
    images = [img.resize((img_width, img_height)) for img in images]
    
    # Create figure and axes
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    # Remove all space between subplots
    fig.subplots_adjust(wspace=0, hspace=0, left=0.01, right=0.99, top=0.99, bottom=0.01)
    
    # Plot images without axes
    for ax, img in zip(axes.flat, images):
        ax.imshow(img)
        ax.axis("off")
    
    # Add X and Y labels *closer* to the image grid
    fig.text(0.5, -0.01, 'Predicted Label', ha='center', va='top', fontsize=40)
    fig.text(-0.01, 0.5, 'True Label', ha='right', va='center', rotation='vertical', fontsize=40)
    
    # Save figure
    plt.savefig("results_normalized/confmat_5class", dpi=600, bbox_inches='tight')
    
def confmat_3class():
    image_paths = [
        "results_normalized/0-6_months_confusion_3class.png", "results_normalized/6-12_months_confusion_3class.png", "results_normalized/1-2_years_confusion_3class.png",
        "results_normalized/2-3_years_confusion_3class.png", "results_normalized/3-4_years_confusion_3class.png", "results_normalized/4-5_years_confusion_3class.png",
        "results_normalized/5-6_years_confusion_3class.png", "results_normalized/6-12_years_confusion_3class.png", "results_normalized/over_12_years_confusion_3class.png"
    ]
    
    
    images = [Image.open(p) for p in image_paths]
    
    # Ensure uniform size
    img_width, img_height = images[0].size
    images = [img.resize((img_width, img_height)) for img in images]
    
    # Create figure and axes
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    # Remove all space between subplots
    fig.subplots_adjust(wspace=0, hspace=0, left=0.01, right=0.99, top=0.99, bottom=0.01)
    
    # Plot images without axes
    for ax, img in zip(axes.flat, images):
        ax.imshow(img)
        ax.axis("off")
    
    # Add X and Y labels *closer* to the image grid
    fig.text(0.5, -0.01, 'Predicted Label', ha='center', va='top', fontsize=40)
    fig.text(-0.01, 0.5, 'True Label', ha='right', va='center', rotation='vertical', fontsize=40)
    
    # Save figure
    plt.savefig("results_normalized/confmat_3class", dpi=600, bbox_inches='tight')
    
confmat_5class()
confmat_3class()
