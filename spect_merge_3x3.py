from PIL import Image, ImageDraw, ImageFont
import os

# === 1. Update Image Paths (9 total) ===
img_paths = [
    "results_normalized/spectrograms/sub-I0003175657475_ses-2_task-psg_eeg.png",
    "results_normalized/spectrograms/sub-I0003175658609_ses-1_task-psg_eeg.png",
    "results_normalized/spectrograms/sub-I0003175663840_ses-2_task-PSG_eeg.png",
    "results_normalized/spectrograms/sub-I0003175572478_ses-1_task-psg_eeg.png",
    "results_normalized/spectrograms/sub-I0003175672602_ses-2_task-PSG_eeg.png",
    "results_normalized/spectrograms/sub-I0003175677177_ses-1_task-psg_eeg.png",
    "results_normalized/spectrograms/sub-I0003175658756_ses-1_task-psg_eeg.png",
    "results_normalized/spectrograms/sub-I0003175677127_ses-1_task-psg_eeg.png",
    "results_normalized/spectrograms/sub-I0003175667957_ses-1_task-psg_eeg.png"
]

# === 2. Labels ===
labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"]

# === 3. Font Settings ===
font_size = 40
try:
    font = ImageFont.truetype("DejaVuSans.ttf", size=font_size)
except:
    font = ImageFont.load_default()

# === 4. Label Size Function ===
def get_label_size(label, font):
    dummy_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), label, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width, height

label_w, label_h = get_label_size("(a)", font)
padding = label_h + 30

# === 5. Generate Labeled Images ===
labeled_images = []
for path, label in zip(img_paths, labels):
    if not os.path.exists(path):
        print(f"Warning: Missing file: {path}")
        continue
    img = Image.open(path)
    W, H = img.size

    new_img = Image.new("RGB", (W, H + padding), color="white")
    new_img.paste(img, (0, 0))

    draw = ImageDraw.Draw(new_img)
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (W - text_w) // 2 - bbox[0]
    y = H + (padding - text_h) // 2
    draw.text((x, y), label, font=font, fill="black")

    labeled_images.append(new_img)

# === 6. Validate Image Count ===
if len(labeled_images) != 9:
    raise ValueError(f"Expected 9 images, got {len(labeled_images)}.")

# === 7. Create Grid ===
img_w, img_h = labeled_images[0].size
grid_cols = 3
grid_rows = 3
grid_img = Image.new("RGB", (grid_cols * img_w, grid_rows * img_h), color="white")

for idx, img in enumerate(labeled_images):
    row = idx // grid_cols
    col = idx % grid_cols
    x = col * img_w
    y = row * img_h
    grid_img.paste(img, (x, y))

# === 8. Scale and Save ===
scale_factor = 2
highres_size = (grid_img.width * scale_factor, grid_img.height * scale_factor)
highres_img = grid_img.resize(highres_size, resample=Image.LANCZOS)

highres_img.save("results_normalized/merged_spectrograms_3x3.png", dpi=(300, 300))
