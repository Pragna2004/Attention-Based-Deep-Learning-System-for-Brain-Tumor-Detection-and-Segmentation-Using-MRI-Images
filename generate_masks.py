import os
import cv2
import numpy as np
from tqdm import tqdm

# -----------------------------
# PATHS (CHANGE IF NEEDED)
# -----------------------------
image_root = "dataset/segmentation/images"
mask_root = "dataset/segmentation/masks"

os.makedirs(mask_root, exist_ok=True)

classes = ["Glioma", "Meningioma", "Pituitary"]  # ❌ NO "No Tumor"

print("Generating masks...")

# -----------------------------
# LOOP THROUGH CLASSES
# -----------------------------
for cls in classes:

    img_folder = os.path.join(image_root, cls, "images")

    print(f"\nProcessing {cls}...")

    for file in tqdm(os.listdir(img_folder)):

        img_path = os.path.join(img_folder, file)

        img = cv2.imread(img_path)
        if img is None:
            continue

        # -----------------------------
        # PREPROCESS
        # -----------------------------
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)

        # -----------------------------
        # THRESHOLD (ADAPTIVE)
        # -----------------------------
        _, thresh = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # -----------------------------
        # REMOVE NOISE
        # -----------------------------
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # -----------------------------
        # KEEP ONLY LARGEST REGION
        # -----------------------------
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)

        if num_labels > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = np.zeros_like(gray)
            mask[labels == largest] = 255
        else:
            mask = thresh

        # -----------------------------
        # OPTIONAL: CENTER FILTER (remove edge noise)
        # -----------------------------
        h, w = mask.shape
        cx, cy = w // 2, h // 2

        final_mask = np.zeros_like(mask)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            if area > 200:  # ignore tiny regions
                x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]

                # keep regions near center (tumor usually central)
                if abs(x - cx) < w//2 and abs(y - cy) < h//2:
                    final_mask[labels == i] = 255

        if np.sum(final_mask) == 0:
            final_mask = mask

        # -----------------------------
        # SAVE MASK
        # -----------------------------
        save_name = file.replace(".jpg", "_mask.png")
        save_path = os.path.join(mask_root, save_name)

        cv2.imwrite(save_path, final_mask)

print("\n✅ Masks generated successfully!")