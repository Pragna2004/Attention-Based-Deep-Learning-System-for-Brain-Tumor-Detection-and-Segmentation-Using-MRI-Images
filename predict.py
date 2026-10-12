import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

IMG_SIZE = 224
IMAGE_PATH = "test.jpg"

# -----------------------------
# Load models
# -----------------------------
detection_model = tf.keras.models.load_model("models/detection_model.h5")
segmentation_model = tf.keras.models.load_model("models/segmentation_model.h5", compile=False)

classes = ["Glioma","Meningioma","No Tumor","Pituitary"]

# -----------------------------
# Load MRI image
# -----------------------------
img = cv2.imread(IMAGE_PATH)

if img is None:
    print("Image not found!")
    exit()

img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
original = img.copy()

img_input = img / 255.0
img_input = np.expand_dims(img_input,axis=0)

# -----------------------------
# Tumor type prediction
# -----------------------------
pred = detection_model.predict(img_input)
class_index = np.argmax(pred)
tumor_type = classes[class_index]

print("Prediction:",tumor_type)

# -----------------------------
# Segmentation prediction
# -----------------------------
mask = segmentation_model.predict(img_input)[0]

mask = mask[:,:,0]

# threshold mask
mask = (mask > 0.4).astype(np.uint8)

# remove noise
kernel = np.ones((5,5),np.uint8)
mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
mask = cv2.morphologyEx(mask,cv2.MORPH_DILATE,kernel)

# -----------------------------
# Keep only largest tumor region
# -----------------------------
contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

final_mask = np.zeros_like(mask)

if contours:
    tumor = max(contours,key=cv2.contourArea)
    cv2.drawContours(final_mask,[tumor],-1,1,-1)

# -----------------------------
# Highlight tumor
# -----------------------------
highlight = original.copy()

highlight[final_mask==1] = [255,0,0]

# -----------------------------
# Show output
# -----------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,3,1)
plt.title("Original MRI")
plt.imshow(cv2.cvtColor(original,cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Tumor Mask")
plt.imshow(final_mask,cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Tumor Highlighted : "+tumor_type)
plt.imshow(cv2.cvtColor(highlight,cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.suptitle("Attention based Brain Tumor Detection and Segmentation using MRI images",fontsize=18)

plt.show()