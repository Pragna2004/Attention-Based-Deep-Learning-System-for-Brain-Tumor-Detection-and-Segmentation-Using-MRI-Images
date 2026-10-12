# Attention-Based Deep Learning System for Brain Tumor Detection and Segmentation Using MRI Images

## Project Overview
This project uses deep learning to classify brain MRI images and identify tumor regions. It aims to assist healthcare professionals by providing automated image analysis and visual results.

## Technologies Used
- Python
- TensorFlow and Keras
- EfficientNetB0
- Attention U-Net
- OpenCV and NumPy
- Streamlit

## Key Features
- Classifies MRI images into Glioma, Meningioma, Pituitary Tumor, and No Tumor.
- Segments and highlights tumor regions using Attention U-Net.
- Provides a graphical user interface for uploading MRI images.
- Displays classification and segmentation results.

## Project Structure
- `app.py` – Streamlit application
- `train_detection.py` – Detection model training
- `train_segmentation.py` – Segmentation model training
- `attention_model.py` – Attention-related model code
- `attention_unet_model.py` – Attention U-Net architecture
- `gradcam.py` – Grad-CAM implementation
- `predict.py` – Prediction functionality
- `models/` – Trained models and class labels

## How to Run
1. Clone this repository.
2. Install the required Python dependencies.
3. Ensure the trained models are available in the `models/` directory.
4. Run the application:

```bash
streamlit run app.py
```

## Disclaimer
This project is intended for academic and research purposes. It is not a replacement for professional medical diagnosis.
