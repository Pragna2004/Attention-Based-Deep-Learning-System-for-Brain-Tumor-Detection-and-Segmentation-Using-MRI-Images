import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import json

IMG_SIZE = 224

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------
st.markdown("""
<style>

.stApp{
    background-color:#f4f7fb;
}

.main-title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#0b3d91;
}

.sub-title{
    text-align:center;
    font-size:18px;
    color:#555;
    margin-bottom:30px;
}

.prediction-box{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 2px 10px rgba(0,0,0,0.1);
    margin-bottom:20px;
}

.image-box{
    background:white;
    padding:10px;
    border-radius:15px;
    box-shadow:0px 2px 8px rgba(0,0,0,0.08);
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------
@st.cache_resource
def load_models():

    det_model = tf.keras.models.load_model(
        "models/detection_model.h5"
    )

    seg_model = tf.keras.models.load_model(
        "models/segmentation_model.h5",
        compile=False
    )

    with open("models/classes.json") as f:
        classes = json.load(f)

    return det_model, seg_model, classes

det_model, seg_model, classes = load_models()

# -------------------------------------------------
# PREPROCESS
# -------------------------------------------------
def preprocess(img):

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = tf.keras.applications.efficientnet.preprocess_input(img)

    return img

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
@st.cache_data
def predict_image(img):

    img = preprocess(img)

    img = np.expand_dims(img, 0)

    pred = det_model.predict(img, verbose=0)[0]

    return pred

# -------------------------------------------------
# SEGMENTATION
# -------------------------------------------------
def get_mask(img):

    inp = cv2.resize(img, (224,224)) / 255.0

    inp = np.expand_dims(inp, 0)

    mask = seg_model.predict(inp, verbose=0)[0][:,:,0]

    mask = (mask > 0.5).astype(np.uint8)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    if num > 1:

        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

        clean = np.zeros_like(mask)

        clean[labels == largest] = 1

        mask = clean

    return mask

# -------------------------------------------------
# TUMOR INFO
# -------------------------------------------------
def get_tumor_info(tumor):

    info = {

        "Glioma":[
            "Glioma occurs in the brain and spinal cord.",
            "It develops from glial support cells.",
            "Gliomas may grow aggressively.",
            "Symptoms include headaches and seizures.",
            "Early diagnosis improves treatment."
        ],

        "Meningioma":[
            "Meningioma arises from brain coverings.",
            "Usually slow-growing and non-cancerous.",
            "Can affect nearby brain structures.",
            "Symptoms include headaches and vision issues.",
            "Often treatable with surgery."
        ],

        "Pituitary":[
            "Pituitary tumors affect hormone glands.",
            "They may cause hormonal imbalance.",
            "Most pituitary tumors are benign.",
            "Vision problems may occur.",
            "Treatment includes medication or surgery."
        ],

        "No Tumor":[
            "No tumor detected in MRI image.",
            "Brain structure appears normal.",
            "No abnormal tissue growth found.",
            "Routine monitoring is recommended.",
            "Consult doctor if symptoms continue."
        ]
    }

    return info.get(tumor, [])

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.markdown(
    "<h1 class='main-title'>Brain Tumor Detection & Segmentation</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='sub-title'>Attention-Based Deep Learning System Using MRI Images</p>",
    unsafe_allow_html=True
)

# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------
file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg","png","jpeg"]
)

# -------------------------------------------------
# MAIN PROCESS
# -------------------------------------------------
if file:

    file_bytes = np.frombuffer(file.read(), np.uint8)

    img = cv2.imdecode(file_bytes, 1)

    original = cv2.resize(img, (224,224))

    # -------------------------------------------------
    # PREDICTION
    # -------------------------------------------------
    pred = predict_image(original)

    tumor = classes[np.argmax(pred)]

    # -------------------------------------------------
    # PREDICTION RESULT
    # -------------------------------------------------
    st.markdown(
        f"""
        <div class="prediction-box">
            <h2 style="color:#d62828;">
            Prediction : {tumor}
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------------------------------------------------
    # TUMOR INFORMATION
    # -------------------------------------------------
    st.subheader("Tumor Information")

    for line in get_tumor_info(tumor):
        st.write("•", line)

    st.write("")

    # -------------------------------------------------
    # IMAGE OUTPUTS
    # -------------------------------------------------
    col1, col2, col3 = st.columns(3)

    # ORIGINAL IMAGE
    with col1:

        st.markdown(
            '<div class="image-box">',
            unsafe_allow_html=True
        )

        st.image(
            original,
            caption="Original MRI",
            width=250
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    # -------------------------------------------------
    # SEGMENTATION
    # -------------------------------------------------
    if tumor != "No Tumor":

        mask = get_mask(original)

        if np.sum(mask) > 500:

            overlay = original.copy()

            overlay[mask == 1] = [255, 0, 0]

            blended = cv2.addWeighted(
                original,
                0.7,
                overlay,
                0.3,
                0
            )

            # MASK IMAGE
            with col2:

                st.markdown(
                    '<div class="image-box">',
                    unsafe_allow_html=True
                )

                st.image(
                    mask * 255,
                    caption="Tumor Mask",
                    width=250
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

            # HIGHLIGHTED IMAGE
            with col3:

                st.markdown(
                    '<div class="image-box">',
                    unsafe_allow_html=True
                )

                st.image(
                    blended,
                    caption="Highlighted Tumor",
                    width=250
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

        else:

            st.warning(
                "Tumor region not clearly detected"
            )

    else:

        st.success(
            "No Tumor Detected"
        )