import streamlit as st
from PIL import Image
from model_utils import load_my_model, predict_image
from styles import apply_styles, DISCLAIMER

# 1. Setup UI
apply_styles()

# 2. Load Model
@st.cache_resource
def get_model():
    return load_my_model("model/pneumonia_model.h5")

try:
    model = get_model()
except Exception as e:
    st.error("Model not found! Please check the model path.")
    st.stop()

# 3. File Upload
uploaded_file = st.file_uploader("Choose an X-Ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-Ray", use_column_width=True)
    
    with st.spinner('Analyzing image...'):
        # Using our helper function from model_utils
        result, confidence = predict_image(model, image)
        
    # 4. Display Results
    st.divider()
    if result == "PNEUMONIA":
        st.error(f"### Result: {result}")
    else:
        st.success(f"### Result: {result}")
        
    st.metric(label="Confidence Level", value=f"{confidence*100:.2f}%")
    st.warning(DISCLAIMER)