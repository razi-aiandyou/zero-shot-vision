import streamlit as st
import base64
from ai_systems import classify_mercedes_car

st.set_page_config(page_title="AI Computer Vision Prototype V0.1", layout="wide")

# Custom CSS for dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stButton>button {
        color: #FFFFFF;
        background-color: #4CAF50;
        border: none;
    }
    .stTextInput>div>div>input {
        color: #FFFFFF;
    }
    .big-font {
        font-size:20px !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("AI Computer Vision Prototype V0.1")
st.write("Created by Razi Ashary from AI&You, trained on a combination of few-shot and zero-shot learning")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Image Upload")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        if st.button("Run AI System"):
            if uploaded_file is not None:
                bytes_data = uploaded_file.getvalue()
                base64_image = base64.b64encode(bytes_data).decode()

                # Run AI system
                result = classify_mercedes_car(base64_image)

                with col2:
                    st.subheader("Result")
                    st.markdown(f'<p class="big-font">{result}</p>', unsafe_allow_html=True)

            else:
                st.error("Please upload an image first")

with col2:
    st.markdown("*The AI system result will appear here after you upload an image and click 'Run AI System'.*")