import glob

import streamlit as st
from PIL import Image
import torch
import os

cfg_model_path = 'models/yolov5s.pt'
model = None


def image_input(data_src):
    img_file = None
    if data_src == 'Sample data':
        # get all sample images
        img_path = glob.glob('data/sample_data/*')
        img_slider = st.slider("Select a test image.", min_value=1, max_value=len(img_path), step=1)
        img_file = img_path[img_slider - 1]
    else:
        img_bytes = st.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
        if img_bytes:
            img_file = "data/uploaded_data/upload." + img_bytes.name.split('.')[-1]
            Image.open(img_bytes).save(img_file)

    if img_file:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_file, caption="Selected Image")
        submit = st.button('Predict!')
        with col2:
            if submit:
                img = infer_image(img_file)
                st.image(img, caption="Model prediction")


def infer_image(img_path: str):
    result = model(img_path)
    result.render()
    image = Image.fromarray(result.ims[0])
    return image


@st.experimental_singleton
def load_model(path, device):
    model_ = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
    model_.to(device)
    return model_


def main():
    global model
    st.title("Object Recognition Dashboard")

    st.sidebar.title("Settings")

    # input src option
    data_src = st.sidebar.radio("Select input source: ", ['Sample data', 'Upload your own data'])

    # input options
    input_option = st.sidebar.radio("Select input type: ", ['image', 'video'])

    # check if model file is available
    if not os.path.isfile(cfg_model_path):
        st.warning("Model file not available!!!, please added to the model folder.", icon="⚠️")
    else:
        # device options
        if torch.cuda.is_available():
            device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=False, index=1)
        else:
            device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=True, index=0)

        # load model
        model = load_model(cfg_model_path, device_option)

        if input_option == 'image':
            image_input(data_src)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
