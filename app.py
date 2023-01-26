import glob

import streamlit as st
from PIL import Image
import torch

cfg_model_path = 'models/yolov5s.pt'


def image_input(device_option, data_src):
    if data_src == 'Sample data':
        # get all sample images
        img_path = glob.glob('data/sample_data/*')
        img_slider = st.slider("Select a test image.", min_value=1, max_value=len(img_path), step=1)
        img_file = img_path[img_slider - 1]
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_file, caption="Selected Image")

        submit = st.button('Predict!')
        with col2:
            if submit and img_file is not None:
                img = infer_image(img_file, device_option)
                st.image(img, caption="Model prediction")


def infer_image(img_path: str, device_option: str):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=cfg_model_path, force_reload=True)
    model.to(device_option)
    result = model(img_path)
    result.render()
    image = Image.fromarray(result.ims[0])
    return image


def main():
    st.title("Object Recognition Dashboard")

    st.sidebar.title("Settings")

    # input src option
    data_src = st.sidebar.radio("Select input source: ", ['Sample data', 'Upload your own data'])

    # input options
    input_option = st.sidebar.radio("Select input type: ", ['image', 'video'])

    # device options
    if torch.cuda.is_available():
        device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=False, index=1)
    else:
        device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=True, index=0)

    if input_option == 'image':
        image_input(device_option, data_src)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
