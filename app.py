import glob

import streamlit as st
from PIL import Image
import torch
import cv2
import os
import time

cfg_model_path = 'models/yolov5s.pt'
model = None
confidence = .25


def image_input(data_src):
    img_file = None
    if data_src == 'Sample data':
        # get all sample images
        img_path = glob.glob('data/sample_images/*')
        img_slider = st.slider("Select a test image.", min_value=1, max_value=len(img_path), step=1)
        img_file = img_path[img_slider - 1]
    else:
        img_bytes = st.sidebar.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
        if img_bytes:
            img_file = "data/uploaded_data/upload." + img_bytes.name.split('.')[-1]
            Image.open(img_bytes).save(img_file)

    if img_file:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_file, caption="Selected Image")
        with col2:
            img = infer_image(img_file)
            st.image(img, caption="Model prediction")


def video_input(data_src):
    vid_file = None
    if data_src == 'Sample data':
        vid_file = "data/sample_videos/sample.mp4"
    else:
        vid_bytes = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mpv', 'avi'])
        if vid_bytes:
            vid_file = "data/uploaded_data/upload." + vid_bytes.name.split('.')[-1]
            with open(vid_file, 'wb') as out:
                out.write(vid_bytes.read())

    if vid_file:
        cap = cv2.VideoCapture(vid_file)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps = 0
        st1, st2, st3 = st.columns(3)
        with st1:
            st.markdown("## Height")
            st1_text = st.markdown(f"{height}")
        with st2:
            st.markdown("## Width")
            st2_text = st.markdown(f"{width}")
        with st3:
            st.markdown("## FPS")
            st3_text = st.markdown(f"{fps}")

        st.markdown("---")
        output = st.empty()
        prev_time = 0
        curr_time = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Can't read frame, stream ended? Exiting ....")
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_img = infer_image(frame)
            output.image(output_img)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            st1_text.markdown(f"**{height}**")
            st2_text.markdown(f"**{width}**")
            st3_text.markdown(f"**{fps:.2f}**")

        cap.release()


def infer_image(img):
    model.conf = confidence
    result = model(img)
    result.render()
    image = Image.fromarray(result.ims[0])
    return image


@st.experimental_singleton
def load_model(path, device):
    model_ = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
    model_.to(device)
    print("model to ", device)
    return model_


def get_user_model():
    global cfg_model_path
    model_src = st.sidebar.radio("Model source", ["file upload", "url"])
    if model_src == "file upload":
        model_bytes = st.sidebar.file_uploader("Upload a model file", type=['pt'])
        if model_bytes:
            model_file = "models/uploaded." + model_bytes.name.split('.')[-1]
            with open(model_file, 'wb') as out:
                out.write(model_bytes.read())

            # set the model path to the new model
            cfg_model_path = model_file
    else:
        pass


def main():
    # global variables
    global model, confidence

    st.title("Object Recognition Dashboard")

    st.sidebar.title("Settings")

    # upload model
    model_src = st.sidebar.radio("Select yolov5 weight file", ["Use our demo model 5s", "Use your own model"])
    # URL, upload file (max 200 mb)
    if model_src == "Use your own model":
        get_user_model()
        st.sidebar.markdown("---")

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
            device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=False, index=0)
        else:
            device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=True, index=0)

        # load model
        model = load_model(cfg_model_path, device_option)

        # confidence slider
        confidence = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=.45)

        # custom classes
        if st.sidebar.checkbox("Custom Classes"):
            model_names = list(model.names.values())
            assigned_class = st.sidebar.multiselect("Select Classes", model_names, default=[model_names[0]])
            classes = [model_names.index(name) for name in assigned_class]
            model.classes = classes
        else:
            model.classes = list(model.names.keys())

        if input_option == 'image':
            image_input(data_src)
        else:
            video_input(data_src)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
