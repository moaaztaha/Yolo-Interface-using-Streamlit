# Yolov5 Real-time Inference using Streamlit

## Features
- Support both images and videos
- Supports both CPU and GPU inference
- Supports **Custom Classes** and changing **Confidence**


## How to run
After cloning the repo:
1. Install requirements
   - `pip install -r requirements.txt`
2. Add sample images to `data/sample_images`
3. Add sample video to `data/sample_videos` and call it `sample.mp4` or change name in the code.
4. Add model file to `models/` and change `cfg_model_path` to its path.
```bash
git clone https://github.com/moaaztaha/Yolo-Interface-using-Streamlit
cd Yolo-Interface-using-Streamlit
streamlit run app.py
```
