# Pose Estimation Workshop

Demonstration of multi-stage inference with custom DSP blocks in Edge Impulse.

Use one of the Colab scripts to break apart video into frames and upload them as training/test samples to your Edge Impulse project.
* cam_video_capture_and_extract_frames.ipynb: use your webcam to capture video. **Chrome only!**
* vlc_frame_extract_and_upload.ipynb: take video with your phone. Upload that video to this Colab and run it to extract frames.

Train your machine learning model on Edge Impulse using the *pose-estimation* custom DSP block (available at workshop and for enterprise customers). Go to the Dashboard and download the float32 TFLite model file. Create a .txt file containing a string for each label (one on each line). 

Install dependencies:

```
sudo apt update
sudo apt install python3-opencv python3-numpy
python -m pip --upgrade pip
python -m pip install tflite-runtime cv2
```

Run the *rpi-pose-inference.py* script on your Raspberry Pi. Make sure *movenet.tflite* is in the same directory. Give it your classifier model and labels file as arguments. The `-r` argument is for camera rotation in degrees (0, 90, 180, 270). Here is an example:

```
python rpi-pose-inference.py -m ymca-classifier-model.lite -l ymca-labels.txt -r 90
``` 

## License

Unless otherwise specified, code samples in this repository are licensed under the APACHE 2.0 open source license.

Copyright 2022 EdgeImpulse, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.