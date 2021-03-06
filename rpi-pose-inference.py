import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
import argparse

from picamera import PiCamera
from picamera.array import PiRGBArray

# Settings
MOVENET_MODEL_PATH = "movenet.tflite"
CAM_WIDTH = 192     # Camera width
CAM_HEIGHT = 192    # Camera height
DRAW_SCALE = 2

# Command line arguments
parser = argparse.ArgumentParser(description="Run pose classification inference")
parser.add_argument('-m',
                    '--model',
                    dest='model',
                    type=str,
                    required=True,
                    help="Path to TFLite model file for classification")
parser.add_argument('-l',
                    '--labels',
                    dest='labels',
                    type=str,
                    required=True,
                    help="Path to .txt file with label on each line")
parser.add_argument('-r',
                    '--rotation',
                    dest='rotation',
                    type=int,
                    default=0,
                    help="Rotation of captured image (0, 90, 180, 270)")

# Parse arguments
args = parser.parse_args()
classifier_model_path = args.model
labels_path = args.labels
cam_rotation = args.rotation

# Parse labels file
with open(labels_path) as f:
    lines = f.readlines()
labels = [s.strip() for s in lines]

# Initialize the TFLite interpreter for pose estimation
pose_interpreter = tflite.Interpreter(model_path="movenet.tflite")
pose_interpreter.allocate_tensors()

# Set up interpreter
pose_input_details = pose_interpreter.get_input_details()
pose_output_details = pose_interpreter.get_output_details()

# Initialize the TFLite interpreter for pose classification
classifier_interpreter = tflite.Interpreter(model_path=classifier_model_path)
classifier_interpreter.allocate_tensors()

# Set up interpreter
classifier_input_details = classifier_interpreter.get_input_details()
classifier_output_details = classifier_interpreter.get_output_details()

# Initial framerate value
fps = 0

# Start the camera
with PiCamera() as camera:
    
    # Configure camera settings
    camera.resolution = (CAM_WIDTH, CAM_HEIGHT)
    camera.rotation = cam_rotation
    
    # Container for our frames
    raw_capture = PiRGBArray(camera, size=(CAM_WIDTH, CAM_HEIGHT))

    # Continuously capture frames (this is our while loop)
    for frame in camera.capture_continuous(raw_capture, 
                                            format='bgr', 
                                            use_video_port=True):
                                            
        # Get timestamp for calculating actual framerate
        timestamp = cv2.getTickCount()
        
        # Get Numpy array that represents the image
        img = frame.array

        # Add "sample" dimension to line up with model input
        img_arr = img[np.newaxis, :]
        
        # Feed image array to model and perform inference
        pose_interpreter.set_tensor(pose_input_details[0]['index'], img_arr)
        pose_interpreter.invoke()

        # Get predicted pose points
        keypoints_with_scores = pose_interpreter.get_tensor(pose_output_details[0]['index'])
        keypoints_flat = keypoints_with_scores.flatten()
        keypoints_flat = keypoints_flat[np.newaxis, np.newaxis, :, np.newaxis]
        
        # Perform inference with pose points
        classifier_interpreter.set_tensor(classifier_input_details[0]['index'], keypoints_flat)
        classifier_interpreter.invoke()

        # Get the model prediction
        classifier_output = classifier_interpreter.get_tensor(classifier_output_details[0]['index'])[0]

        # Find label with the highest probability
        idx_max = np.argmax(classifier_output)
        label_max = labels[idx_max]

        # Make the image a little easier to see
        #draw_img = cv2.resize(img, (CAM_WIDTH * DRAW_SCALE, CAM_HEIGHT * DRAW_SCALE))
        
        # Draw max label on preview window
        cv2.putText(img,
                    label_max,
                    (0, 12),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (255, 255, 255))

        # Draw max probability on preview window
        cv2.putText(img,
                    str(round(classifier_output[idx_max], 2)),
                    (0, 24),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (255, 255, 255))

        # Draw FPS on preview window
        cv2.putText(img,
            "FPS: " + str(round(fps, 2)),
            (0, 190),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 255, 255))

        # Show the frame
        img_resize = cv2.resize(img, (CAM_WIDTH * DRAW_SCALE, CAM_HEIGHT * DRAW_SCALE))
        cv2.imshow("Frame", img_resize)
        
        # Clear the stream to prepare for next frame
        raw_capture.truncate(0)
        
        # Calculate framrate
        frame_time = (cv2.getTickCount() - timestamp) / cv2.getTickFrequency()
        fps = 1 / frame_time
        
        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break
        
# Clean up
cv2.destroyAllWindows()