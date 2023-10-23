import cv2
import time
import numpy as np
import mediapipe as mp

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

RESULT = None

# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):

    global RESULT
    RESULT = result
    # print('Hand landmarker result:\n {}'.format(RESULT.hand_landmarks))
    
    if RESULT.hand_landmarks!=[]:
        print('HandLandmark.INDEX_FINGER_TIP result:\n {}'.format(RESULT.hand_landmarks[0][8]) )#(HandLandmark.INDEX_FINGER_TIP=8)
        px, py, pz = RESULT.hand_landmarks[0][8].x ,RESULT.hand_landmarks[0][8].y , RESULT.hand_landmarks[0][8].z   # GET INDEX_FINGER POSITION
        

def draw_landmarks_on_image(rgb_image):
   """Courtesy of https://github.com/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb"""
   try:
      if type(RESULT) is type(None):
          return rgb_image
      else:
         hand_landmarks_list = RESULT.hand_landmarks
         annotated_image = np.copy(rgb_image)
         # Loop through the detected hands to visualize.
         for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            
            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
               landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
            mp.solutions.drawing_utils.draw_landmarks(
               annotated_image,
               hand_landmarks_proto,
               mp.solutions.hands.HAND_CONNECTIONS,
               mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
               mp.solutions.drawing_styles.get_default_hand_connections_style())
         return annotated_image
   except:
      return rgb_image


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='/model/hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands =1,
    result_callback=print_result)

cap = cv2.VideoCapture(0) # Use OpenCV’s VideoCapture to start capturing from the webcam.

with HandLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.

   # Create a loop to read the latest frame from the camera using VideoCapture#read()
   while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)  # Convert the frame received from OpenCV to a MediaPipe’s Image object.
         
        hand_landmarker_result = landmarker.detect_async(mp_image, int(time.time() * 1000))
        
        frame = draw_landmarks_on_image(frame)
        cv2.imshow('MediaPipe Hands',cv2.flip(frame, 1))
        
        if cv2.waitKey(5) & 0xFF == 27:
            break    
    
  
