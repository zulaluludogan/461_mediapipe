import cv2
import time
import numpy as np
import mediapipe as mp

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

RESULT = None

# Create a hand landmarker instance with the live stream mode:
def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    
    global RESULT
    RESULT = result
    # print('face landmarker result:\n{}'.format(result)) #468 landmark
    if RESULT.face_landmarks!=[]:
        print('FaceLandmark.upperMOUTH position:\n {}'.format(RESULT.face_landmarks[0][13]) )#(HandLandmark.INDEX_FINGER_TIP=8)
        print('FaceLandmark.lowerMOUTH position:\n {}'.format(RESULT.face_landmarks[0][14]) )#(HandLandmark.INDEX_FINGER_TIP=8)
        px, py, pz = RESULT.face_landmarks[0][13].x ,RESULT.face_landmarks[0][13].y , RESULT.face_landmarks[0][13].z   # GET MOUTH POSITION

def draw_landmarks_on_image(rgb_image):
    try:
      if type(RESULT) is type(None):
          return rgb_image
      else:
        
        face_landmarks_list = RESULT.face_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            # Draw the face landmarks.
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_tesselation_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_iris_connections_style())

        return annotated_image
    except:
      return rgb_image

model_asset_path='face_landmarker.task'

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)


cap = cv2.VideoCapture(0) # Use OpenCV’s VideoCapture to start capturing from the webcam.

with FaceLandmarker.create_from_options(options) as landmarker:
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
