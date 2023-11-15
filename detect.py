import cv2
import mediapipe as mp
import sys
import time

from pythonosc import udp_client
from pythonosc.osc_message_builder import OscMessageBuilder

ip = "127.0.0.1"
port = 3001

address_lefthand = "/mediapipe/lefthand"
address_righthand = "/mediapipe/righthand"
address_face = "/mediapipe/face"
address_pose = "/mediapipe/pose"

client = udp_client.SimpleUDPClient(ip, port)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# Variables to calculate FPS
counter, fps = 0, 0
start_time = time.time()

# Visualization parameters
row_size = 20  # pixels
left_margin = 24  # pixels
text_color = (0, 0, 255)  # red
font_size = 1
font_thickness = 1
fps_avg_frame_count = 10

cap = cv2.VideoCapture(1)
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1) as holistic:
  while cap.isOpened():
    counter += 1
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), (0, 0, 0), -1)

    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_tesselation_style())
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_hand_landmarks_style())
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_hand_landmarks_style())
    
    # Get results for face, pose, left hand, right hand and send as OSC data
    if results.face_landmarks:
        face = results.face_landmarks
        message_builder = OscMessageBuilder(address=address_face)
        for landmark in face.landmark:
            message_builder.add_arg(landmark.x)
            message_builder.add_arg(landmark.y)
            message_builder.add_arg(landmark.z)
        message = message_builder.build()
        client.send(message)
        
    if results.pose_landmarks:
        pose = results.pose_landmarks
        message_builder = OscMessageBuilder(address=address_pose)
        for landmark in pose.landmark:
            message_builder.add_arg(landmark.x)
            message_builder.add_arg(landmark.y)
            message_builder.add_arg(landmark.z)
        message = message_builder.build()
        client.send(message)
    
    if results.left_hand_landmarks:
        lefthand = results.left_hand_landmarks
        message_builder = OscMessageBuilder(address=address_lefthand)
        for landmark in lefthand.landmark:
            message_builder.add_arg(landmark.x)
            message_builder.add_arg(landmark.y)
            message_builder.add_arg(landmark.z)
        message = message_builder.build()
        client.send(message)
        
    if results.right_hand_landmarks:
        righthand = results.right_hand_landmarks
        message_builder = OscMessageBuilder(address=address_righthand)
        for landmark in righthand.landmark:
            message_builder.add_arg(landmark.x)
            message_builder.add_arg(landmark.y)
            message_builder.add_arg(landmark.z)
        message = message_builder.build()
        client.send(message)
        
    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
        end_time = time.time()
        fps = fps_avg_frame_count / (end_time - start_time)
        start_time = time.time()

    # Print the FPS
    # print(fps)

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Holistic OSC', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()