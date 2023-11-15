import cv2
import mediapipe as mp
import sys
import time

from pythonosc import udp_client
from pythonosc.osc_message_builder import OscMessageBuilder

# OSC values
ip = "127.0.0.1"
port = 3001

address_face = "/mediapipe/face"
address_pose = "/mediapipe/pose"
address_lefthand = "/mediapipe/lefthand"
address_righthand = "/mediapipe/righthand"

flag_face = True
flag_pose = True
flag_lefthand = True
flag_righthand = True

# Create OSC client for sending data via OSC
client = udp_client.SimpleUDPClient(ip, port)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(1)
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0,
    static_image_mode=False,
    refine_face_landmarks=False) as holistic:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue
    
    image.flags.writeable = False  
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    
    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Draw rectangle over camera input
    cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), (0, 0, 0), -1)

    if flag_face: mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_tesselation_style())
        
    if flag_pose: mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())
        
    if flag_lefthand: mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_hand_landmarks_style())
        
    if flag_righthand: mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_hand_landmarks_style())
    
    # Get results for face, pose, left hand, right hand and send as OSC data
    if results.face_landmarks and flag_face:
        face = results.face_landmarks
        message_builder = OscMessageBuilder(address=address_face)
        for landmark in face.landmark:
            message_builder.add_arg(landmark.x)
            message_builder.add_arg(landmark.y)
            message_builder.add_arg(landmark.z)
        message = message_builder.build()
        client.send(message)
        
    if results.pose_landmarks and flag_pose:
        pose = results.pose_landmarks
        message_builder = OscMessageBuilder(address=address_pose)
        for landmark in pose.landmark:
            message_builder.add_arg(landmark.x)
            message_builder.add_arg(landmark.y)
            message_builder.add_arg(landmark.z)
        message = message_builder.build()
        client.send(message)
    
    if results.left_hand_landmarks and flag_lefthand:
        lefthand = results.left_hand_landmarks
        message_builder = OscMessageBuilder(address=address_lefthand)
        for landmark in lefthand.landmark:
            message_builder.add_arg(landmark.x)
            message_builder.add_arg(landmark.y)
            message_builder.add_arg(landmark.z)
        message = message_builder.build()
        client.send(message)
        
    if results.right_hand_landmarks and flag_righthand:
        righthand = results.right_hand_landmarks
        message_builder = OscMessageBuilder(address=address_righthand)
        for landmark in righthand.landmark:
            message_builder.add_arg(landmark.x)
            message_builder.add_arg(landmark.y)
            message_builder.add_arg(landmark.z)
        message = message_builder.build()
        client.send(message)

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Holistic OSC', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()