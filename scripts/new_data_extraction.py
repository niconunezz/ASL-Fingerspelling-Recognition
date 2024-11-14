import cv2
import mediapipe as mp

"""
    data structure:
     {
        - row = frame,
        - columns :
            - right_hand_landmarks_1_x
            - ...
            - right_hand_landmarks_1_z
            - ...
            - right_hand_landmarks_21_x
            - ...
            - right_hand_landmarks_21_z
            - left_hand_landmarks_1_x
            - ...
            - left_hand_landmarks_21_z
     }
"""

def detect_sign_language(frame):
    # Initialize MediaPipe Hands
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    mp_hands = mp.solutions.hands
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.65
    )
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.65,
        ) as hands:
        # Convert the BGR image to RGB
        
    # Process the frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)
        print("kfhsbdfihdsjk")



    
    hands_landmarks = []
    if results.multi_hand_landmarks: # If hands are detected

        for landmarks in results.multi_hand_landmarks: # For each hand
            kpoints = []
            for point in landmarks.landmark:
                kpoints.append({
                    'x': point.x,
                    'y': point.y,
                    'z': point.z
                })
            hands_landmarks.append(kpoints)
    
    hands.close()
    return frame, hands

def main():

    cols = [f"{hand}_{i}_{dim}" for hand in ["right_hand", "left_hand"] for i in range(22) for dim in ['x', 'y', 'z']]
    dic = {col: [] for col in cols}
    # Read a video
    video_path = 'data/videos/69532.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return 
        
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("End of video stream")
                break
        
            frame, hand_landamarks = detect_sign_language(frame)
            
            # cv2.imshow('Sign Language Detection', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
                
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()