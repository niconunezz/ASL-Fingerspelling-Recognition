import cv2
import mediapipe as mp

"""
    data structure:
     {
        - frame,
        - right_hand_landmarks
            - 1
                - x
                - y
                - z
            - ...
                - x
                - y
                - z
            - 21
                - x
                - y
                - z
        - left_hand_landmarks
            - 1
                - x
                - y
                - z
            - ...
                - x
                - y
                - z
            - 21
                - x
                - y
                - z
     }
"""

def detect_sign_language(frame):
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.65
    )
    
    # Process the frame
    results = hands.process(frame)
    
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
            
            cv2.imshow('Sign Language Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()