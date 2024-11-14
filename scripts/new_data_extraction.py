import cv2
import mediapipe as mp
import pandas as pd


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

def detect_sign_language(frame, data):
    # Initialize MediaPipe Hands
    
    mp_hands = mp.solutions.hands
   
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        ) as hands:
        # Convert the BGR image to RGB
        
    # Process the frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)

    
    if results.multi_hand_landmarks: # If hands are detected
        assert len(results.multi_hand_landmarks) <= 2, "More than 2 hands detected"

        if len(results.multi_hand_landmarks) == 2:
            hand_types = ["right", "left"]
        else:
            hand_types = ["right"]
            

        for landmarks, hand_type in zip(results.multi_hand_landmarks, hand_types): # For each hand
            

            assert len(landmarks.landmark) == 21, "Not all landmarks detected"
            
            for point, n in zip(landmarks.landmark, [i for i in range(21)]):
                if len(hand_types) == 1:
                    
                    data[f"left_hand_{n}_x"].append("nan")
                    data[f"left_hand_{n}_y"].append("nan")
                    data[f"left_hand_{n}_z"].append("nan")

                    data[f"{hand_type}_hand_{n}_x"].append(point.x)
                    data[f"{hand_type}_hand_{n}_y"].append(point.y)
                    data[f"{hand_type}_hand_{n}_z"].append(point.z)
                        
                else:
                    data[f"{hand_type}_hand_{n}_x"].append(point.x)
                    data[f"{hand_type}_hand_{n}_y"].append(point.y)
                    data[f"{hand_type}_hand_{n}_z"].append(point.z)

                    
             
        
    return data

def main():

    cols = [f"{hand}_{i}_{dim}" for hand in ["right_hand", "left_hand"] for i in range(21) for dim in ['x', 'y', 'z']]
    data = {col: [] for col in cols}
    
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
        
            data = detect_sign_language(frame, data)
            # print(data)
            
            
                
    finally:
        
        df = pd.DataFrame(data)

        df.head(5)
        df.to_csv("data/69532.csv", index=False)
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()