import cv2
import mediapipe as mp
import pandas as pd
import os
import time
import subprocess



def detect_sign_language(frame, data, path, hands, debug = False):
    
    t0 = time.time()
    
        
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)
    t1 = time.time()
    if debug:
        print(f"model loading: {(t1-t0)*1000:.2f} ms")
    

    t0 = time.time()
    if results.multi_hand_landmarks: 
        assert len(results.multi_hand_landmarks) <= 3, f"More than 2 hands detected in {path}, got {len(results.multi_hand_landmarks)}"
        if len(results.multi_hand_landmarks) == 3:
            results.multi_hand_landmarks = results.multi_hand_landmarks[:2]

        if len(results.multi_hand_landmarks) == 2:
            hand_types = ["right", "left"]
        
        else:
            hand_types = ["right"]
        

        data["video_id"].append(path)
      
        mhand_landmarks = results.multi_hand_landmarks
        if len(results.multi_hand_landmarks) == 1:
            for point, n in zip(mhand_landmarks[0].landmark, [i for i in range(21)]):
                data[f"left_hand_{n}_x"].append(None)
                data[f"left_hand_{n}_y"].append(None)
                data[f"left_hand_{n}_z"].append(None)

                data[f"right_hand_{n}_x"].append(point.x)
                data[f"right_hand_{n}_y"].append(point.y)
                data[f"right_hand_{n}_z"].append(point.z)
        else:

            for landmarks, hand_type in zip(mhand_landmarks, hand_types): # For each hand
            
                assert len(landmarks.landmark) == 21, "Not all landmarks detected"
                
                for point, n in zip(landmarks.landmark, [i for i in range(21)]):

                    data[f"{hand_type}_hand_{n}_x"].append(point.x)
                    data[f"{hand_type}_hand_{n}_y"].append(point.y)
                    data[f"{hand_type}_hand_{n}_z"].append(point.z)

        
    t1 = time.time()
    if debug:
        print(f"processing: {(t1-t0)*1000:.2f} ms")

    return data

def extract_video(path, data, hands):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print("Error: Could not open video file")
        return 
    
    
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("End of video stream")
            return data
        
        
        data = detect_sign_language(frame, data, path, hands)



def extract_file(data, videos, index, hands):
    
    t0 = time.time()
    for path in videos:
        data = extract_video(f"data/videos/{path}", data, hands)

        
    df = pd.DataFrame(data)
        # TODO: must make video_id index
    df.to_parquet(f"data/files/{index}.parquet", index=False)
        

    t1 = time.time()
    print(f"Time taken: {t1-t0}")




def main():

    cols = [f"{hand}_{i}_{dim}" for hand in ["right_hand", "left_hand"] for i in range(21) for dim in ['x', 'y', 'z']]
    cols.append("video_id")
    data = {col: [] for col in cols}
    
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.65,
            )

    vid_per_file = 5

    videos = os.listdir("data/videos")
    indexes = [i for i in range(0, len(videos),vid_per_file)]

    for index, interval in enumerate(indexes):

        extract_file(data, videos[interval:interval+vid_per_file], index, hands)

    
    
if __name__ == "__main__":
    main()