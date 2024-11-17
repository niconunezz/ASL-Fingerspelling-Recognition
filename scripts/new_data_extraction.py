import cv2
import mediapipe as mp
import pandas as pd
import os
import time
import numpy as np



def detect_sign_language(frame, data, path, hands, video_id, fnum, debug = False):
    
    t0 = time.time()
            
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)
    t1 = time.time()
    if debug:
        print(f"model processing: {(t1-t0)*1000:.2f} ms")
    

    
    if results.multi_hand_landmarks: 
        assert len(results.multi_hand_landmarks) <= 3, f"More than 2 hands detected in {path}, got {len(results.multi_hand_landmarks)}"

        t0 = time.time()

        if len(results.multi_hand_landmarks) == 3:
            results.multi_hand_landmarks = results.multi_hand_landmarks[:2]

        if len(results.multi_hand_landmarks) == 2:
            hand_types = ["right", "left"]
        
        else:
            hand_types = ["right"]

        t1 = time.time()
        if debug:
            print(f"hand type determination: {(t1-t0)*1000:.2f} ms")
        
        
        data["video_id"].append(video_id)
        data["frame"].append(fnum)
        fnum += 1
        mhand_landmarks = results.multi_hand_landmarks

        t0 = time.time()
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
            print(f"landmark extraction: {(t1-t0)*1000:.2f} ms")


    return data, fnum

def extract_video(path, data, hands):
    cap = cv2.VideoCapture(path)

    video_id = path.split("/")[-1].split(".")[0]
    if not cap.isOpened():
        print("Error: Could not open video file")
        return 
    
    
    counter = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("End of video stream")
            return data
        
        data, counter = detect_sign_language(frame, data, path, hands, video_id, fnum = counter, debug=False)



def extract_file(data, videos, index, hands, debug = False):
    
    t0 = time.time()
    for path in videos:
        t2 = time.time()
        data = extract_video(f"data/videos/{path}", data, hands)
        t3 = time.time()
        if debug:
            print(f"Video extraction took: {(t3-t2)*1000:.2f} ms")

    
    df = pd.DataFrame(data)
    df = df.set_index("video_id")

    
    df.to_parquet(f"data/tfiles/{index}.parquet", index=True)
    
    

    t1 = time.time()
    print(f"Time taken: {t1-t0}")




def main():

    cols = [f"{hand}_{i}_{dim}" for hand in ["right_hand", "left_hand"] for i in range(21) for dim in ['x', 'y', 'z']]
    cols.append("video_id")
    cols.append("frame")
    data = {col: [] for col in cols}
    
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity = 1,
            min_detection_confidence=0.65,
            )

    vid_per_file = 1000

    videos = os.listdir("data/videos")
    indexes = [i for i in range(0, len(videos),vid_per_file)]

    for index, interval in enumerate(indexes):
        extract_file(data, videos[interval:interval+vid_per_file], index, hands, False)

    
    
if __name__ == "__main__":
    main()