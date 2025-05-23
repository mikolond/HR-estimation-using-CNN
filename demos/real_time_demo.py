import cv2
import threading
import time
import numpy as np
from queue import Queue
from face_extractor import FaceExtractor
from inference import Inferencer
import os
import matplotlib.pyplot as plt

fe = FaceExtractor()

# Queue for sending frames between threads
frame_queue1 = Queue(maxsize=2)
face_queue = Queue(maxsize=300)

shared_data = {'bb': None,'quit': False, "frame":None, "hr_data":None}
shared_data_lock = threading.Lock()





def extract_face_thread_func(shared_data: dict):
    quit = False
    frame = None
    bb = None
    while True:
        with shared_data_lock:
            frame = shared_data['frame']
            quit = shared_data['quit']
            shared_data['bb'] = bb

        if frame is not None:
            # Process the frame to extract
            face, bb = fe.extract_face(frame, return_bb=True)
        if quit:
            break


def hr_predict_thread_func(face_queue: Queue, shared_data: dict, extractor_model_path: str = "extractor.pth", estimator_model_path: str = "estimator.pth"):
    quit = False
    inferener = Inferencer()
    inferener.load_extractor_weights(extractor_model_path)
    inferener.load_estimator_weights(estimator_model_path)
    face_array = []
    extractor_output_array = np.array([])
    hr_data_array = []
    hr_out = None
    while True:
        if len(hr_data_array) >= 3:
            hr_data_array = hr_data_array[-3:]
        hr_out = np.mean(hr_data_array, axis=0)

        with shared_data_lock:
            quit = shared_data['quit']
            shared_data['hr_data'] = hr_out

        if quit:
            break
        while not face_queue.empty():
            face = face_queue.get()
            face_array.append(face)
        if len(face_array) > 0:
            extractor_output = inferener.extract(np.array(face_array))
            extractor_output_array = np.append(extractor_output_array, extractor_output)
            face_array = []
        if len(extractor_output_array) >=300:
            extractor_output_array = extractor_output_array[-300:]
            fig = plt.figure(figsize=(10, 5))
            plt.plot(extractor_output_array, label='HR Signal')
            plt.title('HR Signal')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.savefig(os.path.join("output", "hr_signal.png"))
            plt.close(fig)
            estimator_output = inferener.estimate(extractor_output_array)
            print("HR:", estimator_output)
            hr_data_array.append(estimator_output.squeeze())
        
        if quit:
            break


def capture_thread_func(queue: Queue, shared_data: dict, fps: int = 30):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Resolution: {int(width)}x{int(height)}")
    quit = False
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            continue

        # Save or enqueue frame
        if not queue.full():
            queue.put(frame)
        with shared_data_lock:
            quit = shared_data['quit']
        if quit:
            break
    cap.release()

def process_thread_func(queue1: Queue, face_queue:Queue, shared_data: dict):
    frame = None
    bb = None
    quit = False
    hr_data = None

    while True:
        new_frame = False
        if not queue1.empty():
            frame = queue1.get()  # Wait for a frame
            with shared_data_lock:
                shared_data['frame'] = frame
            new_frame = True
        else:
            continue
        
        with shared_data_lock:
            bb = shared_data['bb']
            quit = shared_data['quit']
            hr_data = shared_data['hr_data']
        if bb is not None:
            if new_frame:
                face = frame[bb[0][1]:bb[1][1], bb[0][0]:bb[1][0]]
                face = cv2.resize(face, (128, 192))
                # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = np.array(face)
                if not face_queue.full():
                    face_queue.put(face)
                else:
                    print("Face queue is full, dropping frame")
                cv2.rectangle(frame, bb[0], bb[1], (0, 255, 0), 2)
        if hr_data is not None:
            print("hr_data:", hr_data, end="\r")
            cv2.putText(frame, f"HR: {hr_data:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) == ord('q'):
            with shared_data_lock:
                shared_data['quit'] = True
                quit = True
        if quit:
            break
        




    cv2.destroyAllWindows()


extractor_weights_path = os.path.join("weights", "extractor_weights.pth")
estimator_weights_path = os.path.join("weights", "estimator_weights.pth")

# Start capture thread
capture_thread = threading.Thread(target=capture_thread_func, args=(frame_queue1, shared_data))
capture_thread.daemon = True
capture_thread.start()

# Start processing thread
process_thread = threading.Thread(target=process_thread_func, args=(frame_queue1,face_queue, shared_data))
process_thread.daemon = True
process_thread.start()

face_extract_thread = threading.Thread(target=extract_face_thread_func, args=(shared_data,))
face_extract_thread.daemon = True
face_extract_thread.start() 

hr_predict_thread = threading.Thread(target=hr_predict_thread_func, args=(face_queue, shared_data, extractor_weights_path, estimator_weights_path))
hr_predict_thread.daemon = True
hr_predict_thread.start()



# Clean exit
capture_thread.join()
process_thread.join()
face_extract_thread.join()
hr_predict_thread.join()
print("Exiting...")