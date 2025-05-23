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

shared_data = {'bb': None,'quit': False, "frame":None, "hr_data":None, "fps": 0, "rPPG_signal":None}
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


def hr_predict_thread_func(face_queue: Queue, shared_data: dict, extractor_model_path: str = "extractor.py", estimator_model_path: str = "estimator.py", extractor_weights_path: str = "extractor.pth", estimator_weights_path: str = "estimator.pth"):
    quit = False
    alpha = 0.93
    inferener = Inferencer(extractor_path=extractor_model_path, estimator_path=estimator_model_path)
    inferener.load_extractor_weights(extractor_weights_path)
    inferener.load_estimator_weights(estimator_weights_path)
    face_array = []
    extractor_output_array = np.array([])
    hr_data_array = []
    hr_out = None
    rPPG_out = None
    while True:
        if len(hr_data_array) > 0 and hr_out is None:
            hr_out = hr_data_array[-1]
            hr_data_array = []
        elif len(hr_data_array) > 0:
            hr_out = alpha * hr_out + (1 - alpha) * hr_data_array[-1]
            hr_data_array = []

        with shared_data_lock:
            quit = shared_data['quit']
            shared_data['hr_data'] = hr_out
            shared_data['rPPG_signal'] = rPPG_out

        if quit:
            break
        while not face_queue.empty():
            face = face_queue.get()
            face_array.append(face)
        if len(face_array) > 100:
            extractor_output = inferener.extract(np.array(face_array))
            extractor_output_array = np.append(extractor_output_array, extractor_output)
            face_array = []
        if len(extractor_output_array) >=300:
            extractor_output_array = extractor_output_array[-300:]
            rPPG_out = extractor_output_array
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
    fps_out = None
    alpha = 0.999
    desired_fps = 30
    
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            continue

        # Save or enqueue frame
        if not queue.full():
            queue.put(frame)
        with shared_data_lock:
            quit = shared_data['quit']
            shared_data[fps] = fps_out
        if quit:
            break
        loop_time = time.time() - start_time
        fps = 1 / (loop_time+0.00001)
        if fps > desired_fps:
            time.sleep(1 / desired_fps - loop_time)
            fps = 30
        print("FPS:", fps, end="\r")
        if fps_out is None:
            fps_out = fps
        else:
            fps_out = alpha * fps_out + (1 - alpha) * fps
        

    cap.release()

def process_thread_func(queue1: Queue, face_queue:Queue, shared_data: dict):
    frame = None
    bb = None
    quit = False
    hr_data = None
    fps = 0

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
            fps = shared_data['fps']
        if fps is not None and fps > 0:
            print("FPS:", fps)
            hr_data = hr_data * fps / 30
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
extractor_model_path = os.path.join("Models", "extractor_improved.py")
estimator_model_path = os.path.join("Models", "estimator_model_smaller.py")

extractor_weights_path = os.path.join("weights", "best_extractor_weights.pth")
estimator_weights_path = os.path.join("weights", "best_estimator_weights.pth")

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

hr_predict_thread = threading.Thread(target=hr_predict_thread_func, args=(face_queue, shared_data,extractor_model_path, estimator_model_path, extractor_weights_path, estimator_weights_path))
hr_predict_thread.daemon = True
hr_predict_thread.start()



# Clean exit
capture_thread.join()
process_thread.join()
face_extract_thread.join()
hr_predict_thread.join()
print("Exiting...")