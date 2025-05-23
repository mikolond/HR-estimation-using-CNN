from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import cv2
import threading
import time
import numpy as np
from queue import Queue
from face_extractor import FaceExtractor
from inference import Inferencer
import os
import matplotlib.pyplot as plt
import argparse
import yaml

fe = FaceExtractor()



def overlay_rppg_signal(signal: np.ndarray,
                        frame: np.ndarray,
                        history_len: int = 300,
                        plot_size: tuple = (200, 100),
                        position: tuple = (10, 10),
                        alpha: float = 0.8) -> np.ndarray:
    """
    Overlay an rPPG signal plot onto a video frame with tight borders.

    Args:
        signal (np.ndarray): 1D array of length `history_len` containing rPPG values.
        frame (np.ndarray): BGR image onto which the plot will be drawn.
        history_len (int): Expected length of `signal`. Defaults to 300.
        plot_size (tuple): Width and height in pixels of the overlay plot.
        position (tuple): (x_offset, y_offset) from top-left corner of the frame.
        alpha (float): Blending factor for the overlay (0.0 transparent through 1.0 opaque).

    Returns:
        np.ndarray: The same frame with the rPPG plot blended into it.
    """
    w, h = plot_size
    fig = Figure(figsize=(w / 100, h / 100), dpi=100)
    canvas = FigureCanvas(fig)

    # Create axes that fill the figure tightly
    ax = fig.add_axes([0, 0, 1, 1])  # left, bottom, width, height (all relative to figure size)

    # Plot signal
    ax.plot(signal, color='lime', linewidth=1)
    ax.set_xlim(0, history_len - 1)
    ax.set_ylim(np.min(signal), np.max(signal))

    # Remove ticks, labels, and borders
    ax.axis('off')

    # Render the plot to RGB image
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape((h, w, 3))
    plot_bgr = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)

    # Overlay on frame
    fh, fw = frame.shape[:2]
    x_off, y_off = position
    x_end = min(x_off + w, fw)
    y_end = min(y_off + h, fh)

    roi = frame[y_off:y_end, x_off:x_end]
    plot_crop = plot_bgr[0:(y_end - y_off), 0:(x_end - x_off)]

    # Blend
    cv2.addWeighted(plot_crop, alpha, roi, 1 - alpha, 0, roi)

    return frame
# Queue for sending frames between threads
frame_queue1 = Queue(maxsize=2)
face_queue = Queue(maxsize=400)

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
    alpha = 0.98
    inferencer = Inferencer(extractor_path=extractor_model_path, estimator_path=estimator_model_path)
    inferencer.load_extractor_weights(extractor_weights_path)
    inferencer.load_estimator_weights(estimator_weights_path)
    inferencer.set_device_auto()
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

        if len(face_array) > 300:
            extractor_output = inferencer.extract(np.array(face_array))
            extractor_output_array = np.append(extractor_output_array, extractor_output)
            face_array = []

        if len(extractor_output_array) >=300:
            batch_to_estimate = []
            if len(extractor_output_array) > 450:
                extractor_output_array = extractor_output_array[-450:]
            rPPG_out = extractor_output_array[-300:]
            num_of_estimates = len(extractor_output_array) - 300
            real_number_of_estimates = 0
            for i in range(0,num_of_estimates, 10):
                real_number_of_estimates += 1
                if i + 300 > len(extractor_output_array):
                    break
                batch_to_estimate.append(extractor_output_array[i:i+300])
            batch_to_estimate = np.array(batch_to_estimate)
            estimates = inferencer.estimate(batch_to_estimate, real_number_of_estimates)
            if real_number_of_estimates > 0:
                hr_data_array.append(np.mean(estimates))
        
        if quit:
            break


def capture_thread_func(queue: Queue, shared_data: dict, fps: int = 30):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
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
        # print("FPS:", fps, end="\r")
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
    rPPG_signal = None
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
            rPPG_signal = shared_data['rPPG_signal']
        if rPPG_signal is not None:
            cv2.putText(frame, "Extracted rPPG signal:", (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frame = overlay_rppg_signal(rPPG_signal, frame, history_len=300, plot_size=(400, 200), position=(10, 720-210), alpha=0.8)
        if fps is not None and fps > 0:
            # print("FPS:", fps)
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
            # print("hr_data:", hr_data, end="\r")
            cv2.putText(frame, f"HR: {hr_data:.0f} bpm", (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame,"Press 'q' or ESC to quit", (950, 720-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.imshow('Camera', frame)
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:  # 27 is the ESC key
            with shared_data_lock:
                shared_data['quit'] = True
            quit = True
        if quit:
            break
        




    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time heart rate detection")
    parser.add_argument("config_path", type=str, help="Path to the config file", default=None)
    args = parser.parse_args()
    if args.config_path is None:
        raise ValueError("No config path provided")
    # Load configuration file
    if args.config_path is not None:
        config_path = args.config_path
        if not os.path.exists(config_path):
            raise ValueError(f"Config file {config_path} does not exist")
        else:
            config_path = args.config_path

    config_data = yaml.safe_load(open(config_path, 'r'))
    models = config_data["models"]
    weights = config_data["weights"]


    extractor_model_path = models["extractor_model"]
    estimator_model_path = models["estimator_model"]


    extractor_weights_path = weights["extractor_weights"]
    estimator_weights_path = weights["estimator_weights"]

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