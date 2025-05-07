import cv2
import threading
import time
from face_extractor import FaceExtractor

class VideoCaptureProcessor:
    def __init__(self, fps=30, num_processing_threads=2):
        self.fps = fps
        self.frame = None
        self.frame_lock = threading.Lock()
        self.frame_queue = []
        self.fe = FaceExtractor()
        self.process_frames_event = threading.Event()
        self.num_processing_threads = num_processing_threads

    def capture_thread_func(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open webcam")
            return

        frame_time = 1.0 / self.fps
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Save the frame in a thread-safe way
            with self.frame_lock:
                self.frame = frame.copy()

            # Wait to match desired fps
            elapsed = time.time() - start_time
            delay = frame_time - elapsed
            if delay > 0:
                time.sleep(delay)

        cap.release()

    def process_thread_func(self, thread_id):
        while True:
            if self.process_frames_event.is_set():
                with self.frame_lock:
                    if self.frame is not None:
                        frame = self.frame.copy()  # Get the frame to process

                # Example processing: Face detection
                face, bb = self.fe.extract_face(frame, return_bb=True)
                if face is not None:
                    # Process the bounding box (example: just print it)
                    print(f"Thread-{thread_id}: Found face at {bb}")

    def start(self):
        # Start the capture thread
        capture_thread = threading.Thread(target=self.capture_thread_func)
        capture_thread.daemon = True
        capture_thread.start()

        # Start the processing threads
        for i in range(self.num_processing_threads):
            process_thread = threading.Thread(target=self.process_thread_func, args=(i,))
            process_thread.daemon = True
            process_thread.start()

        # Main loop to control processing
        print("Press 'p' to toggle processing, 'q' to quit.")
        while True:
            key = input().strip().lower()
            if key == 'q':
                break
            elif key == 'p':
                if self.process_frames_event.is_set():
                    self.process_frames_event.clear()
                    print("Processing paused.")
                else:
                    self.process_frames_event.set()
                    print("Processing started.")

        # Allow threads to finish
        capture_thread.join(timeout=1)
        for process_thread in threading.enumerate():
            if process_thread is not capture_thread:
                process_thread.join(timeout=1)

if __name__ == "__main__":
    video_processor = VideoCaptureProcessor(fps=30, num_processing_threads=2)
    video_processor.start()
