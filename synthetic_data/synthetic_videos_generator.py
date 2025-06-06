import cv2
import numpy as np
import math
import argparse
import os
import random

WIDTH = 128
HEIGHT = 192

HR_FREQ_MIN = 20
HR_FREQ_MAX = 220

HR_FREQ_START = 60
HR_FREQ_END = 120

AMPLITUDE = 3

DEBUG = False


def draw_rectangle(x,y,w,h,frame, color=(0, 255, 0)):
    '''
    Draw rectangle on the frame
    params: x : int - center of the recantgle
            y : int - center of the recantgle
            w : int
            h : int
            frame : np.array
            color : tuple
    return: frame : np.array
    '''
    x1 = x - w//2
    x2 = x + w//2
    y1 = y - h//2
    y2 = y + h//2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)

    return frame

def create_video(save_path, file_name, f, f_s, length, start_frequency, end_frequency, slope, amplitude):
    #load bajt.mp4
    cap = cv2.VideoCapture("bajt.mp4")

    print(f"HR frequency: {f}, sampling frequency: {f_s}, length: {length}")
    base_color = [180, 80, 100]
    base_color += np.random.randint(-20,20,3)


    fourcc = cv2.VideoWriter_fourcc(*'I420')  # Codec -> Uncompressed format
    if save_path != "test_videos/":
        if file_name == "unknown":
            save_path = save_path + str(f) + "_" + str(f_s) + "_" + str(length) + ".avi"
        else:
            save_path = save_path + file_name + ".avi"
    else:
        # id test_videos doesnt exist create it
        if not os.path.exists("test_videos/"):
            os.makedirs("test_videos/")
        if file_name == "unknown":
            save_path = "test_videos/" + str(f) + "_" + str(f_s) + "_" + str(length) + ".avi"
        else:
            save_path = "test_videos/" + file_name + ".avi"
    out = cv2.VideoWriter(save_path, fourcc, f_s, (WIDTH, HEIGHT))
    # create txt file with HR Frequency in bps
    text_save_path = save_path.replace(".avi", ".txt")
    f_txt = open(text_save_path, "w")
    if f == 0:
        f_array = []
        # generate random HR frequency for args_f_s * length samples
        hr_freq = start_frequency
        increasing = True
        
        for i in range(f_s * length):
            if i % 10 == 0:
                if increasing:
                    hr_freq += slope
                    # hr_freq += random.uniform(-0.5, 0.5)  # introduce subtle randomness
                    if hr_freq >= end_frequency:
                        hr_freq = end_frequency
                        increasing = False
                else:
                    hr_freq -= slope
                    # hr_freq += random.uniform(-0.5, 0.5)  # introduce subtle randomness
                    if hr_freq <= start_frequency:
                        hr_freq = start_frequency
                        increasing = True
            f_array.append(hr_freq)



    amplitude = amplitude
    # center position of the frame:
    x = WIDTH//2
    y = HEIGHT//2
    width = 50
    height = 80


    sin_ic_array = []
    if f != 0:
        c = 2 * np.pi * f / 60 / f_s 
        for i in range(f_s * length):
            #load frame from bajt.mp4
            ret, frame = cap.read()
            # resize frame
            frame = cv2.resize(frame, (WIDTH, HEIGHT))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not ret:
                break
            # draw rectangle on the frame
            r_rand, g_rand, b_rand = np.random.rand(3)*0.1 +1
            color = (base_color[0] + math.sin(i*c)*amplitude*r_rand, base_color[1] - math.sin(i*c)*amplitude*g_rand, base_color[2] - math.sin(i*c)*amplitude*b_rand)
            sin_ic_array.append(math.sin(i*c)*amplitude * r_rand)
            x_new = x  + np.random.randint(-20,20)
            y_new = y + np.random.randint(-20, 20)
            frame = draw_rectangle(x_new, y_new, width, height, frame, color)
            # static_array = np.array([[[100 + math.sin(i*c)*amplitude, 100 + math.sin(i*c)*amplitude, 0]] * 640] * 480, dtype=np.uint8) 
            static_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.write(static_image)
            f_txt.write(str(f) + "\n")
            print("Frame ", i, end="\r")
    elif f == 0:
        frame_count = 0
        phase = 0  # Initialize phase
        for i in range(f_s * length):
            if frame_count == 0:
                c = 2 * np.pi * f_array[i] / 60 / f_s
                period_frames = int(f_s * 60 / f_array[i])
                frame_count = period_frames
            #load frame from bajt.mp4
            ret, frame = cap.read()
            # resize frame
            frame = cv2.resize(frame, (WIDTH, HEIGHT))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not ret:
                break
            # draw rectangle on the frame
            r_rand, g_rand, b_rand = np.random.rand(3)*0.1 +1
            color = (base_color[0] + math.sin(phase)*amplitude*r_rand, base_color[1] - math.sin(phase)*amplitude*g_rand, base_color[2] - math.sin(phase)*amplitude*b_rand)
            sin_ic_array.append(math.sin(phase)*amplitude*r_rand)
            x_new = x  + np.random.randint(-20,20)
            y_new = y + np.random.randint(-20, 20)
            frame = draw_rectangle(x_new, y_new, width, height, frame, color)
            # static_array = np.array([[[100 + math.sin(i*c)*amplitude, 100 + math.sin(i*c)*amplitude, 0]] * 640] * 480, dtype=np.uint8) 
            static_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.write(static_image)
            f_txt.write(str(int(f_array[i])) + "\n")
            print("Frame ", i, end="\r")
            
            phase += c  # Increment phase smoothly
            frame_count -= 1

    if DEBUG:
        from matplotlib import pyplot as plt
        plt.plot(sin_ic_array)
        plt.savefig("sin_ic_array.png")
        plt.show()


    # Release everything when done
    out.release()
    print("Video saved!")
    f_txt.close()

def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Synthetic data generator for hr estimatoin.")
    
    # Add arguments
    parser.add_argument("f", type=int, help="simulated HR frequency in BPM, of 0, the frequency will change in time")
    parser.add_argument("f_s", type=int, help="sampling frequency in Hz")
    parser.add_argument("length", type=int, help="length of the video in seconds")
    parser.add_argument("--save_path", help="Path to where you want to save the file", default="test_videos/")
    parser.add_argument("--file_name", help="The name of the saved file", default="unknown")
    parser.add_argument("--start_frequency", type=int, help="Start frequency for random HR frequency", default=HR_FREQ_START)
    parser.add_argument("--end_frequency", type=int, help="End frequency for random HR frequency", default=HR_FREQ_END)
    parser.add_argument("--slope", type=float, help="Change in HR frequency per frame for random frequency", default=0.5)
    parser.add_argument("--amplitude", type=int, help="Amplitude of the HR frequency", default=AMPLITUDE)

    # Parse the arguments
    args = parser.parse_args()

    create_video(args.save_path, args.file_name, args.f, args.f_s, args.length, args.start_frequency, args.end_frequency, args.slope, args.amplitude)


if __name__ == "__main__":
    main()

    

