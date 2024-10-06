import cv2
import numpy as np
import math
import argparse
import os


def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="A simple script using argparse.")
    
    # Add arguments
    parser.add_argument("f", type=float, help="simulated HR frequency in Hz")
    parser.add_argument("f_s", type=int, help="sampling frequency in Hz")
    parser.add_argument("length", type=int, help="length of the video in seconds")
    parser.add_argument("--save_path", help="Path to where you want to save the file", default="test_videos/")
    parser.add_argument("--file_name", help="The name of the saved file", default="unknown")

    # Parse the arguments
    args = parser.parse_args()

    print(f"HR frequency: {args.f}, sampling frequency: {args.f_s}, length: {args.length}")


    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec
    if args.save_path != "test_videos/":
        if args.file_name == "unknown":
            save_path = args.save_path + str(args.f) + "_" + str(args.f_s) + "_" + str(args.length) + ".avi"
        else:
            save_path = args.save_path + args.file_name + ".avi"
    else:
        # id test_videos doesnt exist create it
        if not os.path.exists("test_videos/"):
            os.makedirs("test_videos/")
        if args.file_name == "unknown":
            save_path = "test_videos/" + str(args.f) + "_" + str(args.f_s) + "_" + str(args.length) + ".avi"
        else:
            save_path = "test_videos/" + args.file_name + ".avi"
    out = cv2.VideoWriter(save_path, fourcc, args.f_s, (640, 480))
    # create txt file with HR Frequency in bps
    text_save_path = save_path.replace(".avi", ".txt")
    f = open(text_save_path, "w")

    c = 2 * np.pi * args.f  / args.f_s 
    amplitude = 50
    for i in range(args.f_s * args.length):
        static_array = np.array([[[100 + math.sin(i*c)*amplitude, 100 + math.sin(i*c)*amplitude, 0]] * 640] * 480, dtype=np.uint8) 
        static_image = cv2.cvtColor(static_array, cv2.COLOR_BGR2RGB)
        out.write(static_image)
        f.write(str(args.f) + "\n")
        print("Frame ", i, end="\r")

    # Release everything when done
    out.release()
    print("Video saved!")
    f.close()





main()