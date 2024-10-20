import cv2

video_path = "test_videos/0.1_20_10.avi"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video stream or file")

else:
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frames per second: {fps}")

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Width: {width}, Height: {height}")
    frame_c = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f"Frame count: {frame_c}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        print("Frame ", cap.get(cv2.CAP_PROP_POS_FRAMES), end="\r")
        print("frame shape: ", frame.shape)
        a = input()
    cap.release()

    cv2.destroyAllWindows()
