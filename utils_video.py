import os
import cv2
import wget
import shutil

frame_dir = "frames"

def download_video(video_url):
    try:
        video_name = wget.download(video_url)
    except:
        print("Error download")
    return video_name

def get_frame(video_path,time_per_frame=1):
    frames = []
    video_name = os.path.basename(video_path).split('.mp4')[0]
    shutil.rmtree("./frames")
    os.mkdir("frames")
    if not os.path.exists(os.path.join(frame_dir, video_name)):
        os.mkdir(os.path.join(frame_dir, video_name))

    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(5) * time_per_frame  # fps : num frames per second
    i = 1
    while cap.isOpened() and i <= 30:
        frame_id = cap.get(1)  # current frame number
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % int(frame_rate) == 0:
            file_name = os.path.join(frame_dir, video_name) + "/" + 'frame_' + str(i) + '.jpg'
            cv2.imwrite(file_name, frame)

            # convert BGR (opencv) to RGB (pil - model use)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            i = i + 1

    os.remove(video_path)
    return frames

def process(url_video):
    video_name = download_video(url_video)
    frames = get_frame(video_name)
    return frames

