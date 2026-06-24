import cv2
import os
def extract_frames(name , data_dir = "SYSTEM/Data"):
    video_path = os.path.join(data_dir, name)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    base_dir = os.path.dirname(video_path)
    out_dir = os.path.join(base_dir, video_name)
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0
    saved_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % round(fps) == 0:  # 1 frame per second
#       if frame_idx % max(1, round(fps / 10)) == 0:     for 10 frames per second   
            out_path = os.path.join(out_dir, f"frame_{saved_idx:04d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_idx += 1
        frame_idx += 1
    cap.release()
    print(f"[{video_name}] Saved {saved_idx} frames to {out_dir}/")
if __name__ == "__main__" :
    extract_frames("x.mp4")
    
    # this creates the folder as per te file name for the frames