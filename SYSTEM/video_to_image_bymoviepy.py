from moviepy.editor import VideoFileClip
import os
def extract_frames(name, data_dir="SYSTEM/Data", frames_per_second=1):
    video_path = os.path.join(data_dir, name)
    if not os.path.exists(video_path):
        print(f"{video_path} not found")
        return None
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    base_dir = os.path.dirname(video_path)
    out_dir = os.path.join(base_dir, video_name)
    os.makedirs(out_dir, exist_ok=True)
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        # Calculate total frames and time interval
        total_frames_to_save = int(duration * frames_per_second)
        interval = 1.0 / frames_per_second
        print(f"[{video_name}] Duration: {duration:.2f}s, Extracting {frames_per_second} frame(s)/sec")
        saved_idx = 0
        for i in range(total_frames_to_save):
            t = i * interval
            # Safety check to not exceed video duration
            if t >= duration:
                break
            out_path = os.path.join(out_dir, f"frame_{saved_idx:04d}.jpg")
            clip.save_frame(out_path, t=t)
            saved_idx += 1
            # Progress tracking
            if saved_idx % 10 == 0:
                progress = (t / duration) * 100
                print(f"Progress: {progress:.1f}% ({saved_idx} frames saved)")
        clip.close()
        print(f"[{video_name}] Saved {saved_idx} frames to {out_dir}/")
        return out_dir
    except Exception as e:
        print(f"Error processing {name}: {str(e)}")
        if 'clip' in locals():
            clip.close()
        return None
def extract_frames_batch(video_names, data_dir="SYSTEM/Data", **kwargs):
    results = {}
    for name in video_names:
        result = extract_frames(name, data_dir, **kwargs)
        results[name] = result
    return results
if __name__ == "__main__":
    # Single video
    extract_frames("x.mp4")
    # Or batch processing
    videos = ["video1.mp4", "video2.mp4", "video3.mp4"]
    extract_frames_batch(videos, frames_per_second=5)