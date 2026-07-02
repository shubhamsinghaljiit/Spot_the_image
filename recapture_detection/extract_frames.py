import cv2
import os
import glob

def process_videos_in_folder(folder_path, frames_per_video=5):
    # Look for common video formats (both lowercase and uppercase extensions)
    video_extensions = ['*.mp4', '*.mov', '*.avi', '*.MP4', '*.MOV', '*.AVI']
    video_paths = []
    
    for ext in video_extensions:
        video_paths.extend(glob.glob(os.path.join(folder_path, ext)))

    if not video_paths:
        print(f"No videos found in {folder_path}. Skipping...")
        return

    print(f"Found {len(video_paths)} videos in {folder_path}. Extracting {frames_per_video} images per video...")

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open {video_path}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            continue

        # Calculate intervals to get evenly spaced frames throughout the video
        step = max(1, total_frames // frames_per_video)
        base_name = os.path.splitext(video_path)[0]

        frame_count = 0
        extracted = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Capture a frame when we hit the calculated interval
            if frame_count % step == 0 and extracted < frames_per_video:
                image_name = f"{base_name}_frame_{extracted}.jpg"
                cv2.imwrite(image_name, frame)
                extracted += 1

            frame_count += 1

        cap.release()
        
        # Delete the original video file to save hard drive space
        os.remove(video_path)
        
    print(f"✅ Finished processing {folder_path}. Videos deleted, images saved.")

if __name__ == "__main__":
    print("Starting video frame extraction...")
    
    # Process BOTH folders
    print("\n--- Checking REAL folder ---")
    process_videos_in_folder("dataset/real", frames_per_video=5)
    
    print("\n--- Checking SCREEN folder ---")
    process_videos_in_folder("dataset/screen", frames_per_video=5)
    
    print("\nDone! Both folders now contain only static images.")
    print("You are completely ready to run: python3 train.py")