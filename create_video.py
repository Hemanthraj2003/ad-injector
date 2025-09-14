#!/usr/bin/env python3
"""
Simple Video Creator from Frames
Creates an MP4 video from PNG frames in the frames/ directory
"""

import cv2
import os
import argparse
from pathlib import Path

def create_video_from_frames(frames_dir="frames", output_path="output_video.mp4", fps=30, quality="high"):
    """
    Create an MP4 video from frames
    
    Args:
        frames_dir (str): Directory containing frame images
        output_path (str): Output video file path
        fps (int): Frames per second for the output video
        quality (str): Video quality - "high", "medium", "low"
    """
    
    print(f"🎬 Creating video from frames in '{frames_dir}/'")
    
    # Check if frames directory exists
    if not os.path.exists(frames_dir):
        print(f"❌ Error: Directory '{frames_dir}' not found!")
        return False
    
    # Get all frame files
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(frame_files) == 0:
        print(f"❌ Error: No image files found in '{frames_dir}'!")
        return False
    
    print(f"📊 Found {len(frame_files)} frames")
    print(f"🎞️ Frame rate: {fps} FPS")
    print(f"📈 Quality: {quality}")
    
    # Read first frame to get dimensions
    first_frame_path = os.path.join(frames_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    
    if first_frame is None:
        print(f"❌ Error: Could not read first frame '{first_frame_path}'")
        return False
    
    height, width, channels = first_frame.shape
    print(f"📐 Video dimensions: {width}x{height}")
    
    # Set up video codec and quality
    if quality == "high":
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        bitrate = None  # Use default high quality
    elif quality == "medium":
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        bitrate = None
    else:  # low quality
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        bitrate = None
    
    # Create video writer
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print(f"❌ Error: Could not open video writer for '{output_path}'")
        return False
    
    print(f"🚀 Creating video: {output_path}")
    
    # Process each frame
    for i, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"⚠️  Warning: Could not read frame '{frame_file}', skipping...")
            continue
        
        # Resize frame if dimensions don't match (shouldn't happen with consistent frames)
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        
        # Write frame to video
        video_writer.write(frame)
        
        # Show progress
        if (i + 1) % 30 == 0 or (i + 1) == len(frame_files):
            progress = (i + 1) / len(frame_files) * 100
            print(f"📹 Progress: {i + 1}/{len(frame_files)} frames ({progress:.1f}%)")
    
    # Release video writer
    video_writer.release()
    
    # Get file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    duration = len(frame_files) / fps
    
    print(f"✅ Video created successfully!")
    print(f"📁 Output: {output_path}")
    print(f"💾 File size: {file_size:.2f} MB")
    print(f"⏱️  Duration: {duration:.2f} seconds")
    print(f"🎞️ Total frames: {len(frame_files)}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Create MP4 video from image frames")
    parser.add_argument("--frames", "-f", default="frames", help="Directory containing frames (default: frames)")
    parser.add_argument("--output", "-o", default="output_video.mp4", help="Output video file (default: output_video.mp4)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument("--quality", choices=["high", "medium", "low"], default="high", help="Video quality (default: high)")
    
    args = parser.parse_args()
    
    print("🎬 Video Creator from Frames")
    print("=" * 40)
    
    success = create_video_from_frames(
        frames_dir=args.frames,
        output_path=args.output,
        fps=args.fps,
        quality=args.quality
    )
    
    if success:
        print("\n🎉 Video creation completed successfully!")
    else:
        print("\n❌ Video creation failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
