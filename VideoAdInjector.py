import cv2
import numpy as np
import torch
import os
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

# ---------------------- Global Variables ----------------------
video_predictor = None
inference_state = None
frame_idx = 0
all_frames = []
frame_names = []
mask = None
original_frame = None
current_frame_display = None
tracking_active = False
ann_obj_id = 1  # Object ID for tracking

# ---------------------- Mouse Callback for Initial Selection ----------------------
def mouse_callback(event, x, y, flags, param):
    global video_predictor, inference_state, frame_idx, mask, original_frame, current_frame_display, tracking_active, ann_obj_id
    
    if event == cv2.EVENT_LBUTTONDOWN and not tracking_active:
        clicked_point = np.array([[x, y]])
        print(f"Clicked at point: ({x}, {y}) on frame {frame_idx}")
        
        # Add point to the first frame for tracking
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=ann_obj_id,
            points=clicked_point,
            labels=np.array([1]),  # 1 = foreground point
        )
        
        # Get the mask for visualization
        mask = (out_mask_logits[0] > 0.0).cpu().numpy()
        
        # Ensure mask has correct dimensions and shape
        if len(mask.shape) == 3:
            mask = mask.squeeze()  # Remove extra dimensions
        
        # Check if mask dimensions match frame dimensions
        frame_height, frame_width = original_frame.shape[:2]
        if mask.shape != (frame_height, frame_width):
            # Resize mask to match frame dimensions
            mask = cv2.resize(mask.astype(np.uint8), (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
        
        # Convert to boolean
        mask = mask.astype(bool)
        
        # Show selection on current frame
        current_frame_display = original_frame.copy()
        current_frame_display[mask] = current_frame_display[mask] * 0.6 + np.array([0, 255, 0]) * 0.4
        cv2.circle(current_frame_display, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow("SAM2 Video Object Selection", current_frame_display)
        
        print(f"Object selected! Press 'T' to start tracking, 'R' to reset selection")

# ---------------------- Logo Overlay Function ----------------------
def overlay_logo_on_mask(frame, mask, logo_path):
    """Apply logo to the masked area of a frame"""
    # Ensure mask has correct dimensions
    if len(mask.shape) == 3:
        mask = mask.squeeze()  # Remove extra dimensions
    
    # Check if mask dimensions match frame dimensions
    frame_height, frame_width = frame.shape[:2]
    if mask.shape != (frame_height, frame_width):
        # Resize mask to match frame dimensions
        mask = cv2.resize(mask.astype(np.uint8), (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(bool)
    
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return frame
    
    # Get bounding box
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    selection_width = x_max - x_min
    selection_height = y_max - y_min
    
    # Load and resize logo
    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    if logo is None:
        print(f"Could not load logo from {logo_path}")
        return frame
    
    logo_height, logo_width = logo.shape[:2]
    
    # Calculate scaling while preserving aspect ratio
    scale_x = selection_width / logo_width
    scale_y = selection_height / logo_height
    scale = min(scale_x, scale_y)
    
    new_width = int(logo_width * scale)
    new_height = int(logo_height * scale)
    
    # Resize logo
    logo_resized = cv2.resize(logo, (new_width, new_height))
    
    # Center the logo
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    start_x = center_x - new_width // 2
    start_y = center_y - new_height // 2
    end_x = start_x + new_width
    end_y = start_y + new_height
    
    # Ensure bounds
    start_x = max(0, start_x)
    start_y = max(0, start_y)
    end_x = min(frame.shape[1], end_x)
    end_y = min(frame.shape[0], end_y)
    
    # Adjust if clipped
    actual_width = end_x - start_x
    actual_height = end_y - start_y
    if actual_width != new_width or actual_height != new_height:
        logo_resized = cv2.resize(logo, (actual_width, actual_height))
    
    # Apply logo with alpha blending
    frame_with_logo = frame.copy()
    if len(logo_resized.shape) == 3 and logo_resized.shape[2] == 4:  # Has alpha
        logo_rgb = logo_resized[:, :, :3]
        logo_alpha = logo_resized[:, :, 3] / 255.0
        
        roi = frame_with_logo[start_y:end_y, start_x:end_x]
        for c in range(3):
            roi[:, :, c] = (logo_alpha * logo_rgb[:, :, c] + (1 - logo_alpha) * roi[:, :, c]).astype(np.uint8)
        frame_with_logo[start_y:end_y, start_x:end_x] = roi
    else:
        frame_with_logo[start_y:end_y, start_x:end_x] = logo_resized
    
    return frame_with_logo

# ---------------------- Process All Frames (Serial Processing) ----------------------
def process_all_frames(logo_path, output_dir="processed_video"):
    global video_predictor, inference_state, frame_names, ann_obj_id
    
    if not tracking_active:
        print("No tracking active! Please select an object first.")
        return
    
    print("Processing all frames with object tracking (serial processing)...")
    
    # Create organized output directory structure
    timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = f"{output_dir}_{timestamp}"
    frames_output_dir = os.path.join(output_base_dir, "frames_with_logo")
    
    os.makedirs(frames_output_dir, exist_ok=True)
    print(f"📁 Created output directory: {output_base_dir}/")
    print(f"📁 Frames will be saved to: {frames_output_dir}/")
    
    # Get all original frame files
    frames_dir = "frames"
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    total_frames = len(frame_files)
    
    print(f"🔄 Serial processing {total_frames} frames (memory efficient)...")
    
    # Propagate masks through all frames first
    print("🎯 Propagating object tracking through all frames...")
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    
    # Process each frame serially (one at a time)
    processed_count = 0
    for frame_idx, frame_file in enumerate(frame_files):
        # Load frame only when needed
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"⚠️  Skipping {frame_file} - could not load")
            continue
        
        # Check if we have a mask for this frame
        if frame_idx in video_segments and ann_obj_id in video_segments[frame_idx]:
            # Get mask for this frame
            mask = video_segments[frame_idx][ann_obj_id]
            
            # Apply logo to frame
            frame_with_logo = overlay_logo_on_mask(frame, mask, logo_path)
            
            # Save processed frame
            output_path = os.path.join(frames_output_dir, frame_file)
            cv2.imwrite(output_path, frame_with_logo)
            processed_count += 1
            
            # Show progress
            if processed_count % 10 == 0:
                print(f"📈 Processed {processed_count}/{total_frames} frames... ({processed_count/total_frames*100:.1f}%)")
        
        # Free memory for this frame
        del frame
        if 'frame_with_logo' in locals():
            del frame_with_logo
        
        # Clear GPU cache periodically
        if frame_idx % 20 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"✅ Serial processing complete! {processed_count} frames saved to {frames_output_dir}/")
    
    # Create a summary file
    summary_path = os.path.join(output_base_dir, "processing_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"SAM2 Video Ad Injection Summary\n")
        f.write(f"=" * 40 + "\n")
        f.write(f"Processing Date: {timestamp}\n")
        f.write(f"Processing Mode: Serial (Memory Efficient)\n")
        f.write(f"Total Frames Processed: {processed_count}\n")
        f.write(f"Logo Used: {logo_path}\n")
        f.write(f"Output Directory: {frames_output_dir}\n")
        f.write(f"Object ID Tracked: {ann_obj_id}\n")
    
    print(f"📋 Summary saved to: {summary_path}")
    
    # Ask if user wants to create video from frames
    print(f"\n🎬 Would you like to create a video from the processed frames?")
    print(f"Press 'V' to create video, or any other key to skip...")
    
    return output_base_dir

# ---------------------- Create Video from Frames ----------------------
def create_video_from_frames(frames_dir, output_path="processed_video.mp4", fps=30):
    """Create an MP4 video from processed frames"""
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    
    if len(frame_files) == 0:
        print("No frames found to create video!")
        return False
    
    # Read first frame to get dimensions
    first_frame_path = os.path.join(frames_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    height, width, _ = first_frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"🎬 Creating video: {output_path}")
    print(f"📐 Video dimensions: {width}x{height}")
    print(f"🎞️ Frame rate: {fps} FPS")
    
    # Write frames to video
    for i, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)
        
        # Show progress
        if (i + 1) % 30 == 0:
            print(f"📹 Added {i + 1}/{len(frame_files)} frames to video...")
    
    video_writer.release()
    print(f"✅ Video created successfully: {output_path}")
    return True

# ---------------------- Main Setup ----------------------
def main():
    global video_predictor, inference_state, frame_idx, all_frames, frame_names
    global original_frame, current_frame_display, tracking_active, mask
    
    print("🎬 SAM2 Video Ad Injector (Serial Processing)")
    print("=" * 50)
    
    # Load SAM2 video model
    print("Loading SAM2 video model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check GPU memory if using CUDA
    if device == "cuda":
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        print(f"GPU Memory: {gpu_memory:.2f} GB")
        
        # If GPU memory is low, use CPU
        if gpu_memory < 6:
            print("⚠️  Low GPU memory detected. Switching to CPU for better stability.")
            device = "cpu"
    
    print(f"Using device: {device}")
    
    sam2_checkpoint = "sam2_hiera_small.pt"
    model_cfg = "sam2_hiera_s.yaml"
    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    
    # Get frame info without loading all frames
    print("Scanning frames for serial processing...")
    frames_dir = "frames"
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    total_frames = len(frame_files)
    
    print(f"📊 Found {total_frames} frames for processing")
    
    # Load only first few frames for preview and selection (memory efficient)
    preview_frames = min(10, total_frames)  # Load max 10 frames for preview
    all_frames = []
    frame_names = []
    
    # Create temporary JPEG directory for SAM2 compatibility
    temp_jpeg_dir = "temp_jpeg_frames"
    os.makedirs(temp_jpeg_dir, exist_ok=True)
    
    print(f"Converting first {preview_frames} frames to JPEG for preview and selection...")
    for i in range(preview_frames):
        frame_file = frame_files[i]
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            # Resize frames to reduce memory usage
            height, width = frame.shape[:2]
            if width > 1024:  # Resize if too large
                scale = 1024 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
                print(f"Resized preview frame {i+1} from {width}x{height} to {new_width}x{new_height}")
            
            all_frames.append(frame)
            frame_names.append(frame_file)
            
            # Save as JPEG with zero-padded naming for SAM2
            jpeg_filename = f"{i:05d}.jpg"
            jpeg_path = os.path.join(temp_jpeg_dir, jpeg_filename)
            cv2.imwrite(jpeg_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    # Convert ALL frames to JPEG for full processing (but don't load into memory)
    print(f"Converting all {total_frames} frames to JPEG format for SAM2...")
    for i in range(preview_frames, total_frames):
        frame_file = frame_files[i]
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            # Resize if needed
            height, width = frame.shape[:2]
            if width > 1024:
                scale = 1024 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Save as JPEG
            jpeg_filename = f"{i:05d}.jpg"
            jpeg_path = os.path.join(temp_jpeg_dir, jpeg_filename)
            cv2.imwrite(jpeg_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # Immediately free memory
            del frame
        
        # Show progress for large conversions
        if (i + 1) % 50 == 0:
            print(f"🔄 Converted {i + 1}/{total_frames} frames to JPEG...")
    
    print(f"✅ Loaded {len(all_frames)} preview frames, converted {total_frames} total frames")
    
    if len(all_frames) == 0:
        print("No frames found! Make sure frames are in the 'frames/' directory")
        return
    
    # Initialize video inference with frames
    print("Initializing SAM2 video inference...")
    try:
        # Clear GPU cache before initialization
        if device == "cuda":
            torch.cuda.empty_cache()
            
        inference_state = video_predictor.init_state(video_path=temp_jpeg_dir)
        print("✅ Video inference state initialized successfully!")
        
    except torch.cuda.OutOfMemoryError:
        print("❌ CUDA out of memory! Switching to CPU...")
        device = "cpu"
        video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
        inference_state = video_predictor.init_state(video_path=temp_jpeg_dir)
        print("✅ Video inference state initialized on CPU!")
    
    # Setup initial display
    frame_idx = 0
    mask = None  # Initialize mask variable
    original_frame = all_frames[frame_idx].copy()
    current_frame_display = original_frame.copy()
    
    # Create window
    cv2.namedWindow("SAM2 Video Object Selection", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("SAM2 Video Object Selection", mouse_callback)
    
    print("\n🎯 Instructions:")
    print("1. Click on an object to select it (use preview frames)")
    print("2. Press 'T' to start tracking through all frames")
    print("3. Press 'P' to process ALL frames with logo (serial processing)")
    print("4. Press 'V' to create video from processed frames")
    print("5. Press 'N'/'B' to go to next/previous preview frame")
    print("6. Press 'R' to reset selection")
    print("7. Press 'Q' to quit")
    print(f"\n📊 Preview: {len(all_frames)} frames loaded, Total: {total_frames} frames available for processing")
    
    # Main loop
    last_output_dir = None
    while True:
        cv2.imshow("SAM2 Video Object Selection", current_frame_display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset selection
            try:
                if device == "cuda":
                    torch.cuda.empty_cache()
                inference_state = video_predictor.init_state(video_path="temp_jpeg_frames")
                original_frame = all_frames[frame_idx].copy()
                current_frame_display = original_frame.copy()
                tracking_active = False
                mask = None
                print("Selection reset")
            except torch.cuda.OutOfMemoryError:
                print("❌ GPU memory issue during reset. Try closing and restarting the script.")
        elif key == ord('t') and mask is not None:
            # Start tracking
            tracking_active = True
            print(f"🎯 Tracking activated! Object will be tracked across all {total_frames} frames.")
            print("🔄 Propagating tracking to preview frames...")
            
            # Update current frame display to show tracking is active
            current_frame_display = original_frame.copy()
            current_frame_display[mask] = current_frame_display[mask] * 0.6 + np.array([0, 255, 0]) * 0.4
            # Add tracking indicator
            cv2.putText(current_frame_display, "TRACKING ACTIVE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("SAM2 Video Object Selection", current_frame_display)
        elif key == ord('p') and tracking_active:
            # Process all frames (serial processing)
            print(f"🚀 Starting serial processing of all {total_frames} frames...")
            last_output_dir = process_all_frames("logo.png")
            print(f"All frames processed and saved to {last_output_dir}/")
        elif key == ord('v') and last_output_dir:
            # Create video from processed frames
            frames_dir = os.path.join(last_output_dir, "frames_with_logo")
            video_output_path = os.path.join(last_output_dir, "final_video_with_ads.mp4")
            create_video_from_frames(frames_dir, video_output_path, fps=30)
        elif key == ord('n'):
            # Next frame (preview only)
            if frame_idx < len(all_frames) - 1:
                frame_idx += 1
                original_frame = all_frames[frame_idx].copy()
                current_frame_display = original_frame.copy()
                
                # If tracking is active, show that tracking is enabled
                if tracking_active:
                    cv2.putText(current_frame_display, "TRACKING ACTIVE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(current_frame_display, f"Press 'P' to process all {total_frames} frames", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                print(f"Preview Frame {frame_idx + 1}/{len(all_frames)} (Total frames: {total_frames})")
        elif key == ord('b'):
            # Previous frame (preview only)
            if frame_idx > 0:
                frame_idx -= 1
                original_frame = all_frames[frame_idx].copy()
                current_frame_display = original_frame.copy()
                
                # If tracking is active, show that tracking is enabled
                if tracking_active:
                    cv2.putText(current_frame_display, "TRACKING ACTIVE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(current_frame_display, f"Press 'P' to process all {total_frames} frames", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                print(f"Preview Frame {frame_idx + 1}/{len(all_frames)} (Total frames: {total_frames})")
    
    cv2.destroyAllWindows()
    
    # Cleanup temporary JPEG directory
    import shutil
    if os.path.exists("temp_jpeg_frames"):
        shutil.rmtree("temp_jpeg_frames")
        print("🧹 Cleaned up temporary JPEG files")
    
    print("👋 Video processing complete!")

if __name__ == "__main__":
    main()
