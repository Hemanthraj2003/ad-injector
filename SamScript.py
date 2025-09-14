import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ---------------------- Global Variables ----------------------
clicked_point = None
frame_display = None
predictor = None
mask = None
original_frame = None  # Store original frame without any overlays

# ---------------------- Mouse Callback ------------------------
def mouse_callback(event, x, y, flags, param):
    global clicked_point, frame_display, predictor, mask, original_frame
    
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = np.array([[x, y]])
        print(f"Clicked at point: ({x}, {y})")
        
        # Reset to original frame before new selection
        frame_display = original_frame.copy()
        
        # Run prediction
        masks, scores, logits = predictor.predict(
            point_coords=clicked_point,
            point_labels=np.array([1]),  # 1 = foreground
            multimask_output=True
        )
        
        # Select best mask
        best_mask_idx = np.argmax(scores)
        mask = masks[best_mask_idx].astype(bool)
        
        # Overlay mask (green semi-transparent) and keep it visible
        frame_display[mask] = frame_display[mask] * 0.6 + np.array([0, 255, 0]) * 0.4  # Apply to main frame_display
        
        # Draw clicked point
        cv2.circle(frame_display, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow("SAM2 Interactive Selection", frame_display)
        print(f"Object segmented (Score: {scores[best_mask_idx]:.3f})")
        print("Press 'S' to overlay logo on selected area")

# ---------------------- Logo Overlay Function -----------------
def overlay_logo(frame, mask, logo_path):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        print("Mask empty! Cannot overlay logo.")
        return
    
    # Get bounding box of selected area
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    selection_width = x_max - x_min
    selection_height = y_max - y_min
    
    # Load logo
    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    if logo is None:
        print(f"Could not load logo from {logo_path}")
        return
    
    logo_height, logo_width = logo.shape[:2]
    
    # Calculate scaling to fit within selection while preserving aspect ratio
    scale_x = selection_width / logo_width
    scale_y = selection_height / logo_height
    scale = min(scale_x, scale_y)  # Use smaller scale to fit within bounds
    
    # Calculate new logo dimensions
    new_width = int(logo_width * scale)
    new_height = int(logo_height * scale)
    
    # Resize logo maintaining aspect ratio
    logo_resized = cv2.resize(logo, (new_width, new_height))
    
    # Center the logo within the selection
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    start_x = center_x - new_width // 2
    start_y = center_y - new_height // 2
    end_x = start_x + new_width
    end_y = start_y + new_height
    
    # Ensure logo stays within frame bounds
    start_x = max(0, start_x)
    start_y = max(0, start_y)
    end_x = min(frame.shape[1], end_x)
    end_y = min(frame.shape[0], end_y)
    
    # Adjust logo size if it was clipped
    actual_width = end_x - start_x
    actual_height = end_y - start_y
    if actual_width != new_width or actual_height != new_height:
        logo_resized = cv2.resize(logo, (actual_width, actual_height))
    
    # Apply logo (fully opaque, no transparency)
    if logo_resized.shape[2] == 4:  # Has alpha channel
        logo_rgb = logo_resized[:, :, :3]
        logo_alpha = logo_resized[:, :, 3] / 255.0
        
        roi = frame[start_y:end_y, start_x:end_x]
        
        # Alpha blending for proper logo overlay
        for c in range(3):
            roi[:, :, c] = (logo_alpha * logo_rgb[:, :, c] + (1 - logo_alpha) * roi[:, :, c]).astype(np.uint8)
        
        frame[start_y:end_y, start_x:end_x] = roi
    else:  # No alpha channel, direct overlay
        frame[start_y:end_y, start_x:end_x] = logo_resized
    
    # Update the main display window instead of creating a new one
    cv2.imshow("SAM2 Interactive Selection", frame)
    cv2.imwrite("frame_with_logo.png", frame)
    print(f"Logo overlaid (scaled to {new_width}x{new_height}) and saved to frame_with_logo.png")

# ---------------------- Load SAM2 Model -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
sam2_checkpoint = "sam2_hiera_small.pt"   # your SAM2 checkpoint
model_cfg = "sam2_hiera_s.yaml"          # your SAM2 config
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)

# ---------------------- Load Frame ----------------------------
frame = cv2.imread("frames/frame_00085.png")
original_frame = frame.copy()  # Store clean original
frame_display = frame.copy()
predictor.set_image(frame)

# ---------------------- Setup Window --------------------------
cv2.namedWindow("SAM2 Interactive Selection", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("SAM2 Interactive Selection", mouse_callback)

print("Click on an object to segment and overlay logo. Press 'q' to quit, 'r' to reset, 's' to apply logo.")

while True:
    cv2.imshow("SAM2 Interactive Selection", frame_display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        frame_display = original_frame.copy()  # Reset to clean original
        cv2.imshow("SAM2 Interactive Selection", frame_display)
        mask = None  # Clear current selection
        print("Reset to original frame")
    elif key == ord('s'):
        if mask is not None:
            # Reset to original frame first, then apply logo (removes green overlay)
            frame_display = original_frame.copy()
            overlay_logo(frame_display, mask, "logo.png")
            print("Logo applied to main frame")
        else:
            print("No object selected! Click on an object first.")

cv2.destroyAllWindows()
