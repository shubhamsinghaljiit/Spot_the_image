import cv2
import numpy as np

def preprocess_image(image_path, target_size=(256, 256)):
    """
    Extracts crops without resizing to preserve sub-pixel screen grids.
    Pads small images using reflection to avoid hard boundary artifacts.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    h, w = img.shape[:2]
    
    # Center crop if large enough
    if h >= target_size[0] and w >= target_size[1]:
        sy = (h - target_size[0]) // 2
        sx = (w - target_size[1]) // 2
        img_crop = img[sy:sy+target_size[0], sx:sx+target_size[1]]
    else:
        # Reflect pad if too small (preserves textures, prevents resizing artifacts)
        pad_y = max(0, target_size[0] - h)
        pad_x = max(0, target_size[1] - w)
        img_crop = cv2.copyMakeBorder(
            img, pad_y//2, pad_y - pad_y//2, pad_x//2, pad_x - pad_x//2,
            cv2.BORDER_REFLECT_101
        )
    
    img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    ycbcr = cv2.cvtColor(img_crop, cv2.COLOR_BGR2YCrCb)
    
    return img_rgb, gray, ycbcr