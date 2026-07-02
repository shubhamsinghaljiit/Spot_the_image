import cv2
import numpy as np
import pywt
from scipy.fftpack import fft2, fftshift
from skimage.feature import local_binary_pattern
from preprocessing import preprocess_image

def extract_fft_robust(gray):
    """Measures energy concentration of high-frequency spikes (Moiré)."""
    f = fft2(gray)
    f_shift = fftshift(f)
    mag = np.abs(f_shift)
    
    h, w = mag.shape
    mag[h//2-2:h//2+3, w//2-2:w//2+3] = 0 # Remove DC component
    
    mag_flat = mag.flatten()
    threshold = np.percentile(mag_flat, 99)
    top_1_energy = np.sum(mag_flat[mag_flat >= threshold])
    total_energy = np.sum(mag_flat) + 1e-5
    
    return [top_1_energy / total_energy, np.std(mag)]

def extract_srm_noise(gray):
    """High-pass filter to catch print halftones and screen pixel gaps."""
    kernel = np.array([[-1,  2, -1],
                       [ 2, -4,  2],
                       [-1,  2, -1]], dtype=np.float32) / 4.0
    residual = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    return [np.mean(np.abs(residual)), np.var(residual), np.percentile(np.abs(residual), 95)]

def extract_chroma_moire(ycbcr):
    """Detects rainbow-colored aliases caused by screen matrices."""
    _, cr, cb = cv2.split(ycbcr)
    cr_var = np.var(cv2.Laplacian(cr, cv2.CV_64F))
    cb_var = np.var(cv2.Laplacian(cb, cv2.CV_64F))
    return [cr_var, cb_var]

def extract_lbp_features(gray):
    """Rotation-invariant texture mapping."""
    radius = 2 
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='nri_uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 60), density=True)
    return hist.tolist()

def extract_glare_and_color(img):
    """Catches backlight clipping (screens) and matte reflections (prints)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = hsv[:,:,2]
    clipping_ratio = np.sum(v > 250) / v.size
    s = hsv[:,:,1]
    return [clipping_ratio, np.mean(s), np.std(s), np.std(v)]

def extract_all_features(image_path):
    img_rgb, gray, ycbcr = preprocess_image(image_path)
    
    features = []
    features.extend(extract_fft_robust(gray))
    features.extend(extract_srm_noise(gray))
    features.extend(extract_chroma_moire(ycbcr))
    features.extend(extract_lbp_features(gray))
    features.extend(extract_glare_and_color(img_rgb))
    
    coeffs = pywt.dwt2(gray, 'db2')
    LL, (LH, HL, HH) = coeffs
    features.extend([np.var(LH), np.var(HL), np.var(HH)])
    
    return np.array(features)
