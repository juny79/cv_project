
import cv2, numpy as np

def deskew(img_gray):
    edges = cv2.Canny(img_gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    if lines is None: return img_gray
    angles = [np.degrees(theta) - 90 for rho, theta in lines[:,0]]
    angle = np.median(angles)
    h, w = img_gray.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img_gray, M, (w, h), borderValue=255)
