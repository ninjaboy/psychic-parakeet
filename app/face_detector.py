"""
Face detection module using face_recognition library (dlib-based ML).
Provides accurate face detection and landmark extraction.
"""

import face_recognition
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import cv2


class FaceDetector:
    """ML-based face detector using dlib's HOG + CNN models."""

    def __init__(self, model: str = "hog"):
        """
        Initialize face detector.

        Args:
            model: Detection model - "hog" (faster, CPU) or "cnn" (more accurate, GPU)
        """
        self.model = model

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect all faces in an image.

        Args:
            image: RGB numpy array of the image

        Returns:
            List of face locations as (top, right, bottom, left) tuples
        """
        # face_recognition expects RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        face_locations = face_recognition.face_locations(image, model=self.model)
        return face_locations

    def get_face_landmarks(self, image: np.ndarray, face_location: Tuple[int, int, int, int]) -> Optional[dict]:
        """
        Get facial landmarks for a detected face.

        Args:
            image: RGB numpy array
            face_location: (top, right, bottom, left) tuple

        Returns:
            Dictionary of facial landmarks or None
        """
        landmarks_list = face_recognition.face_landmarks(image, [face_location])
        if landmarks_list:
            return landmarks_list[0]
        return None

    def get_face_encoding(self, image: np.ndarray, face_location: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Get 128-dimensional face encoding for face matching.

        Args:
            image: RGB numpy array
            face_location: (top, right, bottom, left) tuple

        Returns:
            128-dimensional numpy array encoding
        """
        encodings = face_recognition.face_encodings(image, [face_location])
        if encodings:
            return encodings[0]
        return None

    def extract_face(self, image: np.ndarray, face_location: Tuple[int, int, int, int],
                     padding: float = 0.3) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Extract face region from image with padding.

        Args:
            image: RGB numpy array
            face_location: (top, right, bottom, left) tuple
            padding: Padding ratio around face (0.3 = 30% extra on each side)

        Returns:
            Tuple of (face_image, padded_location)
        """
        top, right, bottom, left = face_location
        height, width = image.shape[:2]

        # Calculate padding
        face_height = bottom - top
        face_width = right - left

        pad_h = int(face_height * padding)
        pad_w = int(face_width * padding)

        # Apply padding with bounds checking
        new_top = max(0, top - pad_h)
        new_bottom = min(height, bottom + pad_h)
        new_left = max(0, left - pad_w)
        new_right = min(width, right + pad_w)

        face_image = image[new_top:new_bottom, new_left:new_right]

        return face_image, (new_top, new_right, new_bottom, new_left)


def create_face_mask(face_image: np.ndarray, landmarks: Optional[dict] = None) -> np.ndarray:
    """
    Create an elliptical mask for smooth face blending.

    Args:
        face_image: Face region as numpy array
        landmarks: Optional facial landmarks for precise masking

    Returns:
        Grayscale mask image
    """
    height, width = face_image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    # Create elliptical mask
    center = (width // 2, height // 2)
    axes = (int(width * 0.45), int(height * 0.48))

    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    # Apply Gaussian blur for smooth edges
    mask = cv2.GaussianBlur(mask, (21, 21), 11)

    return mask
