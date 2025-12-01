"""
Face replacement module.
Combines face detection and image processing to swap faces in images/GIFs.
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
from .face_detector import FaceDetector, create_face_mask


class FaceReplacer:
    """Handles face replacement in images using seamless cloning."""

    def __init__(self, model: str = "hog"):
        """
        Initialize face replacer.

        Args:
            model: Face detection model ("hog" or "cnn")
        """
        self.detector = FaceDetector(model=model)
        self.source_face: Optional[np.ndarray] = None
        self.source_landmarks: Optional[dict] = None

    def set_source_face(self, image: np.ndarray) -> bool:
        """
        Set the source face image to use for replacement.

        Args:
            image: RGB numpy array with a face

        Returns:
            True if face was found and set, False otherwise
        """
        faces = self.detector.detect_faces(image)
        if not faces:
            return False

        # Use the largest face
        largest_face = max(faces, key=lambda f: (f[2] - f[0]) * (f[1] - f[3]))
        self.source_face, _ = self.detector.extract_face(image, largest_face, padding=0.4)
        self.source_landmarks = self.detector.get_face_landmarks(image, largest_face)

        return True

    def replace_faces_in_frame(self, frame: np.ndarray, blend_strength: float = 0.9) -> np.ndarray:
        """
        Replace all detected faces in a frame with the source face.

        Args:
            frame: RGB numpy array of the frame
            blend_strength: How strongly to blend (0-1)

        Returns:
            Frame with faces replaced
        """
        if self.source_face is None:
            return frame

        result = frame.copy()
        faces = self.detector.detect_faces(frame)

        for face_location in faces:
            result = self._replace_single_face(result, face_location, blend_strength)

        return result

    def _replace_single_face(self, frame: np.ndarray, face_location: Tuple[int, int, int, int],
                             blend_strength: float) -> np.ndarray:
        """
        Replace a single face in the frame.

        Args:
            frame: RGB numpy array
            face_location: (top, right, bottom, left) tuple
            blend_strength: Blending strength

        Returns:
            Frame with face replaced
        """
        if self.source_face is None:
            return frame

        top, right, bottom, left = face_location
        face_height = bottom - top
        face_width = right - left

        # Resize source face to match target
        resized_source = cv2.resize(
            self.source_face,
            (face_width, face_height),
            interpolation=cv2.INTER_LANCZOS4
        )

        # Create mask for blending
        mask = create_face_mask(resized_source)

        # Calculate center point for seamless cloning
        center = (left + face_width // 2, top + face_height // 2)

        # Ensure the face fits within frame bounds
        if (center[0] < face_width // 2 or center[0] >= frame.shape[1] - face_width // 2 or
            center[1] < face_height // 2 or center[1] >= frame.shape[0] - face_height // 2):
            # Fallback to simple alpha blending
            return self._simple_blend(frame, resized_source, mask, top, left, blend_strength)

        try:
            # Use seamless cloning for natural blending
            result = cv2.seamlessClone(
                resized_source,
                frame,
                mask,
                center,
                cv2.NORMAL_CLONE
            )

            # Blend with original based on strength
            if blend_strength < 1.0:
                result = cv2.addWeighted(frame, 1 - blend_strength, result, blend_strength, 0)

            return result
        except cv2.error:
            # Fallback to simple blending
            return self._simple_blend(frame, resized_source, mask, top, left, blend_strength)

    def _simple_blend(self, frame: np.ndarray, face: np.ndarray, mask: np.ndarray,
                      top: int, left: int, blend_strength: float) -> np.ndarray:
        """
        Simple alpha blending fallback.

        Args:
            frame: Target frame
            face: Face to blend
            mask: Blending mask
            top, left: Position in frame
            blend_strength: Blend strength

        Returns:
            Blended frame
        """
        result = frame.copy()
        h, w = face.shape[:2]

        # Bounds checking
        frame_h, frame_w = frame.shape[:2]
        end_y = min(top + h, frame_h)
        end_x = min(left + w, frame_w)
        start_y = max(0, top)
        start_x = max(0, left)

        face_start_y = start_y - top
        face_start_x = start_x - left
        face_end_y = face_start_y + (end_y - start_y)
        face_end_x = face_start_x + (end_x - start_x)

        if end_y <= start_y or end_x <= start_x:
            return result

        # Get regions
        face_region = face[face_start_y:face_end_y, face_start_x:face_end_x]
        mask_region = mask[face_start_y:face_end_y, face_start_x:face_end_x]
        frame_region = result[start_y:end_y, start_x:end_x]

        # Normalize mask
        mask_normalized = (mask_region / 255.0 * blend_strength)[:, :, np.newaxis]

        # Blend
        blended = (face_region * mask_normalized + frame_region * (1 - mask_normalized))
        result[start_y:end_y, start_x:end_x] = blended.astype(np.uint8)

        return result

    def process_gif_frames(self, frames: List[np.ndarray], blend_strength: float = 0.9,
                           progress_callback=None) -> List[np.ndarray]:
        """
        Process all frames of a GIF, replacing faces.

        Args:
            frames: List of RGB numpy arrays
            blend_strength: Blending strength
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of processed frames
        """
        processed_frames = []

        for i, frame in enumerate(frames):
            processed = self.replace_faces_in_frame(frame, blend_strength)
            processed_frames.append(processed)

            if progress_callback:
                progress_callback(i + 1, len(frames))

        return processed_frames


class AdvancedFaceReplacer(FaceReplacer):
    """
    Advanced face replacer with additional features:
    - Face tracking across frames
    - Color matching
    - Expression preservation (experimental)
    """

    def __init__(self, model: str = "hog"):
        super().__init__(model)
        self.last_face_locations = []

    def replace_faces_in_frame(self, frame: np.ndarray, blend_strength: float = 0.9) -> np.ndarray:
        """Replace faces with color matching."""
        if self.source_face is None:
            return frame

        result = frame.copy()
        faces = self.detector.detect_faces(frame)

        # If no faces found, use last known locations (temporal smoothing)
        if not faces and self.last_face_locations:
            faces = self.last_face_locations
        else:
            self.last_face_locations = faces

        for face_location in faces:
            result = self._replace_with_color_match(result, face_location, blend_strength)

        return result

    def _replace_with_color_match(self, frame: np.ndarray, face_location: Tuple[int, int, int, int],
                                  blend_strength: float) -> np.ndarray:
        """Replace face with color histogram matching."""
        if self.source_face is None:
            return frame

        top, right, bottom, left = face_location
        face_height = bottom - top
        face_width = right - left

        # Extract target face region for color matching
        target_face = frame[top:bottom, left:right]

        # Resize source face
        resized_source = cv2.resize(
            self.source_face,
            (face_width, face_height),
            interpolation=cv2.INTER_LANCZOS4
        )

        # Color match source to target
        matched_source = self._match_histograms(resized_source, target_face)

        # Create mask and blend
        mask = create_face_mask(matched_source)
        center = (left + face_width // 2, top + face_height // 2)

        try:
            result = cv2.seamlessClone(
                matched_source,
                frame,
                mask,
                center,
                cv2.NORMAL_CLONE
            )

            if blend_strength < 1.0:
                result = cv2.addWeighted(frame, 1 - blend_strength, result, blend_strength, 0)

            return result
        except cv2.error:
            return self._simple_blend(frame, matched_source, mask, top, left, blend_strength)

    def _match_histograms(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Match color histogram of source to target.

        Args:
            source: Source image
            target: Target image for color matching

        Returns:
            Color-matched source image
        """
        # Convert to LAB color space
        source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Calculate mean and std for each channel
        for i in range(3):
            source_mean, source_std = source_lab[:, :, i].mean(), source_lab[:, :, i].std()
            target_mean, target_std = target_lab[:, :, i].mean(), target_lab[:, :, i].std()

            if source_std > 0:
                source_lab[:, :, i] = (source_lab[:, :, i] - source_mean) * (target_std / source_std) + target_mean

        # Clip values and convert back
        source_lab = np.clip(source_lab, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(source_lab, cv2.COLOR_LAB2RGB)

        return result
