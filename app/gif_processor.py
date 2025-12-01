"""
GIF processing module using FFmpeg and Pillow.
Handles frame extraction, manipulation, and GIF reassembly.
"""

import subprocess
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
import numpy as np
import tempfile


class GifProcessor:
    """Handles GIF frame extraction and assembly using FFmpeg."""

    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize GIF processor.

        Args:
            temp_dir: Directory for temporary files. Created if not provided.
        """
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="gif_processor_")
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)

    def extract_frames(self, gif_path: str, output_dir: Optional[str] = None) -> Tuple[List[str], dict]:
        """
        Extract all frames from a GIF using FFmpeg.

        Args:
            gif_path: Path to input GIF file
            output_dir: Directory to save frames (uses temp_dir if not provided)

        Returns:
            Tuple of (list of frame paths, metadata dict with fps, duration, etc.)
        """
        output_dir = output_dir or os.path.join(self.temp_dir, "frames")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Get GIF metadata
        metadata = self._get_gif_metadata(gif_path)

        # Extract frames with FFmpeg
        frame_pattern = os.path.join(output_dir, "frame_%04d.png")

        cmd = [
            "ffmpeg", "-y",
            "-i", gif_path,
            "-vsync", "0",
            frame_pattern
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg frame extraction failed: {result.stderr}")

        # Collect frame paths
        frame_paths = sorted([
            os.path.join(output_dir, f) for f in os.listdir(output_dir)
            if f.startswith("frame_") and f.endswith(".png")
        ])

        return frame_paths, metadata

    def _get_gif_metadata(self, gif_path: str) -> dict:
        """
        Get metadata from GIF file using FFprobe.

        Args:
            gif_path: Path to GIF file

        Returns:
            Dictionary with fps, width, height, duration, frame_count
        """
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
            "-of", "csv=p=0",
            gif_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        metadata = {
            "fps": 10,  # default
            "width": 0,
            "height": 0,
            "duration": 0,
            "frame_count": 0
        }

        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            try:
                if len(parts) >= 2:
                    metadata["width"] = int(parts[0])
                    metadata["height"] = int(parts[1])
                if len(parts) >= 3 and "/" in parts[2]:
                    num, den = parts[2].split("/")
                    metadata["fps"] = float(num) / float(den) if float(den) > 0 else 10
            except (ValueError, ZeroDivisionError):
                pass

        # Fallback: use Pillow to get frame count and fps
        try:
            with Image.open(gif_path) as img:
                metadata["width"] = img.width
                metadata["height"] = img.height
                frames = 0
                durations = []
                try:
                    while True:
                        frames += 1
                        duration = img.info.get("duration", 100)
                        durations.append(duration)
                        img.seek(img.tell() + 1)
                except EOFError:
                    pass
                metadata["frame_count"] = frames
                if durations:
                    avg_duration = sum(durations) / len(durations)
                    metadata["fps"] = 1000 / avg_duration if avg_duration > 0 else 10
        except Exception:
            pass

        return metadata

    def frames_to_numpy(self, frame_paths: List[str]) -> List[np.ndarray]:
        """
        Load frame images as numpy arrays.

        Args:
            frame_paths: List of paths to frame images

        Returns:
            List of numpy arrays (RGB format)
        """
        frames = []
        for path in frame_paths:
            img = Image.open(path).convert("RGB")
            frames.append(np.array(img))
        return frames

    def numpy_to_frames(self, frames: List[np.ndarray], output_dir: str) -> List[str]:
        """
        Save numpy arrays as frame images.

        Args:
            frames: List of numpy arrays
            output_dir: Directory to save frames

        Returns:
            List of saved frame paths
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        frame_paths = []

        for i, frame in enumerate(frames):
            path = os.path.join(output_dir, f"frame_{i:04d}.png")
            img = Image.fromarray(frame.astype(np.uint8))
            img.save(path)
            frame_paths.append(path)

        return frame_paths

    def assemble_gif(self, frame_paths: List[str], output_path: str,
                     fps: float = 10, optimize: bool = True) -> str:
        """
        Assemble frames back into a GIF using FFmpeg.

        Args:
            frame_paths: List of frame image paths (must be in order)
            output_path: Path for output GIF
            fps: Frames per second
            optimize: Whether to optimize GIF size

        Returns:
            Path to created GIF
        """
        if not frame_paths:
            raise ValueError("No frames provided")

        # Create temp directory for ordered frames
        temp_frames_dir = os.path.join(self.temp_dir, "ordered_frames")
        Path(temp_frames_dir).mkdir(parents=True, exist_ok=True)

        # Copy frames with sequential naming
        for i, src_path in enumerate(frame_paths):
            dst_path = os.path.join(temp_frames_dir, f"frame_{i:04d}.png")
            if src_path != dst_path:
                shutil.copy(src_path, dst_path)

        frame_pattern = os.path.join(temp_frames_dir, "frame_%04d.png")

        # Build FFmpeg command with palette for better quality
        palette_path = os.path.join(self.temp_dir, "palette.png")

        # Generate palette
        palette_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-vf", "palettegen=max_colors=256:stats_mode=diff",
            palette_path
        ]

        result = subprocess.run(palette_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Fallback without palette
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", frame_pattern,
                "-loop", "0",
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg GIF assembly failed: {result.stderr}")
        else:
            # Use palette for high quality
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", frame_pattern,
                "-i", palette_path,
                "-lavfi", "paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
                "-loop", "0",
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg GIF assembly failed: {result.stderr}")

        return output_path

    def cleanup(self):
        """Remove temporary files and directories."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)


def load_gif_pillow(gif_path: str) -> Tuple[List[np.ndarray], List[int]]:
    """
    Load GIF frames using Pillow (alternative to FFmpeg).

    Args:
        gif_path: Path to GIF file

    Returns:
        Tuple of (list of frames as numpy arrays, list of frame durations in ms)
    """
    frames = []
    durations = []

    with Image.open(gif_path) as img:
        try:
            while True:
                # Convert to RGB
                frame = img.convert("RGBA")
                frames.append(np.array(frame))
                durations.append(img.info.get("duration", 100))
                img.seek(img.tell() + 1)
        except EOFError:
            pass

    return frames, durations


def save_gif_pillow(frames: List[np.ndarray], output_path: str,
                    durations: Optional[List[int]] = None, loop: int = 0) -> str:
    """
    Save frames as GIF using Pillow.

    Args:
        frames: List of numpy arrays
        output_path: Output file path
        durations: Frame durations in ms (default 100ms each)
        loop: Number of loops (0 = infinite)

    Returns:
        Path to saved GIF
    """
    if not frames:
        raise ValueError("No frames provided")

    durations = durations or [100] * len(frames)

    pil_frames = []
    for frame in frames:
        if frame.shape[2] == 4:  # RGBA
            img = Image.fromarray(frame.astype(np.uint8), "RGBA")
        else:  # RGB
            img = Image.fromarray(frame.astype(np.uint8), "RGB")
        pil_frames.append(img)

    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=durations,
        loop=loop,
        optimize=True
    )

    return output_path
