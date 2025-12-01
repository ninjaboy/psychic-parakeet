"""
FastAPI backend for Face Detection GIF Replacer.
Accepts a source face image and a GIF, replaces faces in GIF with the source face.
"""

import os
import uuid
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .face_detector import FaceDetector
from .face_replacer import FaceReplacer, AdvancedFaceReplacer
from .gif_processor import GifProcessor, load_gif_pillow, save_gif_pillow


app = FastAPI(
    title="Face GIF Replacer",
    description="Replace faces in GIFs with your own face using ML-based detection",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
STATIC_DIR = Path("static")

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def cleanup_file(path: str):
    """Background task to clean up temporary files."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def cleanup_directory(path: str):
    """Background task to clean up temporary directories."""
    try:
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


async def download_file(url: str, dest_path: str) -> str:
    """Download file from URL."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
        response = await client.get(url)
        response.raise_for_status()

        with open(dest_path, "wb") as f:
            f.write(response.content)

    return dest_path


def load_image(path: str) -> np.ndarray:
    """Load image as RGB numpy array."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


@app.get("/")
async def root():
    """Serve the main page."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Face GIF Replacer API", "docs": "/docs"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/api/replace-faces")
async def replace_faces(
    background_tasks: BackgroundTasks,
    face_image: Optional[UploadFile] = File(None),
    face_image_url: Optional[str] = Form(None),
    gif_file: Optional[UploadFile] = File(None),
    gif_url: Optional[str] = Form(None),
    blend_strength: float = Form(0.9),
    use_advanced: bool = Form(True),
    use_ffmpeg: bool = Form(True)
):
    """
    Replace faces in a GIF with a source face.

    Parameters:
    - face_image: Upload face image file
    - face_image_url: OR provide URL to face image
    - gif_file: Upload GIF file
    - gif_url: OR provide URL to GIF
    - blend_strength: How strongly to blend (0.0-1.0)
    - use_advanced: Use advanced replacer with color matching
    - use_ffmpeg: Use FFmpeg for GIF processing (better quality)

    Returns:
    - The processed GIF file
    """
    job_id = str(uuid.uuid4())
    temp_dir = UPLOAD_DIR / job_id
    temp_dir.mkdir(exist_ok=True)

    face_path = None
    gif_path = None

    try:
        # Get face image
        if face_image:
            face_path = str(temp_dir / f"face_{face_image.filename}")
            with open(face_path, "wb") as f:
                content = await face_image.read()
                f.write(content)
        elif face_image_url:
            face_path = str(temp_dir / "face_download.jpg")
            await download_file(face_image_url, face_path)
        else:
            raise HTTPException(status_code=400, detail="Provide face_image or face_image_url")

        # Get GIF
        if gif_file:
            gif_path = str(temp_dir / f"input_{gif_file.filename}")
            with open(gif_path, "wb") as f:
                content = await gif_file.read()
                f.write(content)
        elif gif_url:
            gif_path = str(temp_dir / "input.gif")
            await download_file(gif_url, gif_path)
        else:
            raise HTTPException(status_code=400, detail="Provide gif_file or gif_url")

        # Load source face
        face_array = load_image(face_path)

        # Initialize face replacer
        replacer_class = AdvancedFaceReplacer if use_advanced else FaceReplacer
        replacer = replacer_class(model="hog")

        if not replacer.set_source_face(face_array):
            raise HTTPException(status_code=400, detail="No face detected in the source image")

        # Process GIF
        output_path = str(OUTPUT_DIR / f"{job_id}_output.gif")

        if use_ffmpeg:
            # Use FFmpeg for better quality
            processor = GifProcessor(str(temp_dir / "processing"))

            try:
                frame_paths, metadata = processor.extract_frames(gif_path)
                frames = processor.frames_to_numpy(frame_paths)

                # Replace faces in each frame
                processed_frames = replacer.process_gif_frames(frames, blend_strength)

                # Save processed frames
                processed_dir = str(temp_dir / "processed_frames")
                processed_paths = processor.numpy_to_frames(processed_frames, processed_dir)

                # Assemble final GIF
                fps = metadata.get("fps", 10)
                processor.assemble_gif(processed_paths, output_path, fps=fps)

            finally:
                processor.cleanup()
        else:
            # Use Pillow (simpler, may have lower quality)
            frames, durations = load_gif_pillow(gif_path)

            # Convert RGBA to RGB for processing
            rgb_frames = []
            for frame in frames:
                if frame.shape[2] == 4:
                    # RGBA -> RGB with white background
                    rgb = frame[:, :, :3].copy()
                    alpha = frame[:, :, 3:4] / 255.0
                    white_bg = np.ones_like(rgb) * 255
                    rgb = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)
                    rgb_frames.append(rgb)
                else:
                    rgb_frames.append(frame)

            # Replace faces
            processed_frames = replacer.process_gif_frames(rgb_frames, blend_strength)

            # Save GIF
            save_gif_pillow(processed_frames, output_path, durations)

        # Schedule cleanup of temp directory
        background_tasks.add_task(cleanup_directory, str(temp_dir))

        return FileResponse(
            output_path,
            media_type="image/gif",
            filename=f"face_replaced_{job_id}.gif",
            background=BackgroundTasks([lambda: cleanup_file(output_path)])
        )

    except HTTPException:
        raise
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download file: {str(e)}")
    except Exception as e:
        # Cleanup on error
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/api/detect-faces")
async def detect_faces(
    background_tasks: BackgroundTasks,
    image: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None)
):
    """
    Detect faces in an image.

    Parameters:
    - image: Upload image file
    - image_url: OR provide URL to image

    Returns:
    - List of detected face locations
    """
    temp_path = None

    try:
        if image:
            temp_path = str(UPLOAD_DIR / f"detect_{uuid.uuid4()}_{image.filename}")
            with open(temp_path, "wb") as f:
                content = await image.read()
                f.write(content)
        elif image_url:
            temp_path = str(UPLOAD_DIR / f"detect_{uuid.uuid4()}.jpg")
            await download_file(image_url, temp_path)
        else:
            raise HTTPException(status_code=400, detail="Provide image or image_url")

        # Load and detect
        img_array = load_image(temp_path)
        detector = FaceDetector(model="hog")
        faces = detector.detect_faces(img_array)

        # Schedule cleanup
        if temp_path:
            background_tasks.add_task(cleanup_file, temp_path)

        return JSONResponse({
            "faces_found": len(faces),
            "face_locations": [
                {
                    "top": f[0],
                    "right": f[1],
                    "bottom": f[2],
                    "left": f[3],
                    "width": f[1] - f[3],
                    "height": f[2] - f[0]
                }
                for f in faces
            ],
            "image_size": {
                "width": img_array.shape[1],
                "height": img_array.shape[0]
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.get("/api/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """Check status of a processing job."""
    output_path = OUTPUT_DIR / f"{job_id}_output.gif"

    if output_path.exists():
        return {"status": "completed", "download_url": f"/api/jobs/{job_id}/download"}

    temp_dir = UPLOAD_DIR / job_id
    if temp_dir.exists():
        return {"status": "processing"}

    return {"status": "not_found"}


@app.get("/api/jobs/{job_id}/download")
async def download_result(job_id: str, background_tasks: BackgroundTasks):
    """Download the processed GIF."""
    output_path = OUTPUT_DIR / f"{job_id}_output.gif"

    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Result not found")

    return FileResponse(
        str(output_path),
        media_type="image/gif",
        filename=f"face_replaced_{job_id}.gif"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
