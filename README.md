# Face GIF Replacer

ML-powered application that detects faces in GIFs and replaces them with your face.

## Features

- **ML Face Detection**: Uses dlib's HOG/CNN models via `face_recognition` library
- **Seamless Face Replacement**: Color-matching and seamless cloning for natural results
- **FFmpeg Integration**: High-quality GIF frame extraction and reassembly
- **Web Interface**: Simple drag-and-drop UI
- **REST API**: Programmatic access for integration

## Quick Start

### Using Docker (Recommended)

```bash
docker-compose up --build
```

Then open http://localhost:8000

### Manual Installation

1. Install system dependencies:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg cmake libboost-all-dev libopenblas-dev

# macOS
brew install ffmpeg cmake boost
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Run the server:

```bash
uvicorn app.main:app --reload
```

## API Usage

### Replace Faces in GIF

**POST** `/api/replace-faces`

Form parameters:
- `face_image`: Face image file (or use `face_image_url`)
- `face_image_url`: URL to face image
- `gif_file`: GIF file (or use `gif_url`)
- `gif_url`: URL to GIF
- `blend_strength`: Float 0-1 (default: 0.9)
- `use_advanced`: Boolean - enable color matching (default: true)
- `use_ffmpeg`: Boolean - use FFmpeg for better quality (default: true)

Example with curl:

```bash
curl -X POST http://localhost:8000/api/replace-faces \
  -F "face_image=@my_face.jpg" \
  -F "gif_file=@funny.gif" \
  -F "blend_strength=0.9" \
  --output result.gif
```

Using URLs:

```bash
curl -X POST http://localhost:8000/api/replace-faces \
  -F "face_image_url=https://example.com/face.jpg" \
  -F "gif_url=https://example.com/funny.gif" \
  --output result.gif
```

### Detect Faces

**POST** `/api/detect-faces`

```bash
curl -X POST http://localhost:8000/api/detect-faces \
  -F "image=@photo.jpg"
```

Response:
```json
{
  "faces_found": 2,
  "face_locations": [
    {"top": 100, "right": 200, "bottom": 300, "left": 50, "width": 150, "height": 200},
    {"top": 80, "right": 400, "bottom": 280, "left": 250, "width": 150, "height": 200}
  ],
  "image_size": {"width": 640, "height": 480}
}
```

## Architecture

```
app/
├── main.py           # FastAPI application
├── face_detector.py  # ML face detection (dlib/face_recognition)
├── face_replacer.py  # Face replacement with seamless cloning
└── gif_processor.py  # FFmpeg-based GIF processing

static/
└── index.html        # Web interface
```

## How It Works

1. **Extract Frames**: GIF is split into individual frames using FFmpeg
2. **Detect Faces**: Each frame is analyzed using dlib's face detection model
3. **Replace Faces**: Source face is resized, color-matched, and seamlessly blended
4. **Reassemble GIF**: Processed frames are combined back into GIF with proper timing

## Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST` | Server host | 0.0.0.0 |
| `PORT` | Server port | 8000 |

## License

MIT
