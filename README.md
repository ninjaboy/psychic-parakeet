# Face GIF Replacer

ML-powered application that detects faces in GIFs and replaces them with your face.

Built with **Node.js**, **TensorFlow.js**, and **face-api.js**.

## Features

- **ML Face Detection**: Uses face-api.js (TensorFlow.js) for accurate face detection
- **Seamless Face Replacement**: Color-matching and smooth blending for natural results
- **FFmpeg Integration**: High-quality GIF frame extraction and reassembly
- **Web Interface**: Simple drag-and-drop UI
- **REST API**: Programmatic access for integration
- **Pure JS Fallback**: Works without FFmpeg using gifuct-js

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
sudo apt-get install ffmpeg build-essential libcairo2-dev libpango1.0-dev \
  libjpeg-dev libgif-dev librsvg2-dev

# macOS
brew install pkg-config cairo pango libpng jpeg giflib librsvg ffmpeg
```

2. Install Node.js dependencies:

```bash
npm install
```

3. Download face detection models:

```bash
npm run download-models
```

4. Run the server:

```bash
npm start
# or for development with auto-reload:
npm run dev
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
    {"top": 100, "right": 200, "bottom": 300, "left": 50, "width": 150, "height": 200, "score": 0.95},
    {"top": 80, "right": 400, "bottom": 280, "left": 250, "width": 150, "height": 200, "score": 0.92}
  ],
  "image_size": {"width": 640, "height": 480}
}
```

## Architecture

```
src/
├── index.js          # Express server
├── faceDetector.js   # ML face detection (face-api.js/TensorFlow.js)
├── faceReplacer.js   # Face replacement with color matching
└── gifProcessor.js   # FFmpeg & pure JS GIF processing

models/               # face-api.js pre-trained models
static/
└── index.html        # Web interface
```

## How It Works

1. **Extract Frames**: GIF is split into individual frames using FFmpeg (or pure JS)
2. **Detect Faces**: Each frame is analyzed using face-api.js neural networks
3. **Replace Faces**: Source face is resized, color-matched, and smoothly blended
4. **Reassemble GIF**: Processed frames are combined back into GIF with proper timing

## Tech Stack

- **Node.js 20+** - Runtime
- **Express** - Web framework
- **face-api.js** - Face detection (TensorFlow.js)
- **sharp** - High-performance image processing
- **fluent-ffmpeg** - FFmpeg wrapper
- **canvas** - Node.js canvas implementation
- **gif-encoder-2** - GIF creation
- **gifuct-js** - GIF parsing

## Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | 8000 |
| `NODE_ENV` | Environment | development |

## License

MIT
