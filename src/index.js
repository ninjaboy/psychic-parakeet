/**
 * Face GIF Replacer - Express Backend
 * ML-powered face replacement in GIFs
 */

import express from 'express';
import multer from 'multer';
import path from 'path';
import { fileURLToPath } from 'url';
import { promises as fs } from 'fs';
import { v4 as uuidv4 } from 'uuid';
import fetch from 'node-fetch';

import { loadModels, detectFaces } from './faceDetector.js';
import FaceReplacer from './faceReplacer.js';
import {
  extractFrames,
  assembleGif,
  extractFramesPure,
  createGifPure,
} from './gifProcessor.js';
import { imageDataToBuffer } from './faceReplacer.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 8000;

// Directories
const UPLOAD_DIR = path.join(__dirname, '..', 'uploads');
const OUTPUT_DIR = path.join(__dirname, '..', 'outputs');
const STATIC_DIR = path.join(__dirname, '..', 'static');

// Ensure directories exist
await fs.mkdir(UPLOAD_DIR, { recursive: true });
await fs.mkdir(OUTPUT_DIR, { recursive: true });

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: UPLOAD_DIR,
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname);
    cb(null, `${uuidv4()}${ext}`);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 50 * 1024 * 1024 }, // 50MB limit
  fileFilter: (req, file, cb) => {
    const allowedTypes = /jpeg|jpg|png|gif|webp/i;
    const ext = path.extname(file.originalname).toLowerCase();
    if (allowedTypes.test(ext)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type'));
    }
  },
});

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static(STATIC_DIR));

// CORS
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') {
    return res.sendStatus(200);
  }
  next();
});

// Load models on startup
console.log('Initializing face detection models...');
await loadModels();
console.log('Server ready!');

/**
 * Download file from URL
 */
async function downloadFile(url, destPath) {
  const response = await fetch(url, { timeout: 60000 });
  if (!response.ok) {
    throw new Error(`Failed to download: ${response.status}`);
  }
  const buffer = await response.buffer();
  await fs.writeFile(destPath, buffer);
  return destPath;
}

/**
 * Cleanup file after delay
 */
function scheduleCleanup(filePath, delayMs = 300000) {
  setTimeout(async () => {
    try {
      await fs.unlink(filePath);
    } catch (e) {
      // Ignore
    }
  }, delayMs);
}

/**
 * Cleanup directory
 */
async function cleanupDir(dirPath) {
  try {
    await fs.rm(dirPath, { recursive: true, force: true });
  } catch (e) {
    // Ignore
  }
}

// Routes

app.get('/', (req, res) => {
  res.sendFile(path.join(STATIC_DIR, 'index.html'));
});

app.get('/health', (req, res) => {
  res.json({ status: 'healthy' });
});

/**
 * POST /api/replace-faces
 * Replace faces in a GIF with source face
 */
app.post(
  '/api/replace-faces',
  upload.fields([
    { name: 'face_image', maxCount: 1 },
    { name: 'gif_file', maxCount: 1 },
  ]),
  async (req, res) => {
    const jobId = uuidv4();
    const tempDir = path.join(UPLOAD_DIR, jobId);

    let facePath = null;
    let gifPath = null;

    try {
      await fs.mkdir(tempDir, { recursive: true });

      // Get face image
      if (req.files?.face_image?.[0]) {
        facePath = req.files.face_image[0].path;
      } else if (req.body.face_image_url) {
        facePath = path.join(tempDir, 'face.jpg');
        await downloadFile(req.body.face_image_url, facePath);
      } else {
        return res.status(400).json({ detail: 'Provide face_image or face_image_url' });
      }

      // Get GIF
      if (req.files?.gif_file?.[0]) {
        gifPath = req.files.gif_file[0].path;
      } else if (req.body.gif_url) {
        gifPath = path.join(tempDir, 'input.gif');
        await downloadFile(req.body.gif_url, gifPath);
      } else {
        return res.status(400).json({ detail: 'Provide gif_file or gif_url' });
      }

      // Parse options
      const blendStrength = parseFloat(req.body.blend_strength) || 0.9;
      const useFfmpeg = req.body.use_ffmpeg !== 'false';

      // Initialize face replacer
      const replacer = new FaceReplacer();
      const faceBuffer = await fs.readFile(facePath);

      if (!(await replacer.setSourceFace(faceBuffer))) {
        return res.status(400).json({ detail: 'No face detected in the source image' });
      }

      const outputPath = path.join(OUTPUT_DIR, `${jobId}_output.gif`);

      if (useFfmpeg) {
        // Use FFmpeg for better quality
        const framesDir = path.join(tempDir, 'frames');
        const { framePaths, metadata } = await extractFrames(gifPath, framesDir);

        // Load and process frames
        const processedPaths = [];
        const processedDir = path.join(tempDir, 'processed');
        await fs.mkdir(processedDir, { recursive: true });

        for (let i = 0; i < framePaths.length; i++) {
          const frameBuffer = await fs.readFile(framePaths[i]);
          const processed = await replacer.replaceFacesInFrame(frameBuffer, blendStrength);

          const outPath = path.join(processedDir, `frame_${String(i).padStart(4, '0')}.png`);
          await fs.writeFile(outPath, processed);
          processedPaths.push(outPath);
        }

        // Assemble GIF
        const fps = metadata.fps || 10;
        await assembleGif(processedPaths, outputPath, fps);
      } else {
        // Use pure JavaScript
        const gifBuffer = await fs.readFile(gifPath);
        const { frames, delays, width, height } = await extractFramesPure(gifBuffer);

        // Convert ImageData to buffers and process
        const processedFrames = [];

        for (let i = 0; i < frames.length; i++) {
          const frameBuffer = imageDataToBuffer(frames[i], width, height);
          const processed = await replacer.replaceFacesInFrame(frameBuffer, blendStrength);

          // Convert back to ImageData
          const { createCanvas, loadImage } = await import('canvas');
          const img = await loadImage(processed);
          const canvas = createCanvas(width, height);
          const ctx = canvas.getContext('2d');
          ctx.drawImage(img, 0, 0, width, height);
          processedFrames.push(ctx.getImageData(0, 0, width, height));
        }

        // Create output GIF
        const outputBuffer = await createGifPure(processedFrames, width, height, delays);
        await fs.writeFile(outputPath, outputBuffer);
      }

      // Schedule cleanup
      cleanupDir(tempDir);
      scheduleCleanup(outputPath, 600000); // 10 minutes

      // Send result
      res.setHeader('Content-Type', 'image/gif');
      res.setHeader('Content-Disposition', `attachment; filename="face_replaced_${jobId}.gif"`);
      res.sendFile(outputPath);

    } catch (error) {
      console.error('Processing error:', error);
      await cleanupDir(tempDir);
      res.status(500).json({ detail: `Processing failed: ${error.message}` });
    }
  }
);

/**
 * POST /api/detect-faces
 * Detect faces in an image
 */
app.post(
  '/api/detect-faces',
  upload.single('image'),
  async (req, res) => {
    let imagePath = null;

    try {
      if (req.file) {
        imagePath = req.file.path;
      } else if (req.body.image_url) {
        imagePath = path.join(UPLOAD_DIR, `detect_${uuidv4()}.jpg`);
        await downloadFile(req.body.image_url, imagePath);
      } else {
        return res.status(400).json({ detail: 'Provide image or image_url' });
      }

      const imageBuffer = await fs.readFile(imagePath);
      const { loadImage } = await import('canvas');
      const img = await loadImage(imageBuffer);

      const faces = await detectFaces(imageBuffer);

      // Cleanup
      scheduleCleanup(imagePath, 60000);

      res.json({
        faces_found: faces.length,
        face_locations: faces.map(f => ({
          top: f.box.y,
          right: f.box.x + f.box.width,
          bottom: f.box.y + f.box.height,
          left: f.box.x,
          width: f.box.width,
          height: f.box.height,
          score: f.score,
        })),
        image_size: {
          width: img.width,
          height: img.height,
        },
      });

    } catch (error) {
      console.error('Detection error:', error);
      if (imagePath) {
        scheduleCleanup(imagePath, 1000);
      }
      res.status(500).json({ detail: `Detection failed: ${error.message}` });
    }
  }
);

/**
 * GET /api/jobs/:jobId/status
 */
app.get('/api/jobs/:jobId/status', async (req, res) => {
  const { jobId } = req.params;
  const outputPath = path.join(OUTPUT_DIR, `${jobId}_output.gif`);

  try {
    await fs.access(outputPath);
    res.json({ status: 'completed', download_url: `/api/jobs/${jobId}/download` });
  } catch {
    const tempDir = path.join(UPLOAD_DIR, jobId);
    try {
      await fs.access(tempDir);
      res.json({ status: 'processing' });
    } catch {
      res.json({ status: 'not_found' });
    }
  }
});

/**
 * GET /api/jobs/:jobId/download
 */
app.get('/api/jobs/:jobId/download', async (req, res) => {
  const { jobId } = req.params;
  const outputPath = path.join(OUTPUT_DIR, `${jobId}_output.gif`);

  try {
    await fs.access(outputPath);
    res.setHeader('Content-Type', 'image/gif');
    res.setHeader('Content-Disposition', `attachment; filename="face_replaced_${jobId}.gif"`);
    res.sendFile(outputPath);
  } catch {
    res.status(404).json({ detail: 'Result not found' });
  }
});

// Error handler
app.use((err, req, res, next) => {
  console.error('Server error:', err);
  res.status(500).json({ detail: err.message || 'Internal server error' });
});

// Start server
app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running on http://0.0.0.0:${PORT}`);
});
