/**
 * Face detection module using face-api.js (TensorFlow.js based)
 * Provides accurate face detection and landmark extraction
 */

import * as faceapi from '@vladmandic/face-api';
import * as tf from '@tensorflow/tfjs-node';
import { Canvas, Image, ImageData, createCanvas, loadImage } from 'canvas';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Patch face-api to use node-canvas
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

let modelsLoaded = false;

/**
 * Load face-api.js models
 */
export async function loadModels() {
  if (modelsLoaded) return;

  const modelsPath = path.join(__dirname, '..', 'models');

  console.log('Loading face detection models...');

  await Promise.all([
    faceapi.nets.ssdMobilenetv1.loadFromDisk(modelsPath),
    faceapi.nets.faceLandmark68Net.loadFromDisk(modelsPath),
    faceapi.nets.faceRecognitionNet.loadFromDisk(modelsPath),
  ]);

  modelsLoaded = true;
  console.log('Face detection models loaded');
}

/**
 * Detect faces in an image buffer
 * @param {Buffer} imageBuffer - Image buffer
 * @returns {Promise<Array>} Array of face detections with landmarks
 */
export async function detectFaces(imageBuffer) {
  await loadModels();

  const img = await loadImage(imageBuffer);
  const canvas = createCanvas(img.width, img.height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);

  const detections = await faceapi
    .detectAllFaces(canvas)
    .withFaceLandmarks()
    .withFaceDescriptors();

  return detections.map(d => ({
    box: {
      x: Math.round(d.detection.box.x),
      y: Math.round(d.detection.box.y),
      width: Math.round(d.detection.box.width),
      height: Math.round(d.detection.box.height),
    },
    landmarks: d.landmarks.positions.map(p => ({ x: p.x, y: p.y })),
    descriptor: Array.from(d.descriptor),
    score: d.detection.score,
  }));
}

/**
 * Detect the largest face in an image
 * @param {Buffer} imageBuffer - Image buffer
 * @returns {Promise<Object|null>} Face detection or null
 */
export async function detectLargestFace(imageBuffer) {
  const faces = await detectFaces(imageBuffer);

  if (faces.length === 0) return null;

  // Find largest face by area
  return faces.reduce((largest, face) => {
    const area = face.box.width * face.box.height;
    const largestArea = largest.box.width * largest.box.height;
    return area > largestArea ? face : largest;
  });
}

/**
 * Extract face region from image with padding
 * @param {Buffer} imageBuffer - Image buffer
 * @param {Object} box - Face bounding box
 * @param {number} padding - Padding ratio (0.3 = 30%)
 * @returns {Promise<{buffer: Buffer, box: Object}>}
 */
export async function extractFaceRegion(imageBuffer, box, padding = 0.3) {
  const img = await loadImage(imageBuffer);

  const padW = Math.round(box.width * padding);
  const padH = Math.round(box.height * padding);

  const x = Math.max(0, box.x - padW);
  const y = Math.max(0, box.y - padH);
  const width = Math.min(img.width - x, box.width + 2 * padW);
  const height = Math.min(img.height - y, box.height + 2 * padH);

  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, x, y, width, height, 0, 0, width, height);

  return {
    buffer: canvas.toBuffer('image/png'),
    box: { x, y, width, height },
  };
}

/**
 * Create an elliptical mask for face blending
 * @param {number} width - Mask width
 * @param {number} height - Mask height
 * @returns {Buffer} Grayscale mask buffer
 */
export function createFaceMask(width, height) {
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');

  // Create radial gradient for smooth edges
  const centerX = width / 2;
  const centerY = height / 2;
  const radiusX = width * 0.45;
  const radiusY = height * 0.48;

  // Draw ellipse with gradient
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, width, height);

  ctx.beginPath();
  ctx.ellipse(centerX, centerY, radiusX, radiusY, 0, 0, Math.PI * 2);
  ctx.fillStyle = 'white';
  ctx.fill();

  return canvas.toBuffer('image/png');
}

export default {
  loadModels,
  detectFaces,
  detectLargestFace,
  extractFaceRegion,
  createFaceMask,
};
