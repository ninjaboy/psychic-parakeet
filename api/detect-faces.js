/**
 * Face detection API endpoint for Vercel
 * Uses face-api.js with TensorFlow.js
 */

import * as faceapi from '@vladmandic/face-api';
import * as tf from '@tensorflow/tfjs-node';
import { Canvas, Image, ImageData, createCanvas, loadImage } from 'canvas';
import path from 'path';
import { promises as fs } from 'fs';

// Patch face-api for Node.js
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

let modelsLoaded = false;

async function loadModels() {
  if (modelsLoaded) return;

  const modelsPath = path.join(process.cwd(), 'models');

  await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelsPath);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(modelsPath);

  modelsLoaded = true;
}

export const config = {
  api: {
    bodyParser: {
      sizeLimit: '10mb',
    },
  },
  maxDuration: 60,
};

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    await loadModels();

    const { image } = req.body;

    if (!image) {
      return res.status(400).json({ error: 'No image provided. Send base64 image in body.' });
    }

    // Decode base64 image
    const base64Data = image.replace(/^data:image\/\w+;base64,/, '');
    const imageBuffer = Buffer.from(base64Data, 'base64');

    // Load and detect faces
    const img = await loadImage(imageBuffer);
    const canvas = createCanvas(img.width, img.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);

    const detections = await faceapi
      .detectAllFaces(canvas)
      .withFaceLandmarks();

    const faces = detections.map(d => ({
      box: {
        x: Math.round(d.detection.box.x),
        y: Math.round(d.detection.box.y),
        width: Math.round(d.detection.box.width),
        height: Math.round(d.detection.box.height),
      },
      score: d.detection.score,
    }));

    res.status(200).json({
      faces_found: faces.length,
      faces,
      image_size: { width: img.width, height: img.height },
    });

  } catch (error) {
    console.error('Detection error:', error);
    res.status(500).json({ error: error.message });
  }
}
