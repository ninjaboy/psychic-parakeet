/**
 * Face replacement API endpoint for Vercel
 * Processes GIF frames and replaces faces
 */

import * as faceapi from '@vladmandic/face-api';
import * as tf from '@tensorflow/tfjs-node';
import { Canvas, Image, ImageData, createCanvas, loadImage } from 'canvas';
import { parseGIF, decompressFrames } from 'gifuct-js';
import GIFEncoder from 'gif-encoder-2';
import path from 'path';

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

async function detectFacesInCanvas(canvas) {
  const detections = await faceapi.detectAllFaces(canvas);
  return detections.map(d => d.box);
}

function extractGifFrames(gifBuffer) {
  const gif = parseGIF(gifBuffer);
  const frames = decompressFrames(gif, true);

  const width = gif.lsd.width;
  const height = gif.lsd.height;

  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');

  const processedFrames = [];
  const delays = [];

  for (const frame of frames) {
    if (frame.disposalType === 2) {
      ctx.clearRect(0, 0, width, height);
    }

    const imageData = ctx.createImageData(frame.dims.width, frame.dims.height);
    imageData.data.set(frame.patch);
    ctx.putImageData(imageData, frame.dims.left, frame.dims.top);

    const fullFrame = ctx.getImageData(0, 0, width, height);
    processedFrames.push(fullFrame);
    delays.push(frame.delay || 100);
  }

  return { frames: processedFrames, delays, width, height };
}

async function replaceFaceInFrame(frameData, width, height, sourceFaceCanvas, faceBox, blendStrength) {
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');
  ctx.putImageData(frameData, 0, 0);

  // Resize source face to target size
  const tempCanvas = createCanvas(faceBox.width, faceBox.height);
  const tempCtx = tempCanvas.getContext('2d');
  tempCtx.drawImage(sourceFaceCanvas, 0, 0, faceBox.width, faceBox.height);

  // Create elliptical mask
  const maskCanvas = createCanvas(faceBox.width, faceBox.height);
  const maskCtx = maskCanvas.getContext('2d');

  const gradient = maskCtx.createRadialGradient(
    faceBox.width / 2, faceBox.height / 2, Math.min(faceBox.width, faceBox.height) * 0.2,
    faceBox.width / 2, faceBox.height / 2, Math.max(faceBox.width, faceBox.height) * 0.5
  );
  gradient.addColorStop(0, `rgba(255, 255, 255, ${blendStrength})`);
  gradient.addColorStop(0.7, `rgba(255, 255, 255, ${blendStrength * 0.6})`);
  gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');

  maskCtx.fillStyle = gradient;
  maskCtx.beginPath();
  maskCtx.ellipse(faceBox.width / 2, faceBox.height / 2, faceBox.width * 0.45, faceBox.height * 0.48, 0, 0, Math.PI * 2);
  maskCtx.fill();

  // Blend
  const sourceData = tempCtx.getImageData(0, 0, faceBox.width, faceBox.height);
  const maskData = maskCtx.getImageData(0, 0, faceBox.width, faceBox.height);
  const targetData = ctx.getImageData(faceBox.x, faceBox.y, faceBox.width, faceBox.height);

  for (let i = 0; i < sourceData.data.length; i += 4) {
    const alpha = maskData.data[i] / 255;
    if (alpha > 0) {
      targetData.data[i] = Math.round(sourceData.data[i] * alpha + targetData.data[i] * (1 - alpha));
      targetData.data[i + 1] = Math.round(sourceData.data[i + 1] * alpha + targetData.data[i + 1] * (1 - alpha));
      targetData.data[i + 2] = Math.round(sourceData.data[i + 2] * alpha + targetData.data[i + 2] * (1 - alpha));
    }
  }

  ctx.putImageData(targetData, faceBox.x, faceBox.y);
  return ctx.getImageData(0, 0, width, height);
}

function createGifFromFrames(frames, width, height, delays) {
  const encoder = new GIFEncoder(width, height, 'neuquant', true);
  encoder.start();
  encoder.setRepeat(0);
  encoder.setQuality(10);

  for (let i = 0; i < frames.length; i++) {
    encoder.setDelay(delays[i]);
    const canvas = createCanvas(width, height);
    const ctx = canvas.getContext('2d');
    ctx.putImageData(frames[i], 0, 0);
    encoder.addFrame(ctx);
  }

  encoder.finish();
  return encoder.out.getData();
}

export const config = {
  api: {
    bodyParser: {
      sizeLimit: '20mb',
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

    const { faceImage, gifData, blendStrength = 0.9 } = req.body;

    if (!faceImage || !gifData) {
      return res.status(400).json({
        error: 'Missing required fields',
        required: { faceImage: 'base64 face image', gifData: 'base64 GIF' }
      });
    }

    // Decode images
    const faceBase64 = faceImage.replace(/^data:image\/\w+;base64,/, '');
    const faceBuffer = Buffer.from(faceBase64, 'base64');

    const gifBase64 = gifData.replace(/^data:image\/\w+;base64,/, '');
    const gifBuffer = Buffer.from(gifBase64, 'base64');

    // Load source face and detect
    const faceImg = await loadImage(faceBuffer);
    const faceCanvas = createCanvas(faceImg.width, faceImg.height);
    const faceCtx = faceCanvas.getContext('2d');
    faceCtx.drawImage(faceImg, 0, 0);

    const sourceFaces = await detectFacesInCanvas(faceCanvas);
    if (sourceFaces.length === 0) {
      return res.status(400).json({ error: 'No face detected in source image' });
    }

    // Extract source face region with padding
    const srcBox = sourceFaces[0];
    const padding = 0.3;
    const padX = srcBox.width * padding;
    const padY = srcBox.height * padding;
    const x = Math.max(0, srcBox.x - padX);
    const y = Math.max(0, srcBox.y - padY);
    const w = Math.min(faceImg.width - x, srcBox.width + 2 * padX);
    const h = Math.min(faceImg.height - y, srcBox.height + 2 * padY);

    const sourceFaceCanvas = createCanvas(w, h);
    const srcFaceCtx = sourceFaceCanvas.getContext('2d');
    srcFaceCtx.drawImage(faceImg, x, y, w, h, 0, 0, w, h);

    // Extract GIF frames
    const { frames, delays, width, height } = extractGifFrames(gifBuffer);

    // Process each frame
    const processedFrames = [];

    for (let i = 0; i < frames.length; i++) {
      const frameCanvas = createCanvas(width, height);
      const frameCtx = frameCanvas.getContext('2d');
      frameCtx.putImageData(frames[i], 0, 0);

      // Detect faces in frame
      const frameFaces = await detectFacesInCanvas(frameCanvas);

      let processedFrame = frames[i];

      for (const faceBox of frameFaces) {
        processedFrame = await replaceFaceInFrame(
          processedFrame, width, height, sourceFaceCanvas, faceBox, blendStrength
        );
      }

      processedFrames.push(processedFrame);
    }

    // Create output GIF
    const outputBuffer = createGifFromFrames(processedFrames, width, height, delays);

    // Return as base64
    const outputBase64 = outputBuffer.toString('base64');

    res.status(200).json({
      success: true,
      framesProcessed: frames.length,
      gif: `data:image/gif;base64,${outputBase64}`,
    });

  } catch (error) {
    console.error('Processing error:', error);
    res.status(500).json({ error: error.message });
  }
}
