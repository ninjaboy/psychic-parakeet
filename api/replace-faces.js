/**
 * Face replacement API - Pure JS version for Vercel
 * Face detection happens client-side, server just does GIF processing
 */

import { parseGIF, decompressFrames } from 'gifuct-js';
import GIFEncoder from 'gif-encoder-2';
import Jimp from 'jimp';

export const config = {
  api: {
    bodyParser: {
      sizeLimit: '20mb',
    },
  },
  maxDuration: 60,
};

/**
 * Extract frames from GIF buffer
 */
function extractGifFrames(gifBuffer) {
  const gif = parseGIF(gifBuffer);
  const frames = decompressFrames(gif, true);

  const width = gif.lsd.width;
  const height = gif.lsd.height;

  const processedFrames = [];
  const delays = [];

  // Create a cumulative frame buffer for proper GIF rendering
  let previousFrame = new Uint8ClampedArray(width * height * 4);

  for (const frame of frames) {
    const currentFrame = new Uint8ClampedArray(previousFrame);

    // Apply frame patch
    for (let y = 0; y < frame.dims.height; y++) {
      for (let x = 0; x < frame.dims.width; x++) {
        const srcIdx = (y * frame.dims.width + x) * 4;
        const dstX = frame.dims.left + x;
        const dstY = frame.dims.top + y;
        const dstIdx = (dstY * width + dstX) * 4;

        // Only copy non-transparent pixels
        if (frame.patch[srcIdx + 3] > 0) {
          currentFrame[dstIdx] = frame.patch[srcIdx];
          currentFrame[dstIdx + 1] = frame.patch[srcIdx + 1];
          currentFrame[dstIdx + 2] = frame.patch[srcIdx + 2];
          currentFrame[dstIdx + 3] = frame.patch[srcIdx + 3];
        }
      }
    }

    processedFrames.push(new Uint8ClampedArray(currentFrame));
    delays.push(frame.delay || 100);

    // Handle disposal
    if (frame.disposalType !== 2) {
      previousFrame = new Uint8ClampedArray(currentFrame);
    }
  }

  return { frames: processedFrames, delays, width, height };
}

/**
 * Replace face region in a frame
 */
async function replaceFaceInFrame(frameData, width, height, faceImageBuffer, faceBox, blendStrength) {
  // Load face image with Jimp
  const faceImage = await Jimp.read(faceImageBuffer);

  // Resize face to target box size
  faceImage.resize(faceBox.width, faceBox.height);

  // Create result frame
  const result = new Uint8ClampedArray(frameData);

  // Get face pixels
  const facePixels = faceImage.bitmap.data;

  // Create elliptical mask and blend
  const centerX = faceBox.width / 2;
  const centerY = faceBox.height / 2;
  const radiusX = faceBox.width * 0.45;
  const radiusY = faceBox.height * 0.48;

  for (let y = 0; y < faceBox.height; y++) {
    for (let x = 0; x < faceBox.width; x++) {
      const targetX = faceBox.x + x;
      const targetY = faceBox.y + y;

      // Skip if outside frame bounds
      if (targetX < 0 || targetX >= width || targetY < 0 || targetY >= height) continue;

      // Calculate ellipse distance for mask
      const dx = (x - centerX) / radiusX;
      const dy = (y - centerY) / radiusY;
      const dist = Math.sqrt(dx * dx + dy * dy);

      // Smooth falloff
      let alpha = 0;
      if (dist < 0.7) {
        alpha = blendStrength;
      } else if (dist < 1.0) {
        alpha = blendStrength * (1 - (dist - 0.7) / 0.3);
      }

      if (alpha > 0) {
        const srcIdx = (y * faceBox.width + x) * 4;
        const dstIdx = (targetY * width + targetX) * 4;

        result[dstIdx] = Math.round(facePixels[srcIdx] * alpha + result[dstIdx] * (1 - alpha));
        result[dstIdx + 1] = Math.round(facePixels[srcIdx + 1] * alpha + result[dstIdx + 1] * (1 - alpha));
        result[dstIdx + 2] = Math.round(facePixels[srcIdx + 2] * alpha + result[dstIdx + 2] * (1 - alpha));
      }
    }
  }

  return result;
}

/**
 * Create GIF from frames
 */
function createGif(frames, width, height, delays) {
  const encoder = new GIFEncoder(width, height, 'neuquant', true);
  encoder.start();
  encoder.setRepeat(0);
  encoder.setQuality(10);
  encoder.setTransparent(0x000000);

  for (let i = 0; i < frames.length; i++) {
    encoder.setDelay(delays[i]);

    // Convert RGBA to RGB for encoder
    const rgb = [];
    for (let j = 0; j < frames[i].length; j += 4) {
      rgb.push(frames[i][j], frames[i][j + 1], frames[i][j + 2]);
    }

    encoder.addFrame(rgb);
  }

  encoder.finish();
  return Buffer.from(encoder.out.getData());
}

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { faceImage, gifData, faces, blendStrength = 0.9 } = req.body;

    if (!faceImage || !gifData) {
      return res.status(400).json({
        error: 'Missing required fields',
        usage: {
          faceImage: 'base64 encoded face image',
          gifData: 'base64 encoded GIF',
          faces: 'array of {x, y, width, height} for each frame (optional, from client-side detection)',
          blendStrength: 'number 0-1 (default 0.9)'
        }
      });
    }

    // Decode base64
    const faceBase64 = faceImage.replace(/^data:image\/\w+;base64,/, '');
    const faceBuffer = Buffer.from(faceBase64, 'base64');

    const gifBase64 = gifData.replace(/^data:image\/\w+;base64,/, '');
    const gifBuffer = Buffer.from(gifBase64, 'base64');

    // Extract GIF frames
    const { frames, delays, width, height } = extractGifFrames(gifBuffer);

    // Process each frame
    const processedFrames = [];

    for (let i = 0; i < frames.length; i++) {
      let frameResult = frames[i];

      // Get face boxes for this frame
      const frameFaces = faces?.[i] || faces?.[0] || null;

      if (frameFaces) {
        const faceList = Array.isArray(frameFaces) ? frameFaces : [frameFaces];

        for (const faceBox of faceList) {
          if (faceBox && faceBox.width > 0 && faceBox.height > 0) {
            frameResult = await replaceFaceInFrame(
              frameResult, width, height, faceBuffer, faceBox, blendStrength
            );
          }
        }
      }

      processedFrames.push(frameResult);
    }

    // Create output GIF
    const outputBuffer = createGif(processedFrames, width, height, delays);
    const outputBase64 = outputBuffer.toString('base64');

    res.status(200).json({
      success: true,
      framesProcessed: frames.length,
      dimensions: { width, height },
      gif: `data:image/gif;base64,${outputBase64}`,
    });

  } catch (error) {
    console.error('Processing error:', error);
    res.status(500).json({ error: error.message });
  }
}
