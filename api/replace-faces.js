/**
 * Face replacement API using Replicate's face-swap model
 * Uses InsightFace-based model for quality face swapping
 */

const { parseGIF, decompressFrames } = require('gifuct-js');
const GIFEncoder = require('gif-encoder-2');

module.exports.config = {
  api: {
    bodyParser: { sizeLimit: '10mb' },
  },
  maxDuration: 300, // 5 minutes for processing multiple frames
};

// Call Replicate API for face swap
async function swapFace(targetImageBase64, sourceImageBase64, apiKey) {
  const response = await fetch('https://api.replicate.com/v1/predictions', {
    method: 'POST',
    headers: {
      'Authorization': `Token ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      version: 'cff87316e31787df12002c9e20a78a017b8535625d5a9e6fba98a9eb8c4ca04a',
      input: {
        input_image: targetImageBase64,
        swap_image: sourceImageBase64,
      },
    }),
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`Replicate API error: ${response.status} - ${err}`);
  }

  const prediction = await response.json();

  // Poll for completion
  let result = prediction;
  while (result.status !== 'succeeded' && result.status !== 'failed') {
    await new Promise(r => setTimeout(r, 1000));
    const pollResponse = await fetch(`https://api.replicate.com/v1/predictions/${result.id}`, {
      headers: { 'Authorization': `Token ${apiKey}` },
    });
    result = await pollResponse.json();
  }

  if (result.status === 'failed') {
    throw new Error(`Face swap failed: ${result.error || 'Unknown error'}`);
  }

  return result.output;
}

// Convert image URL to base64
async function urlToBase64(url) {
  const response = await fetch(url);
  const buffer = await response.arrayBuffer();
  return `data:image/png;base64,${Buffer.from(buffer).toString('base64')}`;
}

// Extract frames from GIF
function extractGifFrames(gifBuffer) {
  const gif = parseGIF(gifBuffer);
  const frames = decompressFrames(gif, true);
  const width = gif.lsd.width;
  const height = gif.lsd.height;
  const processedFrames = [];
  const delays = [];
  let previousFrame = new Uint8ClampedArray(width * height * 4);

  for (const frame of frames) {
    const currentFrame = new Uint8ClampedArray(previousFrame);
    for (let y = 0; y < frame.dims.height; y++) {
      for (let x = 0; x < frame.dims.width; x++) {
        const srcIdx = (y * frame.dims.width + x) * 4;
        const dstX = frame.dims.left + x;
        const dstY = frame.dims.top + y;
        const dstIdx = (dstY * width + dstX) * 4;
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
    if (frame.disposalType !== 2) {
      previousFrame = new Uint8ClampedArray(currentFrame);
    }
  }
  return { frames: processedFrames, delays, width, height };
}

// Convert RGBA frame to PNG base64
function frameToPngBase64(frameData, width, height) {
  // Create PNG manually (simple uncompressed PNG)
  const Jimp = require('jimp');
  return new Promise((resolve) => {
    new Jimp(width, height, (err, image) => {
      if (err) throw err;
      for (let i = 0; i < frameData.length; i += 4) {
        const x = (i / 4) % width;
        const y = Math.floor((i / 4) / width);
        const color = Jimp.rgbaToInt(
          frameData[i],
          frameData[i + 1],
          frameData[i + 2],
          frameData[i + 3]
        );
        image.setPixelColor(color, x, y);
      }
      image.getBase64(Jimp.MIME_PNG, (err, base64) => {
        resolve(base64);
      });
    });
  });
}

// Convert base64 image to RGBA frame data
async function base64ToFrameData(base64Url, width, height) {
  const Jimp = require('jimp');
  const buffer = Buffer.from(base64Url.replace(/^data:image\/\w+;base64,/, ''), 'base64');
  const image = await Jimp.read(buffer);
  image.resize(width, height);

  const frameData = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const color = image.getPixelColor(x, y);
      const rgba = Jimp.intToRGBA(color);
      const idx = (y * width + x) * 4;
      frameData[idx] = rgba.r;
      frameData[idx + 1] = rgba.g;
      frameData[idx + 2] = rgba.b;
      frameData[idx + 3] = rgba.a;
    }
  }
  return frameData;
}

// Create GIF from frames
function createGif(frames, width, height, delays) {
  const encoder = new GIFEncoder(width, height, 'neuquant', true);
  encoder.start();
  encoder.setRepeat(0);
  encoder.setQuality(10);

  for (let i = 0; i < frames.length; i++) {
    encoder.setDelay(delays[i]);
    const rgb = [];
    for (let j = 0; j < frames[i].length; j += 4) {
      rgb.push(frames[i][j], frames[i][j + 1], frames[i][j + 2]);
    }
    encoder.addFrame(rgb);
  }

  encoder.finish();
  return Buffer.from(encoder.out.getData());
}

module.exports = async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { faceImage, gifData, apiKey, maxFrames = 10 } = req.body;

    if (!faceImage || !gifData) {
      return res.status(400).json({ error: 'Missing faceImage or gifData' });
    }

    if (!apiKey) {
      return res.status(400).json({
        error: 'Missing Replicate API key',
        info: 'Get your API key at https://replicate.com/account/api-tokens'
      });
    }

    console.log('Extracting GIF frames...');
    const gifBuffer = Buffer.from(gifData.replace(/^data:image\/\w+;base64,/, ''), 'base64');
    const { frames, delays, width, height } = extractGifFrames(gifBuffer);

    // Limit frames to process
    const framesToProcess = Math.min(frames.length, maxFrames);
    console.log(`Processing ${framesToProcess} of ${frames.length} frames`);

    const processedFrames = [];

    for (let i = 0; i < framesToProcess; i++) {
      console.log(`Processing frame ${i + 1}/${framesToProcess}...`);

      // Convert frame to base64 PNG
      const frameBase64 = await frameToPngBase64(frames[i], width, height);

      try {
        // Call Replicate face swap
        const swappedUrl = await swapFace(frameBase64, faceImage, apiKey);

        // Convert result back to frame data
        const swappedBase64 = await urlToBase64(swappedUrl);
        const swappedFrame = await base64ToFrameData(swappedBase64, width, height);
        processedFrames.push(swappedFrame);
      } catch (swapErr) {
        console.log(`Frame ${i + 1} swap failed, using original:`, swapErr.message);
        processedFrames.push(frames[i]); // Use original on failure
      }
    }

    // Add remaining frames unchanged if we limited processing
    for (let i = framesToProcess; i < frames.length; i++) {
      processedFrames.push(frames[i]);
    }

    console.log('Creating output GIF...');
    const outputBuffer = createGif(processedFrames, width, height, delays);
    const base64Output = outputBuffer.toString('base64');

    res.status(200).json({
      success: true,
      framesProcessed: framesToProcess,
      totalFrames: frames.length,
      gif: `data:image/gif;base64,${base64Output}`,
    });

  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({
      error: error.message,
      stack: error.stack?.split('\n').slice(0, 5)
    });
  }
};
