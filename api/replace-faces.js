/**
 * Face replacement API - Pure JS version for Vercel
 */

const { parseGIF, decompressFrames } = require('gifuct-js');
const GIFEncoder = require('gif-encoder-2');
const Jimp = require('jimp');

module.exports.config = {
  api: {
    bodyParser: { sizeLimit: '20mb' },
  },
  maxDuration: 60,
};

function extractGifFrames(gifBuffer) {
  console.log('DEBUG API: Parsing GIF, buffer length:', gifBuffer.length);
  const gif = parseGIF(gifBuffer);
  console.log('DEBUG API: GIF parsed, dimensions:', gif.lsd.width, 'x', gif.lsd.height);
  const frames = decompressFrames(gif, true);
  console.log('DEBUG API: Frames decompressed, count:', frames.length);

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

async function replaceFaceInFrame(frameData, width, height, faceImageBuffer, faceBox, blendStrength) {
  const faceImage = await Jimp.read(faceImageBuffer);
  faceImage.resize(faceBox.width, faceBox.height);
  const result = new Uint8ClampedArray(frameData);
  const facePixels = faceImage.bitmap.data;
  const centerX = faceBox.width / 2;
  const centerY = faceBox.height / 2;
  const radiusX = faceBox.width * 0.45;
  const radiusY = faceBox.height * 0.48;

  for (let y = 0; y < faceBox.height; y++) {
    for (let x = 0; x < faceBox.width; x++) {
      const targetX = faceBox.x + x;
      const targetY = faceBox.y + y;
      if (targetX < 0 || targetX >= width || targetY < 0 || targetY >= height) continue;
      const dx = (x - centerX) / radiusX;
      const dy = (y - centerY) / radiusY;
      const dist = Math.sqrt(dx * dx + dy * dy);
      let alpha = 0;
      if (dist < 0.7) alpha = blendStrength;
      else if (dist < 1.0) alpha = blendStrength * (1 - (dist - 0.7) / 0.3);
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

function createGif(frames, width, height, delays) {
  console.log('DEBUG API: Creating GIF, frames:', frames.length, 'size:', width, 'x', height);
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
  const result = Buffer.from(encoder.out.getData());
  console.log('DEBUG API: GIF created, size:', result.length);
  return result;
}

module.exports = async function handler(req, res) {
  console.log('DEBUG API: Request received, method:', req.method);

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    console.log('DEBUG API: Body type:', typeof req.body);
    console.log('DEBUG API: Body keys:', Object.keys(req.body || {}));

    const { faceImage, gifData, faces, blendStrength = 0.9 } = req.body;

    console.log('DEBUG API: faceImage length:', faceImage?.length);
    console.log('DEBUG API: gifData length:', gifData?.length);
    console.log('DEBUG API: faces:', JSON.stringify(faces));
    console.log('DEBUG API: blendStrength:', blendStrength);

    if (!faceImage || !gifData) {
      return res.status(400).json({
        error: 'Missing required fields',
        debug: { hasFaceImage: !!faceImage, hasGifData: !!gifData }
      });
    }

    console.log('DEBUG API: Decoding base64...');
    const faceBuffer = Buffer.from(faceImage.replace(/^data:image\/\w+;base64,/, ''), 'base64');
    const gifBuffer = Buffer.from(gifData.replace(/^data:image\/\w+;base64,/, ''), 'base64');
    console.log('DEBUG API: Face buffer:', faceBuffer.length, 'GIF buffer:', gifBuffer.length);

    console.log('DEBUG API: Extracting frames...');
    const { frames, delays, width, height } = extractGifFrames(gifBuffer);
    console.log('DEBUG API: Extracted', frames.length, 'frames');

    const processedFrames = [];

    for (let i = 0; i < frames.length; i++) {
      let frameResult = frames[i];
      const frameFaces = faces?.[i] || faces?.[0] || null;
      if (frameFaces) {
        const faceList = Array.isArray(frameFaces) ? frameFaces : [frameFaces];
        for (const faceBox of faceList) {
          if (faceBox && faceBox.width > 0 && faceBox.height > 0) {
            console.log('DEBUG API: Processing frame', i, 'face:', JSON.stringify(faceBox));
            frameResult = await replaceFaceInFrame(frameResult, width, height, faceBuffer, faceBox, blendStrength);
          }
        }
      }
      processedFrames.push(frameResult);
    }

    console.log('DEBUG API: Creating output GIF...');
    const outputBuffer = createGif(processedFrames, width, height, delays);
    const base64Output = outputBuffer.toString('base64');
    console.log('DEBUG API: Output base64 length:', base64Output.length);

    res.status(200).json({
      success: true,
      framesProcessed: frames.length,
      dimensions: { width, height },
      gif: `data:image/gif;base64,${base64Output}`,
    });

  } catch (error) {
    console.error('DEBUG API: Error:', error.message);
    console.error('DEBUG API: Stack:', error.stack);
    res.status(500).json({
      error: error.message,
      stack: error.stack?.split('\n').slice(0, 5)
    });
  }
};
