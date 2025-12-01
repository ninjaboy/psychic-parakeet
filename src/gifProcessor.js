/**
 * GIF processing module using FFmpeg
 * Handles frame extraction, manipulation, and GIF reassembly
 */

import ffmpeg from 'fluent-ffmpeg';
import { promises as fs } from 'fs';
import path from 'path';
import { createCanvas, loadImage } from 'canvas';
import GIFEncoder from 'gif-encoder-2';
import { parseGIF, decompressFrames } from 'gifuct-js';

/**
 * Get GIF metadata using FFprobe
 * @param {string} gifPath - Path to GIF file
 * @returns {Promise<Object>} Metadata object
 */
export async function getGifMetadata(gifPath) {
  return new Promise((resolve, reject) => {
    ffmpeg.ffprobe(gifPath, (err, metadata) => {
      if (err) {
        // Fallback to default values
        resolve({ fps: 10, width: 0, height: 0, duration: 0 });
        return;
      }

      const videoStream = metadata.streams.find(s => s.codec_type === 'video');

      if (!videoStream) {
        resolve({ fps: 10, width: 0, height: 0, duration: 0 });
        return;
      }

      // Parse frame rate
      let fps = 10;
      if (videoStream.r_frame_rate) {
        const [num, den] = videoStream.r_frame_rate.split('/').map(Number);
        if (den > 0) fps = num / den;
      }

      resolve({
        fps: Math.min(fps, 50), // Cap at 50 fps
        width: videoStream.width || 0,
        height: videoStream.height || 0,
        duration: parseFloat(metadata.format.duration) || 0,
        frameCount: parseInt(videoStream.nb_frames) || 0,
      });
    });
  });
}

/**
 * Extract frames from GIF using FFmpeg
 * @param {string} gifPath - Path to GIF file
 * @param {string} outputDir - Directory to save frames
 * @returns {Promise<{framePaths: string[], metadata: Object}>}
 */
export async function extractFrames(gifPath, outputDir) {
  await fs.mkdir(outputDir, { recursive: true });

  const metadata = await getGifMetadata(gifPath);
  const framePattern = path.join(outputDir, 'frame_%04d.png');

  return new Promise((resolve, reject) => {
    ffmpeg(gifPath)
      .outputOptions(['-vsync', '0'])
      .output(framePattern)
      .on('end', async () => {
        const files = await fs.readdir(outputDir);
        const framePaths = files
          .filter(f => f.startsWith('frame_') && f.endsWith('.png'))
          .sort()
          .map(f => path.join(outputDir, f));

        resolve({ framePaths, metadata });
      })
      .on('error', reject)
      .run();
  });
}

/**
 * Load frames as canvas images
 * @param {string[]} framePaths - Array of frame file paths
 * @returns {Promise<Image[]>} Array of loaded images
 */
export async function loadFrames(framePaths) {
  return Promise.all(framePaths.map(p => loadImage(p)));
}

/**
 * Assemble frames back into GIF using FFmpeg
 * @param {string[]} framePaths - Paths to frame images
 * @param {string} outputPath - Output GIF path
 * @param {number} fps - Frames per second
 * @returns {Promise<string>} Output path
 */
export async function assembleGif(framePaths, outputPath, fps = 10) {
  if (framePaths.length === 0) {
    throw new Error('No frames provided');
  }

  // Create temp directory with sequential frame names
  const tempDir = path.join(path.dirname(outputPath), 'temp_frames_' + Date.now());
  await fs.mkdir(tempDir, { recursive: true });

  // Copy frames with sequential naming
  for (let i = 0; i < framePaths.length; i++) {
    const destPath = path.join(tempDir, `frame_${String(i).padStart(4, '0')}.png`);
    await fs.copyFile(framePaths[i], destPath);
  }

  const framePattern = path.join(tempDir, 'frame_%04d.png');
  const palettePath = path.join(tempDir, 'palette.png');

  try {
    // Generate palette for better quality
    await new Promise((resolve, reject) => {
      ffmpeg(framePattern)
        .inputOptions(['-framerate', String(fps)])
        .outputOptions(['-vf', 'palettegen=max_colors=256:stats_mode=diff'])
        .output(palettePath)
        .on('end', resolve)
        .on('error', reject)
        .run();
    });

    // Create GIF with palette
    await new Promise((resolve, reject) => {
      ffmpeg()
        .input(framePattern)
        .inputOptions(['-framerate', String(fps)])
        .input(palettePath)
        .complexFilter(['paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle'])
        .outputOptions(['-loop', '0'])
        .output(outputPath)
        .on('end', resolve)
        .on('error', reject)
        .run();
    });
  } finally {
    // Cleanup temp directory
    await fs.rm(tempDir, { recursive: true, force: true });
  }

  return outputPath;
}

/**
 * Extract frames using pure JavaScript (gifuct-js)
 * Fallback when FFmpeg is not available
 * @param {Buffer} gifBuffer - GIF file buffer
 * @returns {Promise<{frames: ImageData[], delays: number[], width: number, height: number}>}
 */
export async function extractFramesPure(gifBuffer) {
  const gif = parseGIF(gifBuffer);
  const frames = decompressFrames(gif, true);

  const width = gif.lsd.width;
  const height = gif.lsd.height;

  // Create full frames (handling disposal)
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');

  const processedFrames = [];
  const delays = [];

  for (const frame of frames) {
    // Handle disposal
    if (frame.disposalType === 2) {
      ctx.clearRect(0, 0, width, height);
    }

    // Create ImageData from frame
    const imageData = ctx.createImageData(frame.dims.width, frame.dims.height);
    imageData.data.set(frame.patch);

    // Draw frame at correct position
    ctx.putImageData(imageData, frame.dims.left, frame.dims.top);

    // Capture full canvas
    const fullFrame = ctx.getImageData(0, 0, width, height);
    processedFrames.push(fullFrame);
    delays.push(frame.delay || 100);
  }

  return { frames: processedFrames, delays, width, height };
}

/**
 * Create GIF from ImageData frames using pure JavaScript
 * @param {ImageData[]} frames - Array of ImageData frames
 * @param {number} width - GIF width
 * @param {number} height - GIF height
 * @param {number[]} delays - Frame delays in ms
 * @returns {Promise<Buffer>} GIF buffer
 */
export async function createGifPure(frames, width, height, delays) {
  const encoder = new GIFEncoder(width, height, 'neuquant', true);

  encoder.start();
  encoder.setRepeat(0); // Loop forever
  encoder.setQuality(10);

  for (let i = 0; i < frames.length; i++) {
    encoder.setDelay(delays[i] || 100);

    const canvas = createCanvas(width, height);
    const ctx = canvas.getContext('2d');
    ctx.putImageData(frames[i], 0, 0);

    encoder.addFrame(ctx);
  }

  encoder.finish();

  return encoder.out.getData();
}

export default {
  getGifMetadata,
  extractFrames,
  loadFrames,
  assembleGif,
  extractFramesPure,
  createGifPure,
};
