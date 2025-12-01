/**
 * Face replacement module
 * Combines face detection and image processing to swap faces
 */

import { createCanvas, loadImage } from 'canvas';
import sharp from 'sharp';
import { detectFaces, detectLargestFace, createFaceMask } from './faceDetector.js';

/**
 * Face replacer class - handles face swapping in images
 */
export class FaceReplacer {
  constructor() {
    this.sourceFace = null;
    this.sourceFaceBuffer = null;
  }

  /**
   * Set the source face to use for replacement
   * @param {Buffer} imageBuffer - Image buffer containing a face
   * @returns {Promise<boolean>} True if face was found
   */
  async setSourceFace(imageBuffer) {
    const face = await detectLargestFace(imageBuffer);

    if (!face) {
      return false;
    }

    this.sourceFace = face;

    // Extract face region with padding
    const img = await loadImage(imageBuffer);
    const padding = 0.4;
    const padW = Math.round(face.box.width * padding);
    const padH = Math.round(face.box.height * padding);

    const x = Math.max(0, face.box.x - padW);
    const y = Math.max(0, face.box.y - padH);
    const width = Math.min(img.width - x, face.box.width + 2 * padW);
    const height = Math.min(img.height - y, face.box.height + 2 * padH);

    const canvas = createCanvas(width, height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, x, y, width, height, 0, 0, width, height);

    this.sourceFaceBuffer = canvas.toBuffer('image/png');

    return true;
  }

  /**
   * Replace faces in a frame
   * @param {Buffer} frameBuffer - Frame image buffer
   * @param {number} blendStrength - Blend strength (0-1)
   * @returns {Promise<Buffer>} Processed frame buffer
   */
  async replaceFacesInFrame(frameBuffer, blendStrength = 0.9) {
    if (!this.sourceFaceBuffer) {
      return frameBuffer;
    }

    const faces = await detectFaces(frameBuffer);

    if (faces.length === 0) {
      return frameBuffer;
    }

    let result = frameBuffer;

    for (const face of faces) {
      result = await this.replaceSingleFace(result, face.box, blendStrength);
    }

    return result;
  }

  /**
   * Replace a single face in an image
   * @param {Buffer} frameBuffer - Frame buffer
   * @param {Object} box - Face bounding box
   * @param {number} blendStrength - Blend strength
   * @returns {Promise<Buffer>} Processed frame
   */
  async replaceSingleFace(frameBuffer, box, blendStrength) {
    const frameImg = await loadImage(frameBuffer);
    const sourceImg = await loadImage(this.sourceFaceBuffer);

    // Create output canvas
    const canvas = createCanvas(frameImg.width, frameImg.height);
    const ctx = canvas.getContext('2d');

    // Draw original frame
    ctx.drawImage(frameImg, 0, 0);

    // Resize source face to match target face size
    const resizedSource = await sharp(this.sourceFaceBuffer)
      .resize(box.width, box.height, { fit: 'fill' })
      .toBuffer();

    const resizedImg = await loadImage(resizedSource);

    // Create elliptical mask
    const maskCanvas = createCanvas(box.width, box.height);
    const maskCtx = maskCanvas.getContext('2d');

    // Draw ellipse mask with gradient for smooth edges
    const centerX = box.width / 2;
    const centerY = box.height / 2;
    const radiusX = box.width * 0.45;
    const radiusY = box.height * 0.48;

    // Create radial gradient for soft edges
    const gradient = maskCtx.createRadialGradient(
      centerX, centerY, Math.min(radiusX, radiusY) * 0.5,
      centerX, centerY, Math.max(radiusX, radiusY)
    );
    gradient.addColorStop(0, `rgba(255, 255, 255, ${blendStrength})`);
    gradient.addColorStop(0.7, `rgba(255, 255, 255, ${blendStrength * 0.8})`);
    gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');

    maskCtx.fillStyle = gradient;
    maskCtx.beginPath();
    maskCtx.ellipse(centerX, centerY, radiusX, radiusY, 0, 0, Math.PI * 2);
    maskCtx.fill();

    // Apply color matching
    const colorMatchedSource = await this.matchColors(
      resizedSource,
      frameBuffer,
      box
    );
    const colorMatchedImg = await loadImage(colorMatchedSource);

    // Create temporary canvas for blending
    const tempCanvas = createCanvas(box.width, box.height);
    const tempCtx = tempCanvas.getContext('2d');

    // Draw color-matched source face
    tempCtx.drawImage(colorMatchedImg, 0, 0);

    // Get image data for manual alpha blending
    const sourceData = tempCtx.getImageData(0, 0, box.width, box.height);
    const maskData = maskCtx.getImageData(0, 0, box.width, box.height);

    // Get target region
    const targetX = Math.max(0, box.x);
    const targetY = Math.max(0, box.y);
    const targetWidth = Math.min(box.width, frameImg.width - targetX);
    const targetHeight = Math.min(box.height, frameImg.height - targetY);

    if (targetWidth <= 0 || targetHeight <= 0) {
      return frameBuffer;
    }

    const targetData = ctx.getImageData(targetX, targetY, targetWidth, targetHeight);

    // Blend pixels
    for (let y = 0; y < targetHeight; y++) {
      for (let x = 0; x < targetWidth; x++) {
        const srcIdx = (y * box.width + x) * 4;
        const dstIdx = (y * targetWidth + x) * 4;

        // Get mask alpha value
        const maskAlpha = maskData.data[srcIdx + 3] / 255;

        if (maskAlpha > 0) {
          // Blend source and target based on mask
          targetData.data[dstIdx] = Math.round(
            sourceData.data[srcIdx] * maskAlpha +
            targetData.data[dstIdx] * (1 - maskAlpha)
          );
          targetData.data[dstIdx + 1] = Math.round(
            sourceData.data[srcIdx + 1] * maskAlpha +
            targetData.data[dstIdx + 1] * (1 - maskAlpha)
          );
          targetData.data[dstIdx + 2] = Math.round(
            sourceData.data[srcIdx + 2] * maskAlpha +
            targetData.data[dstIdx + 2] * (1 - maskAlpha)
          );
        }
      }
    }

    // Put blended data back
    ctx.putImageData(targetData, targetX, targetY);

    return canvas.toBuffer('image/png');
  }

  /**
   * Match colors of source to target region
   * @param {Buffer} sourceBuffer - Source image buffer
   * @param {Buffer} targetBuffer - Target frame buffer
   * @param {Object} box - Target region box
   * @returns {Promise<Buffer>} Color-matched source buffer
   */
  async matchColors(sourceBuffer, targetBuffer, box) {
    try {
      // Get target region for color matching
      const targetRegion = await sharp(targetBuffer)
        .extract({
          left: Math.max(0, box.x),
          top: Math.max(0, box.y),
          width: box.width,
          height: box.height,
        })
        .toBuffer();

      // Get stats for both images
      const [sourceStats, targetStats] = await Promise.all([
        sharp(sourceBuffer).stats(),
        sharp(targetRegion).stats(),
      ]);

      // Calculate adjustment factors for each channel
      const adjustments = sourceStats.channels.map((srcCh, i) => {
        const tgtCh = targetStats.channels[i];
        return {
          meanDiff: tgtCh.mean - srcCh.mean,
          stdRatio: srcCh.stdev > 0 ? tgtCh.stdev / srcCh.stdev : 1,
        };
      });

      // Apply color correction using sharp's modulate
      // This is a simplified version - for better results, LAB color space would be ideal
      const avgBrightness = adjustments.reduce((sum, a) => sum + a.meanDiff, 0) / 3;

      let result = sharp(sourceBuffer);

      // Adjust brightness
      if (Math.abs(avgBrightness) > 5) {
        const brightnessFactor = 1 + (avgBrightness / 255);
        result = result.modulate({
          brightness: Math.max(0.5, Math.min(1.5, brightnessFactor)),
        });
      }

      return result.toBuffer();
    } catch (error) {
      // Return original if color matching fails
      return sourceBuffer;
    }
  }

  /**
   * Process multiple GIF frames
   * @param {Buffer[]} frameBuffers - Array of frame buffers
   * @param {number} blendStrength - Blend strength
   * @param {Function} onProgress - Progress callback (current, total)
   * @returns {Promise<Buffer[]>} Processed frame buffers
   */
  async processFrames(frameBuffers, blendStrength = 0.9, onProgress = null) {
    const results = [];

    for (let i = 0; i < frameBuffers.length; i++) {
      const processed = await this.replaceFacesInFrame(frameBuffers[i], blendStrength);
      results.push(processed);

      if (onProgress) {
        onProgress(i + 1, frameBuffers.length);
      }
    }

    return results;
  }
}

/**
 * Convert canvas ImageData to Buffer
 * @param {ImageData} imageData - Canvas ImageData
 * @param {number} width - Image width
 * @param {number} height - Image height
 * @returns {Buffer} PNG buffer
 */
export function imageDataToBuffer(imageData, width, height) {
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');
  ctx.putImageData(imageData, 0, 0);
  return canvas.toBuffer('image/png');
}

/**
 * Convert Buffer to canvas ImageData
 * @param {Buffer} buffer - Image buffer
 * @returns {Promise<{imageData: ImageData, width: number, height: number}>}
 */
export async function bufferToImageData(buffer) {
  const img = await loadImage(buffer);
  const canvas = createCanvas(img.width, img.height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);

  return {
    imageData: ctx.getImageData(0, 0, img.width, img.height),
    width: img.width,
    height: img.height,
  };
}

export default FaceReplacer;
