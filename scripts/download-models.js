/**
 * Script to download face-api.js models
 */

import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import https from 'https';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const MODELS_DIR = path.join(__dirname, '..', 'models');
const BASE_URL = 'https://raw.githubusercontent.com/vladmandic/face-api/master/model';

const MODELS = [
  // SSD MobileNet v1
  'ssd_mobilenetv1_model-weights_manifest.json',
  'ssd_mobilenetv1_model-shard1',
  'ssd_mobilenetv1_model-shard2',
  // Face Landmark 68
  'face_landmark_68_model-weights_manifest.json',
  'face_landmark_68_model-shard1',
  // Face Recognition
  'face_recognition_model-weights_manifest.json',
  'face_recognition_model-shard1',
  'face_recognition_model-shard2',
];

async function downloadFile(url, destPath) {
  return new Promise((resolve, reject) => {
    const file = require('fs').createWriteStream(destPath);

    https.get(url, (response) => {
      if (response.statusCode === 302 || response.statusCode === 301) {
        // Follow redirect
        https.get(response.headers.location, (res) => {
          res.pipe(file);
          file.on('finish', () => {
            file.close();
            resolve();
          });
        }).on('error', reject);
      } else if (response.statusCode === 200) {
        response.pipe(file);
        file.on('finish', () => {
          file.close();
          resolve();
        });
      } else {
        reject(new Error(`HTTP ${response.statusCode}`));
      }
    }).on('error', reject);
  });
}

async function downloadWithFetch(url, destPath) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  const buffer = await response.arrayBuffer();
  await fs.writeFile(destPath, Buffer.from(buffer));
}

async function main() {
  console.log('Creating models directory...');
  await fs.mkdir(MODELS_DIR, { recursive: true });

  console.log('Downloading face-api.js models...\n');

  for (const model of MODELS) {
    const url = `${BASE_URL}/${model}`;
    const destPath = path.join(MODELS_DIR, model);

    try {
      // Check if already exists
      await fs.access(destPath);
      console.log(`✓ ${model} (already exists)`);
    } catch {
      process.stdout.write(`Downloading ${model}...`);
      try {
        await downloadWithFetch(url, destPath);
        console.log(' ✓');
      } catch (error) {
        console.log(` ✗ (${error.message})`);
      }
    }
  }

  console.log('\nDone! Models saved to:', MODELS_DIR);
}

main().catch(console.error);
