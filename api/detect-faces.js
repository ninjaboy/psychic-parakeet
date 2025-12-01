/**
 * Face detection info endpoint
 * Actual detection happens client-side with TensorFlow.js
 */

export default function handler(req, res) {
  res.status(200).json({
    message: 'Face detection runs client-side using TensorFlow.js',
    info: 'Use the web UI at / for full functionality',
    clientLibraries: [
      'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs',
      'https://cdn.jsdelivr.net/npm/@tensorflow-models/face-detection'
    ]
  });
}
