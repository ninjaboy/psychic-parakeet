/**
 * Face detection info endpoint
 */

module.exports = function handler(req, res) {
  res.status(200).json({
    message: 'Face detection runs client-side using TensorFlow.js',
    info: 'Use the web UI at / for full functionality'
  });
};
