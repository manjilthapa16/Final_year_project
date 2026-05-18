const express = require('express');
const cors = require('cors');

const { analyzePose } = require('./analyzePose');

const app = express();

const PORT = Number(process.env.PORT || 4000);
const ALLOWED_POSES = ['squat', 'plank', 'downdog', 'tree', 'warrior2', 'goddess', 'sitting'];

app.use(cors());
app.use(express.json({ limit: '2mb' }));

app.get('/api/health', (_req, res) => {
  res.json({ ok: true });
});

app.get('/api/poses', (_req, res) => {
  res.json({ poses: ALLOWED_POSES });
});

app.post('/api/analyze', async (req, res) => {
  const selectedPose = (req.body.selectedPose || '').toLowerCase().trim();
  const landmarks = req.body.landmarks;
  const previousFeedback = req.body.previousFeedback;
  const previousMetrics = req.body.previousMetrics;

  if (!landmarks || !Array.isArray(landmarks)) {
    res.status(400).json({ error: 'Missing or invalid landmarks' });
    return;
  }

  if (!ALLOWED_POSES.includes(selectedPose)) {
    res.status(400).json({ error: 'Invalid selectedPose' });
    return;
  }

  try {
    const result = await analyzePose({
      landmarks,
      selectedPose,
      previousFeedback,
      previousMetrics,
    });
    res.json(result);
  } catch (err) {
    res.status(500).json({
      error: 'Failed to analyze pose',
      detail: err.message
    });
  }
});

app.listen(PORT, () => {
  console.log(`Your Gym Buddy API running on http://localhost:${PORT}`);
});
