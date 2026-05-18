const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');
const crypto = require('crypto');

function analyzePose({ landmarks, selectedPose, previousFeedback, previousMetrics }) {
  return new Promise((resolve, reject) => {
    const projectRoot = path.resolve(__dirname, '..', '..');
    const pose = (selectedPose || '').toLowerCase().trim();
    const specialistScripts = {
      squat: 'squat_analysis.py',
      plank: 'plank_analysis.py',
      downdog: 'downdog_analysis.py',
      tree: 'tree_analysis.py',
      goddess: 'goddess_analysis.py',
      sitting: 'sitting_pose.py',
    };
    const scriptPath = path.join(projectRoot, 'scripts', specialistScripts[pose] || 'web_pose_feedback.py');
    const pythonCandidates = [
      process.env.GYM_BUDDY_PYTHON,
      path.join(projectRoot, '.venv39', 'bin', 'python'),
      path.join(projectRoot, '.venv', 'bin', 'python'),
      path.join(projectRoot, 'venv', 'bin', 'python'),
      'python3',
    ].filter(Boolean);

    const pythonBin = pythonCandidates.find((candidate) => {
      if (candidate === 'python3') {
        return true;
      }
      return fs.existsSync(candidate);
    }) || 'python3';

    const tempFilePath = path.join(os.tmpdir(), `landmarks_${crypto.randomBytes(6).toString('hex')}.json`);
    fs.writeFileSync(tempFilePath, JSON.stringify(landmarks));

    const args = [scriptPath, '--landmarks', tempFilePath];
    if (pose === 'squat' || pose === 'sitting' || !specialistScripts[pose]) {
      args.push('--pose', selectedPose);
      if (previousFeedback) args.push('--previous_feedback', previousFeedback);
      if (previousMetrics) args.push('--previous_metrics', JSON.stringify(previousMetrics));
    }

    const py = spawn(pythonBin, args, {
      cwd: projectRoot,
      stdio: ['ignore', 'pipe', 'pipe']
    });

    let stdout = '';
    let stderr = '';

    const timeoutId = setTimeout(() => {
      py.kill('SIGKILL');
      fs.unlink(tempFilePath, () => {});
      reject(new Error('Python analyzer timed out'));
    }, 20000);

    py.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
    });

    py.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });

    py.on('error', (err) => {
      clearTimeout(timeoutId);
      fs.unlink(tempFilePath, () => {});
      reject(err);
    });

    py.on('close', (code) => {
      clearTimeout(timeoutId);
      fs.unlink(tempFilePath, () => {});
      if (code !== 0) {
        reject(new Error(`Analyzer failed (${code}): ${stderr || stdout}`));
        return;
      }

      try {
        const parsed = JSON.parse(stdout.trim());
        resolve(parsed);
      } catch (err) {
        reject(new Error(`Invalid analyzer output: ${stdout || stderr}`));
      }
    });
  });
}

module.exports = { analyzePose };
