# HeartFace – Heart Rate Monitor

A contact-free heart rate detector that works entirely in your browser, using **Eulerian Video Magnification** (EVM) concepts.
Point your front camera at your face, keep still for ~10 seconds, and your BPM appears.

## How it works

The algorithm mirrors the [MIT CSAIL EVM paper (Wu et al., 2012)](http://people.csail.mit.edu/mrub/vidmag/):

| Step | What happens |
|------|-------------|
| **Gaussian pyramid** | Each video frame is spatially decomposed into a multi-scale pyramid. The coarsest level gives a spatially-averaged colour per channel — the EVM "spatial decomposition". |
| **Temporal signal** | Mean R/G/B from the forehead ROI is accumulated over 500+ frames, forming a time-series. Blood pumping through capillaries causes a ~0.5 % periodic colour shift. |
| **CHROM normalisation** | De Haan & Jeanne's chrominance projection `3R − 2G` cancels illumination noise and isolates haemoglobin absorption. |
| **Bandpass + FFT** | Signal is filtered to 0.75–4 Hz (45–240 BPM) via an IIR cascade, then a Hanning-windowed FFT with parabolic peak interpolation yields the dominant frequency → BPM. |
| **EVM visualisation** | "Show Pulse Glow" amplifies the bandpassed signal as a real-time colour overlay on your face — red during systole, cyan during diastole — making the pulse *visible*, as in the MIT demos. |

All processing runs **entirely in the browser**. No video is ever sent to a server.

## Quick start

```bash
# 1. Clone
git clone https://github.com/RichardNewcombe/heart-face.git
cd heart-face

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Run
python app.py
# → open http://localhost:5000 in Chrome / Safari
```

For HTTPS (required for camera on non-localhost):

```bash
# Using a reverse proxy (e.g. ngrok for testing)
ngrok http 5000
```

## Deploy

The app includes a `Procfile` for Heroku / Railway:

```bash
heroku create
git push heroku main
```

Or deploy to any platform that runs Python (Fly.io, Render, etc.).

## Installing as a PWA

- **Android (Chrome):** tap the "Install App" button that appears, or use the browser menu.
- **iOS (Safari):** tap Share → "Add to Home Screen".

## Project structure

```
app.py                  Flask server (serves static files + templates)
requirements.txt
Procfile
templates/
  index.html            Single-page PWA
static/
  manifest.json         PWA manifest
  service-worker.js     Offline cache
  css/style.css
  js/app.js             All signal processing + UI (no external dependencies)
  icons/                SVG + PNG app icons
generate_icons.py       Regenerate PNG icons from SVG (needs cairosvg)
```

## References

- Wu, H.-Y., Rubinstein, M., Shih, E., Guttag, J., Durand, F., & Freeman, W. T. (2012). **Eulerian Video Magnification for Revealing Subtle Changes in the World.** *ACM SIGGRAPH*.
- De Haan, G., & Jeanne, V. (2013). **Robust Pulse Rate From Chrominance-Based rPPG.** *IEEE Trans. Biomed. Eng.*
