"use strict";

/**
 * HeartFace – Heart Rate via Eulerian Video Magnification
 *
 * Algorithm (EVM-inspired, De Haan & Jeanne CHROM method):
 *  1. For each video frame, extract the face / forehead ROI
 *  2. Build a Gaussian spatial pyramid; use coarsest level → mean RGB per frame
 *     (This IS the EVM spatial decomposition — we work at the lowest pyramid level
 *      to capture the spatially-averaged colour change caused by blood flow.)
 *  3. Accumulate a temporal RGB signal buffer
 *  4. Apply CHROM normalisation to cancel illumination variation
 *  5. Bandpass filter 0.75–4 Hz  (45–240 BPM) via IIR cascade
 *  6. FFT on filtered buffer → dominant frequency → BPM
 *  7. EVM visualisation: amplify bandpassed signal as coloured overlay on the face
 *     to make the pulse *visible*, as in the original MIT EVM paper.
 */

// ═══════════════════════════════════════════════════════════════════════════
//  DSP Utilities
// ═══════════════════════════════════════════════════════════════════════════

const DSP = {
  /** Cooley-Tukey radix-2 in-place FFT (Float64Arrays, length must be power-of-2) */
  fft(re, im) {
    const n = re.length;
    // Bit-reversal permutation
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        [re[i], re[j]] = [re[j], re[i]];
        [im[i], im[j]] = [im[j], im[i]];
      }
    }
    // Butterfly stages
    for (let len = 2; len <= n; len <<= 1) {
      const halfLen = len >> 1;
      const ang = (-2 * Math.PI) / len;
      const wCos = Math.cos(ang);
      const wSin = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let cr = 1.0,
          ci = 0.0;
        for (let k = 0; k < halfLen; k++) {
          const ur = re[i + k],
            ui = im[i + k];
          const vr = re[i + k + halfLen] * cr - im[i + k + halfLen] * ci;
          const vi = re[i + k + halfLen] * ci + im[i + k + halfLen] * cr;
          re[i + k] = ur + vr;
          im[i + k] = ui + vi;
          re[i + k + halfLen] = ur - vr;
          im[i + k + halfLen] = ui - vi;
          const tmp = cr * wCos - ci * wSin;
          ci = cr * wSin + ci * wCos;
          cr = tmp;
        }
      }
    }
  },

  nextPow2(n) {
    let p = 1;
    while (p < n) p <<= 1;
    return p;
  },

  /** Remove linear trend from signal */
  detrend(arr) {
    const n = arr.length;
    let sx = 0,
      sy = 0,
      sxy = 0,
      sx2 = 0;
    for (let i = 0; i < n; i++) {
      sx += i;
      sy += arr[i];
      sxy += i * arr[i];
      sx2 += i * i;
    }
    const denom = n * sx2 - sx * sx;
    if (Math.abs(denom) < 1e-10) return arr.slice();
    const slope = (n * sxy - sx * sy) / denom;
    const bias = (sy - slope * sx) / n;
    return arr.map((v, i) => v - (slope * i + bias));
  },

  /** Hanning window array of length n */
  hanning(n) {
    return Float64Array.from({ length: n }, (_, i) =>
      0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1))
    );
  },

  /**
   * Estimate dominant heart-rate frequency in `signal` sampled at `fps`.
   * Returns { bpm, frequency, confidence, spectrum } or null.
   */
  estimateHeartRate(signal, fps, fMin = 0.75, fMax = 4.0) {
    const n = signal.length;
    if (n < 30) return null;

    const detrended = DSP.detrend(signal);
    const win = DSP.hanning(n);

    const fftLen = DSP.nextPow2(n); // zero-pad → higher freq resolution
    const re = new Float64Array(fftLen);
    const im = new Float64Array(fftLen);
    for (let i = 0; i < n; i++) re[i] = detrended[i] * win[i];

    DSP.fft(re, im);

    const df = fps / fftLen; // Hz per bin
    const iMin = Math.max(1, Math.ceil(fMin / df));
    const iMax = Math.min(Math.floor(fftLen / 2) - 1, Math.floor(fMax / df));

    let maxPow = -Infinity,
      maxIdx = iMin,
      totalPow = 0;
    const spectrum = [];
    const pow = (i) => re[i] * re[i] + im[i] * im[i];

    for (let i = iMin; i <= iMax; i++) {
      const p = pow(i);
      totalPow += p;
      spectrum.push({ freq: i * df, bpm: i * df * 60, power: p });
      if (p > maxPow) {
        maxPow = p;
        maxIdx = i;
      }
    }

    // Parabolic sub-bin interpolation for smoother estimate
    let freq = maxIdx * df;
    if (maxIdx > iMin && maxIdx < iMax) {
      const p0 = Math.sqrt(pow(maxIdx - 1));
      const p1 = Math.sqrt(pow(maxIdx));
      const p2 = Math.sqrt(pow(maxIdx + 1));
      const d2 = 2 * p1 - p0 - p2;
      if (Math.abs(d2) > 1e-10) {
        const delta = Math.max(-0.5, Math.min(0.5, (p2 - p0) / (2 * d2)));
        freq = (maxIdx + delta) * df;
      }
    }

    const avgPow = totalPow / (iMax - iMin + 1);
    // SNR-based confidence: 0 when SNR~1 (flat), 1 when dominant peak >> noise
    const snr = maxPow / (avgPow + 1e-10);
    const confidence = Math.min(1, Math.max(0, (snr - 1.5) / 8.5));

    return { bpm: freq * 60, frequency: freq, confidence, spectrum };
  },

  /**
   * IIR bandpass filter using first-order HP + LP cascade.
   * Returns new float array.
   */
  bandpass(signal, fps, fLow, fHigh) {
    const n = signal.length;
    if (n === 0) return [];

    // High-pass: y[n] = a*(y[n-1] + x[n] - x[n-1])
    const aHP = Math.exp((-2 * Math.PI * fLow) / fps);
    const hp = new Float64Array(n);
    for (let i = 1; i < n; i++) {
      hp[i] = aHP * (hp[i - 1] + signal[i] - signal[i - 1]);
    }

    // Low-pass on HP output: y[n] = a*y[n-1] + (1-a)*x[n]
    const aLP = Math.exp((-2 * Math.PI * fHigh) / fps);
    const bp = new Float64Array(n);
    bp[0] = hp[0];
    for (let i = 1; i < n; i++) {
      bp[i] = aLP * bp[i - 1] + (1 - aLP) * hp[i];
    }

    return Array.from(bp);
  },
};

// ═══════════════════════════════════════════════════════════════════════════
//  Gaussian Pyramid – EVM spatial decomposition
// ═══════════════════════════════════════════════════════════════════════════

class GaussianPyramid {
  constructor(levels = 3) {
    this.levels = levels;
    // Reusable off-screen canvas for downsampling
    this._canvas = document.createElement("canvas");
    this._ctx = this._canvas.getContext("2d", { willReadFrequently: true });
  }

  /**
   * Extract the mean colour at the coarsest pyramid level from `roi` in `source`.
   * @param {HTMLVideoElement} source
   * @param {{ x, y, w, h }} roi
   * @returns {{ r, g, b }}
   */
  extractMean(source, roi) {
    const { x, y, w, h } = roi;
    this._canvas.width = w;
    this._canvas.height = h;
    this._ctx.drawImage(source, x, y, w, h, 0, 0, w, h);

    let imgData = this._ctx.getImageData(0, 0, w, h);
    let level = { data: imgData.data, w, h };

    for (let l = 0; l < this.levels; l++) {
      level = this._downsample(level.data, level.w, level.h);
    }

    let r = 0,
      g = 0,
      b = 0;
    const n = level.w * level.h;
    for (let i = 0; i < n; i++) {
      r += level.data[i * 4];
      g += level.data[i * 4 + 1];
      b += level.data[i * 4 + 2];
    }
    return { r: r / n, g: g / n, b: b / n };
  }

  /**
   * Build a full Gaussian pyramid from the ROI, returning all levels.
   * Level 0 = original ROI pixels, level N = N times downsampled.
   * @param {HTMLVideoElement} source
   * @param {{ x, y, w, h }} roi
   * @param {number} numLevels
   * @returns {{ data: Uint8ClampedArray, w: number, h: number }[]}
   */
  buildPyramid(source, roi, numLevels) {
    const { x, y, w, h } = roi;
    this._canvas.width = w;
    this._canvas.height = h;
    this._ctx.drawImage(source, x, y, w, h, 0, 0, w, h);
    const imgData = this._ctx.getImageData(0, 0, w, h);

    const pyramid = [{ data: imgData.data, w, h }];
    let current = pyramid[0];
    for (let l = 0; l < numLevels; l++) {
      current = this._downsample(current.data, current.w, current.h);
      pyramid.push(current);
    }
    return pyramid;
  }

  /** 2× Gaussian downsample (box filter over 2×2 blocks) */
  _downsample(src, w, h) {
    const nw = Math.max(1, w >> 1);
    const nh = Math.max(1, h >> 1);
    const dst = new Uint8ClampedArray(nw * nh * 4);

    for (let y = 0; y < nh; y++) {
      for (let x = 0; x < nw; x++) {
        let r = 0,
          g = 0,
          b = 0,
          cnt = 0;
        for (let dy = 0; dy < 2; dy++) {
          for (let dx = 0; dx < 2; dx++) {
            const px = x * 2 + dx,
              py = y * 2 + dy;
            if (px < w && py < h) {
              const si = (py * w + px) * 4;
              r += src[si];
              g += src[si + 1];
              b += src[si + 2];
              cnt++;
            }
          }
        }
        const di = (y * nw + x) * 4;
        dst[di] = r / cnt;
        dst[di + 1] = g / cnt;
        dst[di + 2] = b / cnt;
        dst[di + 3] = 255;
      }
    }
    return { data: dst, w: nw, h: nh };
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Signal Buffer & Heart Rate Estimator
// ═══════════════════════════════════════════════════════════════════════════

class SignalEngine {
  /** Maximum number of frames to keep (~17 s at 30 fps) */
  static MAX_FRAMES = 512;
  /** Minimum frames before attempting estimation (~2 s) */
  static MIN_FRAMES = 60;

  constructor() {
    this.pyramid = new GaussianPyramid(3);
    /** @type {{ r: number, g: number, b: number }[]} */
    this.buffer = [];
    /** @type {number[]} DOMHighResTimeStamp per frame */
    this.timestamps = [];
    this.fps = 30;
  }

  /** Add one frame's colour sample to the buffer. */
  addSample(source, roi, timestamp) {
    const mean = this.pyramid.extractMean(source, roi);
    this.buffer.push(mean);
    this.timestamps.push(timestamp);

    if (this.buffer.length > SignalEngine.MAX_FRAMES) {
      this.buffer.shift();
      this.timestamps.shift();
    }

    // Estimate fps from last 10 frames
    const len = this.timestamps.length;
    if (len > 10) {
      const elapsed = this.timestamps[len - 1] - this.timestamps[len - 11];
      if (elapsed > 0) this.fps = (10 * 1000) / elapsed;
    }
  }

  /**
   * CHROM signal (De Haan & Jeanne 2013) – more robust to lighting changes
   * than a single green channel.
   * Xs = 3Rn − 2Gn  (highly correlated with pulse)
   */
  getSignal() {
    const n = this.buffer.length;
    if (n === 0) return [];
    const WINSIZE = Math.min(n, 64); // sliding normalisation window

    const sig = new Array(n).fill(0);
    for (let i = WINSIZE - 1; i < n; i++) {
      let mR = 0,
        mG = 0,
        mB = 0;
      for (let k = i - WINSIZE + 1; k <= i; k++) {
        mR += this.buffer[k].r;
        mG += this.buffer[k].g;
        mB += this.buffer[k].b;
      }
      mR /= WINSIZE;
      mG /= WINSIZE;
      mB /= WINSIZE;
      const rn = this.buffer[i].r / (mR + 1e-6) - 1;
      const gn = this.buffer[i].g / (mG + 1e-6) - 1;
      const bn = this.buffer[i].b / (mB + 1e-6) - 1; // eslint-disable-line no-unused-vars
      // CHROM Xs component
      sig[i] = 3 * rn - 2 * gn;
    }
    return sig;
  }

  /** Bandpassed version of the CHROM signal (for EVM overlay visualisation). */
  getBandpassedSignal() {
    const raw = this.getSignal();
    if (raw.length < 10) return raw;
    return DSP.bandpass(raw, this.fps, 0.75, 4.0);
  }

  /**
   * Estimate heart rate.
   * @returns {{ status: 'collecting'|'ok'|'error', bpm?, confidence?, framesNeeded? }}
   */
  estimate() {
    if (this.buffer.length < SignalEngine.MIN_FRAMES) {
      return {
        status: "collecting",
        framesNeeded: SignalEngine.MIN_FRAMES - this.buffer.length,
      };
    }
    const sig = this.getSignal();
    const result = DSP.estimateHeartRate(sig, this.fps);
    if (!result) return { status: "error" };
    return { status: "ok", ...result };
  }

  clear() {
    this.buffer = [];
    this.timestamps = [];
  }

  get frameCount() {
    return this.buffer.length;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  ROI / Face Detector
// ═══════════════════════════════════════════════════════════════════════════

class ROITracker {
  constructor() {
    /** @type {FaceDetector|null} */
    this._detector = null;
    this._lastROI = null;
    this._lastDetectTime = 0;
    this._detectInterval = 800; // ms between face-api calls
    this._busy = false;

    if ("FaceDetector" in window) {
      try {
        this._detector = new window.FaceDetector({
          fastMode: true,
          maxDetectedFaces: 1,
        });
      } catch (_) {
        this._detector = null;
      }
    }
  }

  /**
   * Returns best-effort { x, y, w, h, faceBox } in *video* pixel coordinates.
   * faceBox is the full face; x/y/w/h is the forehead ROI used for extraction.
   */
  async getROI(video) {
    const vw = video.videoWidth,
      vh = video.videoHeight;
    if (!vw || !vh) return null;

    const now = performance.now();
    const stale = now - this._lastDetectTime > this._detectInterval;

    if (stale && !this._busy && this._detector) {
      this._busy = true;
      this._lastDetectTime = now;
      try {
        const faces = await this._detector.detect(video);
        if (faces.length > 0) {
          const bb = faces[0].boundingBox;
          this._lastROI = this._roiFromFace(bb.x, bb.y, bb.width, bb.height);
        }
      } catch (_) {
        // FaceDetector can fail — just keep last ROI
      } finally {
        this._busy = false;
      }
    }

    return this._lastROI || this._centerROI(vw, vh);
  }

  _roiFromFace(fx, fy, fw, fh) {
    return {
      // Forehead: horizontally centred 60% of face width, top 25% of face height
      x: Math.round(fx + fw * 0.2),
      y: Math.round(fy + fh * 0.05),
      w: Math.round(fw * 0.6),
      h: Math.round(fh * 0.25),
      faceBox: { x: Math.round(fx), y: Math.round(fy), w: Math.round(fw), h: Math.round(fh) },
      detected: true,
    };
  }

  _centerROI(vw, vh) {
    // Fallback: assume face is roughly centred in a 640×480 frame
    const cx = vw / 2,
      cy = vh / 2;
    const fw = vw * 0.5,
      fh = vh * 0.6;
    return {
      x: Math.round(cx - fw * 0.3),
      y: Math.round(cy - fh * 0.4),
      w: Math.round(fw * 0.6),
      h: Math.round(fh * 0.25),
      faceBox: {
        x: Math.round(cx - fw / 2),
        y: Math.round(cy - fh * 0.45),
        w: Math.round(fw),
        h: Math.round(fh),
      },
      detected: false,
    };
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Canvas Renderers
// ═══════════════════════════════════════════════════════════════════════════

class WaveformRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this._history = [];
    this._MAX = 250;
  }

  push(filteredSignal) {
    if (!filteredSignal.length) return;
    this._history.push(filteredSignal[filteredSignal.length - 1]);
    if (this._history.length > this._MAX) this._history.shift();
  }

  draw(bpm, confidence) {
    const ctx = this.ctx;
    const W = this.canvas.width,
      H = this.canvas.height;

    ctx.fillStyle = "#0a0a1a";
    ctx.fillRect(0, 0, W, H);

    // Grid lines
    ctx.strokeStyle = "rgba(255,255,255,0.04)";
    ctx.lineWidth = 1;
    for (let gx = 0; gx < W; gx += 40) {
      ctx.beginPath();
      ctx.moveTo(gx, 0);
      ctx.lineTo(gx, H);
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.moveTo(0, H / 2);
    ctx.lineTo(W, H / 2);
    ctx.stroke();

    const sig = this._history;
    if (sig.length < 2) return;

    const absMax = sig.reduce((m, v) => Math.max(m, Math.abs(v)), 0) || 1;
    const norm = sig.map((v) => v / absMax);

    const alpha = Math.max(0.3, Math.min(1, 0.3 + confidence * 0.7));
    ctx.strokeStyle = `rgba(0,255,136,${alpha})`;
    ctx.lineWidth = 2;
    ctx.shadowColor = "#00ff88";
    ctx.shadowBlur = confidence > 0.5 ? 6 : 0;

    ctx.beginPath();
    for (let i = 0; i < norm.length; i++) {
      const px = (i / (this._MAX - 1)) * W;
      const py = H / 2 - norm[i] * (H / 2 - 6);
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    if (bpm && confidence > 0.25) {
      ctx.fillStyle = `rgba(0,255,136,${alpha})`;
      ctx.font = "bold 12px -apple-system,sans-serif";
      ctx.fillText(`${Math.round(bpm)} BPM`, 8, 16);
    }
  }
}

class OverlayRenderer {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {HTMLVideoElement} video  – needed to map ROI coords (video space) to canvas space
   */
  constructor(canvas, video) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.video = video;
  }

  /**
   * Compute the transform that maps video-pixel coordinates to CSS-pixel coordinates
   * on the overlay canvas, accounting for object-fit: cover scaling.
   * Returns { s, ox, oy } where  canvasX = roi.x * s + ox.
   */
  _getVideoTransform() {
    const v = this.video;
    if (!v || !v.videoWidth) return null;
    const cw = this.canvas.offsetWidth;
    const ch = this.canvas.offsetHeight;
    if (!cw || !ch) return null;
    const s = Math.max(cw / v.videoWidth, ch / v.videoHeight);
    return { s, ox: (cw - v.videoWidth * s) / 2, oy: (ch - v.videoHeight * s) / 2 };
  }

  /**
   * Draw face overlays. Only draws when a face is actually detected.
   * When no face detected, nothing is drawn so the static guide oval shows through.
   * Returns true if a detected face was drawn (so the caller can hide the static oval).
   */
  draw(roi, result, filteredSignal, evmEnabled) {
    const ctx = this.ctx;
    const W = this.canvas.offsetWidth;
    const H = this.canvas.offsetHeight;
    ctx.clearRect(0, 0, W, H);

    // Only draw overlays when a real face is detected — otherwise let the
    // static SVG guide oval tell the user to position their face.
    if (!roi || !roi.detected) return false;

    const t = this._getVideoTransform();
    if (!t) return false;

    const mx = (x) => t.ox + x * t.s;
    const my = (y) => t.oy + y * t.s;
    const ms = (s) => s * t.s;

    const conf = result?.confidence || 0;
    const faceBox = roi.faceBox;

    // ── EVM colour-amplification overlay ──────────────────────────────────
    if (evmEnabled && filteredSignal.length > 0 && faceBox) {
      const sigVal = filteredSignal[filteredSignal.length - 1];
      const amp = Math.tanh(Math.abs(sigVal) * 40) * 0.28;
      const [r, g, b] = sigVal > 0 ? [255, 60, 60] : [60, 200, 255];
      ctx.save();
      ctx.globalAlpha = amp;
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.beginPath();
      ctx.ellipse(
        mx(faceBox.x + faceBox.w / 2),
        my(faceBox.y + faceBox.h / 2),
        ms(faceBox.w / 2),
        ms(faceBox.h / 2),
        0, 0, Math.PI * 2
      );
      ctx.fill();
      ctx.restore();
    }

    // ── Face bounding-box corners ──────────────────────────────────────────
    if (faceBox) {
      const fx = mx(faceBox.x), fy = my(faceBox.y);
      const fw = ms(faceBox.w), fh = ms(faceBox.h);
      const color = conf > 0.55 ? "#00ff88" : conf > 0.25 ? "#ffcc00" : "rgba(255,255,255,0.5)";
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      const c = Math.min(fw, fh) * 0.14;
      this._corner(ctx, fx, fy, c, c);
      this._corner(ctx, fx + fw, fy, -c, c);
      this._corner(ctx, fx, fy + fh, c, -c);
      this._corner(ctx, fx + fw, fy + fh, -c, -c);
    }

    // ── Forehead ROI (the actual sampling region) ──────────────────────────
    const rx = mx(roi.x), ry = my(roi.y), rw = ms(roi.w), rh = ms(roi.h);
    ctx.strokeStyle = "rgba(0,255,136,0.6)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.strokeRect(rx, ry, rw, rh);
    ctx.setLineDash([]);

    return true; // face was drawn — hide the static guide oval
  }

  _corner(ctx, x, y, dx, dy) {
    ctx.beginPath();
    ctx.moveTo(x + dx, y);
    ctx.lineTo(x, y);
    ctx.lineTo(x, y + dy);
    ctx.stroke();
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Debug Renderer – amplified raw video + pyramid level views
// ═══════════════════════════════════════════════════════════════════════════

class DebugRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { willReadFrequently: true });
    // Persistent canvas for pyramid level upscaling
    this._lvlCanvas = document.createElement("canvas");
    this._lvlCtx = this._lvlCanvas.getContext("2d");
    // Pyramid builder for full-frame levels
    this._pyramid = new GaussianPyramid(3);
  }

  /**
   * Draw the debug view.
   * @param {HTMLVideoElement} video
   * @param {object} roi  – ROI in video pixel space
   * @param {number[]} filteredSignal
   * @param {number} pyramidLevel  -1 = EVM amplified, 0 = raw, 1–3 = pyramid level
   * @param {boolean} isMirrored   – true for front camera (match CSS scaleX(-1) on video)
   */
  draw(video, roi, filteredSignal, pyramidLevel, isMirrored) {
    if (!video || video.readyState < 2 || !video.videoWidth) return;

    if (pyramidLevel === -1) {
      this._drawEVM(video, roi, filteredSignal, isMirrored);
    } else {
      this._drawPyramidLevel(video, roi, pyramidLevel, isMirrored);
    }
  }

  /** EVM amplified view – CHROM signal coloured onto full video frame */
  _drawEVM(video, roi, filteredSignal, isMirrored) {
    const W = this.canvas.width;
    const H = this.canvas.height;
    const ctx = this.ctx;

    ctx.save();
    if (isMirrored) { ctx.translate(W, 0); ctx.scale(-1, 1); }
    ctx.drawImage(video, 0, 0, W, H);
    ctx.restore();

    // Apply CHROM amplification
    const sigVal = filteredSignal.length > 0
      ? filteredSignal[filteredSignal.length - 1]
      : 0;
    const GAIN = 150;
    const dR = sigVal * GAIN * 3;
    const dG = -sigVal * GAIN * 2;
    if (Math.abs(sigVal) > 0.0001) {
      const imgData = ctx.getImageData(0, 0, W, H);
      const d = imgData.data;
      for (let i = 0; i < d.length; i += 4) {
        d[i]     = Math.min(255, Math.max(0, d[i]     + dR));
        d[i + 1] = Math.min(255, Math.max(0, d[i + 1] + dG));
      }
      ctx.putImageData(imgData, 0, 0);
    }

    this._drawROIBoxes(video, roi, isMirrored);
    this._drawInfoBar(`EVM ×150  sig: ${sigVal.toFixed(4)}`, sigVal > 0 ? "#ff6060" : "#60c8ff");
  }

  /**
   * Show the video frame at a given Gaussian pyramid level (blurred/downsampled
   * then upscaled back with nearest-neighbour so pixelation is visible).
   * Level 0 = raw, level 1–3 = progressively blurred.
   */
  _drawPyramidLevel(video, roi, level, isMirrored) {
    const W = this.canvas.width;
    const H = this.canvas.height;
    const ctx = this.ctx;

    if (level === 0) {
      // Raw frame
      ctx.save();
      if (isMirrored) { ctx.translate(W, 0); ctx.scale(-1, 1); }
      ctx.drawImage(video, 0, 0, W, H);
      ctx.restore();
    } else {
      // Build pyramid on the forehead ROI region and display it
      // (we use the ROI so performance stays reasonable)
      if (!roi) {
        ctx.fillStyle = "#000";
        ctx.fillRect(0, 0, W, H);
      } else {
        const levels = this._pyramid.buildPyramid(video, roi, level);
        const lvl = levels[level];

        // Put downsampled level into a temporary canvas
        this._lvlCanvas.width = lvl.w;
        this._lvlCanvas.height = lvl.h;
        this._lvlCtx.putImageData(new ImageData(lvl.data, lvl.w, lvl.h), 0, 0);

        // Black background
        ctx.fillStyle = "#111";
        ctx.fillRect(0, 0, W, H);

        // Show the raw video (dimmed) as context behind the pyramid ROI
        ctx.save();
        ctx.globalAlpha = 0.35;
        if (isMirrored) { ctx.translate(W, 0); ctx.scale(-1, 1); }
        ctx.drawImage(video, 0, 0, W, H);
        ctx.restore();

        // Compute display rect for the ROI (same scaling as the overlay)
        const s = Math.max(W / video.videoWidth, H / video.videoHeight);
        const ox = (W - video.videoWidth * s) / 2;
        const oy = (H - video.videoHeight * s) / 2;

        // Destination rect in debug canvas (mirrored if needed)
        let destX = ox + roi.x * s;
        const destY = oy + roi.y * s;
        const destW = roi.w * s;
        const destH = roi.h * s;
        if (isMirrored) destX = W - destX - destW;

        // Upscale pyramid level (nearest-neighbour) into ROI rect
        ctx.save();
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(this._lvlCanvas, destX, destY, destW, destH);
        ctx.restore();

        // Border around the ROI
        ctx.strokeStyle = "rgba(0,255,136,0.8)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([]);
        ctx.strokeRect(destX, destY, destW, destH);
      }
    }

    this._drawROIBoxes(video, roi, isMirrored);
    const label = level === 0 ? "Raw (L0)" : `Gaussian pyramid L${level}`;
    this._drawInfoBar(label, "#00ff88");
  }

  /** Draw face bounding box and forehead ROI on the debug canvas */
  _drawROIBoxes(video, roi, isMirrored) {
    if (!roi) return;
    const W = this.canvas.width;
    const H = this.canvas.height;
    const ctx = this.ctx;

    const sx = W / video.videoWidth;
    const sy = H / video.videoHeight;

    if (roi.faceBox) {
      const fb = roi.faceBox;
      let fbX = fb.x * sx;
      if (isMirrored) fbX = W - fbX - fb.w * sx;
      ctx.strokeStyle = "rgba(255,220,0,0.75)";
      ctx.lineWidth = 1;
      ctx.setLineDash([]);
      ctx.strokeRect(fbX, fb.y * sy, fb.w * sx, fb.h * sy);
    }

    let roiX = roi.x * sx;
    if (isMirrored) roiX = W - roiX - roi.w * sx;
    ctx.strokeStyle = "rgba(0,255,136,0.9)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 2]);
    ctx.strokeRect(roiX, roi.y * sy, roi.w * sx, roi.h * sy);
    ctx.setLineDash([]);
  }

  _drawInfoBar(text, color) {
    const ctx = this.ctx;
    const W = this.canvas.width;
    ctx.fillStyle = "rgba(0,0,0,0.6)";
    ctx.fillRect(0, 0, W, 14);
    ctx.fillStyle = color;
    ctx.font = "9px -apple-system,sans-serif";
    ctx.fillText(text, 3, 10);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Main Application
// ═══════════════════════════════════════════════════════════════════════════

class HeartFaceApp {
  constructor() {
    // DOM refs
    this.$video = document.getElementById("video");
    this.$overlay = document.getElementById("overlay");
    this.$waveform = document.getElementById("waveform");
    this.$bpmValue = document.getElementById("bpm-value");
    this.$statusMsg = document.getElementById("status-msg");
    this.$confFill = document.getElementById("conf-fill");
    this.$confLabel = document.getElementById("conf-label");
    this.$evmBtn = document.getElementById("evm-btn");
    this.$installBtn = document.getElementById("install-btn");
    this.$startScreen = document.getElementById("start-screen");
    this.$startBtn = document.getElementById("start-btn");
    this.$cameraErr = document.getElementById("camera-err");
    this.$evmPanel = document.getElementById("evm-panel");
    this.$infoBtn = document.getElementById("info-btn");
    this.$infoModal = document.getElementById("info-modal");
    this.$infoClose = document.getElementById("info-close");
    this.$camBtn = document.getElementById("cam-btn");
    this.$debugBtn = document.getElementById("debug-btn");
    this.$debugCanvas = document.getElementById("debug-canvas");
    this.$debugLevelSelect = document.getElementById("debug-level-select");
    this.$cameraPreSelect = document.getElementById("camera-pre-select");
    this.$faceGuide = document.getElementById("face-guide");
    this.$beatSyncLabel = document.getElementById("beat-sync-label");

    // Engines
    this.signalEngine = new SignalEngine();
    this.roiTracker = new ROITracker();
    this.waveformRenderer = new WaveformRenderer(this.$waveform);
    // OverlayRenderer receives the video element for coordinate scaling
    this.overlayRenderer = new OverlayRenderer(this.$overlay, this.$video);
    this.debugRenderer = new DebugRenderer(this.$debugCanvas);

    // State
    this.running = false;
    this.evmEnabled = false;
    /** 0 = off, 1 = full-page */
    this.debugMode = 0;
    /** -1 = EVM amplified, 0 = raw, 1-3 = pyramid level */
    this.debugLevel = -1;
    this.facingMode = "user";
    this.stream = null;
    this.frameCount = 0;
    this.currentROI = null;
    this.currentResult = null;
    this.filteredSignal = [];
    this.bpmHistory = [];
    this._deferredInstall = null;
    this._beatInterval = null;  // timer for BPM-synced pulsing

    this._bindEvents();
    this._initPWA();
    this._onResize();
    window.addEventListener("resize", () => this._onResize());
  }

  get debugEnabled() { return this.debugMode > 0; }

  // ── Setup ──────────────────────────────────────────────────────────────

  _bindEvents() {
    this.$startBtn.addEventListener("click", () => this._start());
    this.$evmBtn.addEventListener("click", () => this._toggleEVM());
    this.$installBtn.addEventListener("click", () => this._install());
    this.$infoBtn.addEventListener("click", () => {
      this.$infoModal.classList.remove("hidden");
    });
    this.$infoClose.addEventListener("click", () => {
      this.$infoModal.classList.add("hidden");
    });
    this.$infoModal.addEventListener("click", (e) => {
      if (e.target === this.$infoModal) this.$infoModal.classList.add("hidden");
    });
    this.$cameraPreSelect.addEventListener("change", (e) => {
      this.facingMode = e.target.value;
    });
    this.$camBtn.addEventListener("click", () => this._switchCamera());
    this.$debugBtn.addEventListener("click", () => this._toggleDebug());
    this.$debugLevelSelect.addEventListener("change", (e) => {
      this.debugLevel = parseInt(e.target.value, 10);
    });
  }

  _initPWA() {
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker
        .register("/service-worker.js")
        .catch(() => {});
    }
    window.addEventListener("beforeinstallprompt", (e) => {
      e.preventDefault();
      this._deferredInstall = e;
      this.$installBtn.classList.remove("hidden");
    });
  }

  _onResize() {
    [this.$overlay, this.$waveform].forEach((c) => {
      c.width = c.offsetWidth * devicePixelRatio;
      c.height = c.offsetHeight * devicePixelRatio;
      c.getContext("2d").scale(devicePixelRatio, devicePixelRatio);
    });
    // Re-size full-page debug canvas if active
    if (this.debugMode === 1) {
      this.$debugCanvas.width = this.$debugCanvas.offsetWidth;
      this.$debugCanvas.height = this.$debugCanvas.offsetHeight;
    }
  }

  async _install() {
    if (!this._deferredInstall) return;
    this._deferredInstall.prompt();
    const { outcome } = await this._deferredInstall.userChoice;
    if (outcome === "accepted") this.$installBtn.classList.add("hidden");
    this._deferredInstall = null;
  }

  _toggleEVM() {
    this.evmEnabled = !this.evmEnabled;
    this.$evmBtn.classList.toggle("active", this.evmEnabled);
    this.$evmBtn.textContent = this.evmEnabled
      ? "Hide Pulse Glow"
      : "Show Pulse Glow";
    this.$evmPanel.classList.toggle("visible", this.evmEnabled);
  }

  // ── Camera start ───────────────────────────────────────────────────────

  _buildVideoConstraints(facingMode) {
    return {
      facingMode,
      width: { ideal: 640 },
      height: { ideal: 480 },
      frameRate: { ideal: 30 },
    };
  }

  _updateCamBtn() {
    const label = this.facingMode === "user" ? "Front" : "Back";
    this.$camBtn.textContent = `Camera: ${label}`;
    // Mirror the video for front camera (selfie); back camera shows natural orientation
    const mirror = this.facingMode === "user" ? "scaleX(-1)" : "none";
    this.$video.style.transform = mirror;
    // The overlay canvas draws in video-native coordinate space.
    // For front camera: apply the same CSS mirror so the overlay aligns with the video.
    // For back camera: no mirror needed.
    this.$overlay.style.transform = mirror;
  }

  async _start() {
    this.$startBtn.disabled = true;
    this.$startBtn.textContent = "Requesting camera…";
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: this._buildVideoConstraints(this.facingMode),
        audio: false,
      });
      this.stream = stream;
      this.$video.srcObject = stream;
      await this.$video.play();
      this._updateCamBtn();
      this.$startScreen.classList.add("hidden");
      this.running = true;
      requestAnimationFrame((t) => this._loop(t));
    } catch (err) {
      this.$startBtn.disabled = false;
      this.$startBtn.textContent = "Start Measuring";
      this.$cameraErr.textContent =
        err.name === "NotAllowedError"
          ? "Camera access denied — please allow camera and reload."
          : `Camera error: ${err.message}`;
      this.$cameraErr.classList.remove("hidden");
    }
  }

  async _switchCamera() {
    if (!this.running) return;

    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
    }

    this.facingMode = this.facingMode === "user" ? "environment" : "user";

    this.signalEngine.clear();
    this.frameCount = 0;
    this.bpmHistory = [];
    this.currentResult = null;
    this.filteredSignal = [];
    this.$bpmValue.textContent = "--";
    this.$statusMsg.textContent = "Switching camera…";

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: this._buildVideoConstraints(this.facingMode),
        audio: false,
      });
      this.stream = stream;
      this.$video.srcObject = stream;
      await this.$video.play();
      this._updateCamBtn();
    } catch (err) {
      this.$statusMsg.textContent = `Camera error: ${err.message}`;
    }
  }

  /** Toggle debug: off → full-page → off */
  _toggleDebug() {
    this.debugMode = this.debugMode === 0 ? 1 : 0;

    const labels = ["Debug View", "Debug: Full ▼"];
    this.$debugBtn.textContent = labels[this.debugMode];
    this.$debugBtn.classList.toggle("active", this.debugMode > 0);
    this.$debugLevelSelect.classList.toggle("hidden", this.debugMode === 0);

    if (this.debugMode === 0) {
      // Off
      this.$debugCanvas.classList.add("hidden");
      this.$debugCanvas.classList.remove("debug-fullpage");
    } else {
      // Full-page
      this.$debugCanvas.classList.remove("hidden");
      this.$debugCanvas.classList.add("debug-fullpage");
      // Size to fill the camera-wrap area
      this.$debugCanvas.width = this.$debugCanvas.offsetWidth;
      this.$debugCanvas.height = this.$debugCanvas.offsetHeight;
    }
  }

  // ── Main render loop ───────────────────────────────────────────────────

  async _loop(timestamp) {
    if (!this.running) return;

    if (this.$video.readyState >= 2) {
      const roi = await this.roiTracker.getROI(this.$video);
      this.currentROI = roi;

      if (roi) {
        this.signalEngine.addSample(this.$video, roi, timestamp);
      }

      this.frameCount++;

      if (this.frameCount % 30 === 0) {
        this._computeHeartRate();
      }

      this._render();
    }

    requestAnimationFrame((t) => this._loop(t));
  }

  // ── Heart rate computation ─────────────────────────────────────────────

  _computeHeartRate() {
    const r = this.signalEngine.estimate();

    if (r.status === "collecting") {
      const pct = Math.round(
        ((SignalEngine.MIN_FRAMES - r.framesNeeded) / SignalEngine.MIN_FRAMES) *
          100
      );
      this.$statusMsg.textContent = `Collecting signal… ${pct}%`;
      this.$confFill.style.width = `${pct}%`;
      this.$confFill.style.background = "#ffcc00";
      this.$confLabel.textContent = "";
      return;
    }

    if (r.status === "ok") {
      this.bpmHistory.push(Math.round(r.bpm));
      if (this.bpmHistory.length > 5) this.bpmHistory.shift();
      const smoothBPM = Math.round(
        this.bpmHistory.reduce((a, b) => a + b, 0) / this.bpmHistory.length
      );

      this.currentResult = { ...r, bpm: smoothBPM };
      this.$bpmValue.textContent = smoothBPM;

      const pct = Math.round(r.confidence * 100);
      this.$confFill.style.width = `${pct}%`;
      this.$confFill.style.background =
        r.confidence > 0.6
          ? "#00ff88"
          : r.confidence > 0.3
          ? "#ffcc00"
          : "#ff4466";

      if (r.confidence > 0.55) {
        this.$statusMsg.textContent = "Good signal \u2014 keep still";
        this.$confLabel.textContent = `${pct}% confidence`;
        this._startBeatPulse(smoothBPM);
      } else if (r.confidence > 0.25) {
        this.$statusMsg.textContent = "Stay still for better accuracy";
        this.$confLabel.textContent = `${pct}% confidence`;
      } else {
        this.$statusMsg.textContent = "Ensure good lighting \u2014 face centred";
        this.$confLabel.textContent = "Low signal";
        this._stopBeatPulse();
      }
    }
  }

  /**
   * Start pulsing the BPM display at the detected heart rate.
   * The pulse interval matches the actual BPM so the user can see
   * the text "beating" in sync with their estimated heart rate.
   */
  _startBeatPulse(bpm) {
    if (!bpm || bpm < 30) { this._stopBeatPulse(); return; }
    const interval = 60000 / bpm; // ms between beats

    // Only restart if BPM changed significantly (avoid jitter)
    if (this._beatInterval && this._lastBeatBPM &&
        Math.abs(bpm - this._lastBeatBPM) < 3) return;

    this._stopBeatPulse();
    this._lastBeatBPM = bpm;

    const doBeat = () => {
      this.$bpmValue.classList.remove("beat");
      void this.$bpmValue.offsetWidth;
      this.$bpmValue.classList.add("beat");
    };
    doBeat(); // immediate first beat
    this._beatInterval = setInterval(doBeat, interval);
    this.$beatSyncLabel.innerHTML = '<span class="heart-dot">❤</span> pulsing at your heart rate';
  }

  _stopBeatPulse() {
    if (this._beatInterval) {
      clearInterval(this._beatInterval);
      this._beatInterval = null;
    }
    this._lastBeatBPM = null;
    this.$bpmValue.classList.remove("beat");
    this.$beatSyncLabel.textContent = '';
  }

  // ── Rendering ─────────────────────────────────────────────────────────

  _render() {
    const filtered =
      this.signalEngine.frameCount >= SignalEngine.MIN_FRAMES
        ? this.signalEngine.getBandpassedSignal()
        : [];

    this.filteredSignal = filtered;

    this.waveformRenderer.push(filtered);
    this.waveformRenderer.draw(
      this.currentResult?.bpm,
      this.currentResult?.confidence || 0
    );

    const faceDrawn = this.overlayRenderer.draw(
      this.currentROI,
      this.currentResult,
      filtered,
      this.evmEnabled
    );
    // Show the static guide oval only when no face has been detected yet
    this.$faceGuide.style.opacity = faceDrawn ? "0" : "";

    if (this.debugEnabled) {
      const isMirrored = this.facingMode === "user";
      this.debugRenderer.draw(
        this.$video,
        this.currentROI,
        filtered,
        this.debugLevel,
        isMirrored
      );
    }
  }
}

// ── Bootstrap ───────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  window._app = new HeartFaceApp();
});
