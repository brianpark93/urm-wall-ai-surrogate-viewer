// ── State ──────────────────────────────────────────────────────────────────
let geometry   = null;
let modelPlain = null;
let modelPinn  = null;
let debounceTimer = null;

const DIFF_MAX = 0.4; // colorbar cap for the |plain - PINN| disagreement panel

// ── Tiny MLP forward pass (mirrors F_deep_surrogate.py's MLP exactly) ──────
// layers[i] = {w: (out x in) array of arrays, b: (out,) array}
// All layers ReLU except the last, which is Sigmoid.
function forward(model, x) {
  let v = x;
  const n = model.layers.length;
  for (let li = 0; li < n; li++) {
    const { w, b } = model.layers[li];
    const out = new Float32Array(b.length);
    for (let o = 0; o < b.length; o++) {
      let s = b[o];
      const row = w[o];
      for (let i = 0; i < v.length; i++) s += row[i] * v[i];
      out[o] = s;
    }
    if (li < n - 1) {
      for (let o = 0; o < out.length; o++) out[o] = Math.max(0, out[o]); // ReLU
    } else {
      for (let o = 0; o < out.length; o++) out[o] = 1 / (1 + Math.exp(-out[o])); // Sigmoid
    }
    v = out;
  }
  return v[0];
}

// Predict damage for every LW element at a given raw PGA value.
function predictAll(model, pgaRaw) {
  const pgaNorm = pgaRaw / model.pga_max;
  const feats = geometry.xyz_features;
  const n = feats.length;
  const damage = new Float32Array(n);
  // reuse one input buffer (1 + 27 = 28 dims)
  const inDim = 1 + feats[0].length;
  const xbuf = new Float32Array(inDim);
  xbuf[0] = pgaNorm;
  for (let e = 0; e < n; e++) {
    const f = feats[e];
    for (let k = 0; k < f.length; k++) xbuf[k + 1] = f[k];
    damage[e] = forward(model, xbuf);
  }
  return damage;
}

// ── Coordinate mapping (auto-fit to projected bbox) ───────────────────────
function makeXform(canvas) {
  const { xmin, xmax, zmin, zmax } = geometry.bbox;
  const PAD_PX = 16;
  const W = canvas.width  - 2 * PAD_PX;
  const H = canvas.height - 2 * PAD_PX;
  const sx = W / (xmax - xmin);
  const sz = H / (zmax - zmin);
  const s  = Math.min(sx, sz);
  const ox = PAD_PX + (W - s * (xmax - xmin)) / 2;
  const oz = PAD_PX + (H - s * (zmax - zmin)) / 2;
  return {
    tx: xi => ox + (xi - xmin) * s,
    tz: zi => oz + (zmax - zi) * s,   // screen Y increases downward
  };
}

// ── Color scales ─────────────────────────────────────────────────────────
// Damage: white (0) -> black (1), same convention as the rest of the project.
function damageColor(d) {
  if (d < 0.05) return 'rgb(255,255,255)';
  const g = Math.round(255 * (1 - d));
  return `rgb(${g},${g},${g})`;
}
// Disagreement: white (0) -> dark red (DIFF_MAX+), same 'Reds' convention
// used for error maps elsewhere in this project (light=low, dark=high).
function diffColor(d) {
  const t = Math.min(1, d / DIFF_MAX);
  if (t < 0.03) return 'rgb(255,255,255)';
  const r = Math.round(255 - t * (255 - 130));
  const g = Math.round(255 - t * 235);
  const b = Math.round(255 - t * 235);
  return `rgb(${r},${g},${b})`;
}

function renderWall(canvasId, damage, colorFn, diagColor) {
  const canvas = document.getElementById(canvasId);
  const ctx    = canvas.getContext('2d');
  const { tx, tz } = makeXform(canvas);
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // elements are pre-sorted back-to-front (painter's algorithm) in geometry_lw.json
  geometry.elements.forEach((elem) => {
    ctx.fillStyle = colorFn(damage[elem.col]);
    ctx.beginPath();
    elem.poly.forEach(([xi, zi], j) => {
      if (j === 0) ctx.moveTo(tx(xi), tz(zi));
      else         ctx.lineTo(tx(xi), tz(zi));
    });
    ctx.closePath();
    ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,0.15)';
    ctx.lineWidth = 0.3;
    ctx.stroke();
  });

  const diag = geometry.diagonal;
  const [lineC, bandC] = diagColor || ['rgba(200,0,0,0.85)', 'rgba(200,0,0,0.40)'];
  strokePolyline(ctx, diag.line,     tx, tz, lineC, 1.5, []);
  strokePolyline(ctx, diag.band_pos, tx, tz, bandC, 1.0, [5, 4]);
  strokePolyline(ctx, diag.band_neg, tx, tz, bandC, 1.0, [5, 4]);
}

function strokePolyline(ctx, pts, tx, tz, color, width, dash) {
  if (!pts || pts.length === 0) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth   = width;
  ctx.setLineDash(dash);
  ctx.beginPath();
  pts.forEach(([x, z], j) => {
    if (j === 0) ctx.moveTo(tx(x), tz(z));
    else         ctx.lineTo(tx(x), tz(z));
  });
  ctx.stroke();
  ctx.restore();
}

// ── Colorbars (horizontal, with headroom so labels never clip) ────────────
function drawColorbar(canvasId, colorStops, ticks, title) {
  const canvas = document.getElementById(canvasId);
  const ctx    = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const left = 24, right = canvas.width - 24, y0 = 10, h = 16;

  const grad = ctx.createLinearGradient(left, 0, right, 0);
  colorStops.forEach(([stop, color]) => grad.addColorStop(stop, color));
  ctx.fillStyle = grad;
  ctx.fillRect(left, y0, right - left, h);
  ctx.strokeStyle = '#aaa';
  ctx.strokeRect(left, y0, right - left, h);

  ctx.fillStyle = '#555';
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ticks.forEach(([frac, label]) => {
    const x = left + frac * (right - left);
    ctx.fillText(label, x, y0 + h + 6);
  });
  ctx.fillStyle = '#666';
  ctx.font = '11px sans-serif';
  ctx.fillText(title, (left + right) / 2, y0 + h + 24);
}

function renderColorbars() {
  drawColorbar('colorbar-canvas',
    [[0, '#fff'], [1, '#000']],
    [[0, '0'], [0.25, '0.25'], [0.5, '0.5'], [0.75, '0.75'], [1, '1.0']],
    'predicted damage (white = intact, black = fully damaged)');
  drawColorbar('colorbar-canvas-diff',
    [[0, '#fff'], [1, `rgb(130,20,20)`]],
    [[0, '0'], [0.25, (0.25*DIFF_MAX).toFixed(2)], [0.5, (0.5*DIFF_MAX).toFixed(2)],
     [0.75, (0.75*DIFF_MAX).toFixed(2)], [1, `${DIFF_MAX.toFixed(2)}+`]],
    'plain vs PINN disagreement (|Δdamage|)');
}

// ── Zone-averaged stats, written out as sentences ──────────────────────────
function zoneStats(damage) {
  let sLw=0,nLw=0, sBas=0,nBas=0, sInb=0,nInb=0;
  geometry.elements.forEach(e => {
    const d = damage[e.col];
    sLw += d; nLw++;
    if (e.zone === 'base')   { sBas += d; nBas++; }
    if (e.zone === 'inband') { sInb += d; nInb++; }
  });
  return { lw: sLw/nLw, base: nBas?sBas/nBas:0, inband: nInb?sInb/nInb:0 };
}

function pct(x) { return Math.round(x * 100); }

function statsSentence(s) {
  return `
    <p>On average, <strong>${pct(s.lw)}%</strong> of the long wall is damaged.</p>
    <p>The base bed-joint is <strong>${pct(s.base)}%</strong> damaged on average.</p>
    <p>The in-band diagonal zone is <strong>${pct(s.inband)}%</strong> damaged on average.</p>
  `;
}

function diffStats(dPlain, dPinn) {
  let sum = 0, max = 0;
  const n = dPlain.length;
  for (let i = 0; i < n; i++) {
    const d = Math.abs(dPlain[i] - dPinn[i]);
    sum += d;
    if (d > max) max = d;
  }
  const mean = sum / n;
  return `
    <p>The two models disagree by <strong>${pct(mean)}%</strong> damage on average across all elements.</p>
    <p>The single largest disagreement at any one element is <strong>${pct(max)}%</strong>.</p>
  `;
}

// ── Main update ─────────────────────────────────────────────────────────────
function updatePrediction(pgaRaw) {
  document.getElementById('loading').style.display = 'inline';
  // let the loading indicator paint before the (synchronous) heavy work
  setTimeout(() => {
    const dPlain = predictAll(modelPlain, pgaRaw);
    const dPinn  = predictAll(modelPinn,  pgaRaw);
    const dDiff  = new Float32Array(dPlain.length);
    for (let i = 0; i < dDiff.length; i++) dDiff[i] = Math.abs(dPlain[i] - dPinn[i]);

    renderWall('wall-canvas-plain', dPlain, damageColor);
    renderWall('wall-canvas-pinn',  dPinn,  damageColor);
    renderWall('wall-canvas-diff',  dDiff,  diffColor, ['rgba(0,90,200,0.85)', 'rgba(0,90,200,0.40)']);

    document.getElementById('stats-plain').innerHTML = statsSentence(zoneStats(dPlain));
    document.getElementById('stats-pinn').innerHTML  = statsSentence(zoneStats(dPinn));
    document.getElementById('stats-diff').innerHTML  = diffStats(dPlain, dPinn);

    const badge = document.getElementById('range-badge');
    if (pgaRaw > modelPlain.pga_max) {
      badge.textContent = `extrapolation (>${modelPlain.pga_max.toFixed(2)}g training max)`;
      badge.classList.add('extrapolated');
    } else {
      badge.textContent = 'within training range (0.01–' + modelPlain.pga_max.toFixed(2) + 'g)';
      badge.classList.remove('extrapolated');
    }
    document.getElementById('loading').style.display = 'none';
  }, 10);
}

// source: 'input' (typed, full precision, don't touch the box while they're
// typing) or 'slider' (dragged, browser already snapped it to a step -- fine
// to mirror that value into the number box for display).
function onPgaChange(value, source) {
  const pga = Math.max(0.001, parseFloat(value) || 0.35);
  if (source === 'slider') {
    document.getElementById('pga-input').value = pga.toFixed(3);
  } else {
    document.getElementById('pga-slider').value = Math.min(Math.max(pga, 0.01), 0.90);
  }
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => updatePrediction(pga), 40);
}

// ── Boot ───────────────────────────────────────────────────────────────────
async function init() {
  [geometry, modelPlain, modelPinn] = await Promise.all([
    fetch('data/geometry_lw.json').then(r => r.json()),
    fetch('data/model_plain.json').then(r => r.json()),
    fetch('data/model_pinn.json').then(r => r.json()),
  ]);

  renderColorbars();

  const input  = document.getElementById('pga-input');
  const slider = document.getElementById('pga-slider');
  input.addEventListener('input',  e => onPgaChange(e.target.value, 'input'));
  slider.addEventListener('input', e => onPgaChange(e.target.value, 'slider'));

  updatePrediction(parseFloat(input.value));
}

init().catch(err => {
  console.error(err);
  document.body.innerHTML +=
    `<p style="color:red;padding:20px">
       Error loading model/geometry data. Run <code>export_web_model.py</code> first
       (from <code>06_Lshape_large_simulations_AI/</code>), then serve this folder with<br>
       <code>python -m http.server 8000</code> and open <code>localhost:8000</code>.
     </p>`;
});
