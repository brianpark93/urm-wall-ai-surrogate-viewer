// ── State ──────────────────────────────────────────────────────────────────
let geometry   = null;
let modelPlain = null;
let modelPinn  = null;
let debounceTimer = null;

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

function renderWall(canvasId, damage) {
  const canvas = document.getElementById(canvasId);
  const ctx    = canvas.getContext('2d');
  const { tx, tz } = makeXform(canvas);
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // elements are pre-sorted back-to-front (painter's algorithm) in geometry_lw.json
  geometry.elements.forEach((elem) => {
    const d = damage[elem.col];
    if (d < 0.05) {
      ctx.fillStyle = 'rgb(255,255,255)';
    } else {
      const gray = Math.round(255 * (1 - d));
      ctx.fillStyle = `rgb(${gray},${gray},${gray})`;
    }
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
  strokePolyline(ctx, diag.line,     tx, tz, 'rgba(200,0,0,0.85)', 1.5, []);
  strokePolyline(ctx, diag.band_pos, tx, tz, 'rgba(200,0,0,0.45)', 1.0, [5, 4]);
  strokePolyline(ctx, diag.band_neg, tx, tz, 'rgba(200,0,0,0.45)', 1.0, [5, 4]);
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

// ── Colorbar (horizontal) ──────────────────────────────────────────────────
function renderColorbar() {
  const canvas = document.getElementById('colorbar-canvas');
  const ctx    = canvas.getContext('2d');
  const left = 20, right = canvas.width - 20, y0 = 8, h = 14;
  const grad = ctx.createLinearGradient(left, 0, right, 0);
  grad.addColorStop(0, '#fff');
  grad.addColorStop(1, '#000');
  ctx.fillStyle = grad;
  ctx.fillRect(left, y0, right - left, h);
  ctx.strokeStyle = '#aaa';
  ctx.strokeRect(left, y0, right - left, h);
  ctx.fillStyle = '#555';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'center';
  [0, 0.25, 0.5, 0.75, 1.0].forEach(v => {
    const x = left + v * (right - left);
    ctx.fillText(v.toFixed(2), x, y0 + h + 14);
  });
  ctx.fillText('damage (predicted)', (left + right) / 2, y0 + h + 30);
}

// ── Zone-averaged stats ─────────────────────────────────────────────────────
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

// ── Main update ─────────────────────────────────────────────────────────────
function updatePrediction(pgaRaw) {
  document.getElementById('loading').style.display = 'inline';
  // let the loading indicator paint before the (synchronous) heavy work
  setTimeout(() => {
    const dPlain = predictAll(modelPlain, pgaRaw);
    const dPinn  = predictAll(modelPinn,  pgaRaw);
    renderWall('wall-canvas-plain', dPlain);
    renderWall('wall-canvas-pinn',  dPinn);

    const sp = zoneStats(dPlain), sn = zoneStats(dPinn);
    document.getElementById('stats-plain').textContent =
      `LW=${sp.lw.toFixed(3)}  base=${sp.base.toFixed(3)}  in-band=${sp.inband.toFixed(3)}`;
    document.getElementById('stats-pinn').textContent =
      `LW=${sn.lw.toFixed(3)}  base=${sn.base.toFixed(3)}  in-band=${sn.inband.toFixed(3)}`;

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

  renderColorbar();

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
