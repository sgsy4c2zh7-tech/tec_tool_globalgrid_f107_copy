/* SWIFT-TEC v4 add-on: 30-day NOAA TEC archive + selectable replay + multi-GNSS DOP/TEC.
   Load this after the original SWIFT-TEC script. */
(function () {
  const TEC_INDEX_URL = "data/tec/index.json";
  const TEC_BASE_URL = "data/tec/";
  const SATELLITE_JS_URL = "vendor/satellite.min.js";
  const KP_AI_COEFF_URL = "data/ai/kp_coefficients.json";
  const KP_AI_PERF_URL = "data/ai/kp_performance.json";
  const KP_AI_GRID_COEFF_URL = "data/ai/kp_grid_coefficients.json";

  const GNSS_SOURCES = {
    gps:     { label: "GPS",     url: "data/gnss/gps_latest.tle",     liveUrl: "https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=tle",     checked: true  },
    galileo: { label: "Galileo", url: "data/gnss/galileo_latest.tle", liveUrl: "https://celestrak.org/NORAD/elements/gp.php?GROUP=galileo&FORMAT=tle", checked: false },
    glonass: { label: "GLONASS", url: "data/gnss/glonass_latest.tle", liveUrl: "https://celestrak.org/NORAD/elements/gp.php?GROUP=glo-ops&FORMAT=tle", checked: false },
    beidou:  { label: "BeiDou",  url: "data/gnss/beidou_latest.tle",  liveUrl: "https://celestrak.org/NORAD/elements/gp.php?GROUP=beidou&FORMAT=tle",  checked: false },
    qzss:    { label: "QZSS",    url: "data/gnss/qzss_latest.tle",    liveUrl: "https://celestrak.org/NORAD/elements/gp.php?GROUP=qzss&FORMAT=tle",    checked: true  },
  };

  let archiveIndex = null;
  let gnssSatList = [];
  let gnssLoaded = false;
  let dopFrameCache = new Map();
  let tecInterpCache = new Map();

  let originalDrawTecOverlay = null;
  let originalUpdateLegend = null;
  let originalSampleAtLatLon = null;
  let originalChangeMapMode = null;
  let originalReadTecInputsForForecastApi = null;
  let originalFillForecastStartCandidatesForForecastApi = null;
  let originalBuildNoaaTecTodForForecastApi = null;
  let originalRunForecastForForecastApi = null;
  let forecastTecApiReady = false;
  let forecastTecApiSourceLabel = "未取得";

  // data/tecに保存されるTECはNOAA 30分値。UIではTEC系30分、DOP系10分へ展開する。
  const TEC_REPLAY_STEP_MIN = 30;
  const DOP_REPLAY_STEP_MIN = 10;
  const TEC_FORECAST_HOURS = 72;
  const DEFAULT_TEC_FORECAST_BASE_DAYS = 7;
  let rawDisplayFrames = [];
  let activeTimelineStepMin = TEC_REPLAY_STEP_MIN;
  let tecSmoothCache = new Map();
  let selectionVersion = 0;

  let kpAiCoefficients = null;
  let kpAiPerformance = null;
  let kpAiGridCoefficients = null;
  let kpAiLoaded = false;
  let kpAiLoadError = null;
  let kpAiRenderMode = "hitrate";

  const dopColorScale = [
    { limit: 2,  color: "#00ff00" },
    { limit: 4,  color: "#ffff00" },
    { limit: 8,  color: "#ff9900" },
    { limit: 16, color: "#ff0000" },
  ];

  const dopTecColorScale = [
    { limit: 5,  color: "#00ff00" },
    { limit: 10, color: "#ffff00" },
    { limit: 20, color: "#ff9900" },
    { limit: 40, color: "#ff0000" },
  ];

  const satCountColorScale = [
    { limit: 4,  color: "#ff0000" },
    { limit: 6,  color: "#ff9900" },
    { limit: 8,  color: "#ffff00" },
    { limit: 16, color: "#00ff00" },
  ];

  function c(v, a, b) {
    if (typeof clamp === "function") return clamp(v, a, b);
    return v < a ? a : (v > b ? b : v);
  }

  function isoNoMs(t) {
    if (!(t instanceof Date) || isNaN(t.getTime())) return "--";
    return t.toISOString().replace(".000Z", "Z");
  }

  function setV4Status(msg) {
    const el = document.getElementById("v4ArchiveStatus");
    if (el) el.textContent = msg || "";
    if (typeof logInfo === "function") logInfo(msg || "");
  }

  async function readJsonMaybeGz(url) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${url}`);
    if (!url.endsWith(".gz")) return await res.json();

    const buf = await res.arrayBuffer();
    if (!("DecompressionStream" in window)) {
      throw new Error("このブラウザは gzip 展開(DecompressionStream)に未対応です。json.gzではなくjson保存にしてください。: " + url);
    }
    const ds = new DecompressionStream("gzip");
    const stream = new Blob([buf]).stream().pipeThrough(ds);
    const text = await new Response(stream).text();
    return JSON.parse(text);
  }

  async function loadTecArchiveIndex(force = false) {
    if (archiveIndex && !force) return archiveIndex;
    archiveIndex = await readJsonMaybeGz(TEC_INDEX_URL);
    const frames = Array.isArray(archiveIndex.frames) ? archiveIndex.frames : [];
    frames.sort((a, b) => String(a.time_utc).localeCompare(String(b.time_utc)));
    archiveIndex.frames = frames;
    populateArchiveSelectors();
    return archiveIndex;
  }

  function populateArchiveSelectors() {
    const startSel = document.getElementById("archiveStartSelect");
    const endSel = document.getElementById("archiveEndSelect");
    const info = document.getElementById("archiveIndexInfo");
    if (!startSel || !endSel || !archiveIndex) return;

    startSel.innerHTML = "";
    endSel.innerHTML = "";
    const frames = archiveIndex.frames || [];
    for (const f of frames) {
      const label = String(f.time_utc || "").replace(".000Z", "Z");
      const opt1 = document.createElement("option");
      opt1.value = f.time_utc;
      opt1.textContent = label;
      startSel.appendChild(opt1);

      const opt2 = document.createElement("option");
      opt2.value = f.time_utc;
      opt2.textContent = label;
      endSel.appendChild(opt2);
    }

    if (frames.length) {
      startSel.value = frames[Math.max(0, frames.length - 13)].time_utc; // default: roughly latest 24h at 2h step
      endSel.value = frames[frames.length - 1].time_utc;
    }
    if (info) {
      info.textContent = frames.length
        ? `履歴 ${frames.length}枚 / ${frames[0].time_utc} 〜 ${frames[frames.length - 1].time_utc} / updated=${archiveIndex.updated_utc || "--"}`
        : "履歴なし";
    }
  }

  async function loadArchiveFrame(entry) {
    const url = TEC_BASE_URL + entry.file;
    const frame = await readJsonMaybeGz(url);
    const time = new Date(frame.time_utc || entry.time_utc);
    return {
      time,
      gridMeta: {
        latArr: frame.lat_arr,
        lonArr: frame.lon_arr,
        nLat: frame.n_lat || (frame.lat_arr ? frame.lat_arr.length : 0),
        nLon: frame.n_lon || (frame.lon_arr ? frame.lon_arr.length : 0),
      },
      grid: frame.grid,
      sourceFile: frame.source_file || entry.file,
    };
  }

  function selectedArchiveEntries() {
    if (!archiveIndex || !Array.isArray(archiveIndex.frames)) return [];
    const startVal = document.getElementById("archiveStartSelect")?.value;
    const endVal = document.getElementById("archiveEndSelect")?.value;
    const start = startVal ? new Date(startVal).getTime() : -Infinity;
    const end = endVal ? new Date(endVal).getTime() : Infinity;
    const a = Math.min(start, end);
    const b = Math.max(start, end);
    return archiveIndex.frames.filter(f => {
      const t = new Date(f.time_utc).getTime();
      return isFinite(t) && t >= a && t <= b;
    });
  }

  function isDopLikeMode() {
    return [
      "satcount",
      "gdop", "pdop", "hdop", "vdop", "tdop",
      "gdoptec", "pdoptec", "hdoptec", "vdoptec",
      "dop", "doptec",
    ].includes(mapMode);
  }

  function buildUniformTimeline(frames, stepMinutes) {
    if (!frames.length) return [];
    const start = frames[0].time.getTime();
    const end = frames[frames.length - 1].time.getTime();
    const stepMs = Math.max(1, stepMinutes) * 60000;
    const out = [];
    for (let t = start; t <= end + 1000; t += stepMs) out.push(new Date(t));
    if (out.length && out[out.length - 1].getTime() < end) out.push(new Date(end));
    return out;
  }

  function setSliderForTimeline() {
    const slider = document.getElementById("timeSlider");
    if (slider) {
      slider.min = "0";
      slider.max = String(Math.max(0, gForecastTimes.length - 1));
      slider.value = String(c(currentStepIndex, 0, Math.max(0, gForecastTimes.length - 1)));
    }
  }

  function getTimelineLabel() {
    if (isDopLikeMode()) return `DOP系: ${DOP_REPLAY_STEP_MIN}分刻み`;
    return `TEC系: ${TEC_REPLAY_STEP_MIN}分刻み`;
  }

  function adoptCurrentForecastAsRawFramesIfNeeded() {
    if (rawDisplayFrames.length) return;
    if (!gGrid || !Array.isArray(gForecastFrames) || !Array.isArray(gForecastTimes)) return;
    if (!gForecastFrames.length || !Array.isArray(gForecastFrames[0])) return;
    rawDisplayFrames = gForecastFrames.map((grid, i) => ({
      time: gForecastTimes[i],
      grid,
      gridMeta: gGrid,
      sourceFile: "forecast",
    })).filter(f => f.time instanceof Date && !isNaN(f.time.getTime()) && Array.isArray(f.grid));
  }

  function applyTimelineForCurrentMode(resetIndex = true) {
    adoptCurrentForecastAsRawFramesIfNeeded();
    if (!rawDisplayFrames.length) return;
    tecInterpCache.clear();
    tecSmoothCache.clear();

    activeTimelineStepMin = isDopLikeMode() ? DOP_REPLAY_STEP_MIN : TEC_REPLAY_STEP_MIN;
    gForecastTimes = buildUniformTimeline(rawDisplayFrames, activeTimelineStepMin);
    // 保存TECは30分値。TEC表示では基本的にそのまま使い、DOP系では10分時刻へ時間補間する。
    gForecastFrames = new Array(gForecastTimes.length).fill(null);

    if (resetIndex) currentStepIndex = 0;
    currentStepIndex = c(currentStepIndex, 0, Math.max(0, gForecastTimes.length - 1));
    gForecastStart = gForecastTimes[0] || null;
    setSliderForTimeline();
    dynamicOnSliderChange();
  }

  function interpolateGridAtTime(t) {
    if (!rawDisplayFrames.length) return null;
    if (!(t instanceof Date) || isNaN(t.getTime())) return rawDisplayFrames[0].grid;

    const strength = getFourierSmoothStrength();
    const smoothKey = `${t.toISOString()}|s=${strength.toFixed(2)}|v=${selectionVersion}`;
    if (tecSmoothCache.has(smoothKey)) return tecSmoothCache.get(smoothKey);

    const rawKey = t.toISOString();
    let grid = tecInterpCache.get(rawKey);
    if (!grid) {
      const targetMs = t.getTime();
      if (rawDisplayFrames.length === 1 || targetMs <= rawDisplayFrames[0].time.getTime()) {
        grid = rawDisplayFrames[0].grid;
      } else {
        const last = rawDisplayFrames[rawDisplayFrames.length - 1];
        if (targetMs >= last.time.getTime()) {
          grid = last.grid;
        } else {
          let lo = rawDisplayFrames[0];
          let hi = rawDisplayFrames[rawDisplayFrames.length - 1];
          for (let k = 0; k < rawDisplayFrames.length - 1; k++) {
            const a = rawDisplayFrames[k];
            const b = rawDisplayFrames[k + 1];
            if (a.time.getTime() <= targetMs && targetMs <= b.time.getTime()) {
              lo = a; hi = b; break;
            }
          }
          const span = Math.max(1, hi.time.getTime() - lo.time.getTime());
          const w = c((targetMs - lo.time.getTime()) / span, 0, 1);
          if (w <= 1e-9) grid = lo.grid;
          else if (w >= 1 - 1e-9) grid = hi.grid;
          else {
            const nLat = gGrid.nLat, nLon = gGrid.nLon;
            const out = Array.from({ length: nLat }, () => Array(nLon).fill(0));
            for (let i = 0; i < nLat; i++) {
              const rowA = lo.grid[i] || [];
              const rowB = hi.grid[i] || [];
              const rowO = out[i];
              for (let j = 0; j < nLon; j++) {
                const rawA = rowA[j];
                const rawB = rowB[j];
                const a = (rawA === null || rawA === undefined || rawA === "") ? NaN : Number(rawA);
                const b = (rawB === null || rawB === undefined || rawB === "") ? NaN : Number(rawB);
                if (isFinite(a) && isFinite(b)) rowO[j] = a * (1 - w) + b * w;
                else if (isFinite(a)) rowO[j] = a;
                else if (isFinite(b)) rowO[j] = b;
                else rowO[j] = NaN;
              }
            }
            grid = out;
          }
        }
      }
      tecInterpCache.set(rawKey, grid);
      if (tecInterpCache.size > 60) {
        const firstKey = tecInterpCache.keys().next().value;
        tecInterpCache.delete(firstKey);
      }
    }

    // 0 TECの孤立穴だけを表示用に補正する。
    // NOAAの実データ表示方針は維持し、時間補間以外の平滑化・再スケールは行わない。
    return patchIsolatedZeroHoles(grid);
  }


  function getFourierSmoothStrength() {
    const el = document.getElementById("fourierSmoothSelect");
    const v = parseFloat(el?.value ?? "0.20");
    return isFinite(v) ? c(v, 0, 0.95) : 0.20;
  }

  function medianFinite(vals) {
    const a = vals.filter(v => isFinite(v)).sort((x, y) => x - y);
    if (!a.length) return 0;
    return a[Math.floor(a.length / 2)];
  }

  function cloneGrid(g) {
    return (g || []).map(row => (row || []).map(v => {
      const x = Number(v);
      return isFinite(x) ? Math.max(0, x) : NaN;
    }));
  }

  function inpaintGrid(grid, iterations = 6) {
    const nLat = gGrid?.nLat || grid.length;
    const nLon = gGrid?.nLon || (grid[0] ? grid[0].length : 0);
    let out = cloneGrid(grid);
    const observed = Array.from({ length: nLat }, () => Array(nLon).fill(false));
    const all = [];
    for (let i = 0; i < nLat; i++) {
      for (let j = 0; j < nLon; j++) {
        if (isFinite(out[i]?.[j])) {
          observed[i][j] = true;
          all.push(out[i][j]);
        }
      }
    }
    const fallback = medianFinite(all);
    for (let i = 0; i < nLat; i++) {
      if (!out[i]) out[i] = Array(nLon).fill(NaN);
      for (let j = 0; j < nLon; j++) {
        if (!isFinite(out[i][j])) out[i][j] = NaN;
      }
    }

    // 欠損点だけを近傍平均で埋める。観測点は一切平均化しない。
    for (let iter = 0; iter < iterations; iter++) {
      const next = out.map(r => r.slice());
      let changed = 0;
      for (let i = 0; i < nLat; i++) {
        for (let j = 0; j < nLon; j++) {
          if (observed[i][j] && isFinite(out[i][j])) continue;
          const vals = [];
          for (let di = -1; di <= 1; di++) {
            for (let dj = -1; dj <= 1; dj++) {
              if (di === 0 && dj === 0) continue;
              const ii = i + di;
              let jj = j + dj;
              if (ii < 0 || ii >= nLat) continue;
              if (jj < 0) jj += nLon;
              if (jj >= nLon) jj -= nLon;
              const v = out[ii]?.[jj];
              if (isFinite(v)) vals.push(v);
            }
          }
          if (vals.length) {
            next[i][j] = vals.reduce((a, b) => a + b, 0) / vals.length;
            changed++;
          }
        }
      }
      out = next;
      if (!changed) break;
    }

    for (let i = 0; i < nLat; i++) {
      for (let j = 0; j < nLon; j++) {
        if (!isFinite(out[i][j])) out[i][j] = fallback;
        out[i][j] = Math.max(0, out[i][j]);
      }
    }
    return out;
  }

  function lowPassFourier1D(arr, keepHarmonics) {
    const n = arr.length;
    if (n <= 2) return arr.slice();
    const K = Math.max(0, Math.min(Math.floor(n / 2), keepHarmonics));
    const re = new Array(K + 1).fill(0);
    const im = new Array(K + 1).fill(0);
    for (let k = 0; k <= K; k++) {
      let sr = 0, si = 0;
      for (let x = 0; x < n; x++) {
        const ang = -2 * Math.PI * k * x / n;
        const v = Number(arr[x]) || 0;
        sr += v * Math.cos(ang);
        si += v * Math.sin(ang);
      }
      re[k] = sr;
      im[k] = si;
    }
    const out = new Array(n).fill(0);
    for (let x = 0; x < n; x++) {
      let v = re[0] / n;
      for (let k = 1; k <= K; k++) {
        const ang = 2 * Math.PI * k * x / n;
        v += (2 / n) * (re[k] * Math.cos(ang) - im[k] * Math.sin(ang));
      }
      out[x] = Math.max(0, v);
    }
    return out;
  }

  function finiteGridValues(grid) {
    const vals = [];
    for (const row of (grid || [])) {
      for (const v0 of (row || [])) {
        const v = Number(v0);
        if (isFinite(v)) vals.push(Math.max(0, v));
      }
    }
    vals.sort((a, b) => a - b);
    return vals;
  }

  function percentileFiniteGrid(grid, p) {
    const vals = finiteGridValues(grid);
    return percentileSorted(vals, p);
  }

  function matchTecDistribution(smoothed, reference) {
    // 旧版の見え方に寄せるため、フーリエで山が潰れてもTECの物理的なレンジを戻す。
    // 中央値を基準に、P05/P95の幅をreferenceと一致させる。
    const refVals = finiteGridValues(reference);
    const smVals = finiteGridValues(smoothed);
    if (refVals.length < 8 || smVals.length < 8) return smoothed;

    const refP05 = percentileSorted(refVals, 0.05);
    const refP50 = percentileSorted(refVals, 0.50);
    const refP95 = percentileSorted(refVals, 0.95);
    const smP05  = percentileSorted(smVals, 0.05);
    const smP50  = percentileSorted(smVals, 0.50);
    const smP95  = percentileSorted(smVals, 0.95);

    const refAmp = Math.max(1e-6, refP95 - refP05);
    const smAmp  = Math.max(1e-6, smP95 - smP05);
    let gain = c(refAmp / smAmp, 0.45, 3.50);

    const refMax = percentileSorted(refVals, 0.995);
    const hardMax = Math.max(120, refMax * 1.25);

    return smoothed.map(row => row.map(v0 => {
      const v = Number(v0);
      if (!isFinite(v)) return NaN;
      const restored = refP50 + (v - smP50) * gain;
      return c(restored, 0, hardMax);
    }));
  }

  function fourierSmoothGrid(grid, strength = 0.20) {
    const nLat = gGrid?.nLat || grid.length;
    const nLon = gGrid?.nLon || (grid[0] ? grid[0].length : 0);

    // 旧版は格子値をそのまま矩形で塗っていたため、TEC振幅はそのまま残っていた。
    // ここでは欠損だけ埋め、観測済みセルのTEC値は基準値として保持する。
    const filled = inpaintGrid(grid, 3);

    // 歯抜けを消すために低域は使うが、以前より高周波を多めに残す。
    // nLon=72, nLat=91程度なら keepLon/keepLat が十分大きく、局所的なTEC山を潰しにくい。
    const keepLon = Math.max(14, Math.round(nLon * (0.72 - 0.22 * strength)));
    const keepLat = Math.max(14, Math.round(nLat * (0.72 - 0.22 * strength)));

    const rowSmooth = Array.from({ length: nLat }, () => Array(nLon).fill(0));
    for (let i = 0; i < nLat; i++) rowSmooth[i] = lowPassFourier1D(filled[i], keepLon);

    const colSmooth = Array.from({ length: nLat }, () => Array(nLon).fill(0));
    for (let j = 0; j < nLon; j++) {
      const col = new Array(nLat);
      for (let i = 0; i < nLat; i++) col[i] = rowSmooth[i][j];
      const sm = lowPassFourier1D(col, keepLat);
      for (let i = 0; i < nLat; i++) colSmooth[i][j] = sm[i];
    }

    // 値を小さくしすぎないよう、平滑値の混合率は控えめ。
    const blend = c(0.06 + 0.28 * strength, 0, 0.32);
    const out = Array.from({ length: nLat }, () => Array(nLon).fill(0));
    for (let i = 0; i < nLat; i++) {
      for (let j = 0; j < nLon; j++) {
        const raw = isFinite(filled[i][j]) ? filled[i][j] : colSmooth[i][j];
        const sm = isFinite(colSmooth[i][j]) ? colSmooth[i][j] : raw;
        out[i][j] = Math.max(0, raw * (1 - blend) + sm * blend);
      }
    }

    return matchTecDistribution(out, filled);
  }

  function patchIsolatedZeroHoles(grid, zeroThreshold = 0.05, minNeighborCount = 4, neighborPositiveThreshold = 0.2) {
    if (!Array.isArray(grid) || !grid.length || !Array.isArray(grid[0])) return grid;
    const nLat = grid.length;
    const nLon = grid[0].length;
    const out = grid.map(row => row.slice());

    for (let i = 0; i < nLat; i++) {
      for (let j = 0; j < nLon; j++) {
        const v = Number(grid[i]?.[j]);
        if (isFinite(v) && v > zeroThreshold) continue;

        const vals = [];
        for (let di = -1; di <= 1; di++) {
          for (let dj = -1; dj <= 1; dj++) {
            if (di === 0 && dj === 0) continue;
            const ii = i + di;
            let jj = j + dj;
            if (ii < 0 || ii >= nLat) continue;
            if (jj < 0) jj += nLon;
            if (jj >= nLon) jj -= nLon;
            const nv = Number(grid[ii]?.[jj]);
            if (isFinite(nv) && nv > neighborPositiveThreshold) vals.push(nv);
          }
        }
        if (vals.length < minNeighborCount) continue;
        vals.sort((a, b) => a - b);
        const med = vals[Math.floor(vals.length / 2)];
        out[i][j] = med;
      }
    }
    return out;
  }

  function normalizeLon180(lon) {
    let x = Number(lon);
    if (!isFinite(x)) return 0;
    while (x < -180) x += 360;
    while (x >= 180) x -= 360;
    return x;
  }

  function kpAiRegionId(lat, lon) {
    const la = Number(lat);
    const lo = normalizeLon180(lon);
    const latBand = la < -30 ? 0 : (la < 30 ? 1 : 2); // 0=S,1=EQ,2=N
    let lonBand = Math.floor((lo + 180) / 60);
    lonBand = c(lonBand, 0, 5);
    return `R${String(latBand * 6 + lonBand + 1).padStart(2, "0")}`;
  }

  function kpAiMonthKey(t) {
    return (t instanceof Date && !isNaN(t.getTime())) ? String(t.getUTCMonth() + 1) : String((new Date()).getUTCMonth() + 1);
  }

  function kpAiEnabled() {
    const el = document.getElementById("kpAiCorrectionEnabled");
    return !!(el && el.checked && kpAiCoefficients && kpAiLoaded);
  }

  function kpAiClipLimit() {
    const el = document.getElementById("kpAiCorrectionClip");
    const v = parseFloat(el?.value || kpAiCoefficients?.model?.correction_clip_tecu || "20");
    return isFinite(v) ? c(v, 1, 60) : 20;
  }

  function kpAiKpAtTime(t) {
    try {
      if (Array.isArray(gKpSeries) && gKpSeries.length) {
        let best = null, bestDiff = Infinity;
        for (const r of gKpSeries) {
          const rt = r.t instanceof Date ? r.t : new Date(r.t || r.time || r.time_utc || 0);
          const kp = Number(r.kp ?? r.Kp ?? r.value);
          if (!isFinite(kp) || isNaN(rt.getTime())) continue;
          const d = Math.abs(rt.getTime() - t.getTime());
          if (d < bestDiff) { bestDiff = d; best = kp; }
        }
        if (best !== null && bestDiff <= 4 * 3600000) return best;
      }
    } catch {}
    const el = document.getElementById("kpAiFallbackKp");
    const fallback = parseFloat(el?.value || "3");
    return isFinite(fallback) ? fallback : 3.0;
  }

  function kpAiCoeffFor(regionId, monthKey) {
    const cfs = kpAiCoefficients?.coefficients || {};
    const r = cfs[regionId] || {};
    return r[monthKey] || r[String(parseInt(monthKey, 10))] || null;
  }

  function kpAiCorrectionValue(lat, lon, t, kp) {
    if (!kpAiEnabled()) return 0;
    const rid = kpAiRegionId(lat, lon);
    const mk = kpAiMonthKey(t);
    const cf = kpAiCoeffFor(rid, mk);
    if (!cf || Number(cf.sample_count || 0) < 8) return 0;
    const x = (isFinite(kp) ? kp : 3.0) - 3.0;
    const k0 = Number(cf.k0 || 0), k1 = Number(cf.k1 || 0), k2 = Number(cf.k2 || 0), k3 = Number(cf.k3 || 0);
    const y = k0 + k1 * x + k2 * x * x + k3 * x * x * x;
    const lim = kpAiClipLimit();
    return isFinite(y) ? c(y, -lim, lim) : 0;
  }


  function kpAiGridMonthFor(monthKey) {
    const g = kpAiGridCoefficients?.coefficients_grid || kpAiGridCoefficients?.grid_coefficients || {};
    return g[monthKey] || g[String(parseInt(monthKey, 10))] || null;
  }

  function kpAiNearestIndex(arr, value) {
    if (!Array.isArray(arr) || !arr.length) return -1;
    let best = 0, bestD = Infinity;
    const v = Number(value);
    for (let i = 0; i < arr.length; i++) {
      const d = Math.abs(Number(arr[i]) - v);
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  }

  function kpAiGridCoeffAt(monthGrid, i, j, lat, lon) {
    if (!monthGrid) return null;
    let ii = i, jj = j;
    const latArr = kpAiGridCoefficients?.lat_arr || kpAiGridCoefficients?.latArr;
    const lonArr = kpAiGridCoefficients?.lon_arr || kpAiGridCoefficients?.lonArr;
    if (Array.isArray(latArr) && Array.isArray(lonArr)) {
      if (!monthGrid.k0?.[ii] || monthGrid.k0?.[ii]?.[jj] === undefined) {
        ii = kpAiNearestIndex(latArr, lat);
        jj = kpAiNearestIndex(lonArr, lon);
      }
    }
    const n = Number(monthGrid.sample_count?.[ii]?.[jj] ?? monthGrid.count?.[ii]?.[jj] ?? 0);
    if (!isFinite(n) || n < 4) return null;
    return {
      k0: Number(monthGrid.k0?.[ii]?.[jj] || 0),
      k1: Number(monthGrid.k1?.[ii]?.[jj] || 0),
      k2: Number(monthGrid.k2?.[ii]?.[jj] || 0),
      k3: Number(monthGrid.k3?.[ii]?.[jj] || 0),
      sample_count: n,
    };
  }

  function kpAiBaseKpAtTime(t) {
    try {
      const v = todValueAt(gBaseKpTod, t);
      if (isFinite(v)) return Number(v);
    } catch {}
    // If base Kp is not ready, neutral Kp=3 gives F(KpB)=k0 and keeps the correction bounded.
    return 3.0;
  }

  function kpAiFValue(cf, kp) {
    const x = (isFinite(kp) ? kp : 3.0) - 3.0;
    return Number(cf.k0 || 0) + Number(cf.k1 || 0) * x + Number(cf.k2 || 0) * x * x + Number(cf.k3 || 0) * x * x * x;
  }

  function applyKpAiCorrectionToGrid(grid, t) {
    if (!kpAiEnabled() || !grid || !gGrid) return grid;

    // User model:
    //   BaseTEC = PrevObservedTEC - F(KpB)
    //   ForecastTEC = BaseTEC + F(KpF)
    // Therefore the AI Kp term is not F(KpF) alone.
    // It is the net Kp term:
    //   AI_delta = F(KpF) - F(KpB)
    const kpF = kpAiKpAtTime(t);
    const kpB = kpAiBaseKpAtTime(t);
    const mk = kpAiMonthKey(t);
    const monthGrid = kpAiGridMonthFor(mk);
    const lim = kpAiClipLimit();

    const out = Array.from({ length: gGrid.nLat }, () => Array(gGrid.nLon).fill(NaN));
    for (let i = 0; i < gGrid.nLat; i++) {
      const lat = gGrid.latArr[i];
      for (let j = 0; j < gGrid.nLon; j++) {
        const v = Number(grid?.[i]?.[j]);
        if (!isFinite(v)) { out[i][j] = NaN; continue; }

        let cf = kpAiGridCoeffAt(monthGrid, i, j, lat, gGrid.lonArr[j]);
        if (!cf) {
          const rid = kpAiRegionId(lat, gGrid.lonArr[j]);
          cf = kpAiCoeffFor(rid, mk);
        }

        let corr = 0;
        if (cf && Number(cf.sample_count || 0) >= 4) {
          const y = kpAiFValue(cf, kpF) - kpAiFValue(cf, kpB);
          corr = isFinite(y) ? c(y, -lim, lim) : 0;
        }
        out[i][j] = Math.max(0, v + corr);
      }
    }
    return out;
  }

  async function loadKpAiData(force = false) {
    if (kpAiLoaded && !force) return;
    kpAiLoadError = null;
    try {
      const [coeff, perf, gridCoeff] = await Promise.all([
        fetch(KP_AI_COEFF_URL, { cache: "no-store" }).then(r => r.ok ? r.json() : null),
        fetch(KP_AI_PERF_URL, { cache: "no-store" }).then(r => r.ok ? r.json() : null),
        fetch(KP_AI_GRID_COEFF_URL, { cache: "no-store" }).then(r => r.ok ? r.json() : null).catch(() => null),
      ]);
      kpAiCoefficients = coeff;
      kpAiPerformance = perf;
      kpAiGridCoefficients = gridCoeff;
      kpAiLoaded = !!(coeff || gridCoeff);
      renderKpAiPanel();
    } catch (e) {
      kpAiLoadError = e.message;
      kpAiLoaded = false;
      renderKpAiPanel();
    }
  }

  function kpAiPerfRows() {
    const rows = [];
    const metrics = kpAiPerformance?.metrics || {};
    for (const [rid, byMonth] of Object.entries(metrics)) {
      for (const [month, m] of Object.entries(byMonth || {})) {
        rows.push({ region_id: rid, month: Number(month), ...m });
      }
    }
    return rows;
  }

  function kpAiLatestMonthRows() {
    const rows = kpAiPerfRows();
    if (!rows.length) return [];
    const nowMonth = (new Date()).getUTCMonth() + 1;
    const hasNow = rows.some(r => Number(r.month) === nowMonth);
    const month = hasNow ? nowMonth : rows[rows.length - 1].month;
    const selected = rows.filter(r => Number(r.month) === Number(month));
    selected.sort((a, b) => String(a.region_id).localeCompare(String(b.region_id)));
    return selected;
  }

  function renderKpAiPanel() {
    const status = document.getElementById("kpAiStatus");
    const table = document.getElementById("kpAiRegionTable");
    const bars = document.getElementById("kpAiHitRateBars");
    if (status) {
      if (kpAiLoaded) {
        const updated = kpAiCoefficients?.updated_utc || "--";
        status.textContent = `AI係数読込OK: grid-cell + 18地域表示 / updated=${updated}`;
      } else if (kpAiLoadError) {
        status.textContent = `AI係数未読込: ${kpAiLoadError}`;
      } else {
        status.textContent = "AI係数未読込。Train Kp AI Corrector workflow実行後に表示されます。";
      }
    }
    const rows = kpAiLatestMonthRows();
    if (bars) {
      if (!rows.length) bars.innerHTML = '<div class="small">性能データなし</div>';
      else bars.innerHTML = rows.map(r => {
        const hit = Number(r.corrected_hit_rate ?? r.hit_rate ?? 0) * 100;
        const raw = Number(r.raw_hit_rate ?? 0) * 100;
        const w = c(hit, 0, 100);
        return `<div class="small" style="margin:2px 0;">${r.region_id} <span class="mono">${hit.toFixed(0)}%</span> <span style="opacity:.65;">raw ${raw.toFixed(0)}%</span><div style="height:6px;background:#111827;border:1px solid #334;border-radius:999px;overflow:hidden;"><div style="height:100%;width:${w}%;background:#3b82f6;"></div></div></div>`;
      }).join("");
    }
    if (table) {
      if (!rows.length) table.innerHTML = '<div class="small">係数・性能データなし</div>';
      else table.innerHTML = `<table><thead><tr><th>地域</th><th>月</th><th>N</th><th>Hit</th><th>RMSE</th><th>k0/k1/k2/k3</th></tr></thead><tbody>${rows.map(r => {
        const cf = kpAiCoeffFor(r.region_id, String(r.month)) || {};
        const hit = Number(r.corrected_hit_rate ?? 0) * 100;
        const rmse = Number(r.corrected_rmse ?? r.rmse ?? NaN);
        return `<tr><td class="mono">${r.region_id}</td><td>${r.month}</td><td>${r.sample_count || 0}</td><td>${isFinite(hit)?hit.toFixed(0):"--"}%</td><td>${isFinite(rmse)?rmse.toFixed(2):"--"}</td><td class="mono">${Number(cf.k0||0).toFixed(2)} / ${Number(cf.k1||0).toFixed(2)} / ${Number(cf.k2||0).toFixed(2)} / ${Number(cf.k3||0).toFixed(2)}</td></tr>`;
      }).join("")}</tbody></table>`;
    }
  }

  function ensureKpAiUi() {
    if (document.getElementById("kpAiCorrectorCard")) return;
    const card = document.createElement("div");
    card.className = "card";
    card.id = "kpAiCorrectorCard";
    card.innerHTML = `
      <div class="card-header"><h2>8d. Kp Cubic AI Corrector</h2><span>全格子AI解析 / 18地域表示</span></div>
      <div class="small">
        ・Kpによる残差だけを <b>k0 + k1(Kp-3) + k2(Kp-3)^2 + k3(Kp-3)^3</b> で補正します。<br>
        ・係数は <b>全格子×12か月</b> で学習し、UIでは18地域に集約して表示します。
      </div>
      <div class="row small" style="margin-top:6px;">
        <label><input id="kpAiCorrectionEnabled" type="checkbox" onchange="tecInterpCache.clear(); requestDraw();"> AI Kp補正ON</label>
        <div>補正上限[TECU]:<br><input id="kpAiCorrectionClip" type="number" value="20" min="1" max="60" step="1" style="width:80px;" onchange="tecInterpCache.clear(); requestDraw();"></div>
        <div>Fallback Kp:<br><input id="kpAiFallbackKp" type="number" value="3" min="0" max="9" step="0.33" style="width:80px;" onchange="tecInterpCache.clear(); requestDraw();"></div>
      </div>
      <div class="row">
        <button onclick="loadKpAiData(true)">AI係数/的中率を読込</button>
        <button class="secondary" onclick="renderKpAiPanel()">グラフ再表示</button>
      </div>
      <div class="small" id="kpAiStatus">AI係数未読込</div>
      <div id="kpAiHitRateBars" style="max-height:210px;overflow:auto;border:1px solid #222b3f;border-radius:6px;padding:4px;margin-top:4px;"></div>
      <div id="kpAiRegionTable" style="max-height:210px;overflow:auto;border:1px solid #222b3f;border-radius:6px;padding:4px;margin-top:4px;"></div>
    `;
    const sidebar = document.querySelector(".sidebar");
    const after = document.getElementById("forecastTecApiCard") || document.getElementById("satelliteSelectionList")?.closest(".card") || document.getElementById("v4ArchiveStatus")?.closest(".card");
    if (sidebar && after) after.insertAdjacentElement("afterend", card);
    else if (sidebar) sidebar.appendChild(card);
    renderKpAiPanel();
  }

  function currentTecGrid() {
    const t = gForecastTimes[currentStepIndex];
    if (!(t instanceof Date) || isNaN(t.getTime())) return null;
    const grid = interpolateGridAtTime(t);
    return applyKpAiCorrectionToGrid(grid, t);
  }

  function maybeAutoPlayAfterLoad() {
    const chk = document.getElementById("movieAutoPlayAfterLoad");
    if (chk && chk.checked) playArchiveMovie();
  }

  function setDisplayedFrames(frames, sourceLabel) {
    if (!frames.length) throw new Error("表示できるTECフレームがありません。期間を変えてください。");
    const meta = frames[0].gridMeta;
    gGrid = { latArr: meta.latArr, lonArr: meta.lonArr, nLat: meta.nLat, nLon: meta.nLon };
    rawDisplayFrames = frames.slice().sort((a, b) => a.time - b.time);
    dopFrameCache.clear();
    tecInterpCache.clear();
    tecSmoothCache.clear();
    currentStepIndex = 0;

    if (typeof initMapIfNeeded === "function") initMapIfNeeded();
    if (typeof updateLegend === "function") updateLegend();
    applyTimelineForCurrentMode(true);
    if (typeof updateKpLabels === "function") updateKpLabels();
    setV4Status(`${sourceLabel}: base ${rawDisplayFrames.length}枚を読み込み。表示=${getTimelineLabel()} / ${isoNoMs(rawDisplayFrames[0].time)} 〜 ${isoNoMs(rawDisplayFrames[rawDisplayFrames.length - 1].time)}`);
    maybeAutoPlayAfterLoad();
  }

  async function loadTecArchiveRange() {
    try {
      setV4Status("TEC履歴indexを読み込み中…");
      await loadTecArchiveIndex();
      const entries = selectedArchiveEntries();
      if (!entries.length) throw new Error("選択期間にTEC履歴がありません。GitHub Actions実行後に再確認してください。");
      setV4Status(`TEC履歴を読み込み中… ${entries.length}枚`);
      const frames = [];
      for (const e of entries) frames.push(await loadArchiveFrame(e));
      setDisplayedFrames(frames, "過去TEC");
    } catch (e) {
      console.error(e);
      setV4Status("TEC履歴読み込み失敗: " + e.message);
    }
  }

  async function loadTecArchivePlusCurrentForecast() {
    try {
      const existingFrames = Array.isArray(gForecastFrames) ? gForecastFrames.slice() : [];
      const existingTimes = Array.isArray(gForecastTimes) ? gForecastTimes.slice() : [];
      await loadTecArchiveIndex();
      const entries = selectedArchiveEntries();
      const historyFrames = [];
      for (const e of entries) historyFrames.push(await loadArchiveFrame(e));

      const combined = historyFrames.slice();
      if (existingFrames.length && existingTimes.length) {
        const lastHist = combined.length ? combined[combined.length - 1].time.getTime() : -Infinity;
        for (let i = 0; i < existingFrames.length; i++) {
          const t = existingTimes[i];
          if (t instanceof Date && !isNaN(t.getTime()) && t.getTime() > lastHist && Array.isArray(existingFrames[i])) {
            combined.push({ time: t, grid: existingFrames[i], gridMeta: gGrid, sourceFile: "forecast" });
          }
        }
      }
      setDisplayedFrames(combined, "過去TEC + 現在の予報");
    } catch (e) {
      console.error(e);
      setV4Status("過去+予報の読み込み失敗: " + e.message);
    }
  }


  function getTecForecastBaseDays() {
    const el = document.getElementById("tecForecastBaseDays");
    const v = parseFloat(el?.value || String(DEFAULT_TEC_FORECAST_BASE_DAYS));
    return isFinite(v) ? c(v, 1, 30) : DEFAULT_TEC_FORECAST_BASE_DAYS;
  }

  function timeOfDayHours(t) {
    return t.getUTCHours() + t.getUTCMinutes() / 60 + t.getUTCSeconds() / 3600;
  }

  function buildTemporalFourierForecastFrames(historyFrames, forecastHours = TEC_FORECAST_HOURS, stepMinutes = 120) {
    if (!historyFrames.length) throw new Error("予報に使えるTEC履歴がありません。");
    const frames = historyFrames.slice().sort((a, b) => a.time - b.time);
    const meta = frames[0].gridMeta;
    const nLat = meta.nLat, nLon = meta.nLon;
    const H = 3; // 24h周期の第3高調波まで。過学習を避けつつ日変化を残す。
    const lastTime = frames[frames.length - 1].time;
    const stepMs = Math.max(1, stepMinutes) * 60000;
    const nFuture = Math.round(forecastHours * 60 / stepMinutes);
    const targetTimes = [];
    for (let k = 1; k <= nFuture; k++) targetTimes.push(new Date(lastTime.getTime() + k * stepMs));

    // 各時刻の位相だけ先に計算。
    const histPhase = frames.map(f => 2 * Math.PI * timeOfDayHours(f.time) / 24);
    const targPhase = targetTimes.map(t => 2 * Math.PI * timeOfDayHours(t) / 24);

    const outFrames = targetTimes.map(t => ({
      time: t,
      gridMeta: meta,
      grid: Array.from({ length: nLat }, () => Array(nLon).fill(0)),
      sourceFile: "data-driven temporal Fourier 3-day forecast",
      forecast: true,
    }));

    for (let i = 0; i < nLat; i++) {
      for (let j = 0; j < nLon; j++) {
        const y = [];
        const p = [];
        for (let n = 0; n < frames.length; n++) {
          const v = Number(frames[n].grid?.[i]?.[j]);
          if (isFinite(v)) { y.push(Math.max(0, v)); p.push(histPhase[n]); }
        }
        if (!y.length) continue;
        const mean = y.reduce((a, b) => a + b, 0) / y.length;
        const a = new Array(H + 1).fill(0);
        const b = new Array(H + 1).fill(0);
        for (let h = 1; h <= H; h++) {
          let ca = 0, sb = 0;
          for (let n = 0; n < y.length; n++) {
            ca += y[n] * Math.cos(h * p[n]);
            sb += y[n] * Math.sin(h * p[n]);
          }
          a[h] = (2 / y.length) * ca;
          b[h] = (2 / y.length) * sb;
        }
        for (let k = 0; k < targetTimes.length; k++) {
          let pred = mean;
          for (let h = 1; h <= H; h++) pred += a[h] * Math.cos(h * targPhase[k]) + b[h] * Math.sin(h * targPhase[k]);
          // 外挿暴れを抑える。履歴平均から大きく外れた場合は少し丸める。
          pred = Math.max(0, pred);
          outFrames[k].grid[i][j] = pred;
        }
      }
    }
    return outFrames;
  }

  async function loadTecDataDriven3DayForecast() {
    try {
      setV4Status("data/tec履歴から3日TEC予報を作成中…");
      await loadTecArchiveIndex();
      const frames = archiveIndex.frames || [];
      if (!frames.length) throw new Error("data/tec/index.jsonに履歴がありません。先にNOAA TEC取得workflowを回してください。");
      const endVal = document.getElementById("archiveEndSelect")?.value || frames[frames.length - 1].time_utc;
      const endMs = new Date(endVal).getTime();
      const baseDays = getTecForecastBaseDays();
      const startMs = endMs - baseDays * 24 * 3600000;
      const entries = frames.filter(f => {
        const t = new Date(f.time_utc).getTime();
        return isFinite(t) && t >= startMs && t <= endMs;
      });
      if (entries.length < 6) throw new Error("予報に使う履歴が少なすぎます。基準日数を増やすか、data/tecを蓄積してください。");
      setV4Status(`TEC履歴読み込み中… 基準${baseDays}日 / ${entries.length}枚`);
      const history = [];
      for (const e of entries) history.push(await loadArchiveFrame(e));
      history.sort((a, b) => a.time - b.time);
      const forecast = buildTemporalFourierForecastFrames(history, TEC_FORECAST_HOURS, 120);
      // 表示は履歴末尾24h + 3日予報。計算用rawは2時間値、UIで30分/10分へ補間。
      const histTailStart = history[history.length - 1].time.getTime() - 24 * 3600000;
      const histTail = history.filter(f => f.time.getTime() >= histTailStart);
      setDisplayedFrames(histTail.concat(forecast), `data基準3日TEC予報（Fourier日周期 / 基準${baseDays}日）`);
    } catch (e) {
      console.error(e);
      setV4Status("data基準3日TEC予報の作成失敗: " + e.message);
    }
  }

  function dynamicOnSliderChange() {
    const slider = document.getElementById("timeSlider");
    const maxStep = Math.max(0, (gForecastTimes && gForecastTimes.length ? gForecastTimes.length - 1 : N_STEPS));
    const v = parseInt(slider?.value || "0", 10) || 0;
    currentStepIndex = c(v, 0, maxStep);
    const t = gForecastTimes[currentStepIndex] || gForecastStart;
    const hours = gForecastStart && t ? (t.getTime() - gForecastStart.getTime()) / 3600000 : (currentStepIndex * DT_MINUTES) / 60;
    const label = document.getElementById("timeLabel");
    const utc = document.getElementById("utcLabel");
    if (label) label.textContent = gForecastTimes.length
      ? `frame ${currentStepIndex + 1} / ${gForecastTimes.length}  (t=${hours.toFixed(1)}h / ${getTimelineLabel()})`
      : `t = ${hours.toFixed(1)} h`;
    if (utc) utc.textContent = `UTC: ${(t ? isoNoMs(t) : "--")}`;
    if (typeof setOverlayTime === "function") setOverlayTime(t);
    if (typeof updateKpLabels === "function") updateKpLabels();
    updateGnssQuickStatus();
    if (typeof requestDraw === "function") requestDraw();
  }

  function playArchiveMovie() {
    const speedSel = document.getElementById("movieSpeedSelect");
    const speed = Math.max(1, parseFloat(speedSel?.value || "1") || 1);
    const slider = document.getElementById("timeSlider");
    if (!slider) return;
    stopArchiveMovie();
    setV4Status(`自動再生中: ${getTimelineLabel()} / speed x${speed}`);
    window._swiftTecV4Timer = setInterval(() => {
      const max = parseInt(slider.max || "0", 10) || 0;
      let v = parseInt(slider.value || "0", 10) || 0;
      v += 1;
      if (v > max) v = 0;
      slider.value = String(v);
      dynamicOnSliderChange();
    }, 800 / speed);
  }

  function stopArchiveMovie() {
    if (window._swiftTecV4Timer) {
      clearInterval(window._swiftTecV4Timer);
      window._swiftTecV4Timer = null;
    }
  }

  function loadScriptOnce(url) {
    const urls = [
      url,
      "vendor/satellite.min.js",
      "https://unpkg.com/satellite.js/dist/satellite.min.js",
      "https://cdn.jsdelivr.net/npm/satellite.js/dist/satellite.min.js",
      "https://unpkg.com/satellite.js@5.0.0/dist/satellite.min.js",
      "https://cdn.jsdelivr.net/npm/satellite.js@5.0.0/dist/satellite.min.js",
    ].filter((v, i, a) => v && a.indexOf(v) === i);

    return new Promise((resolve, reject) => {
      if (window.satellite) return resolve();

      let idx = 0;
      const errors = [];
      const tryNext = () => {
        if (window.satellite) return resolve();
        if (idx >= urls.length) {
          return reject(new Error("satellite.js の読み込みに失敗: " + errors.join(" / ")));
        }
        const src = urls[idx++];
        const s = document.createElement("script");
        s.src = src;
        s.async = true;
        s.onload = () => {
          if (window.satellite) resolve();
          else {
            errors.push(src + " loaded but window.satellite missing");
            tryNext();
          }
        };
        s.onerror = () => {
          errors.push(src);
          tryNext();
        };
        document.head.appendChild(s);
      };
      tryNext();
    });
  }

  function parseTleText(tle) {
    const lines = String(tle || "").split(/\r?\n/).map(x => x.trim()).filter(Boolean);
    const out = [];
    for (let i = 0; i < lines.length - 2; i++) {
      if (lines[i + 1].startsWith("1 ") && lines[i + 2].startsWith("2 ")) {
        out.push({ name: lines[i], l1: lines[i + 1], l2: lines[i + 2] });
        i += 2;
      } else if (lines[i].startsWith("1 ") && lines[i + 1].startsWith("2 ")) {
        out.push({ name: "SAT-" + lines[i].slice(2, 7).trim(), l1: lines[i], l2: lines[i + 1] });
        i += 1;
      }
    }
    return out;
  }

  function noradFromLine1(l1) {
    const s = String(l1 || "");
    return s.length >= 7 ? s.slice(2, 7).trim() : "";
  }

  function shortSatName(name, constellation, norad) {
    const s = String(name || "").replace(/\s+/g, " ").trim();
    const prn = (s.match(/PRN\s*([A-Z0-9]+)/i) || [])[1];
    if (prn) return `${GNSS_SOURCES[constellation]?.label || constellation} PRN ${prn}`;
    const cospar = (s.match(/\(([A-Z0-9 \-]+)\)/) || [])[1];
    if (cospar) return `${GNSS_SOURCES[constellation]?.label || constellation} ${cospar}`;
    return s || `${GNSS_SOURCES[constellation]?.label || constellation} ${norad}`;
  }

  function selectedConstellationsFromUi() {
    const out = [];
    for (const [key, src] of Object.entries(GNSS_SOURCES)) {
      const cbVisible = document.getElementById(`gnssConstV66_${key}`);
      const cb = cbVisible || document.getElementById(`gnssConst_${key}`);
      if (!cb) {
        if (src.checked) out.push(key);
      } else if (cb.checked) {
        out.push(key);
      }
    }
    return out;
  }

  async function fetchTleIfExists(url) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) return null;
    const text = await res.text();
    return text;
  }

  function markSelectionChanged() {
    selectionVersion++;
    dopFrameCache.clear();
    tecSmoothCache.clear();
    updateGnssQuickStatus();
    if (typeof requestDraw === "function") requestDraw();
  }

  async function loadGnssDopData() {
    try {
      setV4Status("GNSS TLEを読み込み中… local data/gnss → 失敗時CelesTrak live");
      await loadScriptOnce(SATELLITE_JS_URL);

      const selectedConst = selectedConstellationsFromUi();
      if (!selectedConst.length) throw new Error("GNSSコンステレーションが未選択です。GPSなどを選択してください。");

      const previous = new Map(gnssSatList.map(s => [s.id, { selected: s.selected, active: s.active }]));
      const loaded = [];
      const failed = [];

      for (const key of selectedConst) {
        const src = GNSS_SOURCES[key];
        try {
          let tle = await fetchTleIfExists(src.url);
          let tleSource = "local";
          if (!tle && src.liveUrl) {
            tle = await fetchTleIfExists(src.liveUrl);
            tleSource = "CelesTrak";
          }
          if (!tle) {
            failed.push(`${src.label}: not found local/live`);
            continue;
          }
          const records = parseTleText(tle);
          for (const r of records) {
            const norad = noradFromLine1(r.l1);
            const id = `${key}:${norad || r.name}`;
            const prev = previous.get(id);
            loaded.push({
              id,
              constellation: key,
              constellationLabel: src.label,
              norad,
              name: r.name,
              displayName: shortSatName(r.name, key, norad),
              l1: r.l1,
              l2: r.l2,
              satrec: satellite.twoline2satrec(r.l1, r.l2),
              selected: prev ? prev.selected : true,
              active: prev ? prev.active : true,
              lastOk: true,
            });
          }
        } catch (e) {
          failed.push(`${src.label}: ${e.message}`);
        }
      }

      gnssSatList = loaded;
      gnssLoaded = gnssSatList.length > 0;
      selectionVersion++;
      dopFrameCache.clear();
      renderSatelliteSelection();

      const byConst = summarizeByConstellation();
      const summary = Object.entries(byConst).map(([k, n]) => `${GNSS_SOURCES[k]?.label || k}=${n}`).join(" / ");
      const warn = failed.length ? ` / 未取得: ${failed.join(", ")}` : "";
      setV4Status(`GNSS DOP準備OK: ${gnssSatList.length}機 (${summary})${warn}`);
      updateGnssQuickStatus();
      if (typeof requestDraw === "function") requestDraw();
    } catch (e) {
      console.error(e);
      gnssLoaded = false;
      setV4Status("GNSS DOP準備失敗: " + e.message);
    }
  }

  // 後方互換: 既存ボタン名を残す。
  async function loadGpsDopData() {
    return loadGnssDopData();
  }

  function summarizeByConstellation() {
    const out = {};
    for (const s of gnssSatList) out[s.constellation] = (out[s.constellation] || 0) + 1;
    return out;
  }

  function selectedActiveSats() {
    return gnssSatList.filter(s => s.selected && s.active);
  }

  function updateGnssQuickStatus() {
    const el = document.getElementById("gnssQuickStatus");
    if (!el) return;
    const total = gnssSatList.length;
    const active = selectedActiveSats().length;
    const inactive = total - gnssSatList.filter(s => s.active).length;
    const selected = gnssSatList.filter(s => s.selected).length;
    el.textContent = `読込=${total} / 使用=${active} / 選択=${selected} / inactive=${inactive} / mask=${getElevationMaskDeg()}°`;
  }

  function renderSatelliteSelection() {
    const box = document.getElementById("satelliteSelectionList");
    if (!box) return;

    if (!gnssSatList.length) {
      box.innerHTML = `<div class="small">未読込。まず「GNSS TLE読込 / DOP準備」を押してください。</div>`;
      updateGnssQuickStatus();
      return;
    }

    const rows = gnssSatList.map((s, idx) => {
      const statusClass = s.active ? "sat-active" : "sat-inactive";
      return `
        <tr>
          <td><input type="checkbox" ${s.selected ? "checked" : ""} onchange="setSatelliteSelected(${idx}, this.checked)"></td>
          <td class="mono">${s.constellationLabel}</td>
          <td>${escapeHtml(s.displayName)}</td>
          <td class="mono">${escapeHtml(s.norad || "--")}</td>
          <td>
            <select onchange="setSatelliteActive(${idx}, this.value === 'active')" class="${statusClass}">
              <option value="active" ${s.active ? "selected" : ""}>Active</option>
              <option value="inactive" ${!s.active ? "selected" : ""}>Inactive</option>
            </select>
          </td>
        </tr>
      `;
    }).join("");

    box.innerHTML = `
      <div style="max-height:260px; overflow:auto; border:1px solid #222b3f; border-radius:6px;">
        <table>
          <thead>
            <tr><th>使用</th><th>系</th><th>衛星</th><th>NORAD</th><th>状態</th></tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    `;
    updateGnssQuickStatus();
  }

  function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, ch => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;"
    }[ch]));
  }

  function setSatelliteSelected(idx, checked) {
    if (!gnssSatList[idx]) return;
    gnssSatList[idx].selected = !!checked;
    markSelectionChanged();
    renderSatelliteSelection();
  }

  function setSatelliteActive(idx, active) {
    if (!gnssSatList[idx]) return;
    gnssSatList[idx].active = !!active;
    if (!active) gnssSatList[idx].selected = false;
    markSelectionChanged();
    renderSatelliteSelection();
  }

  function setAllSatSelected(selected) {
    for (const s of gnssSatList) {
      if (s.active) s.selected = !!selected;
    }
    markSelectionChanged();
    renderSatelliteSelection();
  }

  function setAllSatActive(active) {
    for (const s of gnssSatList) {
      s.active = !!active;
      s.selected = !!active;
    }
    markSelectionChanged();
    renderSatelliteSelection();
  }

  function getElevationMaskDeg() {
    const el = document.getElementById("dopElevationMaskDeg");
    const v = parseFloat(el?.value || "10");
    return isFinite(v) ? c(v, 0, 45) : 10;
  }

  function geodeticToEcfKm(latRad, lonRad, hKm) {
    const a = 6378.137;
    const e2 = 0.00669437999014;
    const sinLat = Math.sin(latRad);
    const cosLat = Math.cos(latRad);
    const n = a / Math.sqrt(1 - e2 * sinLat * sinLat);
    return {
      x: (n + hKm) * cosLat * Math.cos(lonRad),
      y: (n + hKm) * cosLat * Math.sin(lonRad),
      z: (n * (1 - e2) + hKm) * sinLat,
    };
  }

  function invert4(m) {
    const n = 4;
    const a = m.map((row, i) => row.concat([0, 0, 0, 0].map((_, j) => (i === j ? 1 : 0))));
    for (let col = 0; col < n; col++) {
      let pivot = col;
      for (let r = col + 1; r < n; r++) if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) pivot = r;
      if (Math.abs(a[pivot][col]) < 1e-10) return null;
      if (pivot !== col) [a[pivot], a[col]] = [a[col], a[pivot]];
      const div = a[col][col];
      for (let cc = 0; cc < 2 * n; cc++) a[col][cc] /= div;
      for (let r = 0; r < n; r++) {
        if (r === col) continue;
        const f = a[r][col];
        for (let cc = 0; cc < 2 * n; cc++) a[r][cc] -= f * a[col][cc];
      }
    }
    return a.map(row => row.slice(n));
  }

  function propagatedGnssEcf(time) {
    if (!gnssLoaded || !window.satellite) return [];
    const gmst = satellite.gstime(time);
    const sats = [];
    for (const item of selectedActiveSats()) {
      try {
        const pv = satellite.propagate(item.satrec, time);
        if (!pv || !pv.position) {
          item.lastOk = false;
          continue;
        }
        const ecf = satellite.eciToEcf(pv.position, gmst);
        item.lastOk = isFinite(ecf.x) && isFinite(ecf.y) && isFinite(ecf.z);
        if (item.lastOk) {
          sats.push({
            id: item.id,
            constellation: item.constellation,
            name: item.displayName,
            x: ecf.x, y: ecf.y, z: ecf.z,
          });
        }
      } catch {
        item.lastOk = false;
      }
    }
    return sats;
  }

  function enuLosAndRange(latRad, lonRad, sat, rec) {
    const dx = sat.x - rec.x;
    const dy = sat.y - rec.y;
    const dz = sat.z - rec.z;
    const range = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (!isFinite(range) || range <= 0) return null;

    const sinLat = Math.sin(latRad), cosLat = Math.cos(latRad);
    const sinLon = Math.sin(lonRad), cosLon = Math.cos(lonRad);

    const east = -sinLon * dx + cosLon * dy;
    const north = -sinLat * cosLon * dx - sinLat * sinLon * dy + cosLat * dz;
    const up = cosLat * cosLon * dx + cosLat * sinLon * dy + sinLat * dz;
    const elev = Math.asin(c(up / range, -1, 1));

    return {
      elev,
      ux: dx / range,
      uy: dy / range,
      uz: dz / range,
      east: east / range,
      north: north / range,
      up: up / range,
    };
  }

  function ecefToEnuCov(C, latRad, lonRad) {
    const sinLat = Math.sin(latRad), cosLat = Math.cos(latRad);
    const sinLon = Math.sin(lonRad), cosLon = Math.cos(lonRad);
    const R = [
      [-sinLon,             cosLon,              0],
      [-sinLat * cosLon,   -sinLat * sinLon,    cosLat],
      [ cosLat * cosLon,    cosLat * sinLon,    sinLat],
    ];

    function mulRowCol(r, cidx) {
      let s = 0;
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) s += R[r][a] * C[a][b] * R[cidx][b];
      return s;
    }

    return [
      [mulRowCol(0, 0), mulRowCol(0, 1), mulRowCol(0, 2)],
      [mulRowCol(1, 0), mulRowCol(1, 1), mulRowCol(1, 2)],
      [mulRowCol(2, 0), mulRowCol(2, 1), mulRowCol(2, 2)],
    ];
  }

  function dopAllAt(latDeg, lonDeg, sats, minElevDeg) {
    const latRad = latDeg * Math.PI / 180;
    const lonRad = lonDeg * Math.PI / 180;
    const rec = geodeticToEcfKm(latRad, lonRad, 0);
    const minElevRad = minElevDeg * Math.PI / 180;
    const rows = [];
    let visibleCount = 0;

    for (const sat of sats) {
      const los = enuLosAndRange(latRad, lonRad, sat, rec);
      if (!los) continue;
      if (los.elev < minElevRad) continue;
      visibleCount++;
      // 符号はDOP値に影響しない。clock項を含む4未知数。
      rows.push([-los.ux, -los.uy, -los.uz, 1]);
    }

    if (rows.length < 4) {
      return { count: visibleCount, gdop: NaN, pdop: NaN, hdop: NaN, vdop: NaN, tdop: NaN };
    }

    const n = 4;
    const ata = Array.from({ length: n }, () => Array(n).fill(0));
    for (const r of rows) {
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) ata[i][j] += r[i] * r[j];
    }

    const inv = invert4(ata);
    if (!inv) return { count: visibleCount, gdop: NaN, pdop: NaN, hdop: NaN, vdop: NaN, tdop: NaN };

    const qxx = inv[0][0], qyy = inv[1][1], qzz = inv[2][2], qtt = inv[3][3];
    const Cxyz = [
      [inv[0][0], inv[0][1], inv[0][2]],
      [inv[1][0], inv[1][1], inv[1][2]],
      [inv[2][0], inv[2][1], inv[2][2]],
    ];
    const Cenu = ecefToEnuCov(Cxyz, latRad, lonRad);
    const hdop = Math.sqrt(Math.max(0, Cenu[0][0] + Cenu[1][1]));
    const vdop = Math.sqrt(Math.max(0, Cenu[2][2]));
    const pdop = Math.sqrt(Math.max(0, qxx + qyy + qzz));
    const tdop = Math.sqrt(Math.max(0, qtt));
    const gdop = Math.sqrt(Math.max(0, qxx + qyy + qzz + qtt));

    return { count: visibleCount, gdop, pdop, hdop, vdop, tdop };
  }

  function dopCacheKey(stepIndex) {
    const t = gForecastTimes[stepIndex];
    const ids = selectedActiveSats().map(s => s.id).join(",");
    return `${t ? t.toISOString() : "no-time"}|v${selectionVersion}|mask${getElevationMaskDeg()}|${ids}`;
  }

  function getDopAllFrame(stepIndex) {
    if (!gGrid || !gForecastTimes.length || !gnssLoaded) return null;
    const t = gForecastTimes[stepIndex];
    if (!(t instanceof Date) || isNaN(t.getTime())) return null;
    const key = dopCacheKey(stepIndex);
    if (dopFrameCache.has(key)) return dopFrameCache.get(key);

    const sats = propagatedGnssEcf(t);
    const minElev = getElevationMaskDeg();
    const frame = {
      count: Array.from({ length: gGrid.nLat }, () => Array(gGrid.nLon).fill(0)),
      gdop:  Array.from({ length: gGrid.nLat }, () => Array(gGrid.nLon).fill(NaN)),
      pdop:  Array.from({ length: gGrid.nLat }, () => Array(gGrid.nLon).fill(NaN)),
      hdop:  Array.from({ length: gGrid.nLat }, () => Array(gGrid.nLon).fill(NaN)),
      vdop:  Array.from({ length: gGrid.nLat }, () => Array(gGrid.nLon).fill(NaN)),
      tdop:  Array.from({ length: gGrid.nLat }, () => Array(gGrid.nLon).fill(NaN)),
    };

    for (let i = 0; i < gGrid.nLat; i++) {
      const lat = gGrid.latArr[i];
      for (let j = 0; j < gGrid.nLon; j++) {
        const d = dopAllAt(lat, gGrid.lonArr[j], sats, minElev);
        frame.count[i][j] = d.count;
        frame.gdop[i][j] = d.gdop;
        frame.pdop[i][j] = d.pdop;
        frame.hdop[i][j] = d.hdop;
        frame.vdop[i][j] = d.vdop;
        frame.tdop[i][j] = d.tdop;
      }
    }

    dopFrameCache.set(key, frame);
    // 10分再生で増えすぎないよう簡易上限。
    if (dopFrameCache.size > 10) {
      const firstKey = dopFrameCache.keys().next().value;
      dopFrameCache.delete(firstKey);
    }
    return frame;
  }

  function normalizeDopMode(mode) {
    if (mode === "dop") return "pdop";
    if (mode === "doptec") return "pdoptec";
    return mode;
  }

  function metricForMode(mode) {
    const m = normalizeDopMode(mode);
    if (m === "satcount") return "count";
    if (["gdop", "pdop", "hdop", "vdop", "tdop"].includes(m)) return m;
    if (["gdoptec", "pdoptec", "hdoptec", "vdoptec"].includes(m)) return m.replace("tec", "");
    return null;
  }

  function modeValue(baseTec, i, j, cfg) {
    const m = normalizeDopMode(mapMode);
    if (m === "gps") return baseTec * cfg.kL1;
    if (m === "satcount") {
      const df = getDopAllFrame(currentStepIndex);
      return df ? df.count[i][j] : NaN;
    }
    if (["gdop", "pdop", "hdop", "vdop", "tdop"].includes(m)) {
      const df = getDopAllFrame(currentStepIndex);
      return df ? df[m][i][j] : NaN;
    }
    if (["gdoptec", "pdoptec", "hdoptec", "vdoptec"].includes(m)) {
      const metric = m.replace("tec", "");
      const df = getDopAllFrame(currentStepIndex);
      const dop = df ? df[metric][i][j] : NaN;
      return isFinite(dop) ? dop * baseTec * cfg.kL1 : NaN;
    }
    return baseTec;
  }

  function scaleForMode() {
    const m = normalizeDopMode(mapMode);
    if (m === "gps") return gGpsColorScale;
    if (m === "satcount") return satCountColorScale;
    if (["gdop", "pdop", "hdop", "vdop", "tdop"].includes(m)) return dopColorScale;
    if (["gdoptec", "pdoptec", "hdoptec", "vdoptec"].includes(m)) return dopTecColorScale;
    return gTecColorScale;
  }

  function titleForMode() {
    const m = normalizeDopMode(mapMode);
    if (m === "gps") return "GPS L1 [m]";
    if (m === "satcount") return "Visible GNSS Sats";
    if (m === "gdop") return "GDOP";
    if (m === "pdop") return "PDOP";
    if (m === "hdop") return "HDOP";
    if (m === "vdop") return "VDOP";
    if (m === "tdop") return "TDOP";
    if (m === "gdoptec") return "GDOP×L1 [m]";
    if (m === "pdoptec") return "PDOP×L1 [m]";
    if (m === "hdoptec") return "HDOP×L1 [m]";
    if (m === "vdoptec") return "VDOP×L1 [m]";
    return "TEC [TECU]";
  }

  function parseCssColorToRgb(css) {
    const s = String(css || "").trim();
    if (!s) return [255, 255, 255];
    if (s.startsWith("#")) {
      let h = s.slice(1);
      if (h.length === 3) h = h.split("").map(ch => ch + ch).join("");
      if (h.length === 6) {
        return [
          parseInt(h.slice(0, 2), 16),
          parseInt(h.slice(2, 4), 16),
          parseInt(h.slice(4, 6), 16),
        ];
      }
    }
    const m = s.match(/rgba?\(([^)]+)\)/i);
    if (m) {
      const parts = m[1].split(",").map(v => Number(v.trim()));
      return [parts[0] || 0, parts[1] || 0, parts[2] || 0];
    }
    return [255, 255, 255];
  }

  function colorToRgbaArray(css, alpha = 255) {
    const [r, g, b] = parseCssColorToRgb(css);
    return [r, g, b, alpha];
  }

  function sampleArrayBilinear(arr, iFloat, jFloat) {
    if (!arr) return NaN;
    const i0 = Math.floor(iFloat), j0 = Math.floor(jFloat);
    const i1 = Math.min(gGrid.nLat - 1, i0 + 1), j1 = Math.min(gGrid.nLon - 1, j0 + 1);
    const fi = c(iFloat - i0, 0, 1), fj = c(jFloat - j0, 0, 1);
    const v00 = Number(arr?.[i0]?.[j0]);
    const v10 = Number(arr?.[i1]?.[j0]);
    const v01 = Number(arr?.[i0]?.[j1]);
    const v11 = Number(arr?.[i1]?.[j1]);
    const vals = [v00, v10, v01, v11].filter(v => isFinite(v));
    if (!vals.length) return NaN;
    const fallback = vals.reduce((a, b) => a + b, 0) / vals.length;
    const a00 = isFinite(v00) ? v00 : fallback;
    const a10 = isFinite(v10) ? v10 : fallback;
    const a01 = isFinite(v01) ? v01 : fallback;
    const a11 = isFinite(v11) ? v11 : fallback;
    const top = a00 * (1 - fj) + a01 * fj;
    const bottom = a10 * (1 - fj) + a11 * fj;
    return top * (1 - fi) + bottom * fi;
  }

  function sampleModeValueBilinear(tecFrame, dopFrame, iFloat, jFloat, cfg) {
    const m = normalizeDopMode(mapMode);
    const tec = sampleArrayBilinear(tecFrame, iFloat, jFloat);
    if (m === "gps") return isFinite(tec) ? tec * cfg.kL1 : NaN;
    if (m === "satcount") return sampleArrayBilinear(dopFrame?.count, iFloat, jFloat);
    if (["gdop", "pdop", "hdop", "vdop", "tdop"].includes(m)) {
      return sampleArrayBilinear(dopFrame?.[m], iFloat, jFloat);
    }
    if (["gdoptec", "pdoptec", "hdoptec", "vdoptec"].includes(m)) {
      const metric = m.replace("tec", "");
      const dop = sampleArrayBilinear(dopFrame?.[metric], iFloat, jFloat);
      return (isFinite(dop) && isFinite(tec)) ? dop * tec * cfg.kL1 : NaN;
    }
    return tec;
  }

  function getAdaptiveColorEnabled() {
    const el = document.getElementById("adaptiveColorScaleSelect");
    if (!el) return true;
    return el.value !== "fixed";
  }

  function percentileSorted(sorted, p) {
    if (!sorted.length) return NaN;
    const x = c(p, 0, 1) * (sorted.length - 1);
    const i = Math.floor(x);
    const f = x - i;
    const a = sorted[i];
    const b = sorted[Math.min(sorted.length - 1, i + 1)];
    return a * (1 - f) + b * f;
  }

  function adaptiveScaleForFrame(frame, cfg, baseScale) {
    if (!getAdaptiveColorEnabled()) return baseScale;
    const m = normalizeDopMode(mapMode);
    // DOP単独と衛星数は物理的な固定レンジの方が見やすい。
    if (["satcount", "gdop", "pdop", "hdop", "vdop", "tdop"].includes(m)) return baseScale;

    const dopFrame = isDopLikeMode() ? getDopAllFrame(currentStepIndex) : null;
    const vals = [];
    const nLat = gGrid?.nLat || 0;
    const nLon = gGrid?.nLon || 0;
    const stepI = Math.max(1, Math.floor(nLat / 80));
    const stepJ = Math.max(1, Math.floor(nLon / 120));
    for (let i = 0; i < nLat; i += stepI) {
      for (let j = 0; j < nLon; j += stepJ) {
        let v;
        if (["gdoptec", "pdoptec", "hdoptec", "vdoptec"].includes(m)) {
          const metric = m.replace("tec", "");
          const dop = Number(dopFrame?.[metric]?.[i]?.[j]);
          const tec = Number(frame?.[i]?.[j]);
          v = (isFinite(dop) && isFinite(tec)) ? dop * tec * cfg.kL1 : NaN;
        } else {
          v = modeValue(Number(frame?.[i]?.[j]), i, j, cfg);
        }
        if (isFinite(v)) vals.push(v);
      }
    }
    if (vals.length < 8) return baseScale;
    vals.sort((a, b) => a - b);
    let p05 = percentileSorted(vals, 0.05);
    let p45 = percentileSorted(vals, 0.45);
    let p75 = percentileSorted(vals, 0.75);
    let p97 = percentileSorted(vals, 0.97);

    const med = percentileSorted(vals, 0.50);
    let range = p97 - p05;
    const minRange = m === "tec" ? Math.max(4, med * 0.35) : Math.max(1, Math.abs(med) * 0.35);
    if (!isFinite(range) || range < minRange) {
      p05 = Math.max(0, med - minRange / 2);
      p97 = med + minRange / 2;
      p45 = p05 + (p97 - p05) * 0.42;
      p75 = p05 + (p97 - p05) * 0.72;
    }

    const colors = baseScale.map(s => s.color);
    return [
      { limit: +p05.toFixed(2), color: colors[0] || "#0000ff" },
      { limit: +p45.toFixed(2), color: colors[1] || "#ffffff" },
      { limit: +p75.toFixed(2), color: colors[2] || "#ffff00" },
      { limit: +p97.toFixed(2), color: colors[3] || "#ff0000" },
    ];
  }

  function drawSmoothHeatmap(frame, cfg, scale) {
    const w = tecCanvas.width, h = tecCanvas.height;
    if (w <= 0 || h <= 0) return;

    // 高解像度でも重くなり過ぎないよう、内部解像度を少し落として描画し、canvasで拡大する。
    const qualityEl = document.getElementById("heatmapQualitySelect");
    const q = qualityEl ? parseFloat(qualityEl.value || "0.65") : 0.65;
    const renderScale = c(q || 0.65, 0.3, 1.0);
    const rw = Math.max(240, Math.round(w * renderScale));
    const rh = Math.max(160, Math.round(h * renderScale));

    const off = document.createElement("canvas");
    off.width = rw;
    off.height = rh;
    const ctx = off.getContext("2d");
    const img = ctx.createImageData(rw, rh);
    const data = img.data;
    const alphaByte = Math.round(c(tecAlpha, 0, 1) * 255);

    const latMax = gGrid.latArr[gGrid.nLat - 1];
    const latMin = gGrid.latArr[0];
    const lonMin = gGrid.lonArr[0];
    const lonMax = gGrid.lonArr[gGrid.nLon - 1];

    const latSpan = Math.max(1e-6, latMax - latMin);
    const lonSpan = Math.max(1e-6, lonMax - lonMin);

    const dopFrame = isDopLikeMode() ? getDopAllFrame(currentStepIndex) : null;

    const latByY = new Array(rh);
    for (let y = 0; y < rh; y++) {
      const py = (y + 0.5) / renderScale;
      const ll = map.containerPointToLatLng([w / 2, py]);
      latByY[y] = c(ll.lat, latMin, latMax);
    }

    const lonByX = new Array(rw);
    for (let x = 0; x < rw; x++) {
      const px = (x + 0.5) / renderScale;
      const ll = map.containerPointToLatLng([px, h / 2]);
      let lon = ll.lng;
      while (lon < lonMin) lon += 360;
      while (lon > lonMax) lon -= 360;
      lonByX[x] = c(lon, lonMin, lonMax);
    }

    for (let y = 0; y < rh; y++) {
      const lat = latByY[y];
      const iFloat = ((lat - latMin) / latSpan) * (gGrid.nLat - 1);
      for (let x = 0; x < rw; x++) {
        const lon = lonByX[x];
        const jFloat = ((lon - lonMin) / lonSpan) * (gGrid.nLon - 1);
        const val = sampleModeValueBilinear(frame, dopFrame, iFloat, jFloat, cfg);
        if (!isFinite(val)) continue;
        const rgba = colorToRgbaArray(valueToColor(val, scale), alphaByte);
        const k = (y * rw + x) * 4;
        data[k] = rgba[0];
        data[k + 1] = rgba[1];
        data[k + 2] = rgba[2];
        data[k + 3] = rgba[3];
      }
    }

    ctx.putImageData(img, 0, 0);
    tecCtx.imageSmoothingEnabled = true;
    tecCtx.clearRect(0, 0, w, h);
    tecCtx.drawImage(off, 0, 0, w, h);
  }

  function installOverrides() {
    if (typeof onSliderChange === "function") onSliderChange = dynamicOnSliderChange;
    if (typeof changeMapMode === "function" && !originalChangeMapMode) originalChangeMapMode = changeMapMode;
    changeMapMode = function () {
      mapMode = document.getElementById("mapModeSelect")?.value || "tec";
      applyTimelineForCurrentMode(false);
      if (typeof updateLegend === "function") updateLegend();
      if (typeof requestDraw === "function") requestDraw();
      setV4Status(`表示モード変更: ${titleForMode()} / ${getTimelineLabel()}`);
    };

    if (typeof drawTecOverlay === "function" && !originalDrawTecOverlay) originalDrawTecOverlay = drawTecOverlay;
    drawTecOverlay = function () {
      if (!map || !tecCanvas || !tecCtx) return;
      if (!gGrid || !gForecastTimes.length) return;

      const w = tecCanvas.width, h = tecCanvas.height;
      tecCtx.clearRect(0, 0, w, h);

      const frame = currentTecGrid();
      if (!frame) return;

      const cfg = getConfigFromUI();
      const scale = scaleForMode();
      tecCtx.imageSmoothingEnabled = false;
      tecCtx.globalAlpha = tecAlpha;

      // v4.5 classic renderer: 旧v3と同じく、格子値をそのまま矩形で塗る。
      // フーリエ平滑化・画面ピクセル補間・自動再スケールはしない。
      for (let i = 0; i < gGrid.nLat - 1; i++) {
        for (let j = 0; j < gGrid.nLon - 1; j++) {
          const lat0 = gGrid.latArr[i];
          const lon0 = gGrid.lonArr[j];
          const lat1 = gGrid.latArr[i + 1];
          const lon1 = gGrid.lonArr[j + 1];

          const p0 = map.latLngToContainerPoint([lat0, lon0]);
          const p1 = map.latLngToContainerPoint([lat1, lon1]);
          const x = Math.min(p0.x, p1.x);
          const y = Math.min(p0.y, p1.y);
          const rw = Math.abs(p1.x - p0.x);
          const rh = Math.abs(p1.y - p0.y);

          const tec = Number(frame?.[i]?.[j]);
          const val = modeValue(isFinite(tec) ? tec : NaN, i, j, cfg);
          if (!isFinite(val)) continue;

          tecCtx.fillStyle = valueToColor(val, scale);
          tecCtx.fillRect(x, y, rw + 1, rh + 1);
        }
      }

      tecCtx.globalAlpha = 1.0;
    };

    if (typeof updateLegend === "function" && !originalUpdateLegend) originalUpdateLegend = updateLegend;
    updateLegend = function () {
      if (!map) return;
      if (tecLegendControl) {
        tecLegendControl.remove();
        tecLegendControl = null;
      }
      const scale = scaleForMode();
      const title = titleForMode();
      tecLegendControl = L.control({ position: "bottomright" });
      tecLegendControl.onAdd = function () {
        const div = L.DomUtil.create("div", "leaflet-control tec-legend");
        const minLabel = scale[0]?.limit ?? 0;
        const maxLabel = scale[scale.length - 1]?.limit ?? "--";
        div.innerHTML = `
          <div class="tec-legend-title">${title}</div>
          <canvas id="legendBar" width="18" height="130" style="display:block;margin:0 auto 4px auto;border:1px solid #444;border-radius:4px;"></canvas>
          <div class="tec-legend-labels"><span>${minLabel}</span><span>${maxLabel}</span></div>
        `;
        setTimeout(() => {
          const c0 = document.getElementById("legendBar");
          if (!c0) return;
          const ctx = c0.getContext("2d");
          const minV = Number(minLabel) || 0;
          const maxV = Number(maxLabel) || 1;
          for (let y = 0; y < c0.height; y++) {
            const v = maxV - (maxV - minV) * (y / Math.max(1, c0.height - 1));
            ctx.fillStyle = valueToColor(v, scale);
            ctx.fillRect(0, y, c0.width, 1);
          }
        }, 0);
        return div;
      };
      tecLegendControl.addTo(map);
    };

    if (typeof sampleAtLatLon === "function" && !originalSampleAtLatLon) originalSampleAtLatLon = sampleAtLatLon;
    sampleAtLatLon = function (lat, lon) {
      if (!gGrid || !gForecastTimes.length) return "未計算です。";
      let bestI = 0, bestJ = 0, bestD = 1e99;
      for (let i = 0; i < gGrid.nLat; i++) {
        const dLat = Math.abs(gGrid.latArr[i] - lat);
        if (dLat > bestD) continue;
        for (let j = 0; j < gGrid.nLon; j++) {
          let dLon = Math.abs(gGrid.lonArr[j] - lon);
          dLon = Math.min(dLon, 360 - dLon);
          const d = dLat + dLon * 0.4;
          if (d < bestD) { bestD = d; bestI = i; bestJ = j; }
        }
      }
      const frame = currentTecGrid();
      const tec = frame?.[bestI]?.[bestJ];
      const cfg = getConfigFromUI();
      const l1 = (isFinite(tec) ? tec : 0) * cfg.kL1;
      const df = getDopAllFrame(currentStepIndex);
      const t = gForecastTimes[currentStepIndex];
      const vals = {
        count: df ? df.count[bestI][bestJ] : NaN,
        gdop: df ? df.gdop[bestI][bestJ] : NaN,
        pdop: df ? df.pdop[bestI][bestJ] : NaN,
        hdop: df ? df.hdop[bestI][bestJ] : NaN,
        vdop: df ? df.vdop[bestI][bestJ] : NaN,
        tdop: df ? df.tdop[bestI][bestJ] : NaN,
      };
      function f(v, digits = 2) { return isFinite(v) ? Number(v).toFixed(digits) : "--"; }
      return [
        `Clicked: lat=${lat.toFixed(3)}, lon=${lon.toFixed(3)}`,
        `Nearest Grid: lat=${gGrid.latArr[bestI].toFixed(2)}, lon=${gGrid.lonArr[bestJ].toFixed(2)}`,
        `Time: ${(t ? isoNoMs(t) : "--")}`,
        `TEC: ${(isFinite(tec) ? tec.toFixed(2) : "NaN")} TECU`,
        `L1 iono error: ${l1.toFixed(2)} m`,
        `Visible GNSS sats: ${isFinite(vals.count) ? vals.count : "--"}`,
        `GDOP: ${f(vals.gdop)} / PDOP: ${f(vals.pdop)} / HDOP: ${f(vals.hdop)} / VDOP: ${f(vals.vdop)} / TDOP: ${f(vals.tdop)}`,
        `GDOP×L1: ${f(vals.gdop * l1)} m`,
        `PDOP×L1: ${f(vals.pdop * l1)} m`,
        `HDOP×L1: ${f(vals.hdop * l1)} m`,
        `VDOP×L1: ${f(vals.vdop * l1)} m`,
        `GNSS selected-active: ${selectedActiveSats().length} / loaded: ${gnssSatList.length}`,
      ].join("\n");
    };
  }

  function addModeOptions() {
    const sel = document.getElementById("mapModeSelect");
    if (!sel) return;
    const opts = [
      ["satcount", "可視GNSS衛星数"],
      ["gdop", "GDOP"],
      ["pdop", "PDOP"],
      ["hdop", "HDOP"],
      ["vdop", "VDOP"],
      ["tdop", "TDOP"],
      ["gdoptec", "GDOP × TEC 測位誤差 [m]"],
      ["pdoptec", "PDOP × TEC 測位誤差 [m]"],
      ["hdoptec", "HDOP × TEC 水平誤差 [m]"],
      ["vdoptec", "VDOP × TEC 垂直誤差 [m]"],
      // 後方互換
      ["dop", "GPS PDOP（旧）"],
      ["doptec", "DOP × TEC（旧）"],
    ];
    for (const [value, label] of opts) {
      if ([...sel.options].some(o => o.value === value)) continue;
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = label;
      sel.appendChild(opt);
    }
  }

  function ensureGnssUi() {
    // 既存HTMLに8bカードが無い場合の最低限フォールバック。
    if (document.getElementById("satelliteSelectionList")) {
      setupConstellationDefaults();
      return;
    }
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <div class="card-header"><h2>8b. GNSS / DOP設定</h2><span>multi-GNSS / satellite select</span></div>
      <div class="small">GPS / Galileo / GLONASS / BeiDou / QZSSを選択してDOPを計算します。</div>
      <div class="row small" id="gnssConstellationBox"></div>
      <div class="row small">
        <div>仰角マスク[deg]:<br><input id="dopElevationMaskDeg" type="number" value="10" min="0" max="45" step="1" style="width:80px;" onchange="markGnssSelectionChanged()"></div>
        <div>描画品質:<br>
          <select id="heatmapQualitySelect" onchange="requestDraw()">
            <option value="0.45">軽量</option>
            <option value="0.65" selected>標準</option>
            <option value="1.0">高精細</option>
          </select>
        </div>
      </div>
      <div class="row">
        <button onclick="loadGnssDopData()">GNSS TLE読込 / DOP準備</button>
        <button class="secondary" onclick="setAllSatSelected(true)">全衛星使用</button>
        <button class="secondary" onclick="setAllSatSelected(false)">全衛星解除</button>
        <button class="secondary" onclick="setAllSatActive(true)">全Active</button>
        <button class="secondary" onclick="setAllSatActive(false)">全Inactive</button>
      </div>
      <div class="small" id="gnssQuickStatus">GNSS未読込</div>
      <div id="satelliteSelectionList"></div>
    `;
    const sidebar = document.querySelector(".sidebar");
    const target = document.getElementById("v4ArchiveStatus")?.closest(".card");
    if (sidebar && target) target.insertAdjacentElement("afterend", card);
    else if (sidebar) sidebar.appendChild(card);
    setupConstellationDefaults();
  }

  function setupConstellationDefaults() {
    const box = document.getElementById("gnssConstellationBox");
    if (box && !box.dataset.ready) {
      box.innerHTML = Object.entries(GNSS_SOURCES).map(([key, src]) => `
        <label class="small" style="white-space:nowrap;">
          <input id="gnssConst_${key}" type="checkbox" ${src.checked ? "checked" : ""}>
          ${src.label}
        </label>
      `).join(" ");
      box.dataset.ready = "1";
    }
    const q = document.getElementById("heatmapQualitySelect");
    if (q && !q.value) q.value = "0.65";
    renderSatelliteSelection();
  }



  /* =========================================================
   * Forecast TEC API source selector
   * - NOAA direct API: SWPC geojson_2d_urt latest 24h / 30min slots
   * - Archive data API: docs/data/tec/index.json + json.gz latest/selected 24h
   * The original SWIFT-TEC forecast model is kept. Only the input TEC source is switched.
   * ========================================================= */
  const NOAA_GLOTEC_INDEX_DIRECT_URL = "https://services.swpc.noaa.gov/products/glotec/geojson_2d_urt.json";
  const NOAA_GLOTEC_BASE_DIRECT_URL = "https://services.swpc.noaa.gov/products/glotec/geojson_2d_urt/";
  const FORECAST_TEC_SOURCE_STORAGE_KEY = "swifttec_forecast_tec_api_source_v1";
  const FORECAST_TEC_SLOT_MIN = 30;
  const FORECAST_TEC_WINDOW_HOURS = 24;
  const FORECAST_TEC_MAX_DIFF_MIN = 24;

  function parseUtcFromGloTecFilenameV48(name) {
    const m = String(name || "").match(/(\d{8})T(\d{6})Z/i);
    if (!m) return null;
    const y = Number(m[1].slice(0, 4));
    const mo = Number(m[1].slice(4, 6));
    const d = Number(m[1].slice(6, 8));
    const hh = Number(m[2].slice(0, 2));
    const mm = Number(m[2].slice(2, 4));
    const ss = Number(m[2].slice(4, 6));
    const t = new Date(Date.UTC(y, mo - 1, d, hh, mm, ss));
    return isNaN(t.getTime()) ? null : t;
  }

  function basenameV48(path) {
    return String(path || "").replace(/\/$/, "").split("/").pop() || "";
  }

  function asNoaaGloTecUrlV48(path) {
    const p = String(path || "");
    if (p.startsWith("http://") || p.startsWith("https://")) return p;
    if (p.includes("/")) return "https://services.swpc.noaa.gov/" + p.replace(/^\/+/, "");
    return NOAA_GLOTEC_BASE_DIRECT_URL + p;
  }

  function normalizeNoaaGloTecIndexV48(obj) {
    let items = [];
    if (Array.isArray(obj)) items = obj;
    else if (obj && typeof obj === "object") {
      for (const key of ["files", "data", "items"]) {
        if (Array.isArray(obj[key])) { items = obj[key]; break; }
      }
      if (!items.length) {
        for (const v of Object.values(obj)) {
          if (Array.isArray(v)) { items = v; break; }
        }
      }
    }
    return items.map(x => {
      if (typeof x === "string") return x;
      if (x && typeof x === "object") return x.url || x.href || x.path || x.name || x.file || x.filename || "";
      return String(x || "");
    }).filter(x => x.toLowerCase().includes(".geojson"));
  }

  function floorUtcToSlotV48(t, stepMin = FORECAST_TEC_SLOT_MIN) {
    const d = new Date(Date.UTC(t.getUTCFullYear(), t.getUTCMonth(), t.getUTCDate(), t.getUTCHours(), t.getUTCMinutes(), 0));
    const total = d.getUTCHours() * 60 + d.getUTCMinutes();
    const floored = Math.floor(total / stepMin) * stepMin;
    return new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate(), 0, floored, 0));
  }

  function buildForecastTecTargetSlotsV48(endTime, hours = FORECAST_TEC_WINDOW_HOURS, stepMin = FORECAST_TEC_SLOT_MIN) {
    const end = floorUtcToSlotV48(endTime, stepMin);
    const n = Math.round((hours * 60) / stepMin);
    const start = new Date(end.getTime() - (n - 1) * stepMin * 60000);
    const out = [];
    for (let k = 0; k < n; k++) out.push(new Date(start.getTime() + k * stepMin * 60000));
    return out;
  }

  function nearestEntryForSlotV48(entries, slot, maxDiffMin = FORECAST_TEC_MAX_DIFF_MIN) {
    let best = null;
    let bestDiff = Infinity;
    for (const e of entries) {
      const diff = Math.abs(e.time.getTime() - slot.getTime());
      if (diff < bestDiff) { bestDiff = diff; best = e; }
    }
    if (!best || bestDiff > maxDiffMin * 60000) return null;
    return best;
  }

  function setForecastTecApiStatus(msg) {
    const el = document.getElementById("forecastTecApiStatus");
    if (el) el.textContent = msg || "";
    if (msg && typeof logInfo === "function") logInfo(msg);
  }

  function getForecastTecApiMode() {
    return document.getElementById("forecastTecApiSourceSelect")?.value || "noaa_direct_30m";
  }

  function setForecastTecApiMode(mode) {
    const sel = document.getElementById("forecastTecApiSourceSelect");
    if (sel) sel.value = mode;
    localStorage.setItem(FORECAST_TEC_SOURCE_STORAGE_KEY, mode);
  }

  function populateForecastTecArchiveEndSelect() {
    const sel = document.getElementById("forecastTecArchiveEndSelect");
    if (!sel || !archiveIndex || !Array.isArray(archiveIndex.frames)) return;
    const old = sel.value;
    sel.innerHTML = "";
    const frames = archiveIndex.frames.slice().sort((a, b) => String(a.time_utc).localeCompare(String(b.time_utc)));
    for (const f of frames) {
      const opt = document.createElement("option");
      opt.value = f.time_utc;
      opt.textContent = String(f.time_utc).replace(".000Z", "Z");
      sel.appendChild(opt);
    }
    if (old && [...sel.options].some(o => o.value === old)) sel.value = old;
    else if (frames.length) sel.value = frames[frames.length - 1].time_utc;
  }

  function ensureForecastTecApiUi() {
    if (document.getElementById("forecastTecApiCard")) return;
    const card = document.createElement("div");
    card.className = "card";
    card.id = "forecastTecApiCard";
    card.innerHTML = `
      <div class="card-header"><h2>8c. 予報用TEC API入力</h2><span>NOAA直取得 / data蓄積</span></div>
      <div class="small">
        ・TEC予報計算に使う入力TECを、<b>NOAA API直取得</b> または <b>docs/data/tecの取りため済みAPI</b> から選べます。<br>
        ・どちらも <b>過去24時間・30分間隔</b> を取得し、初期版SWIFT-TEC方式のBase抽出＋Kp/Flare加算モデルに渡します。
      </div>
      <div class="row small" style="margin-top:6px;">
        <div style="flex:1.4;">
          予報用TECソース:<br>
          <select id="forecastTecApiSourceSelect" style="width:100%;" onchange="setForecastTecApiMode(this.value)">
            <option value="noaa_direct_30m">NOAA APIから直取得（最新24h / 30分）</option>
            <option value="archive_data_30m">取りため済みdata API（選択時刻まで24h / 30分）</option>
          </select>
        </div>
        <div style="flex:1;">
          data API終了UTC:<br>
          <select id="forecastTecArchiveEndSelect" style="width:100%;"></select>
        </div>
      </div>
      <div class="row" style="margin-top:4px;">
        <button onclick="loadForecastTecFromSelectedApi(false)">予報用TECをAPI取得</button>
        <button class="secondary" onclick="loadForecastTecFromSelectedApi(true)">再取得して予報計算</button>
      </div>
      <div class="row small" style="margin-top:4px;">
        <label><input type="checkbox" id="forecastTecAutoFetch" checked> TEC予報計算前に、このソースから自動取得</label>
        <span class="pill" id="forecastTecReadyPill">TEC入力: 未取得</span>
      </div>
      <div class="small" id="forecastTecApiStatus">予報用TEC API未取得</div>
    `;
    const sidebar = document.querySelector(".sidebar");
    const target = document.getElementById("v4ArchiveStatus")?.closest(".card");
    if (sidebar && target) target.insertAdjacentElement("afterend", card);
    else if (sidebar) sidebar.appendChild(card);

    const saved = localStorage.getItem(FORECAST_TEC_SOURCE_STORAGE_KEY);
    if (saved) setForecastTecApiMode(saved);
    populateForecastTecArchiveEndSelect();
  }

  async function loadForecastTecFromNoaaDirectApi30m() {
    setForecastTecApiStatus("NOAA APIから予報用TEC 30分値を取得中…");
    const idx = await fetch(NOAA_GLOTEC_INDEX_DIRECT_URL, { cache: "no-store" });
    if (!idx.ok) throw new Error(`NOAA index HTTP ${idx.status}`);
    const obj = await idx.json();
    const paths = normalizeNoaaGloTecIndexV48(obj);
    if (!paths.length) throw new Error("NOAA GloTEC indexからgeojsonが見つかりません。");

    const entries = paths.map(p => {
      const fn = basenameV48(p);
      const time = parseUtcFromGloTecFilenameV48(fn) || parseUtcFromGloTecFilenameV48(p);
      return time ? { fn, time, url: asNoaaGloTecUrlV48(p) } : null;
    }).filter(Boolean).sort((a, b) => a.time - b.time);
    if (!entries.length) throw new Error("NOAA GeoJSONファイル名からUTCを抽出できません。");

    const latest = entries[entries.length - 1].time;
    const targets = buildForecastTecTargetSlotsV48(latest, FORECAST_TEC_WINDOW_HOURS, FORECAST_TEC_SLOT_MIN);
    const picks = targets.map(slot => nearestEntryForSlotV48(entries, slot, FORECAST_TEC_MAX_DIFF_MIN));
    const miss = picks.filter(x => !x).length;
    if (miss) throw new Error(`NOAA APIで30分枠が不足しています: ${miss}/${targets.length}枠`);

    const frames = [];
    for (let k = 0; k < picks.length; k++) {
      const p = picks[k];
      setForecastTecApiStatus(`NOAA API取得中 ${k + 1}/${picks.length}: ${p.fn}`);
      const res = await fetch(p.url, { cache: "no-store" });
      if (!res.ok) throw new Error(`NOAA GeoJSON HTTP ${res.status}: ${p.fn}`);
      const txt = await res.text();
      const f = parseNoaaGloTecGeoJson(txt, targets[k], 2.0, 5.0);
      frames.push({ frame: f, time: targets[k], file: p.fn });
    }
    installForecastTecFramesV48(frames, "NOAA API直取得 30分値");
  }

  async function loadForecastTecFromArchiveDataApi30m() {
    setForecastTecApiStatus("data/tec APIから予報用TEC 30分値を取得中…");
    await loadTecArchiveIndex(true);
    populateForecastTecArchiveEndSelect();
    const frames = archiveIndex?.frames || [];
    if (!frames.length) throw new Error("data/tec/index.jsonに履歴がありません。Fetch NOAA TEC archiveを先に実行してください。");

    const endSel = document.getElementById("forecastTecArchiveEndSelect");
    const endTime = endSel?.value ? new Date(endSel.value) : new Date(frames[frames.length - 1].time_utc);
    if (isNaN(endTime.getTime())) throw new Error("data API終了UTCが不正です。");
    const targets = buildForecastTecTargetSlotsV48(endTime, FORECAST_TEC_WINDOW_HOURS, FORECAST_TEC_SLOT_MIN);

    const entries = frames.map(f => ({
      time: new Date(f.time_utc),
      file: f.file,
      raw: f,
    })).filter(e => !isNaN(e.time.getTime())).sort((a, b) => a.time - b.time);

    const picks = targets.map(slot => nearestEntryForSlotV48(entries, slot, 16));
    const miss = picks.filter(x => !x).length;
    if (miss) throw new Error(`data/tec APIで30分枠が不足しています: ${miss}/${targets.length}枠。archive workflowを回してください。`);

    const loaded = [];
    for (let k = 0; k < picks.length; k++) {
      const p = picks[k];
      setForecastTecApiStatus(`data API取得中 ${k + 1}/${picks.length}: ${p.file}`);
      const fr = await loadArchiveFrame(p.raw);
      loaded.push({
        frame: {
          latArr: fr.gridMeta.latArr,
          lonArr: fr.gridMeta.lonArr,
          nLat: fr.gridMeta.nLat,
          nLon: fr.gridMeta.nLon,
          grid: fr.grid,
          validTime: targets[k],
        },
        time: targets[k],
        file: p.file,
      });
    }
    installForecastTecFramesV48(loaded, "data/tec取りためAPI 30分値");
  }

  function installForecastTecFramesV48(items, label) {
    if (!items.length) throw new Error("予報用TECフレームが空です。");
    gNoaaDayFrames = items.map(x => x.frame);
    gNoaaDayTimes = items.map(x => x.time);
    gNoaaDayFiles = items.map(x => x.file || "");
    const first = gNoaaDayTimes[0];
    gNoaaDayKey = first instanceof Date && !isNaN(first.getTime()) ? first.toISOString().slice(0, 10) : null;
    forecastTecApiReady = true;
    forecastTecApiSourceLabel = label;

    const tecSel = document.getElementById("tecSourceSelect");
    if (tecSel) tecSel.value = "noaa";
    const pill = document.getElementById("forecastTecReadyPill");
    if (pill) pill.textContent = `TEC入力: ${label} / ${items.length}枚`;

    if (typeof renderNoaa12Table === "function") renderNoaa12Table();
    if (typeof fillForecastStartCandidates === "function") fillForecastStartCandidates();
    setForecastTecApiStatus(`${label}: ${items.length}枚取得OK / ${isoNoMs(gNoaaDayTimes[0])} 〜 ${isoNoMs(gNoaaDayTimes[gNoaaDayTimes.length - 1])}`);
  }

  async function loadForecastTecFromSelectedApi(runAfter = false) {
    const mode = getForecastTecApiMode();
    setForecastTecApiMode(mode);
    if (mode === "archive_data_30m") await loadForecastTecFromArchiveDataApi30m();
    else await loadForecastTecFromNoaaDirectApi30m();
    if (runAfter && typeof runForecast === "function") runForecast();
  }

  function buildGenericNoaaTecTodFromFramesV48(frames, times, stepMinutes = 30) {
    if (!frames || frames.length < 2 || !times || times.length < 2) return null;
    const good = frames.map((frame, i) => ({ frame, time: times[i] }))
      .filter(x => x.time instanceof Date && !isNaN(x.time.getTime()) && x.frame && Array.isArray(x.frame.grid))
      .sort((a, b) => a.time - b.time);
    if (good.length < 2) return null;

    const first = good[0].time;
    const dayStart = new Date(Date.UTC(first.getUTCFullYear(), first.getUTCMonth(), first.getUTCDate(), 0, 0, 0));
    const n = Math.round(24 * 60 / stepMinutes);
    const out = [];

    function todMinutes(t) { return t.getUTCHours() * 60 + t.getUTCMinutes() + t.getUTCSeconds() / 60; }
    const source = good.map(x => ({ ...x, mins: todMinutes(x.time) })).sort((a, b) => a.mins - b.mins);

    function bracket(mins) {
      let lo = source[0], hi = source[source.length - 1];
      for (let i = 0; i < source.length - 1; i++) {
        if (source[i].mins <= mins && mins <= source[i + 1].mins) return [source[i], source[i + 1], false];
      }
      // wrap around midnight
      if (mins < source[0].mins) return [source[source.length - 1], source[0], true];
      return [source[source.length - 1], source[0], true];
    }

    for (let k = 0; k < n; k++) {
      const mins = k * stepMinutes;
      const [lo, hi, wrap] = bracket(mins);
      let loM = lo.mins;
      let hiM = hi.mins;
      let m = mins;
      if (wrap) {
        if (hiM < loM) hiM += 1440;
        if (m < loM) m += 1440;
      }
      const span = Math.max(1, hiM - loM);
      const f = c((m - loM) / span, 0, 1);
      const A = lo.frame;
      const B = hi.frame;
      const nLat = A.nLat, nLon = A.nLon;
      const grid = Array.from({ length: nLat }, () => Array(nLon).fill(0));
      for (let i = 0; i < nLat; i++) {
        for (let j = 0; j < nLon; j++) {
          const v0 = Number(A.grid?.[i]?.[j]);
          const v1 = Number(B.grid?.[i]?.[j]);
          const a = isFinite(v0) ? v0 : (isFinite(v1) ? v1 : 0);
          const b = isFinite(v1) ? v1 : a;
          grid[i][j] = a + (b - a) * f;
        }
      }
      out.push({ latArr: A.latArr, lonArr: A.lonArr, nLat, nLon, grid });
    }

    return { stepMinutes, frames: out, dayKey: toDayKeyUtc(dayStart), gridMeta: good[0].frame, sourceCount: good.length };
  }

  function installForecastTecApiOverrides() {
    if (typeof readTecInputs === "function" && !originalReadTecInputsForForecastApi) {
      originalReadTecInputsForForecastApi = readTecInputs;
      readTecInputs = function () {
        const src = document.getElementById("tecSourceSelect")?.value || "noaa";
        if (src === "noaa") {
          if (!gNoaaDayFrames || gNoaaDayFrames.length < 2) {
            throw new Error("NOAA入力が未取得です。8cの『予報用TECをAPI取得』を押すか、自動取得をONにしてください。");
          }
          return { source: "noaa", frames: gNoaaDayFrames, times: gNoaaDayTimes || [], gridMeta: gNoaaDayFrames[0] };
        }
        return originalReadTecInputsForForecastApi();
      };
    }

    if (typeof buildNoaaTecTodFrom12Frames === "function" && !originalBuildNoaaTecTodForForecastApi) {
      originalBuildNoaaTecTodForForecastApi = buildNoaaTecTodFrom12Frames;
      buildNoaaTecTodFrom12Frames = function (frames, times, stepMinutes = 30) {
        return buildGenericNoaaTecTodFromFramesV48(frames, times, stepMinutes);
      };
    }

    if (typeof fillForecastStartCandidates === "function" && !originalFillForecastStartCandidatesForForecastApi) {
      originalFillForecastStartCandidatesForForecastApi = fillForecastStartCandidates;
      fillForecastStartCandidates = function () {
        return originalFillForecastStartCandidatesForForecastApi();
      };
    }

    if (typeof runForecast === "function" && !originalRunForecastForForecastApi) {
      originalRunForecastForForecastApi = runForecast;
      runForecast = async function () {
        try {
          const auto = !!document.getElementById("forecastTecAutoFetch")?.checked;
          const src = document.getElementById("tecSourceSelect")?.value || "noaa";
          if (src === "noaa" && auto) {
            await loadForecastTecFromSelectedApi(false);
          }
          return originalRunForecastForForecastApi();
        } catch (e) {
          console.error(e);
          setForecastTecApiStatus("予報用TEC API取得/予報計算失敗: " + e.message);
          if (typeof logInfo === "function") logInfo("予報計算失敗: " + e.message);
        }
      };
    }
  }

  function bootAddon() {
    installOverrides();
    installForecastTecApiOverrides();
    addModeOptions();
    ensureGnssUi();
    ensureForecastTecApiUi();
    ensureKpAiUi();
    loadKpAiData(false);
    loadTecArchiveIndex(false).then(() => {
      populateForecastTecArchiveEndSelect();
    }).catch(() => {
      const info = document.getElementById("archiveIndexInfo");
      if (info) info.textContent = "TEC履歴index未作成。GitHub Actionsを実行すると表示されます。";
      populateForecastTecArchiveEndSelect();
    });
  }


  function nearestGridCellForPointSeries(lat, lon) {
    if (!gGrid || !gGrid.latArr || !gGrid.lonArr) return null;
    let bestI = 0, bestJ = 0, bestD = Infinity;
    for (let i = 0; i < gGrid.nLat; i++) {
      const dLat = Math.abs(Number(gGrid.latArr[i]) - lat);
      if (dLat > bestD) continue;
      for (let j = 0; j < gGrid.nLon; j++) {
        let dLon = Math.abs(Number(gGrid.lonArr[j]) - lon);
        dLon = Math.min(dLon, 360 - dLon);
        const d = dLat + dLon * 0.4;
        if (d < bestD) { bestD = d; bestI = i; bestJ = j; }
      }
    }
    return { i: bestI, j: bestJ, lat: Number(gGrid.latArr[bestI]), lon: Number(gGrid.lonArr[bestJ]) };
  }

  function pointTecAtTimeForSeries(t, i, j) {
    const raw = interpolateGridAtTime(t);
    if (!raw) return NaN;
    const grid = applyKpAiCorrectionToGrid(raw, t);
    const v = Number(grid?.[i]?.[j]);
    return isFinite(v) ? v : NaN;
  }

  function pointDopAtTimeForSeries(lat, lon, t) {
    if (!gnssLoaded || !gnssSatList.length) return null;
    const sats = propagatedGnssEcf(t);
    return dopAllAt(lat, lon, sats, getElevationMaskDeg());
  }

  function buildPointDopSeries(lat, lon, opts = {}) {
    const stepMin = Math.max(1, Number(opts.stepMin || 5));
    const metric = String(opts.metric || "pdop");
    if (!gGrid || !gForecastTimes.length) throw new Error("先にTEC/予報ヒートマップを作成してください。");
    if (!gnssLoaded || !gnssSatList.length) throw new Error("先にGNSS TLE読込 / DOP準備を実行してください。");

    const cell = nearestGridCellForPointSeries(Number(lat), Number(lon));
    if (!cell) throw new Error("選択地点に対応する格子を取得できません。");

    const start = gForecastTimes[0];
    const end = gForecastTimes[gForecastTimes.length - 1];
    if (!(start instanceof Date) || isNaN(start.getTime()) || !(end instanceof Date) || isNaN(end.getTime())) {
      throw new Error("予報時刻列がありません。");
    }

    const cfg = (typeof getConfigFromUI === "function") ? getConfigFromUI() : { kL1: 0.162 };
    const kL1 = Number(cfg.kL1 || 0.162);
    const rows = [];
    const maxN = Math.min(2000, Math.floor((end.getTime() - start.getTime()) / (stepMin * 60000)) + 1);

    for (let n = 0; n < maxN; n++) {
      const t = new Date(start.getTime() + n * stepMin * 60000);
      if (t > end) break;
      const d = pointDopAtTimeForSeries(cell.lat, cell.lon, t) || {};
      const tec = pointTecAtTimeForSeries(t, cell.i, cell.j);
      const l1 = isFinite(tec) ? tec * kL1 : NaN;

      const row = {
        time: t.toISOString().replace(".000Z", "Z"),
        time_ms: t.getTime(),
        lat: cell.lat,
        lon: cell.lon,
        selected_lat: Number(lat),
        selected_lon: Number(lon),
        tec,
        l1,
        count: Number(d.count),
        gdop: Number(d.gdop),
        pdop: Number(d.pdop),
        hdop: Number(d.hdop),
        vdop: Number(d.vdop),
        tdop: Number(d.tdop),
      };
      row.gdoptec = isFinite(row.gdop) && isFinite(l1) ? row.gdop * l1 : NaN;
      row.pdoptec = isFinite(row.pdop) && isFinite(l1) ? row.pdop * l1 : NaN;
      row.hdoptec = isFinite(row.hdop) && isFinite(l1) ? row.hdop * l1 : NaN;
      row.vdoptec = isFinite(row.vdop) && isFinite(l1) ? row.vdop * l1 : NaN;
      row.tdoptec = isFinite(row.tdop) && isFinite(l1) ? row.tdop * l1 : NaN;
      row.value = Number(row[metric]);
      rows.push(row);
    }

    return {
      metric,
      step_min: stepMin,
      start_utc: start.toISOString().replace(".000Z", "Z"),
      end_utc: end.toISOString().replace(".000Z", "Z"),
      cell,
      rows,
      gnss_total: gnssSatList.length,
      gnss_active_selected: selectedActiveSats().length,
      elevation_mask_deg: getElevationMaskDeg(),
    };
  }

  function applyGnssPrnHealthMap(healthMap) {
    if (!healthMap || typeof healthMap !== "object") return { applied: 0, inactive: 0 };
    let applied = 0;
    let inactive = 0;
    for (const s of gnssSatList) {
      if (s.constellation !== "gps") continue;
      const prnRaw = (String(s.name || "").match(/PRN\s*([0-9]+)/i) || String(s.displayName || "").match(/PRN\s*([0-9]+)/i) || [])[1];
      if (!prnRaw) continue;
      const prn = String(Number(prnRaw)).padStart(2, "0");
      if (!(prn in healthMap)) continue;
      const ok = Number(healthMap[prn]) === 0;
      s.active = ok;
      s.health = Number(healthMap[prn]);
      applied++;
      if (!ok) inactive++;
    }
    selectionVersion++;
    dopFrameCache.clear();
    renderSatelliteSelection();
    updateGnssQuickStatus();
    if (typeof requestDraw === "function") requestDraw();
    return { applied, inactive };
  }



  async function debugGnssLoadTest() {
    const selected = selectedConstellationsFromUi();
    const out = {
      selected,
      satellite_js_loaded_before: !!window.satellite,
      satellite_js_loaded_after: false,
      sources: [],
      current_loaded_count: gnssSatList.length,
    };
    try {
      await loadScriptOnce(SATELLITE_JS_URL);
      out.satellite_js_loaded_after = !!window.satellite;
    } catch (e) {
      out.satellite_js_error = e.message;
    }

    for (const key of selected) {
      const src = GNSS_SOURCES[key];
      const row = { key, label: src?.label || key, local: null, live: null, parsed_records: 0 };
      for (const mode of ["local", "live"]) {
        const url = mode === "local" ? src?.url : src?.liveUrl;
        if (!url) continue;
        try {
          const res = await fetch(url, { cache: "no-store" });
          const text = res.ok ? await res.text() : "";
          const records = res.ok ? parseTleText(text) : [];
          row[mode] = {
            url,
            ok: res.ok,
            status: res.status,
            bytes: text.length,
            records: records.length,
            first_line: text.split(/\r?\n/).find(Boolean) || "",
          };
          if (!row.parsed_records && records.length) row.parsed_records = records.length;
        } catch (e) {
          row[mode] = { url, ok: false, error: e.message };
        }
      }
      out.sources.push(row);
    }
    out.current_loaded_count = gnssSatList.length;
    return out;
  }


  window.loadTecArchiveIndex = loadTecArchiveIndex;
  window.loadTecArchiveRange = loadTecArchiveRange;
  window.loadTecArchivePlusCurrentForecast = loadTecArchivePlusCurrentForecast;
  window.loadTecDataDriven3DayForecast = loadTecDataDriven3DayForecast;
  window.loadForecastTecFromSelectedApi = loadForecastTecFromSelectedApi;
  window.loadForecastTecFromNoaaDirectApi30m = loadForecastTecFromNoaaDirectApi30m;
  window.loadForecastTecFromArchiveDataApi30m = loadForecastTecFromArchiveDataApi30m;
  window.setForecastTecApiMode = setForecastTecApiMode;
  window.loadKpAiData = loadKpAiData;
  window.renderKpAiPanel = renderKpAiPanel;
  window.playArchiveMovie = playArchiveMovie;
  window.stopArchiveMovie = stopArchiveMovie;

  window.loadGpsDopData = loadGpsDopData;
  window.loadGnssDopData = loadGnssDopData;
  window.setSatelliteSelected = setSatelliteSelected;
  window.setSatelliteActive = setSatelliteActive;
  window.setAllSatSelected = setAllSatSelected;
  window.setAllSatActive = setAllSatActive;
  window.renderSatelliteSelection = renderSatelliteSelection;
  window.markGnssSelectionChanged = markSelectionChanged;
  window.swiftDebugGnssLoadTest = debugGnssLoadTest;
  window.swiftBuildPointDopSeries = buildPointDopSeries;
  window.swiftApplyGnssPrnHealthMap = applyGnssPrnHealthMap;

  document.addEventListener("DOMContentLoaded", bootAddon);
})();



/* =========================================================
 * SWIFT-TEC v5.1 Clean Dashboard UI
 * Keeps the existing calculation engine, but replaces the busy sidebar
 * with a compact forecast + AI learning dashboard.
 * ========================================================= */
(function () {
  const AI_BASE = "data/ai/";
  const REGION_LABELS = (() => {
    const latBands = ["南緯帯", "赤道帯", "北緯帯"];
    const lonBands = ["180W-120W", "120W-60W", "60W-0", "0-60E", "60E-120E", "120E-180E"];
    const out = [];
    let id = 1;
    for (const lat of latBands) {
      for (const lon of lonBands) out.push({ id: `R${String(id++).padStart(2, "0")}`, label: `${lat} / ${lon}` });
    }
    return out;
  })();

  let cleanAiPerf = null;
  let cleanAiCoeff = null;
  let cleanAiHistory = null;
  let cleanTecIndex = null;

  function ready(fn) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn);
    else setTimeout(fn, 0);
  }

  function q(id) { return document.getElementById(id); }
  function pct(v) {
    const n = Number(v);
    return Number.isFinite(n) ? `${(n * 100).toFixed(1)}%` : "--";
  }
  function num(v, d = 2) {
    const n = Number(v);
    return Number.isFinite(n) ? n.toFixed(d) : "--";
  }
  function safeArray(x) { return Array.isArray(x) ? x : []; }

  async function fetchJsonSafe(url) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`${url} HTTP ${res.status}`);
    return await res.json();
  }

  function injectCleanStyle() {
    if (q("swiftCleanDashboardStyle")) return;
    const style = document.createElement("style");
    style.id = "swiftCleanDashboardStyle";
    style.textContent = `
      body.swift-clean-ui .sidebar { width: 460px; min-width: 430px; max-width: 520px; background: linear-gradient(180deg, #08111f 0%, #050816 55%, #03040b 100%); }
      body.swift-clean-ui .sidebar > .card.swift-legacy-hidden { display: none !important; }
      body.swift-clean-ui .main { gap: 10px; }
      body.swift-clean-ui .map-card { min-height: 66vh; }
      body.swift-clean-ui .slider-card { background: rgba(7, 12, 24, 0.96); border-color: #1f355a; }
      body.swift-clean-ui .output-card { display: none; }
      .swift-clean-card { background: rgba(7, 14, 28, 0.98); border: 1px solid #1f355a; border-radius: 14px; padding: 12px; margin-bottom: 10px; box-shadow: 0 12px 30px rgba(0,0,0,.28); }
      .swift-clean-hero { display:flex; justify-content:space-between; gap:10px; align-items:flex-start; }
      .swift-clean-title { font-size: 18px; font-weight: 800; letter-spacing:.04em; }
      .swift-clean-sub { font-size: 10px; color:#a9b8d2; line-height:1.4; margin-top:3px; }
      .swift-clean-pill { display:inline-flex; align-items:center; gap:5px; border:1px solid #2a4774; border-radius:999px; padding:2px 8px; font-size:10px; background:#08152a; color:#d8e7ff; white-space:nowrap; }
      .swift-dot { width:7px; height:7px; border-radius:999px; background:#64748b; display:inline-block; }
      .swift-dot.ok { background:#22c55e; box-shadow:0 0 8px rgba(34,197,94,.55); }
      .swift-dot.warn { background:#f59e0b; box-shadow:0 0 8px rgba(245,158,11,.55); }
      .swift-clean-stats { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:10px; }
      .swift-stat { border:1px solid #1f355a; border-radius:12px; background:#061020; padding:8px; min-height:58px; }
      .swift-stat-label { font-size:10px; color:#9fb0cc; }
      .swift-stat-value { margin-top:3px; font-size:18px; font-weight:800; color:#f8fafc; }
      .swift-stat-note { margin-top:2px; font-size:9px; color:#8090ad; }
      .swift-clean-tabs { display:grid; grid-template-columns:repeat(5, 1fr); gap:5px; margin:10px 0 8px; }
      .swift-clean-tab { border:1px solid #243a60; background:#08152a; color:#c7d6f0; padding:6px 3px; border-radius:10px; font-size:10px; cursor:pointer; }
      .swift-clean-tab.active { background:#1d4ed8; border-color:#60a5fa; color:white; font-weight:700; }
      .swift-clean-panel { display:none; }
      .swift-clean-panel.active { display:block; }
      .swift-clean-row { display:flex; gap:7px; align-items:center; flex-wrap:wrap; margin:6px 0; }
      .swift-clean-row > div { flex:1; min-width:120px; }
      .swift-clean-btn { border-radius:10px; border:1px solid #3b82f6; background:#1d4ed8; color:white; padding:7px 10px; font-size:11px; cursor:pointer; font-weight:650; }
      .swift-clean-btn.secondary { border-color:#334155; background:#0f172a; color:#d7e5ff; }
      .swift-clean-btn.ghost { border-color:#1f355a; background:transparent; color:#bcd0ee; }
      .swift-clean-btn.warn { border-color:#b45309; background:#92400e; }
      .swift-clean-select, .swift-clean-input { width:100%; background:#061020; border:1px solid #2a4774; color:#eaf2ff; border-radius:9px; padding:6px 8px; font-size:11px; }
      .swift-mini-label { font-size:10px; color:#9fb0cc; margin-bottom:3px; }
      .swift-clean-chart { width:100%; height:150px; border:1px solid #1f355a; background:#030712; border-radius:12px; display:block; }
      .swift-region-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:6px; max-height:265px; overflow:auto; padding-right:2px; }
      .swift-region-tile { border:1px solid #1f355a; border-radius:10px; background:#061020; padding:7px; cursor:pointer; }
      .swift-region-tile.active { border-color:#60a5fa; background:#082451; }
      .swift-region-id { font-size:11px; font-weight:800; }
      .swift-region-label { font-size:9px; color:#9fb0cc; margin-top:2px; min-height:22px; }
      .swift-region-hit { font-size:14px; font-weight:800; margin-top:4px; }
      .swift-clean-table { width:100%; border-collapse:collapse; font-size:10px; }
      .swift-clean-table th, .swift-clean-table td { border:1px solid #1f355a; padding:4px 5px; }
      .swift-clean-table th { background:#0b1730; color:#bcd0ee; }
      .swift-clean-status { font-size:10px; color:#a9b8d2; min-height:16px; margin-top:4px; }
      .swift-advanced-open .sidebar > .card.swift-legacy-hidden { display:block !important; opacity:.72; }
      .swift-advanced-open #swiftCleanDashboard { position:sticky; top:0; z-index:30; }
    `;
    document.head.appendChild(style);
  }

  function setHiddenLegacy(hide) {
    const dash = q("swiftCleanDashboard");
    document.querySelectorAll(".sidebar > .card").forEach(card => {
      if (card === dash) return;
      card.classList.toggle("swift-legacy-hidden", hide);
    });
  }

  function setStatus(msg) {
    const el = q("cleanDashboardStatus");
    if (el) el.textContent = msg || "";
    if (typeof window.logInfo === "function") window.logInfo(msg || "");
  }

  function syncForecastSourceToLegacy() {
    const src = q("cleanForecastTecSource")?.value || "archive_data_30m";
    const legacy = q("forecastTecApiSourceSelect");
    if (legacy) legacy.value = src;
    if (typeof window.setForecastTecApiMode === "function") window.setForecastTecApiMode(src);
  }

  function syncAiControlsToLegacy() {
    const enabled = !!q("cleanAiEnabled")?.checked;
    const clip = Number(q("cleanAiClip")?.value || 20);
    const e = q("kpAiCorrectionEnabled");
    const c = q("kpAiCorrectionClip");
    if (e) e.checked = enabled;
    if (c && Number.isFinite(clip)) c.value = String(clip);
    try { window.markGnssSelectionChanged?.(); } catch {}
    try { window.requestDraw?.(); } catch {}
  }

  async function cleanLoadForecastTec() {
    syncForecastSourceToLegacy();
    setStatus("予報用TECを読み込み中…");
    await window.loadForecastTecFromSelectedApi?.(false);
    setStatus("予報用TECを読み込みました。続けて『予報実行』できます。");
  }

  async function cleanRunForecast() {
    syncForecastSourceToLegacy();
    syncAiControlsToLegacy();
    setStatus("TEC予報を計算中…");
    const auto = q("forecastTecAutoFetch");
    if (auto) auto.checked = true;
    await window.runForecast?.();
    setStatus("予報を実行しました。地図とスライダーで確認してください。");
  }

  async function cleanLoadAiData() {
    setStatus("AI学習結果を読み込み中…");
    const results = await Promise.allSettled([
      fetchJsonSafe(AI_BASE + "kp_performance.json"),
      fetchJsonSafe(AI_BASE + "kp_coefficients.json"),
      fetchJsonSafe(AI_BASE + "kp_learning_history.json"),
      fetchJsonSafe("data/tec/index.json"),
    ]);
    cleanAiPerf = results[0].status === "fulfilled" ? results[0].value : null;
    cleanAiCoeff = results[1].status === "fulfilled" ? results[1].value : null;
    cleanAiHistory = results[2].status === "fulfilled" ? results[2].value : null;
    cleanTecIndex = results[3].status === "fulfilled" ? results[3].value : null;
    try { await window.loadKpAiData?.(false); } catch {}
    updateCleanStats();
    renderCleanLearning();
    setStatus(cleanAiPerf ? "AI学習結果を読み込みました。" : "AI学習結果はまだありません。Actionsで Train Kp AI Corrector を実行してください。");
  }

  function monthValue() {
    const v = Number(q("cleanAiMonth")?.value || (new Date()).getUTCMonth() + 1);
    return Number.isFinite(v) ? String(Math.max(1, Math.min(12, Math.round(v)))) : String((new Date()).getUTCMonth() + 1);
  }

  function regionRowsForMonth(month) {
    const metrics = cleanAiPerf?.metrics || {};
    return REGION_LABELS.map(r => {
      const m = metrics[r.id]?.[String(month)] || {};
      const coeff = cleanAiCoeff?.coefficients?.[r.id]?.[String(month)] || {};
      return { ...r, metrics: m, coeff };
    });
  }

  function summary() { return cleanAiPerf?.summary || {}; }

  function updateCleanStats() {
    const s = summary();
    const rawHit = Number(s.raw_hit_rate);
    const corrHit = Number(s.corrected_hit_rate);
    const rawRmse = Number(s.raw_rmse);
    const corrRmse = Number(s.corrected_rmse);
    const updated = cleanAiPerf?.updated_utc || cleanAiCoeff?.updated_utc || "--";
    const frames = safeArray(cleanTecIndex?.frames);

    const elHit = q("cleanStatHit");
    if (elHit) elHit.textContent = `${pct(rawHit)} → ${pct(corrHit)}`;
    const elRmse = q("cleanStatRmse");
    if (elRmse) elRmse.textContent = `${num(rawRmse)} → ${num(corrRmse)}`;
    const elData = q("cleanStatData");
    if (elData) elData.textContent = frames.length ? `${frames.length} frames` : "--";
    const elGroups = q("cleanStatGroups");
    if (elGroups) elGroups.textContent = `${s.updated_groups ?? 0} groups`;
    const elAi = q("cleanAiUpdated");
    if (elAi) elAi.textContent = String(updated).replace("T", " ").replace("Z", "Z");
    const dot = q("cleanAiDot");
    if (dot) dot.className = "swift-dot " + (cleanAiPerf ? "ok" : "warn");
    const dataNote = q("cleanStatDataNote");
    if (dataNote && frames.length) dataNote.textContent = `${frames[0].time_utc || "--"} 〜 ${frames[frames.length - 1].time_utc || "--"}`;
  }

  function canvasClear(ctx, w, h) {
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#030712";
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = "rgba(148,163,184,.18)";
    ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
      const y = (h - 28) * i / 4 + 10;
      ctx.beginPath(); ctx.moveTo(34, y); ctx.lineTo(w - 10, y); ctx.stroke();
    }
  }

  function drawLineChart(canvas, rows) {
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(320, Math.floor(rect.width * dpr));
    const h = Math.max(130, Math.floor(rect.height * dpr));
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext("2d");
    canvasClear(ctx, w, h);
    ctx.font = `${11 * dpr}px system-ui`;
    ctx.fillStyle = "#94a3b8";
    ctx.fillText("Hit rate trend", 12 * dpr, 16 * dpr);
    if (!rows.length) { ctx.fillText("No learning history yet", 12 * dpr, 45 * dpr); return; }
    const plot = { x0: 38 * dpr, y0: 22 * dpr, x1: w - 12 * dpr, y1: h - 22 * dpr };
    const maxY = 1;
    const minY = 0;
    const xAt = i => rows.length === 1 ? (plot.x0 + plot.x1) / 2 : plot.x0 + (plot.x1 - plot.x0) * i / (rows.length - 1);
    const yAt = v => plot.y1 - (plot.y1 - plot.y0) * ((v - minY) / (maxY - minY));
    function line(key, color) {
      ctx.strokeStyle = color; ctx.lineWidth = 2 * dpr; ctx.beginPath();
      rows.forEach((r, i) => { const x = xAt(i), y = yAt(Number(r[key]) || 0); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
      ctx.stroke();
      ctx.fillStyle = color;
      rows.forEach((r, i) => { const x = xAt(i), y = yAt(Number(r[key]) || 0); ctx.beginPath(); ctx.arc(x, y, 2.3 * dpr, 0, Math.PI * 2); ctx.fill(); });
    }
    line("raw_hit_rate", "#64748b");
    line("corrected_hit_rate", "#60a5fa");
    ctx.fillStyle = "#64748b"; ctx.fillText("raw", plot.x0, h - 6 * dpr);
    ctx.fillStyle = "#60a5fa"; ctx.fillText("AI corrected", plot.x0 + 44 * dpr, h - 6 * dpr);
    ctx.fillStyle = "#94a3b8"; ctx.fillText("100%", 4 * dpr, plot.y0 + 3 * dpr); ctx.fillText("0%", 12 * dpr, plot.y1);
  }

  function drawRegionBars(canvas, rows) {
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(320, Math.floor(rect.width * dpr));
    const h = Math.max(150, Math.floor(rect.height * dpr));
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext("2d");
    canvasClear(ctx, w, h);
    ctx.font = `${10 * dpr}px system-ui`;
    ctx.fillStyle = "#94a3b8";
    ctx.fillText("18 regions hit rate", 12 * dpr, 16 * dpr);
    const plot = { x0: 44 * dpr, y0: 25 * dpr, x1: w - 10 * dpr, y1: h - 12 * dpr };
    const n = rows.length || 18;
    const gap = 2 * dpr;
    const barH = Math.max(3 * dpr, (plot.y1 - plot.y0 - gap * (n - 1)) / n);
    rows.forEach((r, i) => {
      const y = plot.y0 + i * (barH + gap);
      const raw = Number(r.metrics.raw_hit_rate || 0);
      const corr = Number(r.metrics.corrected_hit_rate || 0);
      ctx.fillStyle = "#94a3b8"; ctx.fillText(r.id, 10 * dpr, y + barH * .75);
      ctx.fillStyle = "rgba(100,116,139,.45)"; ctx.fillRect(plot.x0, y, (plot.x1 - plot.x0) * raw, barH);
      ctx.fillStyle = "#3b82f6"; ctx.fillRect(plot.x0, y + barH * .45, (plot.x1 - plot.x0) * corr, barH * .55);
    });
  }

  function renderRegionGrid(rows) {
    const box = q("cleanRegionGrid");
    if (!box) return;
    box.innerHTML = rows.map(r => {
      const hit = Number(r.metrics.corrected_hit_rate || 0);
      const raw = Number(r.metrics.raw_hit_rate || 0);
      const rmse = Number(r.metrics.corrected_rmse);
      const good = hit >= 0.75 ? "ok" : (hit >= 0.55 ? "warn" : "");
      return `<div class="swift-region-tile" data-region="${r.id}" onclick="window.swiftCleanSelectRegion('${r.id}')">
        <div style="display:flex;justify-content:space-between;align-items:center;"><span class="swift-region-id">${r.id}</span><span class="swift-dot ${good}"></span></div>
        <div class="swift-region-label">${r.label}</div>
        <div class="swift-region-hit">${pct(hit)}</div>
        <div class="swift-stat-note">raw ${pct(raw)} / RMSE ${num(rmse)}</div>
      </div>`;
    }).join("");
  }

  function renderRegionDetail(regionId, month) {
    const box = q("cleanRegionDetail");
    if (!box) return;
    const row = regionRowsForMonth(month).find(r => r.id === regionId) || regionRowsForMonth(month)[0];
    if (!row) { box.innerHTML = ""; return; }
    document.querySelectorAll(".swift-region-tile").forEach(x => x.classList.toggle("active", x.dataset.region === row.id));
    const m = row.metrics || {}, k = row.coeff || {};
    box.innerHTML = `<table class="swift-clean-table">
      <tbody>
        <tr><th>地域</th><td>${row.id} / ${row.label}</td></tr>
        <tr><th>Hit</th><td>${pct(m.raw_hit_rate)} → <b>${pct(m.corrected_hit_rate)}</b></td></tr>
        <tr><th>RMSE</th><td>${num(m.raw_rmse)} → <b>${num(m.corrected_rmse)}</b></td></tr>
        <tr><th>Bias</th><td>${num(m.raw_bias)} → <b>${num(m.corrected_bias)}</b></td></tr>
        <tr><th>N</th><td>${m.sample_count || 0}</td></tr>
        <tr><th>k0/k1/k2/k3</th><td class="mono">${num(k.k0,3)} / ${num(k.k1,3)} / ${num(k.k2,3)} / ${num(k.k3,3)}</td></tr>
      </tbody>
    </table>`;
  }

  function renderCleanLearning() {
    const month = monthValue();
    const rows = regionRowsForMonth(month);
    const histRows = safeArray(cleanAiHistory?.runs).slice(-30);
    drawLineChart(q("cleanTrendChart"), histRows);
    drawRegionBars(q("cleanRegionBarChart"), rows);
    renderRegionGrid(rows);
    renderRegionDetail(window.swiftCleanSelectedRegion || "R01", month);
    const updated = q("cleanLearningUpdated");
    if (updated) updated.textContent = cleanAiPerf?.updated_utc ? `updated: ${cleanAiPerf.updated_utc}` : "AI学習データなし";
  }

  window.swiftCleanSelectRegion = function (rid) {
    window.swiftCleanSelectedRegion = rid;
    renderRegionDetail(rid, monthValue());
  };

  function setTab(name) {
    document.querySelectorAll(".swift-clean-tab").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
    document.querySelectorAll(".swift-clean-panel").forEach(p => p.classList.toggle("active", p.dataset.panel === name));
  }

  function installCleanDashboard() {
    if (q("swiftCleanDashboard")) return;
    injectCleanStyle();
    document.body.classList.add("swift-clean-ui");
    setHiddenLegacy(true);

    const sidebar = document.querySelector(".sidebar");
    if (!sidebar) return;
    const dash = document.createElement("div");
    dash.className = "swift-clean-card";
    dash.id = "swiftCleanDashboard";
    dash.innerHTML = `
      <div class="swift-clean-hero">
        <div>
          <div class="swift-clean-title">SWIFT-TEC AI Forecast</div>
          <div class="swift-clean-sub">NOAA 30分TEC / GNSS DOP / Kp Cubic AI Corrector</div>
        </div>
        <span class="swift-clean-pill"><span id="cleanAiDot" class="swift-dot warn"></span><span id="cleanAiUpdated">AI未読込</span></span>
      </div>
      <div class="swift-clean-stats">
        <div class="swift-stat"><div class="swift-stat-label">的中率</div><div id="cleanStatHit" class="swift-stat-value">--</div><div class="swift-stat-note">raw → AI補正後</div></div>
        <div class="swift-stat"><div class="swift-stat-label">RMSE</div><div id="cleanStatRmse" class="swift-stat-value">--</div><div class="swift-stat-note">raw → AI補正後</div></div>
        <div class="swift-stat"><div class="swift-stat-label">TEC data</div><div id="cleanStatData" class="swift-stat-value">--</div><div id="cleanStatDataNote" class="swift-stat-note">NOAA archive</div></div>
        <div class="swift-stat"><div class="swift-stat-label">学習更新</div><div id="cleanStatGroups" class="swift-stat-value">--</div><div class="swift-stat-note">18地域×月別</div></div>
      </div>
      <div class="swift-clean-tabs">
        <button class="swift-clean-tab active" data-tab="forecast">予報</button>
        <button class="swift-clean-tab" data-tab="learning">的中率</button>
        <button class="swift-clean-tab" data-tab="region">18地域</button>
        <button class="swift-clean-tab" data-tab="gnss">GNSS</button>
        <button class="swift-clean-tab" data-tab="advanced">設定</button>
      </div>
      <div class="swift-clean-panel active" data-panel="forecast">
        <div class="swift-clean-row">
          <div><div class="swift-mini-label">予報TECソース</div><select id="cleanForecastTecSource" class="swift-clean-select"><option value="archive_data_30m">取りため済み data/tec</option><option value="noaa_direct_30m">NOAA API直取得</option></select></div>
          <div><div class="swift-mini-label">AI補正</div><label class="swift-clean-pill" style="width:100%;justify-content:center;"><input id="cleanAiEnabled" type="checkbox"> Kp AI ON</label></div>
        </div>
        <div class="swift-clean-row">
          <div><div class="swift-mini-label">補正上限[TECU]</div><input id="cleanAiClip" class="swift-clean-input" type="number" value="20" min="1" max="60" step="1"></div>
          <div><div class="swift-mini-label">表示モード</div><select id="cleanMapMode" class="swift-clean-select"><option value="tec">TEC</option><option value="gps">GPS L1誤差</option><option value="pdoptec">PDOP×TEC</option><option value="hdoptec">HDOP×TEC</option><option value="satcount">可視衛星数</option></select></div>
        </div>
        <div class="swift-clean-row">
          <button class="swift-clean-btn" id="cleanRunForecastBtn">予報実行</button>
          <button class="swift-clean-btn secondary" id="cleanLoadForecastTecBtn">TEC入力取得</button>
          <button class="swift-clean-btn secondary" onclick="loadTecArchiveRange()">過去TEC表示</button>
        </div>
        <div class="swift-clean-row">
          <button class="swift-clean-btn ghost" onclick="loadTecDataDriven3DayForecast()">data基準3日予報</button>
          <button class="swift-clean-btn ghost" onclick="loadTecArchivePlusCurrentForecast()">過去+予報接続</button>
          <button class="swift-clean-btn ghost" onclick="playArchiveMovie()">▶ 再生</button>
          <button class="swift-clean-btn ghost" onclick="stopArchiveMovie()">停止</button>
        </div>
      </div>
      <div class="swift-clean-panel" data-panel="learning">
        <div class="swift-clean-row">
          <button class="swift-clean-btn" id="cleanReloadAiBtn">学習結果読込</button>
          <div><div class="swift-mini-label">対象月</div><select id="cleanAiMonth" class="swift-clean-select">${Array.from({length:12},(_,i)=>`<option value="${i+1}" ${(i+1)===(new Date()).getUTCMonth()+1?"selected":""}>${i+1}月</option>`).join("")}</select></div>
        </div>
        <div class="swift-clean-status" id="cleanLearningUpdated">AI学習データ未読込</div>
        <canvas id="cleanTrendChart" class="swift-clean-chart"></canvas>
        <div style="height:8px"></div>
        <canvas id="cleanRegionBarChart" class="swift-clean-chart"></canvas>
      </div>
      <div class="swift-clean-panel" data-panel="region">
        <div class="swift-clean-row"><button class="swift-clean-btn secondary" onclick="window.swiftCleanRefreshLearning()">更新</button><span class="swift-clean-status">地域をクリックすると係数と精度を表示</span></div>
        <div id="cleanRegionGrid" class="swift-region-grid"></div>
        <div style="height:8px"></div>
        <div id="cleanRegionDetail"></div>
      </div>
      <div class="swift-clean-panel" data-panel="gnss">
        <div class="swift-clean-row">
          <button class="swift-clean-btn" onclick="loadGnssDopData()">GNSS TLE読込</button>
          <button class="swift-clean-btn secondary" onclick="setAllSatSelected(true)">全衛星使用</button>
          <button class="swift-clean-btn secondary" onclick="setAllSatSelected(false)">全解除</button>
        </div>
        <div class="swift-clean-row">
          <button class="swift-clean-btn ghost" onclick="setAllSatActive(true)">全Active</button>
          <button class="swift-clean-btn ghost" onclick="setAllSatActive(false)">全Inactive</button>
        </div>
        <div class="swift-clean-status" id="cleanGnssNote">詳細な衛星ON/OFFは設定タブで旧UIを表示して調整できます。</div>
      </div>
      <div class="swift-clean-panel" data-panel="advanced">
        <div class="swift-clean-row">
          <button class="swift-clean-btn secondary" id="cleanToggleLegacyBtn">旧UI/詳細設定を表示</button>
          <button class="swift-clean-btn ghost" onclick="loadTecArchiveIndex(true)">index再読込</button>
        </div>
        <div class="swift-clean-status">詳細係数、手動Kp、色設定などは旧UIを開いて調整。</div>
      </div>
      <div id="cleanDashboardStatus" class="swift-clean-status">Ready</div>
    `;
    sidebar.insertBefore(dash, sidebar.firstChild);

    dash.querySelectorAll(".swift-clean-tab").forEach(btn => btn.addEventListener("click", () => setTab(btn.dataset.tab)));
    q("cleanLoadForecastTecBtn")?.addEventListener("click", () => cleanLoadForecastTec().catch(e => setStatus("TEC入力取得失敗: " + e.message)));
    q("cleanRunForecastBtn")?.addEventListener("click", () => cleanRunForecast().catch(e => setStatus("予報実行失敗: " + e.message)));
    q("cleanReloadAiBtn")?.addEventListener("click", () => cleanLoadAiData().catch(e => setStatus("AI読込失敗: " + e.message)));
    q("cleanAiMonth")?.addEventListener("change", renderCleanLearning);
    q("cleanAiEnabled")?.addEventListener("change", syncAiControlsToLegacy);
    q("cleanAiClip")?.addEventListener("change", syncAiControlsToLegacy);
    q("cleanMapMode")?.addEventListener("change", () => {
      const sel = q("mapModeSelect");
      if (sel) { sel.value = q("cleanMapMode").value; window.changeMapMode?.(); }
    });
    q("cleanToggleLegacyBtn")?.addEventListener("click", () => {
      const open = !document.body.classList.contains("swift-advanced-open");
      document.body.classList.toggle("swift-advanced-open", open);
      setHiddenLegacy(!open);
      q("cleanToggleLegacyBtn").textContent = open ? "旧UI/詳細設定を隠す" : "旧UI/詳細設定を表示";
    });

    window.swiftCleanRefreshLearning = () => { updateCleanStats(); renderCleanLearning(); };
    cleanLoadAiData().catch(() => { updateCleanStats(); renderCleanLearning(); });
  }

  ready(() => {
    // bootAddonが既存カードを作った後に被せる。
    setTimeout(installCleanDashboard, 150);
    setTimeout(installCleanDashboard, 800);
  });
})();


/* =========================================================
 * SWIFT-TEC v5.3 Accuracy First Grid Dashboard UI
 * Main focus: hit-rate trend for the past two years.
 * Left input area is minimized; legacy controls can be opened only when needed.
 * ========================================================= */
(function () {
  const AI_BASE = "data/ai/";
  const TWO_YEARS_DAYS = 730;

  const REGION_LABELS_V52 = (() => {
    const latBands = [
      { key: "S", label: "南緯帯" },
      { key: "E", label: "赤道帯" },
      { key: "N", label: "北緯帯" },
    ];
    const lonBands = ["180W-120W", "120W-60W", "60W-0", "0-60E", "60E-120E", "120E-180E"];
    const out = [];
    let id = 1;
    for (const lat of latBands) {
      for (const lon of lonBands) out.push({ id: `R${String(id++).padStart(2, "0")}`, label: `${lat.label} / ${lon}` });
    }
    return out;
  })();

  let perfV52 = null;
  let coeffV52 = null;
  let histV52 = null;
  let tecIndexV52 = null;
  let selectedRegionV52 = "R01";

  function readyV52(fn) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn);
    else setTimeout(fn, 0);
  }

  function q52(id) { return document.getElementById(id); }
  function arr52(x) { return Array.isArray(x) ? x : []; }
  function n52(v) { const x = Number(v); return Number.isFinite(x) ? x : NaN; }
  function clamp52(v, a, b) { return Math.max(a, Math.min(b, v)); }
  function pct52(v, digits = 1) {
    const x = n52(v);
    return Number.isFinite(x) ? `${(x * 100).toFixed(digits)}%` : "--";
  }
  function num52(v, d = 2) {
    const x = n52(v);
    return Number.isFinite(x) ? x.toFixed(d) : "--";
  }
  function dateShort52(s) {
    const d = new Date(s);
    if (isNaN(d.getTime())) return "--";
    return `${String(d.getUTCMonth() + 1).padStart(2, "0")}/${String(d.getUTCDate()).padStart(2, "0")}`;
  }
  function isoDate52(s) {
    const d = new Date(s);
    if (isNaN(d.getTime())) return "--";
    return d.toISOString().slice(0, 10);
  }

  async function fetchJson52(url) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`${url} HTTP ${res.status}`);
    return await res.json();
  }

  function installStyle52() {
    if (q52("swiftAccuracyStyleV52")) return;
    const st = document.createElement("style");
    st.id = "swiftAccuracyStyleV52";
    st.textContent = `
      body.swift-accuracy-ui {
        background: #050816;
      }
      body.swift-accuracy-ui .page {
        display: grid;
        grid-template-columns: 330px minmax(0, 1fr);
        height: 100vh;
      }
      body.swift-accuracy-ui .sidebar {
        width: auto !important;
        min-width: 0 !important;
        max-width: none !important;
        padding: 10px;
        background: linear-gradient(180deg, #07111f 0%, #050816 55%, #03040b 100%);
        border-right: 1px solid #1e3154;
      }
      body.swift-accuracy-ui .sidebar > .card {
        display: none !important;
      }
      body.swift-accuracy-ui.swift-advanced-open .sidebar > .card {
        display: block !important;
        opacity: .78;
      }
      body.swift-accuracy-ui .sidebar #swiftAccuracySide {
        display: block !important;
      }
      body.swift-accuracy-ui .main {
        min-width: 0;
        gap: 10px;
        padding: 10px;
      }
      body.swift-accuracy-ui .slider-card {
        background: rgba(7, 12, 24, 0.96);
        border-color: #1f355a;
        order: 2;
      }
      body.swift-accuracy-ui .map-card {
        order: 3;
        min-height: 52vh;
      }
      body.swift-accuracy-ui .output-card {
        display: none !important;
      }
      #swiftAccuracyMain {
        order: 1;
      }
      .swift-v52-card {
        background: rgba(7, 14, 28, 0.98);
        border: 1px solid #1f355a;
        border-radius: 16px;
        padding: 12px;
        box-shadow: 0 14px 34px rgba(0,0,0,.30);
      }
      .swift-v52-title {
        font-size: 18px;
        font-weight: 850;
        letter-spacing: .04em;
      }
      .swift-v52-sub {
        font-size: 10px;
        color: #9fb0cc;
        line-height: 1.45;
        margin-top: 3px;
      }
      .swift-v52-status {
        min-height: 18px;
        color: #a9b8d2;
        font-size: 10px;
        margin-top: 6px;
      }
      .swift-v52-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        border: 1px solid #2a4774;
        border-radius: 999px;
        padding: 3px 8px;
        font-size: 10px;
        background: #08152a;
        color: #d8e7ff;
        white-space: nowrap;
      }
      .swift-v52-dot {
        width: 7px; height: 7px;
        border-radius: 999px;
        background: #64748b;
        display: inline-block;
      }
      .swift-v52-dot.ok { background:#22c55e; box-shadow:0 0 8px rgba(34,197,94,.55); }
      .swift-v52-dot.warn { background:#f59e0b; box-shadow:0 0 8px rgba(245,158,11,.55); }
      .swift-v52-controls {
        display: grid;
        gap: 8px;
        margin-top: 10px;
      }
      .swift-v52-label {
        font-size: 10px;
        color: #9fb0cc;
        margin-bottom: 3px;
      }
      .swift-v52-select,
      .swift-v52-input {
        width: 100%;
        background: #061020;
        border: 1px solid #2a4774;
        color: #eaf2ff;
        border-radius: 10px;
        padding: 7px 8px;
        font-size: 11px;
      }
      .swift-v52-row {
        display: flex;
        gap: 7px;
        align-items: center;
        flex-wrap: wrap;
      }
      .swift-v52-row > * { flex: 1; min-width: 0; }
      .swift-v52-btn {
        border-radius: 11px;
        border: 1px solid #3b82f6;
        background: #1d4ed8;
        color: #fff;
        padding: 8px 10px;
        font-size: 11px;
        cursor: pointer;
        font-weight: 700;
      }
      .swift-v52-btn.secondary {
        border-color:#334155;
        background:#0f172a;
        color:#d7e5ff;
      }
      .swift-v52-btn.ghost {
        border-color:#1f355a;
        background:transparent;
        color:#bcd0ee;
      }
      .swift-v52-mini-stats {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 7px;
        margin-top: 10px;
      }
      .swift-v52-mini {
        border:1px solid #1f355a;
        border-radius: 12px;
        background:#061020;
        padding:8px;
        min-height:58px;
      }
      .swift-v52-mini-label { color:#9fb0cc; font-size:9px; }
      .swift-v52-mini-value { color:#f8fafc; font-weight:850; font-size:17px; margin-top:4px; }
      .swift-v52-mini-note { color:#7788a6; font-size:9px; margin-top:2px; }
      .swift-v52-grid-main {
        display: grid;
        grid-template-columns: minmax(0, 1.45fr) minmax(280px, .85fr);
        gap: 10px;
      }
      .swift-v52-chart-head {
        display:flex;
        justify-content:space-between;
        align-items:flex-start;
        gap:8px;
        margin-bottom:8px;
      }
      .swift-v52-head-actions {
        display:flex;
        gap:6px;
        flex-wrap:wrap;
        justify-content:flex-end;
      }
      .swift-v52-chip {
        border:1px solid #1f355a;
        background:#08152a;
        color:#bcd0ee;
        border-radius:999px;
        padding:5px 8px;
        font-size:10px;
        cursor:pointer;
      }
      .swift-v52-chip.active {
        border-color:#60a5fa;
        background:#1d4ed8;
        color:white;
        font-weight:800;
      }
      .swift-v52-canvas-big {
        width: 100%;
        height: 285px;
        border: 1px solid #1f355a;
        background:#030712;
        border-radius: 14px;
        display:block;
      }
      .swift-v52-canvas-small {
        width: 100%;
        height: 190px;
        border: 1px solid #1f355a;
        background:#030712;
        border-radius: 14px;
        display:block;
      }
      .swift-v52-region-list {
        display:grid;
        grid-template-columns: repeat(3, 1fr);
        gap:6px;
        max-height: 260px;
        overflow:auto;
        padding-right:2px;
      }
      .swift-v52-region {
        border:1px solid #1f355a;
        border-radius:10px;
        background:#061020;
        padding:7px;
        cursor:pointer;
      }
      .swift-v52-region.active {
        border-color:#60a5fa;
        background:#082451;
      }
      .swift-v52-region-id { font-weight:850; font-size:11px; }
      .swift-v52-region-label { color:#9fb0cc; font-size:8.5px; margin-top:2px; min-height:20px; }
      .swift-v52-region-hit { font-size:14px; font-weight:850; margin-top:4px; }
      .swift-v52-table {
        width:100%;
        border-collapse:collapse;
        font-size:10px;
        margin-top:8px;
      }
      .swift-v52-table th,.swift-v52-table td {
        border:1px solid #1f355a;
        padding:5px 6px;
      }
      .swift-v52-table th {
        background:#0b1730;
        color:#bcd0ee;
        text-align:left;
      }
      .swift-v52-kpi-row {
        display:grid;
        grid-template-columns: repeat(4, 1fr);
        gap:8px;
        margin-bottom:10px;
      }
      .swift-v52-kpi {
        border:1px solid #1f355a;
        border-radius:13px;
        background:#061020;
        padding:10px;
      }
      .swift-v52-kpi-label { color:#9fb0cc; font-size:10px; }
      .swift-v52-kpi-value { color:#f8fafc; font-size:22px; font-weight:900; margin-top:4px; }
      .swift-v52-kpi-note { color:#7788a6; font-size:9px; margin-top:2px; }
      @media (max-width: 980px) {
        body.swift-accuracy-ui .page { grid-template-columns: 1fr; height:auto; }
        body.swift-accuracy-ui .sidebar { border-right:0; border-bottom:1px solid #1e3154; }
        .swift-v52-grid-main { grid-template-columns: 1fr; }
        .swift-v52-kpi-row { grid-template-columns: 1fr 1fr; }
      }
    `;
    document.head.appendChild(st);
  }

  function setV52Status(msg) {
    const el = q52("swiftV52Status");
    if (el) el.textContent = msg || "";
    try { window.logInfo?.(msg || ""); } catch {}
  }

  function historyRowsV52(days = TWO_YEARS_DAYS) {
    const runs = arr52(histV52?.runs).slice().sort((a, b) => new Date(a.time_utc) - new Date(b.time_utc));
    const now = Date.now();
    const minT = now - days * 86400000;
    return runs.filter(r => {
      const t = new Date(r.time_utc).getTime();
      return Number.isFinite(t) && t >= minT;
    });
  }

  function summaryV52() {
    return perfV52?.summary || {};
  }

  function monthV52() {
    const m = Number(q52("swiftV52Month")?.value || (new Date()).getUTCMonth() + 1);
    return String(clamp52(Math.round(Number.isFinite(m) ? m : 1), 1, 12));
  }

  function regionRowsV52(month) {
    const metrics = perfV52?.metrics || {};
    const coeffs = coeffV52?.coefficients || {};
    return REGION_LABELS_V52.map(r => ({
      ...r,
      metrics: metrics[r.id]?.[String(month)] || {},
      coeff: coeffs[r.id]?.[String(month)] || {},
    }));
  }

  async function loadV52Data() {
    setV52Status("AI学習結果を読み込み中…");
    const results = await Promise.allSettled([
      fetchJson52(AI_BASE + "kp_performance.json"),
      fetchJson52(AI_BASE + "kp_coefficients.json"),
      fetchJson52(AI_BASE + "kp_learning_history.json"),
      fetchJson52("data/tec/index.json"),
    ]);
    perfV52 = results[0].status === "fulfilled" ? results[0].value : null;
    coeffV52 = results[1].status === "fulfilled" ? results[1].value : null;
    histV52 = results[2].status === "fulfilled" ? results[2].value : null;
    tecIndexV52 = results[3].status === "fulfilled" ? results[3].value : null;
    try { await window.loadKpAiData?.(false); } catch {}
    renderV52All();
    setV52Status(perfV52 ? "AI学習結果を読み込みました。" : "AI学習結果なし。Actionsで Train Kp AI Corrector を実行してください。");
  }

  function syncV52ForecastControls() {
    const source = q52("swiftV52TecSource")?.value || "archive_data_30m";
    const legacySource = q52("forecastTecApiSourceSelect");
    if (legacySource) legacySource.value = source;
    try { window.setForecastTecApiMode?.(source); } catch {}

    const enabled = !!q52("swiftV52AiEnabled")?.checked;
    const legacyEnabled = q52("kpAiCorrectionEnabled");
    if (legacyEnabled) legacyEnabled.checked = enabled;

    const clip = q52("swiftV52AiClip")?.value || "20";
    const legacyClip = q52("kpAiCorrectionClip");
    if (legacyClip) legacyClip.value = clip;
  }

  async function v52LoadTec() {
    syncV52ForecastControls();
    setV52Status("予報用TECを取得中…");
    await window.loadForecastTecFromSelectedApi?.(false);
    setV52Status("予報用TECを取得しました。");
  }

  async function v52RunForecast() {
    syncV52ForecastControls();
    const auto = q52("forecastTecAutoFetch");
    if (auto) auto.checked = true;
    setV52Status("AI設定を反映してTEC予報を計算中…");
    await window.runForecast?.();
    setV52Status("予報を実行しました。地図・時間スライダーで確認できます。");
  }

  function drawAxes52(ctx, w, h, plot, yLabel) {
    ctx.strokeStyle = "rgba(148,163,184,.26)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = plot.y1 - (plot.y1 - plot.y0) * i / 4;
      ctx.beginPath();
      ctx.moveTo(plot.x0, y);
      ctx.lineTo(plot.x1, y);
      ctx.stroke();
      ctx.fillStyle = "#73839e";
      ctx.font = `${10 * (window.devicePixelRatio || 1)}px system-ui`;
      const label = yLabel === "pct" ? `${i * 25}%` : "";
      if (label) ctx.fillText(label, 6 * (window.devicePixelRatio || 1), y + 3 * (window.devicePixelRatio || 1));
    }
  }

  function setupCanvas52(canvas, minH = 180) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(360, Math.floor(rect.width * dpr));
    const h = Math.max(minH, Math.floor(rect.height * dpr));
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#030712";
    ctx.fillRect(0, 0, w, h);
    return { ctx, w, h, dpr };
  }

  function drawHitTrendV52() {
    const canvas = q52("swiftV52HitTrend");
    if (!canvas) return;
    const { ctx, w, h, dpr } = setupCanvas52(canvas, 240);
    const span = Number(q52("swiftV52Span")?.value || TWO_YEARS_DAYS);
    const rows = historyRowsV52(span);
    const plot = { x0: 44 * dpr, y0: 26 * dpr, x1: w - 16 * dpr, y1: h - 34 * dpr };
    ctx.fillStyle = "#dbeafe";
    ctx.font = `${13 * dpr}px system-ui`;
    ctx.fillText(`的中率推移（過去${span >= 730 ? "2年" : span + "日"}）`, 14 * dpr, 17 * dpr);
    drawAxes52(ctx, w, h, plot, "pct");

    if (!rows.length) {
      ctx.fillStyle = "#94a3b8";
      ctx.font = `${13 * dpr}px system-ui`;
      ctx.fillText("まだ学習履歴がありません。Train Kp AI Corrector を実行してください。", plot.x0, (plot.y0 + plot.y1) / 2);
      return;
    }

    const xAt = i => rows.length === 1 ? (plot.x0 + plot.x1) / 2 : plot.x0 + (plot.x1 - plot.x0) * i / (rows.length - 1);
    const yAt = v => plot.y1 - (plot.y1 - plot.y0) * clamp52(Number(v) || 0, 0, 1);

    function line(key, color, width = 2.2) {
      ctx.strokeStyle = color;
      ctx.lineWidth = width * dpr;
      ctx.beginPath();
      rows.forEach((r, i) => {
        const x = xAt(i), y = yAt(r[key]);
        if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      });
      ctx.stroke();
      ctx.fillStyle = color;
      const step = Math.max(1, Math.floor(rows.length / 18));
      rows.forEach((r, i) => {
        if (i % step && i !== rows.length - 1) return;
        ctx.beginPath();
        ctx.arc(xAt(i), yAt(r[key]), 2.2 * dpr, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    line("raw_hit_rate", "#64748b", 1.8);
    line("corrected_hit_rate", "#60a5fa", 2.6);

    const latest = rows[rows.length - 1] || {};
    ctx.font = `${11 * dpr}px system-ui`;
    ctx.fillStyle = "#64748b";
    ctx.fillText(`raw ${pct52(latest.raw_hit_rate)}`, plot.x0, h - 13 * dpr);
    ctx.fillStyle = "#60a5fa";
    ctx.fillText(`AI補正後 ${pct52(latest.corrected_hit_rate)}`, plot.x0 + 85 * dpr, h - 13 * dpr);
    ctx.fillStyle = "#94a3b8";
    ctx.fillText(`${isoDate52(rows[0].time_utc)} 〜 ${isoDate52(latest.time_utc)}`, plot.x1 - 155 * dpr, h - 13 * dpr);
  }

  function drawRmseTrendV52() {
    const canvas = q52("swiftV52RmseTrend");
    if (!canvas) return;
    const { ctx, w, h, dpr } = setupCanvas52(canvas, 155);
    const rows = historyRowsV52(Number(q52("swiftV52Span")?.value || TWO_YEARS_DAYS));
    const plot = { x0: 44 * dpr, y0: 24 * dpr, x1: w - 12 * dpr, y1: h - 26 * dpr };
    ctx.fillStyle = "#dbeafe";
    ctx.font = `${12 * dpr}px system-ui`;
    ctx.fillText("RMSE推移", 14 * dpr, 16 * dpr);
    if (!rows.length) {
      ctx.fillStyle = "#94a3b8";
      ctx.fillText("No data", plot.x0, (plot.y0 + plot.y1) / 2);
      return;
    }
    const vals = rows.flatMap(r => [n52(r.raw_rmse), n52(r.corrected_rmse)]).filter(Number.isFinite);
    const maxV = Math.max(1, ...vals) * 1.08;
    for (let i = 0; i <= 4; i++) {
      const y = plot.y1 - (plot.y1 - plot.y0) * i / 4;
      ctx.strokeStyle = "rgba(148,163,184,.20)";
      ctx.beginPath(); ctx.moveTo(plot.x0, y); ctx.lineTo(plot.x1, y); ctx.stroke();
      ctx.fillStyle = "#73839e"; ctx.font = `${9 * dpr}px system-ui`;
      ctx.fillText((maxV * i / 4).toFixed(0), 10 * dpr, y + 3 * dpr);
    }
    const xAt = i => rows.length === 1 ? (plot.x0 + plot.x1) / 2 : plot.x0 + (plot.x1 - plot.x0) * i / (rows.length - 1);
    const yAt = v => plot.y1 - (plot.y1 - plot.y0) * (clamp52(n52(v), 0, maxV) / maxV);
    function line(key, color) {
      ctx.strokeStyle = color; ctx.lineWidth = 2 * dpr; ctx.beginPath();
      rows.forEach((r, i) => { const x = xAt(i), y = yAt(r[key]); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
      ctx.stroke();
    }
    line("raw_rmse", "#64748b");
    line("corrected_rmse", "#38bdf8");
  }

  function drawRegionBarsV52() {
    const canvas = q52("swiftV52RegionBars");
    if (!canvas) return;
    const { ctx, w, h, dpr } = setupCanvas52(canvas, 160);
    const rows = regionRowsV52(monthV52());
    const plot = { x0: 42 * dpr, y0: 22 * dpr, x1: w - 12 * dpr, y1: h - 16 * dpr };
    ctx.fillStyle = "#dbeafe";
    ctx.font = `${12 * dpr}px system-ui`;
    ctx.fillText(`${monthV52()}月 18地域 的中率`, 12 * dpr, 15 * dpr);
    const gap = 2 * dpr;
    const barH = Math.max(3 * dpr, (plot.y1 - plot.y0 - gap * 17) / 18);
    rows.forEach((r, i) => {
      const y = plot.y0 + i * (barH + gap);
      const raw = clamp52(n52(r.metrics.raw_hit_rate) || 0, 0, 1);
      const corr = clamp52(n52(r.metrics.corrected_hit_rate) || 0, 0, 1);
      ctx.fillStyle = "#94a3b8";
      ctx.font = `${8.5 * dpr}px system-ui`;
      ctx.fillText(r.id, 8 * dpr, y + barH * .8);
      ctx.fillStyle = "rgba(100,116,139,.45)";
      ctx.fillRect(plot.x0, y, (plot.x1 - plot.x0) * raw, barH);
      ctx.fillStyle = "#60a5fa";
      ctx.fillRect(plot.x0, y + barH * .48, (plot.x1 - plot.x0) * corr, Math.max(2 * dpr, barH * .52));
    });
  }

  function renderRegionListV52() {
    const box = q52("swiftV52Regions");
    if (!box) return;
    const rows = regionRowsV52(monthV52());
    box.innerHTML = rows.map(r => {
      const hit = n52(r.metrics.corrected_hit_rate);
      const raw = n52(r.metrics.raw_hit_rate);
      const cls = hit >= .75 ? "ok" : (hit >= .55 ? "warn" : "");
      return `<div class="swift-v52-region ${selectedRegionV52 === r.id ? "active" : ""}" data-region="${r.id}" onclick="window.swiftV52SelectRegion('${r.id}')">
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <span class="swift-v52-region-id">${r.id}</span><span class="swift-v52-dot ${cls}"></span>
        </div>
        <div class="swift-v52-region-label">${r.label}</div>
        <div class="swift-v52-region-hit">${pct52(hit)}</div>
        <div class="swift-v52-mini-note">raw ${pct52(raw)}</div>
      </div>`;
    }).join("");
  }

  function renderRegionDetailV52() {
    const box = q52("swiftV52RegionDetail");
    if (!box) return;
    const r = regionRowsV52(monthV52()).find(x => x.id === selectedRegionV52) || regionRowsV52(monthV52())[0];
    if (!r) { box.innerHTML = ""; return; }
    const m = r.metrics || {}, k = r.coeff || {};
    box.innerHTML = `
      <table class="swift-v52-table">
        <tbody>
          <tr><th>地域</th><td>${r.id} / ${r.label}</td></tr>
          <tr><th>Hit</th><td>${pct52(m.raw_hit_rate)} → <b>${pct52(m.corrected_hit_rate)}</b></td></tr>
          <tr><th>RMSE</th><td>${num52(m.raw_rmse)} → <b>${num52(m.corrected_rmse)}</b></td></tr>
          <tr><th>Bias</th><td>${num52(m.raw_bias)} → <b>${num52(m.corrected_bias)}</b></td></tr>
          <tr><th>Samples</th><td>${m.sample_count || 0}</td></tr>
          <tr><th>k0/k1/k2/k3</th><td>${num52(k.k0,3)} / ${num52(k.k1,3)} / ${num52(k.k2,3)} / ${num52(k.k3,3)}</td></tr>
        </tbody>
      </table>`;
  }

  function renderKpisV52() {
    const s = summaryV52();
    const rows = historyRowsV52(TWO_YEARS_DAYS);
    const latest = rows[rows.length - 1] || s;
    const set = (id, v) => { const el = q52(id); if (el) el.textContent = v; };
    set("swiftV52KpiHit", `${pct52(latest.raw_hit_rate)} → ${pct52(latest.corrected_hit_rate)}`);
    set("swiftV52KpiRmse", `${num52(latest.raw_rmse)} → ${num52(latest.corrected_rmse)}`);
    set("swiftV52KpiSamples", String(latest.sample_count || s.sample_count || "--"));
    set("swiftV52KpiRuns", String(rows.length || "--"));
    const note = q52("swiftV52HistoryNote");
    if (note) note.textContent = rows.length
      ? `学習履歴: ${isoDate52(rows[0].time_utc)} 〜 ${isoDate52(rows[rows.length - 1].time_utc)} / 最大2年表示 / 係数は全格子学習`
      : "学習履歴なし";
    const dot = q52("swiftV52AiDot");
    if (dot) dot.className = "swift-v52-dot " + (perfV52 ? "ok" : "warn");

    const frames = arr52(tecIndexV52?.frames);
    const dataEl = q52("swiftV52DataCount");
    if (dataEl) dataEl.textContent = frames.length ? `${frames.length} frames` : "--";
    const dataNote = q52("swiftV52DataNote");
    if (dataNote && frames.length) dataNote.textContent = `${frames[0].time_utc || "--"} 〜 ${frames[frames.length - 1].time_utc || "--"}`;
  }

  function renderRecentRunsV52() {
    const box = q52("swiftV52RecentRuns");
    if (!box) return;
    const rows = historyRowsV52(TWO_YEARS_DAYS).slice(-8).reverse();
    if (!rows.length) {
      box.innerHTML = `<div class="swift-v52-sub">まだ学習履歴がありません。</div>`;
      return;
    }
    box.innerHTML = `<table class="swift-v52-table">
      <thead><tr><th>UTC</th><th>Hit raw→AI</th><th>RMSE raw→AI</th><th>N</th></tr></thead>
      <tbody>${rows.map(r => `<tr>
        <td>${isoDate52(r.time_utc)}</td>
        <td>${pct52(r.raw_hit_rate)} → <b>${pct52(r.corrected_hit_rate)}</b></td>
        <td>${num52(r.raw_rmse)} → <b>${num52(r.corrected_rmse)}</b></td>
        <td>${r.sample_count || 0}</td>
      </tr>`).join("")}</tbody>
    </table>`;
  }

  function renderV52All() {
    renderKpisV52();
    drawHitTrendV52();
    drawRmseTrendV52();
    drawRegionBarsV52();
    renderRegionListV52();
    renderRegionDetailV52();
    renderRecentRunsV52();
  }

  function buildSidebarV52() {
    const sidebar = document.querySelector(".sidebar");
    if (!sidebar) return;
    q52("swiftCleanDashboard")?.remove();
    q52("swiftAccuracySide")?.remove();

    const card = document.createElement("div");
    card.id = "swiftAccuracySide";
    card.className = "swift-v52-card";
    card.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px;">
        <div>
          <div class="swift-v52-title">SWIFT-TEC AI</div>
          <div class="swift-v52-sub">全格子AIで育てるTEC予報</div>
        </div>
        <span class="swift-v52-pill"><span id="swiftV52AiDot" class="swift-v52-dot warn"></span> AI</span>
      </div>

      <div class="swift-v52-mini-stats">
        <div class="swift-v52-mini">
          <div class="swift-v52-mini-label">最新Hit raw→AI</div>
          <div class="swift-v52-mini-value" id="swiftV52KpiHit">--</div>
          <div class="swift-v52-mini-note">±5 TECU以内</div>
        </div>
        <div class="swift-v52-mini">
          <div class="swift-v52-mini-label">最新RMSE raw→AI</div>
          <div class="swift-v52-mini-value" id="swiftV52KpiRmse">--</div>
          <div class="swift-v52-mini-note">小さいほど良い</div>
        </div>
        <div class="swift-v52-mini">
          <div class="swift-v52-mini-label">学習サンプル</div>
          <div class="swift-v52-mini-value" id="swiftV52KpiSamples">--</div>
          <div class="swift-v52-mini-note">直近学習回</div>
        </div>
        <div class="swift-v52-mini">
          <div class="swift-v52-mini-label">2年内学習回数</div>
          <div class="swift-v52-mini-value" id="swiftV52KpiRuns">--</div>
          <div class="swift-v52-mini-note">最大730日分</div>
        </div>
      </div>

      <div class="swift-v52-controls">
        <div>
          <div class="swift-v52-label">予報用TEC</div>
          <select id="swiftV52TecSource" class="swift-v52-select" onchange="window.swiftV52SyncForecastControls()">
            <option value="archive_data_30m" selected>取りため済みdata API</option>
            <option value="noaa_direct_30m">NOAA API直取得</option>
          </select>
        </div>
        <div class="swift-v52-row">
          <label class="swift-v52-pill" style="justify-content:center;cursor:pointer;">
            <input id="swiftV52AiEnabled" type="checkbox" onchange="window.swiftV52SyncForecastControls()" checked>
            AI Kp補正ON
          </label>
          <div>
            <div class="swift-v52-label">補正上限[TECU]</div>
            <input id="swiftV52AiClip" class="swift-v52-input" type="number" value="20" min="0" max="50" step="1" onchange="window.swiftV52SyncForecastControls()">
          </div>
        </div>
        <div class="swift-v52-row">
          <button class="swift-v52-btn secondary" onclick="window.swiftV52LoadData()">AI読込</button>
          <button class="swift-v52-btn secondary" onclick="window.swiftV52LoadTec()">TEC取得</button>
        </div>
        <button class="swift-v52-btn" onclick="window.swiftV52RunForecast()">予報実行</button>
        <div class="swift-v52-row">
          <button class="swift-v52-btn secondary" onclick="window.playArchiveMovie?.()">▶ 再生</button>
          <button class="swift-v52-btn secondary" onclick="window.stopArchiveMovie?.()">⏸ 停止</button>
        </div>
        <div class="swift-v52-row">
          <button class="swift-v52-btn ghost" onclick="window.loadGnssDopData?.()">GNSS読込</button>
          <button class="swift-v52-btn ghost" onclick="document.body.classList.toggle('swift-advanced-open')">詳細設定</button>
        </div>
      </div>

      <div class="swift-v52-mini" style="margin-top:10px;">
        <div class="swift-v52-mini-label">TEC蓄積</div>
        <div class="swift-v52-mini-value" id="swiftV52DataCount">--</div>
        <div class="swift-v52-mini-note" id="swiftV52DataNote">--</div>
      </div>
      <div class="swift-v52-status" id="swiftV52Status"></div>
    `;
    sidebar.insertBefore(card, sidebar.firstChild);
  }

  function buildMainV52() {
    const main = document.querySelector(".main");
    if (!main) return;
    q52("swiftAccuracyMain")?.remove();

    const panel = document.createElement("div");
    panel.id = "swiftAccuracyMain";
    panel.className = "swift-v52-card";
    panel.innerHTML = `
      <div class="swift-v52-chart-head">
        <div>
          <div class="swift-v52-title">的中率モニター</div>
          <div class="swift-v52-sub" id="swiftV52HistoryNote">全格子で学習し、18地域に集約した的中率を過去2年分表示します。</div>
        </div>
        <div class="swift-v52-head-actions">
          <select id="swiftV52Span" class="swift-v52-select" style="width:118px;" onchange="window.swiftV52RenderAll()">
            <option value="90">90日</option>
            <option value="365">1年</option>
            <option value="730" selected>2年</option>
          </select>
          <select id="swiftV52Month" class="swift-v52-select" style="width:80px;" onchange="window.swiftV52RenderAll()">
            ${Array.from({ length: 12 }, (_, i) => `<option value="${i+1}" ${(i+1)===(new Date()).getUTCMonth()+1 ? "selected" : ""}>${i+1}月</option>`).join("")}
          </select>
          <button class="swift-v52-btn secondary" style="padding:7px 10px;" onclick="window.swiftV52LoadData()">更新</button>
        </div>
      </div>

      <div class="swift-v52-kpi-row">
        <div class="swift-v52-kpi">
          <div class="swift-v52-kpi-label">最新Hit raw→AI</div>
          <div class="swift-v52-kpi-value" id="swiftV52MainHit">--</div>
          <div class="swift-v52-kpi-note">AI補正で上がるか確認</div>
        </div>
        <div class="swift-v52-kpi">
          <div class="swift-v52-kpi-label">最新RMSE raw→AI</div>
          <div class="swift-v52-kpi-value" id="swiftV52MainRmse">--</div>
          <div class="swift-v52-kpi-note">小さいほど良い</div>
        </div>
        <div class="swift-v52-kpi">
          <div class="swift-v52-kpi-label">2年内学習回数</div>
          <div class="swift-v52-kpi-value" id="swiftV52MainRuns">--</div>
          <div class="swift-v52-kpi-note">最大730回</div>
        </div>
        <div class="swift-v52-kpi">
          <div class="swift-v52-kpi-label">最新サンプル数</div>
          <div class="swift-v52-kpi-value" id="swiftV52MainSamples">--</div>
          <div class="swift-v52-kpi-note">学習に使った点数</div>
        </div>
      </div>

      <div class="swift-v52-grid-main">
        <div>
          <canvas id="swiftV52HitTrend" class="swift-v52-canvas-big"></canvas>
          <div style="margin-top:10px;">
            <canvas id="swiftV52RmseTrend" class="swift-v52-canvas-small"></canvas>
          </div>
        </div>
        <div>
          <canvas id="swiftV52RegionBars" class="swift-v52-canvas-small"></canvas>
          <div style="margin-top:10px;" class="swift-v52-region-list" id="swiftV52Regions"></div>
          <div id="swiftV52RegionDetail"></div>
        </div>
      </div>
      <div id="swiftV52RecentRuns" style="margin-top:10px;"></div>
    `;
    main.insertBefore(panel, main.firstChild);
  }

  function syncMainKpisV52() {
    const rows = historyRowsV52(TWO_YEARS_DAYS);
    const latest = rows[rows.length - 1] || summaryV52();
    const set = (id, val) => { const el = q52(id); if (el) el.textContent = val; };
    set("swiftV52MainHit", `${pct52(latest.raw_hit_rate)} → ${pct52(latest.corrected_hit_rate)}`);
    set("swiftV52MainRmse", `${num52(latest.raw_rmse)} → ${num52(latest.corrected_rmse)}`);
    set("swiftV52MainRuns", String(rows.length || "--"));
    set("swiftV52MainSamples", String(latest.sample_count || "--"));
  }

  const originalRenderV52All = renderV52All;
  renderV52All = function () {
    originalRenderV52All();
    syncMainKpisV52();
  };

  function bootV52() {
    document.body.classList.add("swift-accuracy-ui");
    document.body.classList.remove("swift-advanced-open");
    installStyle52();
    setTimeout(() => {
      q52("swiftCleanDashboard")?.remove();
      buildSidebarV52();
      buildMainV52();
      syncV52ForecastControls();
      loadV52Data().catch(e => {
        console.warn(e);
        renderV52All();
        setV52Status("AI学習結果をまだ読めません。Train Kp AI Correctorを実行してください。");
      });
      window.addEventListener("resize", () => {
        clearTimeout(window.__swiftV52ResizeTimer);
        window.__swiftV52ResizeTimer = setTimeout(renderV52All, 150);
      });
    }, 550);
  }

  window.swiftV52LoadData = loadV52Data;
  window.swiftV52LoadTec = v52LoadTec;
  window.swiftV52RunForecast = v52RunForecast;
  window.swiftV52SyncForecastControls = syncV52ForecastControls;
  window.swiftV52RenderAll = renderV52All;
  window.swiftV52SelectRegion = function (rid) {
    selectedRegionV52 = rid;
    renderRegionListV52();
    renderRegionDetailV52();
  };

  readyV52(bootV52);
})();


/* =========================================================
 * SWIFT-TEC v5.4 heatmap visibility fix
 * Keep the accuracy dashboard as the main view, but reserve stable
 * vertical space for the Leaflet heatmap and force a resize redraw.
 * ========================================================= */
(function () {
  function readyV54(fn) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn);
    else setTimeout(fn, 0);
  }

  function injectHeatmapFixStyleV54() {
    if (document.getElementById("swiftHeatmapFixStyleV54")) return;
    const style = document.createElement("style");
    style.id = "swiftHeatmapFixStyleV54";
    style.textContent = `
      body.swift-accuracy-ui .page {
        height: 100vh;
        overflow: hidden;
      }

      body.swift-accuracy-ui .main {
        height: 100vh;
        min-height: 0;
        overflow: hidden;
        display: flex;
        flex-direction: column;
      }

      body.swift-accuracy-ui #swiftAccuracyMain {
        order: 1;
        flex: 0 0 40vh;
        max-height: 40vh;
        min-height: 285px;
        overflow: auto;
      }

      body.swift-accuracy-ui .slider-card {
        order: 2;
        flex: 0 0 auto;
      }

      body.swift-accuracy-ui .map-card {
        order: 3;
        flex: 1 1 auto !important;
        min-height: 330px !important;
        display: flex !important;
        overflow: hidden;
      }

      body.swift-accuracy-ui #tecMap {
        display: block !important;
        width: 100% !important;
        height: 100% !important;
        min-height: 330px !important;
      }

      body.swift-accuracy-ui .leaflet-container {
        width: 100% !important;
        height: 100% !important;
      }

      body.swift-accuracy-ui .swift-v52-kpi-row {
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 7px;
        margin-bottom: 8px;
      }

      body.swift-accuracy-ui .swift-v52-kpi {
        padding: 8px;
      }

      body.swift-accuracy-ui .swift-v52-kpi-value {
        font-size: 18px;
      }

      body.swift-accuracy-ui .swift-v52-grid-main {
        grid-template-columns: minmax(0, 1.35fr) minmax(260px, .85fr);
        gap: 8px;
      }

      body.swift-accuracy-ui .swift-v52-canvas-big {
        height: 205px;
      }

      body.swift-accuracy-ui .swift-v52-canvas-small {
        height: 125px;
      }

      body.swift-accuracy-ui .swift-v52-region-list {
        max-height: 145px;
      }

      body.swift-accuracy-ui #swiftV52RecentRuns {
        display: none;
      }

      @media (max-height: 780px) {
        body.swift-accuracy-ui #swiftAccuracyMain {
          flex-basis: 34vh;
          max-height: 34vh;
          min-height: 245px;
        }
        body.swift-accuracy-ui .swift-v52-canvas-big {
          height: 170px;
        }
        body.swift-accuracy-ui .swift-v52-canvas-small {
          height: 105px;
        }
        body.swift-accuracy-ui .map-card {
          min-height: 300px !important;
        }
        body.swift-accuracy-ui #tecMap {
          min-height: 300px !important;
        }
      }

      @media (max-width: 980px) {
        body.swift-accuracy-ui .page {
          height: auto;
          overflow: auto;
        }
        body.swift-accuracy-ui .main {
          height: auto;
          overflow: visible;
        }
        body.swift-accuracy-ui #swiftAccuracyMain {
          flex-basis: auto;
          max-height: none;
        }
        body.swift-accuracy-ui .map-card {
          min-height: 430px !important;
        }
        body.swift-accuracy-ui #tecMap {
          min-height: 430px !important;
        }
      }
    `;
    document.head.appendChild(style);
  }

  function forceLeafletResizeV54() {
    // Leaflet is initialized before/while the new dashboard changes layout.
    // Dispatching resize makes Leaflet recompute the map canvas size.
    for (const delay of [80, 250, 700, 1300]) {
      setTimeout(() => {
        try { window.dispatchEvent(new Event("resize")); } catch {}
        try { window.requestDraw?.(); } catch {}
      }, delay);
    }
  }

  function bootHeatmapFixV54() {
    injectHeatmapFixStyleV54();
    forceLeafletResizeV54();

    const mapCard = document.querySelector(".map-card");
    if (mapCard && "ResizeObserver" in window) {
      const ro = new ResizeObserver(() => forceLeafletResizeV54());
      ro.observe(mapCard);
    }

    document.addEventListener("click", (ev) => {
      const t = ev.target;
      if (t && (t.id === "swiftV52Span" || t.id === "swiftV52Month" || String(t.className || "").includes("swift-v52"))) {
        forceLeafletResizeV54();
      }
    }, true);
  }

  readyV54(bootHeatmapFixV54);
})();


/* =========================================================
 * SWIFT-TEC v5.5 point readout + NOAA 3-Day forecast automation
 * - Shows TEC / ionospheric error / DOP values in the clean left UI when the map is clicked.
 * - Forecast execution downloads NOAA 3-Day Geomagnetic Forecast API first, then runs TEC forecast.
 * ========================================================= */
(function () {
  let oldSwiftV52RunForecast = null;

  function readyV55(fn) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn);
    else setTimeout(fn, 0);
  }

  function q55(id) { return document.getElementById(id); }

  function injectStyleV55() {
    if (q55("swiftPointForecastStyleV55")) return;
    const st = document.createElement("style");
    st.id = "swiftPointForecastStyleV55";
    st.textContent = `
      .swift-v55-point-card {
        margin-top: 10px;
        border: 1px solid #1f355a;
        border-radius: 13px;
        background: #061020;
        padding: 9px;
      }
      .swift-v55-point-title {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
        font-size: 11px;
        font-weight: 850;
        color: #eaf2ff;
      }
      .swift-v55-point-hint {
        font-size: 9px;
        color: #8ba0c2;
        margin-top: 2px;
      }
      .swift-v55-point-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 6px;
        margin-top: 8px;
      }
      .swift-v55-metric {
        border: 1px solid #1b2f50;
        border-radius: 10px;
        background: #030712;
        padding: 7px;
        min-height: 50px;
      }
      .swift-v55-metric-label {
        font-size: 9px;
        color: #8ea3c4;
      }
      .swift-v55-metric-value {
        margin-top: 3px;
        font-size: 15px;
        font-weight: 900;
        color: #f8fafc;
        line-height: 1.1;
      }
      .swift-v55-point-detail {
        margin-top: 7px;
        white-space: pre-wrap;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 9px;
        color: #b8c7df;
        max-height: 92px;
        overflow: auto;
        border-top: 1px solid #1f355a;
        padding-top: 6px;
      }
      .swift-v55-switch {
        display: flex;
        align-items: center;
        gap: 6px;
        border: 1px solid #1f355a;
        border-radius: 11px;
        background: #08152a;
        color: #dbeafe;
        padding: 7px 8px;
        font-size: 10px;
        cursor: pointer;
      }
      body.swift-accuracy-ui #swiftV55ForecastOption {
        display: block;
      }
    `;
    document.head.appendChild(st);
  }

  function parsePointInfoV55(text) {
    const s = String(text || "");
    const out = {};
    const mClicked = s.match(/Clicked:\s*lat=([-\d.]+),\s*lon=([-\d.]+)/i);
    if (mClicked) {
      out.lat = mClicked[1];
      out.lon = mClicked[2];
    }
    const mGrid = s.match(/Nearest Grid:\s*lat=([-\d.]+),\s*lon=([-\d.]+)/i);
    if (mGrid) {
      out.gridLat = mGrid[1];
      out.gridLon = mGrid[2];
    }
    const mTime = s.match(/Time:\s*([^\n]+)/i);
    if (mTime) out.time = mTime[1].trim();
    const mTec = s.match(/TEC:\s*([-\d.]+|NaN)\s*TECU/i);
    if (mTec) out.tec = mTec[1];
    const mL1 = s.match(/L1 iono error:\s*([-\d.]+)\s*m/i) || s.match(/GPS L1 error:\s*([-\d.]+)\s*m/i);
    if (mL1) out.l1 = mL1[1];
    const mSats = s.match(/Visible GNSS sats:\s*([-\d.]+|--)/i);
    if (mSats) out.sats = mSats[1];
    const mPdop = s.match(/PDOP:\s*([-\d.]+|--)/i);
    if (mPdop) out.pdop = mPdop[1];
    const mPdopL1 = s.match(/PDOP×L1:\s*([-\d.]+|--)\s*m/i);
    if (mPdopL1) out.pdopL1 = mPdopL1[1];
    return out;
  }

  function updatePointPanelV55(text) {
    const rawEl = q55("swiftV55PointRaw");
    if (rawEl) rawEl.textContent = text || "地図をクリックしてください。";
    const p = parsePointInfoV55(text);
    const set = (id, val) => { const el = q55(id); if (el) el.textContent = val || "--"; };
    set("swiftV55PointTec", p.tec && p.tec !== "NaN" ? `${p.tec} TECU` : "--");
    set("swiftV55PointL1", p.l1 ? `${p.l1} m` : "--");
    set("swiftV55PointPdop", p.pdop || "--");
    set("swiftV55PointPdopL1", p.pdopL1 ? `${p.pdopL1} m` : "--");
    set("swiftV55PointPos", p.lat && p.lon ? `lat ${p.lat}, lon ${p.lon}` : "--");
    set("swiftV55PointTime", p.time || "--");
    set("swiftV55PointSats", p.sats || "--");
  }

  function ensurePointPanelV55() {
    const side = q55("swiftAccuracySide");
    if (!side || q55("swiftV55PointCard")) return;

    const option = document.createElement("div");
    option.id = "swiftV55ForecastOption";
    option.style.marginTop = "8px";
    option.innerHTML = `
      <label class="swift-v55-switch">
        <input id="swiftV55AutoNoaa3DayKp" type="checkbox" checked>
        予報時にNOAA 3-Day Kpを自動取得
      </label>
      <div class="swift-v52-mini-note" style="margin-top:4px;">
        TEC入力API取得 → NOAA 3-Day Forecast API取得 → 予報計算 の順で実行
      </div>
    `;

    const runBtn = [...side.querySelectorAll("button")].find(b => String(b.textContent || "").includes("予報実行"));
    if (runBtn && runBtn.parentElement) runBtn.insertAdjacentElement("beforebegin", option);
    else side.appendChild(option);

    const card = document.createElement("div");
    card.id = "swiftV55PointCard";
    card.className = "swift-v55-point-card";
    card.innerHTML = `
      <div class="swift-v55-point-title">
        <span>クリック地点の値</span>
        <span class="swift-v52-pill">TEC / 誤差</span>
      </div>
      <div class="swift-v55-point-hint">地図をクリックすると、その場所に一番近い格子の値を表示します。</div>
      <div class="swift-v55-point-grid">
        <div class="swift-v55-metric">
          <div class="swift-v55-metric-label">TEC</div>
          <div class="swift-v55-metric-value" id="swiftV55PointTec">--</div>
        </div>
        <div class="swift-v55-metric">
          <div class="swift-v55-metric-label">L1電離圏誤差</div>
          <div class="swift-v55-metric-value" id="swiftV55PointL1">--</div>
        </div>
        <div class="swift-v55-metric">
          <div class="swift-v55-metric-label">PDOP</div>
          <div class="swift-v55-metric-value" id="swiftV55PointPdop">--</div>
        </div>
        <div class="swift-v55-metric">
          <div class="swift-v55-metric-label">PDOP×L1誤差</div>
          <div class="swift-v55-metric-value" id="swiftV55PointPdopL1">--</div>
        </div>
      </div>
      <div class="swift-v55-point-hint" style="margin-top:7px;">
        <span id="swiftV55PointPos">--</span><br>
        <span id="swiftV55PointTime">--</span><br>
        可視衛星数: <span id="swiftV55PointSats">--</span>
      </div>
      <div class="swift-v55-point-detail" id="swiftV55PointRaw">地図をクリックしてください。</div>
    `;
    side.appendChild(card);

    const pointInfo = q55("pointInfo");
    if (pointInfo) {
      updatePointPanelV55(pointInfo.textContent || "");
      const mo = new MutationObserver(() => updatePointPanelV55(pointInfo.textContent || ""));
      mo.observe(pointInfo, { childList: true, characterData: true, subtree: true });
    }
  }

  async function fetchNoaa3DayKpIfEnabledV55() {
    const enabled = q55("swiftV55AutoNoaa3DayKp")?.checked !== false;
    if (!enabled) return;
    const status = q55("swiftV52Status") || q55("swiftV55ForecastStatus");
    if (status) status.textContent = "NOAA 3-Day Kp Forecast APIを取得中…";
    if (typeof window.fetchNoaa3DayGeomagToTextarea === "function") {
      await window.fetchNoaa3DayGeomagToTextarea();
      if (status) status.textContent = "NOAA 3-Day Kp Forecast API取得OK。予報計算へ進みます…";
      return;
    }

    // Fallback: direct text download into the existing textarea.
    const url = "https://services.swpc.noaa.gov/text/3-day-forecast.txt";
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`NOAA 3-Day Forecast HTTP ${res.status}`);
    const txt = await res.text();
    const ta = q55("noaaKpText");
    if (ta) ta.value = txt;
    if (status) status.textContent = "NOAA 3-Day Kp Forecast API取得OK。予報計算へ進みます…";
  }

  function patchForecastButtonV55() {
    if (!oldSwiftV52RunForecast && typeof window.swiftV52RunForecast === "function") {
      oldSwiftV52RunForecast = window.swiftV52RunForecast;
    }
    window.swiftV52RunForecast = async function () {
      try {
        if (typeof window.swiftV52SyncForecastControls === "function") window.swiftV52SyncForecastControls();
        await fetchNoaa3DayKpIfEnabledV55();
        if (oldSwiftV52RunForecast) return await oldSwiftV52RunForecast();
        const auto = q55("forecastTecAutoFetch");
        if (auto) auto.checked = true;
        return await window.runForecast?.();
      } catch (e) {
        console.error(e);
        const status = q55("swiftV52Status") || q55("swiftV55ForecastStatus");
        if (status) status.textContent = "予報失敗: " + e.message;
        try { window.logInfo?.("予報失敗: " + e.message); } catch {}
      }
    };
  }

  function bootV55() {
    injectStyleV55();
    for (const delay of [700, 1200, 2000]) {
      setTimeout(() => {
        ensurePointPanelV55();
        patchForecastButtonV55();
      }, delay);
    }
  }

  readyV55(bootV55);
})();


/* =========================================================
 * SWIFT-TEC v5.6 dynamic KpB/KpF label fix
 * - KpF is computed from the forecast Kp series by time, not only by slider index.
 * - KpB is fetched from NOAA Planetary K-index when forecasting, so it no longer
 *   silently copies KpF when Base Kp is blank.
 * - Adds a compact Kp status panel to the clean sidebar.
 * ========================================================= */
(function () {
  let originalUpdateKpLabelsV56 = null;
  let originalSwiftV52RunForecastV56 = null;

  function readyV56(fn) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn);
    else setTimeout(fn, 0);
  }

  function q56(id) { return document.getElementById(id); }
  function finite56(v) { return Number.isFinite(Number(v)); }
  function fmt56(v, d = 2) { return finite56(v) ? Number(v).toFixed(d) : "--"; }
  function iso56(t) {
    return (t instanceof Date && !isNaN(t.getTime())) ? t.toISOString().replace(".000Z", "Z") : "--";
  }

  function injectStyleV56() {
    if (q56("swiftKpLabelStyleV56")) return;
    const st = document.createElement("style");
    st.id = "swiftKpLabelStyleV56";
    st.textContent = `
      .swift-v56-kp-card {
        margin-top: 10px;
        border: 1px solid #1f355a;
        border-radius: 13px;
        background: #061020;
        padding: 9px;
      }
      .swift-v56-kp-title {
        display:flex;
        justify-content:space-between;
        align-items:center;
        font-size:11px;
        font-weight:850;
        color:#eaf2ff;
      }
      .swift-v56-kp-grid {
        display:grid;
        grid-template-columns:1fr 1fr;
        gap:6px;
        margin-top:8px;
      }
      .swift-v56-kp-metric {
        border:1px solid #1b2f50;
        border-radius:10px;
        background:#030712;
        padding:7px;
      }
      .swift-v56-kp-label {
        font-size:9px;
        color:#8ea3c4;
      }
      .swift-v56-kp-value {
        margin-top:3px;
        font-size:18px;
        font-weight:900;
        color:#f8fafc;
      }
      .swift-v56-kp-note {
        color:#8ba0c2;
        font-size:9px;
        line-height:1.35;
        margin-top:6px;
      }
      .swift-v56-kp-steps {
        margin-top:7px;
        white-space:pre-wrap;
        font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size:9px;
        color:#b8c7df;
        max-height:70px;
        overflow:auto;
        border-top:1px solid #1f355a;
        padding-top:6px;
      }
    `;
    document.head.appendChild(st);
  }

  function kpFromForecastSeriesAtV56(t) {
    if (!(t instanceof Date) || isNaN(t.getTime()) || !Array.isArray(gKpSeries) || !gKpSeries.length) return NaN;
    // Exact/nearest by time. Kp forecast is 3-hourly, but gKpSeries is expanded to each forecast step.
    let best = null;
    let bestDiff = Infinity;
    for (const r of gKpSeries) {
      if (!r || !(r.t instanceof Date) || isNaN(r.t.getTime())) continue;
      const diff = Math.abs(r.t.getTime() - t.getTime());
      if (diff < bestDiff) {
        bestDiff = diff;
        best = r;
      }
    }
    return finite56(best?.kp) ? Number(best.kp) : NaN;
  }

  function kpBaseAtV56(t) {
    try {
      const v = todValueAt(gBaseKpTod, t);
      return finite56(v) ? Number(v) : NaN;
    } catch {
      return NaN;
    }
  }

  function nextKpChangeV56(t) {
    if (!(t instanceof Date) || isNaN(t.getTime()) || !Array.isArray(gKpSeries) || !gKpSeries.length) return null;
    const nowKp = kpFromForecastSeriesAtV56(t);
    const future = gKpSeries
      .filter(r => r?.t instanceof Date && !isNaN(r.t.getTime()) && r.t.getTime() > t.getTime())
      .sort((a, b) => a.t - b.t);
    for (const r of future) {
      if (Math.abs(Number(r.kp) - nowKp) > 0.001) return r;
    }
    return null;
  }

  function updateKpPanelV56(t, kpF, kpB) {
    const set = (id, val) => { const el = q56(id); if (el) el.textContent = val; };
    set("swiftV56KpF", fmt56(kpF));
    set("swiftV56KpB", fmt56(kpB));
    set("swiftV56KpTime", iso56(t));
    const next = nextKpChangeV56(t);
    set("swiftV56KpNext", next ? `${iso56(next.t)} / KpF=${fmt56(next.kp)}` : "--");
    const diff = finite56(kpF) && finite56(kpB) ? Number(kpF) - Number(kpB) : NaN;
    set("swiftV56KpDiff", finite56(diff) ? `${diff >= 0 ? "+" : ""}${diff.toFixed(2)}` : "--");

    const steps = q56("swiftV56KpSteps");
    if (steps && Array.isArray(gKpSeries) && gKpSeries.length) {
      const rows = gKpSeries
        .filter(r => r?.t instanceof Date && !isNaN(r.t.getTime()) && Math.abs(r.t.getTime() - t.getTime()) <= 9 * 3600000)
        .filter((r, idx, a) => {
          if (idx === 0) return true;
          const prev = a[idx - 1];
          return Math.floor(prev.t.getTime() / (3 * 3600000)) !== Math.floor(r.t.getTime() / (3 * 3600000));
        })
        .slice(0, 8)
        .map(r => `${iso56(r.t).slice(11, 16)}  KpF=${fmt56(r.kp)}`);
      steps.textContent = rows.length ? rows.join("\n") : "Kp forecast series not ready";
    }
  }

  function patchedUpdateKpLabelsV56() {
    const t = Array.isArray(gForecastTimes) ? gForecastTimes[currentStepIndex] : null;
    if (!(t instanceof Date) || isNaN(t.getTime())) {
      if (originalUpdateKpLabelsV56) return originalUpdateKpLabelsV56();
      return;
    }

    const kpF = kpFromForecastSeriesAtV56(t);
    const kpB = kpBaseAtV56(t);
    const el = q56("kpNowLabel");
    if (el) el.textContent = `KpF=${fmt56(kpF)} / KpB=${fmt56(kpB)}`;
    updateKpPanelV56(t, kpF, kpB);
  }

  async function fetchBaseKpIfBlankV56() {
    // The old behavior uses KpF as KpB when Base Kp is blank, so both labels become identical.
    // Fetch actual NOAA 1-day K-index before forecasting to make KpB an independent base input.
    const ta = q56("baseKpJson");
    if (ta && String(ta.value || "").trim().length > 8) return;
    if (typeof window.fetchNoaaPlanetaryKIndex1DayToBase === "function") {
      const status = q56("swiftV52Status");
      if (status) status.textContent = "Base用Kp（NOAA K-index 1日分）を取得中…";
      await window.fetchNoaaPlanetaryKIndex1DayToBase();
    }
  }

  function installKpPanelV56() {
    const side = q56("swiftAccuracySide");
    if (!side || q56("swiftV56KpCard")) return;
    const card = document.createElement("div");
    card.id = "swiftV56KpCard";
    card.className = "swift-v56-kp-card";
    card.innerHTML = `
      <div class="swift-v56-kp-title">
        <span>Kp 現在値</span>
        <span class="swift-v52-pill">KpF / KpB</span>
      </div>
      <div class="swift-v56-kp-grid">
        <div class="swift-v56-kp-metric">
          <div class="swift-v56-kp-label">KpF 予報</div>
          <div class="swift-v56-kp-value" id="swiftV56KpF">--</div>
        </div>
        <div class="swift-v56-kp-metric">
          <div class="swift-v56-kp-label">KpB Base</div>
          <div class="swift-v56-kp-value" id="swiftV56KpB">--</div>
        </div>
        <div class="swift-v56-kp-metric">
          <div class="swift-v56-kp-label">差 KpF-KpB</div>
          <div class="swift-v56-kp-value" id="swiftV56KpDiff">--</div>
        </div>
        <div class="swift-v56-kp-metric">
          <div class="swift-v56-kp-label">次のKpF変化</div>
          <div class="swift-v56-kp-value" style="font-size:12px;" id="swiftV56KpNext">--</div>
        </div>
      </div>
      <div class="swift-v56-kp-note">
        KpFはNOAA 3-Day Forecast、KpBはBase抽出用Kpです。KpFは3時間単位なので、30分スライダーでは6コマごとに変わります。
      </div>
      <div class="swift-v56-kp-note">時刻: <span id="swiftV56KpTime">--</span></div>
      <div class="swift-v56-kp-steps" id="swiftV56KpSteps">予報実行後に表示</div>
    `;
    const point = q56("swiftV55PointCard");
    if (point) point.insertAdjacentElement("beforebegin", card);
    else side.appendChild(card);
  }

  function installKpLabelPatchV56() {
    if (!originalUpdateKpLabelsV56 && typeof updateKpLabels === "function") {
      originalUpdateKpLabelsV56 = updateKpLabels;
      updateKpLabels = patchedUpdateKpLabelsV56;
    }

    if (!originalSwiftV52RunForecastV56 && typeof window.swiftV52RunForecast === "function") {
      originalSwiftV52RunForecastV56 = window.swiftV52RunForecast;
      window.swiftV52RunForecast = async function () {
        try {
          await fetchBaseKpIfBlankV56();
        } catch (e) {
          console.warn("Base Kp auto fetch failed:", e);
        }
        const result = await originalSwiftV52RunForecastV56();
        setTimeout(patchedUpdateKpLabelsV56, 80);
        return result;
      };
    }
  }

  function bootV56() {
    injectStyleV56();
    for (const delay of [500, 1000, 1800]) {
      setTimeout(() => {
        installKpPanelV56();
        installKpLabelPatchV56();
        try { patchedUpdateKpLabelsV56(); } catch {}
      }, delay);
    }
  }

  window.swiftV56UpdateKpLabels = patchedUpdateKpLabelsV56;
  readyV56(bootV56);
})();


/* =========================================================
 * SWIFT-TEC v5.7 model-rule note
 * ========================================================= */
(function () {
  function readyV57(fn) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn);
    else setTimeout(fn, 0);
  }
  function q57(id) { return document.getElementById(id); }

  function bootV57() {
    setTimeout(() => {
      const side = q57("swiftAccuracySide");
      if (!side || q57("swiftV57ModelRuleNote")) return;
      const note = document.createElement("div");
      note.id = "swiftV57ModelRuleNote";
      note.className = "swift-v56-kp-card";
      note.innerHTML = `
        <div class="swift-v56-kp-title">
          <span>AIモデルルール</span>
          <span class="swift-v52-pill">v5.7</span>
        </div>
        <div class="swift-v56-kp-note">
          BaseTEC = 前日TEC − F(KpB)<br>
          ForecastTEC = BaseTEC + F(KpF)<br>
          AI補正 = F(KpF) − F(KpB)
        </div>
      `;
      const kpCard = q57("swiftV56KpCard");
      if (kpCard) kpCard.insertAdjacentElement("afterend", note);
      else side.appendChild(note);
    }, 1400);
  }
  readyV57(bootV57);
})();


/* =========================================================
 * SWIFT-TEC v5.8 heatmap fullscreen button
 * UI-only patch.
 * ========================================================= */
(function () {
  function readyV58(fn) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn);
    else setTimeout(fn, 0);
  }

  function q58(id) { return document.getElementById(id); }

  function injectFullscreenStyleV58() {
    if (q58("swiftFullscreenStyleV58")) return;
    const style = document.createElement("style");
    style.id = "swiftFullscreenStyleV58";
    style.textContent = `
      .map-card {
        position: relative;
      }

      #swiftHeatmapFullscreenBtn {
        position: absolute;
        top: 12px;
        right: 12px;
        z-index: 1200;
        border: 1px solid rgba(96,165,250,.85);
        background: rgba(7, 14, 28, .92);
        color: #eaf2ff;
        border-radius: 12px;
        padding: 8px 11px;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: .02em;
        cursor: pointer;
        box-shadow: 0 10px 24px rgba(0,0,0,.35);
        backdrop-filter: blur(6px);
      }

      #swiftHeatmapFullscreenBtn:hover {
        background: rgba(29, 78, 216, .95);
        border-color: #93c5fd;
      }

      #swiftHeatmapFullscreenBtn .swift-fs-icon {
        font-size: 14px;
        margin-right: 4px;
      }

      .map-card:fullscreen {
        background: #020617;
        padding: 10px;
        display: flex !important;
        flex-direction: column;
      }

      .map-card:fullscreen #tecMap {
        min-height: 0 !important;
        height: 100% !important;
        flex: 1 1 auto !important;
        border-radius: 12px;
      }

      .map-card:fullscreen #swiftHeatmapFullscreenBtn {
        top: 18px;
        right: 18px;
      }

      .map-card.swift-fs-fallback {
        position: fixed !important;
        inset: 0 !important;
        z-index: 99999 !important;
        background: #020617 !important;
        padding: 10px !important;
        margin: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        display: flex !important;
        min-height: 100vh !important;
        border-radius: 0 !important;
      }

      .map-card.swift-fs-fallback #tecMap {
        width: 100% !important;
        height: 100% !important;
        min-height: 0 !important;
        flex: 1 1 auto !important;
      }

      body.swift-fs-lock {
        overflow: hidden !important;
      }
    `;
    document.head.appendChild(style);
  }

  function forceMapResizeV58() {
    for (const delay of [60, 180, 420, 900]) {
      setTimeout(() => {
        try { window.dispatchEvent(new Event("resize")); } catch {}
        try { window.requestDraw?.(); } catch {}
      }, delay);
    }
  }

  async function toggleHeatmapFullscreenV58() {
    const mapCard = document.querySelector(".map-card");
    if (!mapCard) return;

    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
        return;
      }
      if (mapCard.requestFullscreen) {
        await mapCard.requestFullscreen();
        return;
      }
    } catch (e) {
      console.warn("Fullscreen API failed; using fallback.", e);
    }

    // Fallback for browsers/settings where Fullscreen API is blocked.
    mapCard.classList.toggle("swift-fs-fallback");
    document.body.classList.toggle("swift-fs-lock", mapCard.classList.contains("swift-fs-fallback"));
    updateButtonLabelV58();
    forceMapResizeV58();
  }

  function updateButtonLabelV58() {
    const btn = q58("swiftHeatmapFullscreenBtn");
    if (!btn) return;
    const fallback = document.querySelector(".map-card")?.classList.contains("swift-fs-fallback");
    const on = !!document.fullscreenElement || fallback;
    btn.innerHTML = on
      ? '<span class="swift-fs-icon">↙</span> 通常表示'
      : '<span class="swift-fs-icon">⛶</span> 全画面';
  }

  function ensureFullscreenButtonV58() {
    const mapCard = document.querySelector(".map-card");
    if (!mapCard || q58("swiftHeatmapFullscreenBtn")) return;

    const btn = document.createElement("button");
    btn.id = "swiftHeatmapFullscreenBtn";
    btn.type = "button";
    btn.innerHTML = '<span class="swift-fs-icon">⛶</span> 全画面';
    btn.title = "ヒートマップを全画面表示";
    btn.addEventListener("click", toggleHeatmapFullscreenV58);
    mapCard.appendChild(btn);

    document.addEventListener("fullscreenchange", () => {
      updateButtonLabelV58();
      forceMapResizeV58();
    });

    document.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape") {
        const mc = document.querySelector(".map-card");
        if (mc?.classList.contains("swift-fs-fallback")) {
          mc.classList.remove("swift-fs-fallback");
          document.body.classList.remove("swift-fs-lock");
          updateButtonLabelV58();
          forceMapResizeV58();
        }
      }
    });

    forceMapResizeV58();
  }

  function bootV58() {
    injectFullscreenStyleV58();
    for (const delay of [300, 900, 1600]) {
      setTimeout(ensureFullscreenButtonV58, delay);
    }
  }

  window.swiftToggleHeatmapFullscreen = toggleHeatmapFullscreenV58;
  readyV58(bootV58);
})();


/* =========================================================
 * SWIFT-TEC v5.9 fullscreen button visibility fix
 * The v5.8 button could be hidden behind the huge UTC overlay because
 * #timeOverlay uses z-index:9999. This patch pins the button inside
 * #tecMap with a higher z-index and lower position.
 * ========================================================= */
(function () {
  function readyV59(fn) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn);
    else setTimeout(fn, 0);
  }

  function q59(id) { return document.getElementById(id); }

  function injectStyleV59() {
    if (q59("swiftFullscreenVisibleStyleV59")) return;
    const style = document.createElement("style");
    style.id = "swiftFullscreenVisibleStyleV59";
    style.textContent = `
      #tecMap {
        position: relative !important;
      }

      #swiftHeatmapFullscreenBtn {
        position: absolute !important;
        top: 92px !important;
        right: 18px !important;
        z-index: 10080 !important;
        display: inline-flex !important;
        align-items: center;
        gap: 4px;
        border: 1px solid rgba(96,165,250,.95) !important;
        background: rgba(7, 14, 28, .94) !important;
        color: #eaf2ff !important;
        border-radius: 12px !important;
        padding: 9px 12px !important;
        font-size: 12px !important;
        font-weight: 850 !important;
        letter-spacing: .02em;
        cursor: pointer !important;
        box-shadow: 0 10px 24px rgba(0,0,0,.42);
        backdrop-filter: blur(6px);
        pointer-events: auto !important;
      }

      #swiftHeatmapFullscreenBtn:hover {
        background: rgba(29, 78, 216, .97) !important;
        border-color: #bfdbfe !important;
      }

      #swiftHeatmapFullscreenBtn .swift-fs-icon {
        font-size: 15px;
        line-height: 1;
      }

      #tecMap:fullscreen {
        background: #020617;
        padding: 10px;
      }

      #tecMap:fullscreen #swiftHeatmapFullscreenBtn {
        top: 92px !important;
        right: 22px !important;
      }

      #tecMap.swift-fs-fallback {
        position: fixed !important;
        inset: 0 !important;
        z-index: 99999 !important;
        background: #020617 !important;
        padding: 10px !important;
        margin: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        min-height: 100vh !important;
        border-radius: 0 !important;
      }

      body.swift-fs-lock {
        overflow: hidden !important;
      }

      @media (max-width: 1100px) {
        #swiftHeatmapFullscreenBtn {
          top: 74px !important;
          right: 14px !important;
          padding: 8px 10px !important;
        }
      }
    `;
    document.head.appendChild(style);
  }

  function forceResizeV59() {
    for (const delay of [50, 160, 360, 800]) {
      setTimeout(() => {
        try { window.dispatchEvent(new Event("resize")); } catch {}
        try { window.requestDraw?.(); } catch {}
      }, delay);
    }
  }

  async function toggleV59() {
    const map = q59("tecMap");
    if (!map) return;

    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
        return;
      }
      if (map.requestFullscreen) {
        await map.requestFullscreen();
        return;
      }
    } catch (e) {
      console.warn("Fullscreen API failed; using fallback", e);
    }

    map.classList.toggle("swift-fs-fallback");
    document.body.classList.toggle("swift-fs-lock", map.classList.contains("swift-fs-fallback"));
    updateLabelV59();
    forceResizeV59();
  }

  function updateLabelV59() {
    const btn = q59("swiftHeatmapFullscreenBtn");
    if (!btn) return;
    const fallback = q59("tecMap")?.classList.contains("swift-fs-fallback");
    const on = !!document.fullscreenElement || fallback;
    btn.innerHTML = on
      ? '<span class="swift-fs-icon">↙</span> 通常表示'
      : '<span class="swift-fs-icon">⛶</span> 全画面';
  }

  function ensureButtonV59() {
    const map = q59("tecMap");
    if (!map) return;

    let btn = q59("swiftHeatmapFullscreenBtn");
    if (!btn) {
      btn = document.createElement("button");
      btn.id = "swiftHeatmapFullscreenBtn";
      btn.type = "button";
      btn.title = "ヒートマップを全画面表示";
      btn.innerHTML = '<span class="swift-fs-icon">⛶</span> 全画面';
      map.appendChild(btn);
    }

    btn.onclick = toggleV59;
    document.addEventListener("fullscreenchange", () => {
      updateLabelV59();
      forceResizeV59();
    });

    document.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape") {
        const mapEl = q59("tecMap");
        if (mapEl?.classList.contains("swift-fs-fallback")) {
          mapEl.classList.remove("swift-fs-fallback");
          document.body.classList.remove("swift-fs-lock");
          updateLabelV59();
          forceResizeV59();
        }
      }
    });

    updateLabelV59();
    forceResizeV59();
  }

  function bootV59() {
    injectStyleV59();
    for (const delay of [200, 700, 1400]) {
      setTimeout(ensureButtonV59, delay);
    }
  }

  window.swiftToggleHeatmapFullscreen = toggleV59;
  readyV59(bootV59);
})();


/* =========================================================
 * SWIFT-TEC v6.0 always-visible fullscreen button
 * Adds a second button in the slider controls row, so the fullscreen
 * control is visible even if the map overlay button is hidden by layout.
 * ========================================================= */
(function () {
  function readyV60(fn) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn);
    else setTimeout(fn, 0);
  }
  function q60(id) { return document.getElementById(id); }

  function injectStyleV60() {
    if (q60("swiftAlwaysFullscreenStyleV60")) return;
    const style = document.createElement("style");
    style.id = "swiftAlwaysFullscreenStyleV60";
    style.textContent = `
      #swiftHeatmapFullscreenTopBtn {
        display: inline-flex !important;
        align-items: center;
        gap: 4px;
        min-height: 24px;
        border-radius: 8px !important;
        padding: 4px 10px !important;
        box-shadow: 0 0 0 1px rgba(96,165,250,.28), 0 6px 14px rgba(0,0,0,.25);
        cursor: pointer !important;
      }
      #swiftHeatmapFullscreenTopBtn:hover {
        background: #2563eb !important;
      }
      #swiftHeatmapFullscreenBtn {
        top: 110px !important;
        right: 20px !important;
        z-index: 11000 !important;
      }
      #tecMap.swift-fs-fallback,
      .map-card.swift-fs-fallback {
        position: fixed !important;
        inset: 0 !important;
        z-index: 99999 !important;
        width: 100vw !important;
        height: 100vh !important;
        min-height: 100vh !important;
        background: #020617 !important;
        padding: 10px !important;
        margin: 0 !important;
      }
      body.swift-fs-lock {
        overflow: hidden !important;
      }
    `;
    document.head.appendChild(style);
  }

  function forceResizeV60() {
    for (const delay of [60, 180, 420, 900]) {
      setTimeout(() => {
        try { window.dispatchEvent(new Event("resize")); } catch {}
        try { window.requestDraw?.(); } catch {}
      }, delay);
    }
  }

  async function toggleV60() {
    const target = document.getElementById("tecMap") || document.querySelector(".map-card");
    if (!target) return;

    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
        return;
      }
      if (target.requestFullscreen) {
        await target.requestFullscreen();
        return;
      }
    } catch (e) {
      console.warn("Fullscreen API failed; using fallback", e);
    }

    target.classList.toggle("swift-fs-fallback");
    document.body.classList.toggle("swift-fs-lock", target.classList.contains("swift-fs-fallback"));
    updateLabelsV60();
    forceResizeV60();
  }

  function updateLabelsV60() {
    const target = document.getElementById("tecMap") || document.querySelector(".map-card");
    const on = !!document.fullscreenElement || !!target?.classList.contains("swift-fs-fallback");
    const labels = on
      ? { top: "↙ 通常表示", map: '<span class="swift-fs-icon">↙</span> 通常表示' }
      : { top: "⛶ ヒートマップ全画面", map: '<span class="swift-fs-icon">⛶</span> 全画面' };
    const topBtn = q60("swiftHeatmapFullscreenTopBtn");
    if (topBtn) topBtn.textContent = labels.top;
    const mapBtn = q60("swiftHeatmapFullscreenBtn");
    if (mapBtn) mapBtn.innerHTML = labels.map;
  }

  function ensureTopButtonV60() {
    let topBtn = q60("swiftHeatmapFullscreenTopBtn");
    if (!topBtn) {
      const sliderRows = document.querySelectorAll(".slider-card .row.small, .slider-card .slider-row");
      const row = sliderRows[sliderRows.length - 1] || document.querySelector(".slider-card");
      if (!row) return;
      topBtn = document.createElement("button");
      topBtn.id = "swiftHeatmapFullscreenTopBtn";
      topBtn.type = "button";
      topBtn.textContent = "⛶ ヒートマップ全画面";
      topBtn.style.fontWeight = "900";
      topBtn.style.borderColor = "#60a5fa";
      topBtn.style.background = "#1d4ed8";
      topBtn.style.color = "white";
      row.insertBefore(topBtn, row.firstChild);
    }
    topBtn.onclick = toggleV60;

    const mapBtn = q60("swiftHeatmapFullscreenBtn");
    if (mapBtn) mapBtn.onclick = toggleV60;

    document.addEventListener("fullscreenchange", () => {
      updateLabelsV60();
      forceResizeV60();
    });

    document.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape") {
        const target = document.getElementById("tecMap") || document.querySelector(".map-card");
        if (target?.classList.contains("swift-fs-fallback")) {
          target.classList.remove("swift-fs-fallback");
          document.body.classList.remove("swift-fs-lock");
          updateLabelsV60();
          forceResizeV60();
        }
      }
    });
    updateLabelsV60();
  }

  function bootV60() {
    injectStyleV60();
    for (const delay of [100, 400, 1000, 1800]) {
      setTimeout(ensureTopButtonV60, delay);
    }
  }

  window.swiftToggleHeatmapFullscreen = toggleV60;
  readyV60(bootV60);
})();


/* =========================================================
 * SWIFT-TEC v6.1 forced fullscreen heatmap
 * Does not depend on browser Fullscreen API. It fixes .map-card to the viewport.
 * This avoids cases where requestFullscreen is blocked or appears to do nothing.
 * ========================================================= */
(function () {
  function readyV61(fn) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn);
    else setTimeout(fn, 0);
  }

  function q61(id) { return document.getElementById(id); }

  function injectStyleV61() {
    if (q61("swiftForcedFullscreenStyleV61")) return;
    const style = document.createElement("style");
    style.id = "swiftForcedFullscreenStyleV61";
    style.textContent = `
      body.swift-map-fs-on {
        overflow: hidden !important;
      }

      body.swift-map-fs-on .map-card {
        position: fixed !important;
        inset: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        min-height: 100vh !important;
        z-index: 2147483000 !important;
        margin: 0 !important;
        padding: 10px !important;
        border-radius: 0 !important;
        background: #020617 !important;
        display: flex !important;
        flex-direction: column !important;
        box-shadow: none !important;
      }

      body.swift-map-fs-on #tecMap {
        width: 100% !important;
        height: 100% !important;
        min-height: 0 !important;
        flex: 1 1 auto !important;
        border-radius: 12px !important;
        overflow: hidden !important;
      }

      body.swift-map-fs-on .leaflet-container {
        width: 100% !important;
        height: 100% !important;
      }

      body.swift-map-fs-on #swiftHeatmapFullscreenBtn {
        display: inline-flex !important;
        position: absolute !important;
        top: 18px !important;
        right: 18px !important;
        z-index: 2147483640 !important;
        background: rgba(185, 28, 28, .95) !important;
        border: 1px solid #fecaca !important;
        color: white !important;
      }

      #swiftHeatmapFullscreenTopBtn {
        display: inline-flex !important;
        align-items: center !important;
        gap: 4px !important;
        min-height: 24px !important;
        border-radius: 8px !important;
        padding: 4px 10px !important;
        border-color: #60a5fa !important;
        background: #1d4ed8 !important;
        color: white !important;
        font-weight: 900 !important;
        cursor: pointer !important;
      }

      #swiftHeatmapFullscreenTopBtn:hover {
        background: #2563eb !important;
      }

      #swiftHeatmapFullscreenBtn {
        position: absolute !important;
        top: 110px !important;
        right: 20px !important;
        z-index: 11000 !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 4px !important;
        border-radius: 10px !important;
        padding: 8px 11px !important;
        border: 1px solid #60a5fa !important;
        background: rgba(7,14,28,.94) !important;
        color: #eaf2ff !important;
        font-weight: 850 !important;
        cursor: pointer !important;
      }
    `;
    document.head.appendChild(style);
  }

  function forceMapResizeV61() {
    for (const delay of [0, 80, 180, 360, 700, 1200]) {
      setTimeout(() => {
        try { window.dispatchEvent(new Event("resize")); } catch {}
        try { window.requestDraw?.(); } catch {}
      }, delay);
    }
  }

  function updateLabelsV61() {
    const on = document.body.classList.contains("swift-map-fs-on");
    const top = q61("swiftHeatmapFullscreenTopBtn");
    if (top) top.textContent = on ? "↙ 通常表示に戻す" : "⛶ ヒートマップ全画面";
    const map = q61("swiftHeatmapFullscreenBtn");
    if (map) map.innerHTML = on
      ? '<span class="swift-fs-icon">↙</span> 通常表示'
      : '<span class="swift-fs-icon">⛶</span> 全画面';
  }

  function toggleForcedFullscreenV61() {
    document.body.classList.toggle("swift-map-fs-on");
    updateLabelsV61();
    forceMapResizeV61();
  }

  function ensureButtonsV61() {
    const sliderCard = document.querySelector(".slider-card");
    let topBtn = q61("swiftHeatmapFullscreenTopBtn");
    if (!topBtn && sliderCard) {
      const row = sliderCard.querySelector(".row.small") || sliderCard.querySelector(".slider-row") || sliderCard;
      topBtn = document.createElement("button");
      topBtn.id = "swiftHeatmapFullscreenTopBtn";
      topBtn.type = "button";
      topBtn.textContent = "⛶ ヒートマップ全画面";
      row.insertBefore(topBtn, row.firstChild);
    }
    if (topBtn) {
      topBtn.onclick = toggleForcedFullscreenV61;
      topBtn.disabled = false;
    }

    const tecMap = q61("tecMap");
    let mapBtn = q61("swiftHeatmapFullscreenBtn");
    if (!mapBtn && tecMap) {
      mapBtn = document.createElement("button");
      mapBtn.id = "swiftHeatmapFullscreenBtn";
      mapBtn.type = "button";
      mapBtn.innerHTML = '<span class="swift-fs-icon">⛶</span> 全画面';
      tecMap.appendChild(mapBtn);
    }
    if (mapBtn) {
      mapBtn.onclick = toggleForcedFullscreenV61;
      mapBtn.disabled = false;
    }

    updateLabelsV61();
  }

  function bootV61() {
    injectStyleV61();
    for (const delay of [50, 250, 700, 1400, 2400]) {
      setTimeout(ensureButtonsV61, delay);
    }

    document.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape" && document.body.classList.contains("swift-map-fs-on")) {
        document.body.classList.remove("swift-map-fs-on");
        updateLabelsV61();
        forceMapResizeV61();
      }
    });
  }

  window.swiftForceHeatmapFullscreen = toggleForcedFullscreenV61;
  window.swiftToggleHeatmapFullscreen = toggleForcedFullscreenV61;
  readyV61(bootV61);
})();


/* =========================================================
 * SWIFT-TEC v6.2 failure analysis UI
 * Shows hit-rate by TEC error threshold and by Kp bin.
 * Purpose: find conditions where forecast misses.
 * ========================================================= */
(function () {
  let failPerfV62 = null;

  function readyV62(fn) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn);
    else setTimeout(fn, 0);
  }
  function q62(id) { return document.getElementById(id); }
  function pct62(v) {
    const n = Number(v);
    return Number.isFinite(n) ? `${(n * 100).toFixed(1)}%` : "--";
  }
  function num62(v, d = 2) {
    const n = Number(v);
    return Number.isFinite(n) ? n.toFixed(d) : "--";
  }

  function injectStyleV62() {
    if (q62("swiftFailAnalysisStyleV62")) return;
    const st = document.createElement("style");
    st.id = "swiftFailAnalysisStyleV62";
    st.textContent = `
      .swift-v62-card {
        background: rgba(7, 14, 28, .98);
        border: 1px solid #1f355a;
        border-radius: 14px;
        padding: 10px;
        margin-top: 10px;
      }
      .swift-v62-title {
        font-size: 13px;
        font-weight: 900;
        color: #eaf2ff;
        margin-bottom: 2px;
      }
      .swift-v62-sub {
        font-size: 10px;
        color: #9fb0cc;
        margin-bottom: 8px;
      }
      .swift-v62-grid {
        display: grid;
        grid-template-columns: minmax(280px, .85fr) minmax(360px, 1.15fr);
        gap: 10px;
      }
      .swift-v62-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 10px;
      }
      .swift-v62-table th,
      .swift-v62-table td {
        border: 1px solid #1f355a;
        padding: 5px 6px;
        text-align: right;
      }
      .swift-v62-table th:first-child,
      .swift-v62-table td:first-child {
        text-align: left;
      }
      .swift-v62-table th {
        background: #0b1730;
        color: #bcd0ee;
      }
      .swift-v62-bad { color: #fecaca; font-weight: 850; }
      .swift-v62-good { color: #bbf7d0; font-weight: 850; }
      .swift-v62-barbox {
        width: 100%;
        height: 9px;
        background: #020617;
        border: 1px solid #1f355a;
        border-radius: 99px;
        overflow: hidden;
      }
      .swift-v62-bar {
        height: 100%;
        background: #60a5fa;
        border-radius: 99px;
      }
      .swift-v62-kp-row {
        display: grid;
        grid-template-columns: 42px 1fr 58px 58px 70px;
        gap: 6px;
        align-items: center;
        font-size: 10px;
        color: #dbeafe;
        margin: 5px 0;
      }
      @media (max-width: 980px) {
        .swift-v62-grid { grid-template-columns: 1fr; }
      }
    `;
    document.head.appendChild(st);
  }

  async function loadFailPerfV62() {
    try {
      const res = await fetch("data/ai/kp_performance.json", { cache: "no-store" });
      if (!res.ok) throw new Error("HTTP " + res.status);
      failPerfV62 = await res.json();
    } catch (e) {
      console.warn("failure analysis load failed", e);
      failPerfV62 = null;
    }
    renderFailPanelV62();
  }

  function renderThresholdTableV62() {
    const th = failPerfV62?.thresholds || failPerfV62?.summary?.thresholds || {};
    const keys = ["5", "10", "15", "20"];
    return `<table class="swift-v62-table">
      <thead>
        <tr><th>閾値</th><th>raw Hit</th><th>AI Hit</th><th>改善</th><th>AI RMSE</th><th>N</th></tr>
      </thead>
      <tbody>
        ${keys.map(k => {
          const r = th[k] || {};
          const raw = Number(r.raw_hit_rate);
          const corr = Number(r.corrected_hit_rate);
          const imp = Number.isFinite(raw) && Number.isFinite(corr) ? corr - raw : NaN;
          const cls = Number.isFinite(imp) && imp < 0 ? "swift-v62-bad" : "swift-v62-good";
          return `<tr>
            <td>±${k} TECU</td>
            <td>${pct62(raw)}</td>
            <td><b>${pct62(corr)}</b></td>
            <td class="${cls}">${Number.isFinite(imp) ? (imp >= 0 ? "+" : "") + (imp * 100).toFixed(1) + "pt" : "--"}</td>
            <td>${num62(r.corrected_rmse)}</td>
            <td>${r.sample_count || 0}</td>
          </tr>`;
        }).join("")}
      </tbody>
    </table>`;
  }

  function renderKpBinsV62() {
    const bins = failPerfV62?.kp_bins || {};
    const keys = ["0-2", "2-3", "3-4", "4-5", "5-6", "6-7", "7+"];
    const rows = keys.map(k => {
      const r = bins[k] || {};
      const t = r.thresholds?.["5"] || r;
      const corr = Number(t.corrected_hit_rate);
      const n = Number(t.sample_count || 0);
      return { k, corr, n, rmse: t.corrected_rmse };
    });
    return `<div>
      <div class="swift-v62-sub">KpFごとの ±5 TECU 的中率。低いKp帯が外れやすい条件です。</div>
      ${rows.map(r => `<div class="swift-v62-kp-row">
        <div>Kp ${r.k}</div>
        <div class="swift-v62-barbox"><div class="swift-v62-bar" style="width:${Math.max(0, Math.min(100, (r.corr || 0) * 100)).toFixed(1)}%"></div></div>
        <div>${pct62(r.corr)}</div>
        <div>N=${r.n}</div>
        <div>RMSE ${num62(r.rmse)}</div>
      </div>`).join("")}
    </div>`;
  }

  function worstKpTextV62() {
    const bins = failPerfV62?.kp_bins || {};
    let worst = null;
    for (const [k, r] of Object.entries(bins)) {
      const t = r.thresholds?.["5"] || r;
      const n = Number(t.sample_count || 0);
      const hit = Number(t.corrected_hit_rate);
      if (n < 100 || !Number.isFinite(hit)) continue;
      if (!worst || hit < worst.hit) worst = { k, hit, n, rmse: t.corrected_rmse };
    }
    if (!worst) return "外れやすいKp帯はまだ判定できません。学習データが増えると表示が安定します。";
    return `現時点で一番外れやすいKp帯: Kp ${worst.k} / Hit ${pct62(worst.hit)} / RMSE ${num62(worst.rmse)} / N=${worst.n}`;
  }

  function ensureFailPanelV62() {
    const main = q62("swiftAccuracyMain");
    if (!main || q62("swiftFailureAnalysisV62")) return null;
    const panel = document.createElement("div");
    panel.id = "swiftFailureAnalysisV62";
    panel.className = "swift-v62-card";
    panel.innerHTML = `<div class="swift-v62-title">外れやすさ分析</div>
      <div class="swift-v62-sub">TEC誤差閾値別・Kp帯別に的中率を見ることで、どの条件で外れるかを確認します。</div>
      <div id="swiftFailureAnalysisBodyV62">読み込み中…</div>`;
    main.appendChild(panel);
    return panel;
  }

  function renderFailPanelV62() {
    ensureFailPanelV62();
    const body = q62("swiftFailureAnalysisBodyV62");
    if (!body) return;
    if (!failPerfV62) {
      body.innerHTML = `<div class="swift-v62-sub">まだ kp_performance.json を読めません。Train Kp AI Corrector を実行してください。</div>`;
      return;
    }
    body.innerHTML = `<div class="swift-v62-grid">
      <div>
        <div class="swift-v62-title">閾値別 的中率</div>
        ${renderThresholdTableV62()}
      </div>
      <div>
        <div class="swift-v62-title">Kp別 的中率</div>
        ${renderKpBinsV62()}
        <div class="swift-v62-sub" style="margin-top:8px;">${worstKpTextV62()}</div>
      </div>
    </div>`;
  }

  function bootV62() {
    injectStyleV62();
    for (const delay of [800, 1500, 2600]) {
      setTimeout(() => {
        ensureFailPanelV62();
        if (!failPerfV62) loadFailPerfV62();
      }, delay);
    }
  }

  window.swiftLoadFailureAnalysis = loadFailPerfV62;
  readyV62(bootV62);
})();


/* =========================================================
 * SWIFT-TEC v6.3 map focus controls
 * Keeps timeline slider and playback controls visible in map-focus mode.
 * ========================================================= */
(function () {
  function readyV63(fn) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn);
    else setTimeout(fn, 0);
  }
  function q63(id) { return document.getElementById(id); }

  function forceResizeV63() {
    for (const delay of [0, 80, 180, 360, 800]) {
      setTimeout(() => {
        try { window.dispatchEvent(new Event("resize")); } catch {}
        try { window.requestDraw?.(); } catch {}
      }, delay);
    }
  }

  function updateFocusButtonV63() {
    const on = document.documentElement.classList.contains("swift-map-focus");
    const b = q63("swiftMapFocusButton");
    if (b) {
      b.textContent = on ? "↙ 通常表示" : "⛶ 地図だけ拡大";
      b.onclick = on ? window.swiftExitMapFocusMode : window.swiftEnterMapFocusMode;
    }
  }

  function patchFocusFunctionsV63() {
    const oldEnter = window.swiftEnterMapFocusMode;
    const oldExit = window.swiftExitMapFocusMode;
    const oldToggle = window.swiftToggleMapFocusMode;

    window.swiftEnterMapFocusMode = function () {
      if (typeof oldEnter === "function") oldEnter();
      else document.documentElement.classList.add("swift-map-focus");
      updateFocusButtonV63();
      forceResizeV63();
    };

    window.swiftExitMapFocusMode = function () {
      if (typeof oldExit === "function") oldExit();
      else document.documentElement.classList.remove("swift-map-focus");
      updateFocusButtonV63();
      forceResizeV63();
    };

    window.swiftToggleMapFocusMode = function () {
      if (typeof oldToggle === "function") oldToggle();
      else document.documentElement.classList.toggle("swift-map-focus");
      updateFocusButtonV63();
      forceResizeV63();
    };
  }

  function bootV63() {
    patchFocusFunctionsV63();
    updateFocusButtonV63();

    const slider = q63("timeSlider");
    if (slider) {
      slider.addEventListener("input", () => {
        if (document.documentElement.classList.contains("swift-map-focus")) forceResizeV63();
      });
    }

    document.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape") {
        setTimeout(updateFocusButtonV63, 0);
      }
    });
  }

  readyV63(bootV63);
})();


/* =========================================================
 * SWIFT-TEC v6.4 persistent timeline dock + GNSS load helper
 * - Shows a floating timeline dock when forecast heatmap exists or map focus is on.
 * - Timeline dock keeps slider/play/stop/speed usable even if the original slider-card is hidden.
 * - GNSS load uses local data/gnss first, then CelesTrak live fallback in the core loader.
 * ========================================================= */
(function () {
  function readyV64(fn) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn);
    else setTimeout(fn, 0);
  }

  function q64(id) { return document.getElementById(id); }

  function injectStyleV64() {
    if (q64("swiftTimelineDockStyleV64")) return;
    const st = document.createElement("style");
    st.id = "swiftTimelineDockStyleV64";
    st.textContent = `
      #swiftTimelineDockV64 {
        position: fixed;
        left: 50%;
        bottom: 12px;
        transform: translateX(-50%);
        width: min(1180px, calc(100vw - 28px));
        z-index: 2147483642;
        display: none;
        flex-direction: column;
        gap: 6px;
        padding: 8px 10px;
        border: 1px solid rgba(96,165,250,.90);
        border-radius: 14px;
        background: rgba(3, 7, 18, .95);
        color: #eaf2ff;
        box-shadow: 0 18px 44px rgba(0,0,0,.58);
        backdrop-filter: blur(9px);
        font-size: 11px;
      }
      #swiftTimelineDockV64.swift-show {
        display: flex !important;
      }
      .swift-v64-timeline-row {
        display: flex;
        align-items: center;
        gap: 8px;
        min-width: 0;
        flex-wrap: nowrap;
      }
      #swiftTimelineSliderV64 {
        flex: 1 1 auto;
        min-width: 280px;
        accent-color: #60a5fa;
      }
      #swiftTimelineDockV64 button,
      #swiftTimelineDockV64 select {
        min-height: 24px;
        border-radius: 8px;
        border: 1px solid #334155;
        background: #0f172a;
        color: #dbeafe;
        padding: 3px 8px;
        font-size: 11px;
      }
      #swiftTimelineDockV64 button {
        cursor: pointer;
        font-weight: 750;
      }
      #swiftTimelineDockV64 button.primary {
        background: #1d4ed8;
        border-color: #60a5fa;
        color: #fff;
      }
      #swiftTimelineDockV64 button.danger {
        background: #7f1d1d;
        border-color: #fecaca;
        color: #fff;
      }
      .swift-v64-label {
        white-space: nowrap;
        color: #c8d8f2;
      }
      .swift-v64-muted {
        color: #8ba0c2;
        font-size: 10px;
      }
      html.swift-map-focus #swiftTimelineDockV64 {
        display: flex !important;
      }
      html.swift-map-focus .slider-card {
        display: none !important;
      }
      html.swift-map-focus .map-card {
        padding-bottom: 78px !important;
      }
      #swiftGnssLoadStatusV64 {
        margin-top: 6px;
        font-size: 10px;
        color: #9fb0cc;
        line-height: 1.35;
      }
      @media (max-width: 760px) {
        #swiftTimelineDockV64 {
          width: calc(100vw - 16px);
          bottom: 8px;
          padding: 7px;
          max-height: 180px;
          overflow: auto;
        }
        .swift-v64-timeline-row {
          flex-wrap: wrap;
        }
        #swiftTimelineSliderV64 {
          min-width: 160px;
        }
      }
    `;
    document.head.appendChild(st);
  }

  function forceResizeV64() {
    for (const delay of [0, 80, 180, 360, 800]) {
      setTimeout(() => {
        try { window.dispatchEvent(new Event("resize")); } catch {}
        try { window.requestDraw?.(); } catch {}
      }, delay);
    }
  }

  function hasForecastV64() {
    const slider = q64("timeSlider");
    const max = Number(slider?.max || 0);
    const utc = String(q64("utcLabel")?.textContent || "");
    if (max > 0 && utc && !utc.includes("--")) return true;
    try {
      if (typeof gForecastTimes !== "undefined" && Array.isArray(gForecastTimes) && gForecastTimes.length > 0) return true;
    } catch {}
    return false;
  }

  function nativeSliderVisibleV64() {
    const card = document.querySelector(".slider-card");
    if (!card) return false;
    const r = card.getBoundingClientRect();
    const style = getComputedStyle(card);
    if (style.display === "none" || style.visibility === "hidden" || Number(style.opacity) === 0) return false;
    return r.width > 50 && r.height > 20 && r.bottom > 0 && r.top < window.innerHeight;
  }

  function ensureDockV64() {
    if (q64("swiftTimelineDockV64")) return q64("swiftTimelineDockV64");
    const dock = document.createElement("div");
    dock.id = "swiftTimelineDockV64";
    dock.innerHTML = `
      <div class="swift-v64-timeline-row">
        <button class="danger" type="button" id="swiftV64ExitFocusBtn">↙ 通常表示</button>
        <span class="swift-v64-label">時間</span>
        <input id="swiftTimelineSliderV64" type="range" min="0" max="192" value="0">
        <span id="swiftTimelineFrameV64" class="swift-v64-label">frame --</span>
        <span id="swiftTimelineUtcV64" class="swift-v64-label">UTC: --</span>
        <span id="swiftTimelineKpV64" class="swift-v64-label">KpF=-- / KpB=--</span>
      </div>
      <div class="swift-v64-timeline-row">
        <button class="primary" type="button" id="swiftV64PlayBtn">▶ 再生</button>
        <button type="button" id="swiftV64StopBtn">⏸ 停止</button>
        <span class="swift-v64-muted">速度</span>
        <select id="swiftV64SpeedSelect">
          <option value="1">x1</option>
          <option value="2">x2</option>
          <option value="4">x4</option>
        </select>
        <span class="swift-v64-muted">表示</span>
        <select id="swiftV64MapModeSelect"></select>
        <button type="button" id="swiftV64HideDockBtn">下部バーを隠す</button>
      </div>
    `;
    document.body.appendChild(dock);

    q64("swiftTimelineSliderV64").addEventListener("input", () => {
      const native = q64("timeSlider");
      const v = q64("swiftTimelineSliderV64").value;
      if (native) {
        native.value = v;
        try { window.onSliderChange?.(); } catch {}
      }
      syncDockV64(true);
      forceResizeV64();
    });

    q64("swiftV64PlayBtn").onclick = () => {
      try { window.playArchiveMovie?.(); } catch {}
      showDockV64(true, true);
    };
    q64("swiftV64StopBtn").onclick = () => {
      try { window.stopArchiveMovie?.(); } catch {}
      showDockV64(true, true);
    };
    q64("swiftV64ExitFocusBtn").onclick = () => {
      try { window.swiftExitMapFocusMode?.(); } catch {}
      document.documentElement.classList.remove("swift-map-focus");
      showDockV64(hasForecastV64() && !nativeSliderVisibleV64(), true);
      forceResizeV64();
    };
    q64("swiftV64HideDockBtn").onclick = () => {
      dock.dataset.userHidden = "1";
      if (!document.documentElement.classList.contains("swift-map-focus")) showDockV64(false, true);
    };
    q64("swiftV64SpeedSelect").onchange = () => {
      const native = q64("movieSpeedSelect");
      if (native) native.value = q64("swiftV64SpeedSelect").value;
    };
    q64("swiftV64MapModeSelect").onchange = () => {
      const native = q64("mapModeSelect");
      if (native) {
        native.value = q64("swiftV64MapModeSelect").value;
        try { window.changeMapMode?.(); } catch {}
      }
      forceResizeV64();
    };

    return dock;
  }

  function populateMapModeV64() {
    const native = q64("mapModeSelect");
    const mini = q64("swiftV64MapModeSelect");
    if (!native || !mini || mini.options.length) return;
    for (const opt of native.options) {
      const o = document.createElement("option");
      o.value = opt.value;
      o.textContent = opt.textContent;
      mini.appendChild(o);
    }
  }

  function syncDockV64(keepUserValue = false) {
    ensureDockV64();
    populateMapModeV64();

    const nativeSlider = q64("timeSlider");
    const miniSlider = q64("swiftTimelineSliderV64");
    if (nativeSlider && miniSlider) {
      miniSlider.min = nativeSlider.min || "0";
      miniSlider.max = nativeSlider.max || "192";
      if (!keepUserValue) miniSlider.value = nativeSlider.value || "0";
    }

    const frame = q64("swiftTimelineFrameV64");
    if (frame && nativeSlider) frame.textContent = `frame ${nativeSlider.value || "0"} / ${nativeSlider.max || "--"}`;

    const utc = q64("swiftTimelineUtcV64");
    if (utc) utc.textContent = q64("utcLabel")?.textContent || "UTC: --";

    const kp = q64("swiftTimelineKpV64");
    if (kp) kp.textContent = q64("kpNowLabel")?.textContent || "KpF=-- / KpB=--";

    const nativeSpeed = q64("movieSpeedSelect");
    const miniSpeed = q64("swiftV64SpeedSelect");
    if (nativeSpeed && miniSpeed) miniSpeed.value = nativeSpeed.value || "1";

    const nativeMode = q64("mapModeSelect");
    const miniMode = q64("swiftV64MapModeSelect");
    if (nativeMode && miniMode) miniMode.value = nativeMode.value || "tec";
  }

  function showDockV64(show, force = false) {
    const dock = ensureDockV64();
    if (!force && dock.dataset.userHidden === "1" && !document.documentElement.classList.contains("swift-map-focus")) return;
    dock.classList.toggle("swift-show", !!show);
    if (show) syncDockV64();
  }

  function autoVisibilityV64() {
    const focus = document.documentElement.classList.contains("swift-map-focus");
    const forecast = hasForecastV64();
    const nativeVisible = nativeSliderVisibleV64();

    if (focus) {
      showDockV64(true, true);
      return;
    }

    if (forecast && !nativeVisible) {
      showDockV64(true);
    } else {
      showDockV64(false, true);
    }
  }

  function patchFocusFunctionsV64() {
    const oldEnter = window.swiftEnterMapFocusMode;
    const oldExit = window.swiftExitMapFocusMode;

    window.swiftEnterMapFocusMode = function () {
      if (typeof oldEnter === "function") oldEnter();
      else document.documentElement.classList.add("swift-map-focus");
      showDockV64(true, true);
      forceResizeV64();
    };

    window.swiftExitMapFocusMode = function () {
      if (typeof oldExit === "function") oldExit();
      else document.documentElement.classList.remove("swift-map-focus");
      autoVisibilityV64();
      forceResizeV64();
    };
  }

  function patchGnssButtonStatusV64() {
    const side = document.getElementById("swiftAccuracySide");
    if (!side || document.getElementById("swiftGnssLoadStatusV64")) return;
    const s = document.createElement("div");
    s.id = "swiftGnssLoadStatusV64";
    s.textContent = "GNSS: local data/gnss → 失敗時CelesTrak liveで読込";
    side.appendChild(s);
  }

  function bootV64() {
    injectStyleV64();
    ensureDockV64();
    patchFocusFunctionsV64();
    patchGnssButtonStatusV64();

    const slider = q64("timeSlider");
    if (slider) slider.addEventListener("input", () => setTimeout(syncDockV64, 0));

    // After forecast execution, the native slider may be pushed outside visible area.
    // Poll lightly so the dock appears exactly when needed.
    setInterval(() => {
      syncDockV64();
      autoVisibilityV64();
    }, 900);

    for (const delay of [300, 1000, 2000]) {
      setTimeout(() => {
        syncDockV64();
        autoVisibilityV64();
        patchGnssButtonStatusV64();
      }, delay);
    }
  }

  window.swiftShowTimelineDock = () => showDockV64(true, true);
  readyV64(bootV64);
})();


/* =========================================================
 * SWIFT-TEC v6.5 selected point 5-minute DOP graph
 * Adds selectable DOP / TECxDOP time series for clicked lat/lon.
 * Also adds a GPS Yuma almanac health loader where available.
 * ========================================================= */
(function () {
  let selectedPointV65 = null;
  let lastSeriesV65 = null;

  const METRICS_V65 = {
    count: "可視衛星数",
    gdop: "GDOP",
    pdop: "PDOP",
    hdop: "HDOP",
    vdop: "VDOP",
    tdop: "TDOP",
    tec: "TEC [TECU]",
    l1: "L1電離圏誤差 [m]",
    gdoptec: "GDOP × L1誤差 [m]",
    pdoptec: "PDOP × L1誤差 [m]",
    hdoptec: "HDOP × L1誤差 [m]",
    vdoptec: "VDOP × L1誤差 [m]",
    tdoptec: "TDOP × L1誤差 [m]",
  };

  function readyV65(fn) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn);
    else setTimeout(fn, 0);
  }
  function q65(id) { return document.getElementById(id); }
  function num65(v, d = 2) {
    const n = Number(v);
    return Number.isFinite(n) ? n.toFixed(d) : "--";
  }
  function parseLatLonFromPointInfoV65(text) {
    const m = String(text || "").match(/Clicked:\s*lat=([-\d.]+),\s*lon=([-\d.]+)/i);
    if (!m) return null;
    return { lat: Number(m[1]), lon: Number(m[2]) };
  }

  function injectStyleV65() {
    if (q65("swiftPointDopGraphStyleV65")) return;
    const st = document.createElement("style");
    st.id = "swiftPointDopGraphStyleV65";
    st.textContent = `
      .swift-v65-card {
        margin-top: 10px;
        border: 1px solid #1f355a;
        border-radius: 14px;
        background: rgba(7, 14, 28, .98);
        padding: 10px;
      }
      .swift-v65-title {
        color: #eaf2ff;
        font-size: 13px;
        font-weight: 900;
      }
      .swift-v65-sub {
        color: #9fb0cc;
        font-size: 10px;
        line-height: 1.35;
        margin-top: 3px;
      }
      .swift-v65-row {
        display: flex;
        align-items: center;
        gap: 7px;
        flex-wrap: wrap;
        margin-top: 8px;
      }
      .swift-v65-row > * {
        min-width: 0;
      }
      .swift-v65-select,
      .swift-v65-input {
        background: #061020;
        border: 1px solid #2a4774;
        color: #eaf2ff;
        border-radius: 9px;
        padding: 6px 8px;
        font-size: 11px;
      }
      .swift-v65-btn {
        border-radius: 9px;
        border: 1px solid #3b82f6;
        background: #1d4ed8;
        color: white;
        padding: 6px 9px;
        font-size: 11px;
        font-weight: 750;
        cursor: pointer;
      }
      .swift-v65-btn.secondary {
        border-color: #334155;
        background: #0f172a;
      }
      #swiftPointDopCanvasV65 {
        width: 100%;
        height: 185px;
        border: 1px solid #1f355a;
        border-radius: 12px;
        background: #030712;
        display: block;
        margin-top: 8px;
      }
      .swift-v65-kpi {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 6px;
        margin-top: 8px;
      }
      .swift-v65-mini {
        border: 1px solid #1f355a;
        border-radius: 10px;
        background: #061020;
        padding: 7px;
      }
      .swift-v65-mini-label {
        color: #8ba0c2;
        font-size: 9px;
      }
      .swift-v65-mini-value {
        color: #f8fafc;
        font-size: 14px;
        font-weight: 900;
        margin-top: 3px;
      }
    `;
    document.head.appendChild(st);
  }

  function ensurePanelV65() {
    const side = q65("swiftAccuracySide") || document.querySelector(".sidebar");
    if (!side || q65("swiftPointDopPanelV65")) return;

    const panel = document.createElement("div");
    panel.id = "swiftPointDopPanelV65";
    panel.className = "swift-v65-card";
    panel.innerHTML = `
      <div class="swift-v65-title">選択地点 5分DOPグラフ</div>
      <div class="swift-v65-sub">
        地図をクリックした緯度経度について、5分間隔でDOP/TEC×DOPを時系列表示します。
      </div>
      <div class="swift-v65-row">
        <select id="swiftPointDopMetricV65" class="swift-v65-select">
          ${Object.entries(METRICS_V65).map(([k, v]) => `<option value="${k}" ${k === "pdoptec" ? "selected" : ""}>${v}</option>`).join("")}
        </select>
        <select id="swiftPointDopStepV65" class="swift-v65-select">
          <option value="5" selected>5分</option>
          <option value="10">10分</option>
          <option value="15">15分</option>
          <option value="30">30分</option>
        </select>
        <button class="swift-v65-btn" onclick="window.swiftRenderPointDopGraphV65()">グラフ更新</button>\n        <button class="swift-v65-btn secondary" onclick="window.swiftExportPointDopExcelV66 && window.swiftExportPointDopExcelV66()">Excel出力</button>
      </div>
      <div class="swift-v65-sub" id="swiftPointDopSelectedV65">地点: 未選択。地図をクリックしてください。</div>
      <canvas id="swiftPointDopCanvasV65"></canvas>
      <div class="swift-v65-kpi">
        <div class="swift-v65-mini"><div class="swift-v65-mini-label">最小</div><div class="swift-v65-mini-value" id="swiftPointDopMinV65">--</div></div>
        <div class="swift-v65-mini"><div class="swift-v65-mini-label">平均</div><div class="swift-v65-mini-value" id="swiftPointDopAvgV65">--</div></div>
        <div class="swift-v65-mini"><div class="swift-v65-mini-label">最大</div><div class="swift-v65-mini-value" id="swiftPointDopMaxV65">--</div></div>
      </div>
      <div class="swift-v65-row">
        <button class="swift-v65-btn secondary" onclick="window.swiftLoadGpsYumaHealthV65()">GPS Almanac Health読込</button>
      </div>
      <div class="swift-v65-sub" id="swiftAlmanacStatusV65">
        Almanac healthはGPS Yumaを取得できる場合のみ反映。TLE運用グループは従来通りlocal→CelesTrakで読込。
      </div>
    `;
    side.appendChild(panel);

    const metric = q65("swiftPointDopMetricV65");
    const step = q65("swiftPointDopStepV65");
    if (metric) metric.addEventListener("change", () => window.swiftRenderPointDopGraphV65?.());
    if (step) step.addEventListener("change", () => window.swiftRenderPointDopGraphV65?.());
  }

  function drawSeriesV65(series) {
    const canvas = q65("swiftPointDopCanvasV65");
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(300, Math.floor(rect.width * dpr));
    const h = Math.max(150, Math.floor(rect.height * dpr));
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#030712";
    ctx.fillRect(0, 0, w, h);

    const rows = series?.rows || [];
    const vals = rows.map(r => Number(r.value)).filter(Number.isFinite);
    const metricLabel = METRICS_V65[series?.metric] || series?.metric || "metric";

    ctx.font = `${12 * dpr}px system-ui`;
    ctx.fillStyle = "#dbeafe";
    ctx.fillText(metricLabel + ` / ${series?.step_min || 5}分間隔`, 12 * dpr, 18 * dpr);

    if (!vals.length) {
      ctx.fillStyle = "#94a3b8";
      ctx.fillText("データなし。予報作成とGNSS読込後に地図をクリックしてください。", 14 * dpr, 55 * dpr);
      return;
    }

    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
    q65("swiftPointDopMinV65").textContent = num65(min);
    q65("swiftPointDopAvgV65").textContent = num65(avg);
    q65("swiftPointDopMaxV65").textContent = num65(max);

    const pad = { l: 42 * dpr, r: 12 * dpr, t: 28 * dpr, b: 24 * dpr };
    const x0 = pad.l, x1 = w - pad.r, y0 = pad.t, y1 = h - pad.b;
    const yMin = Math.min(0, min);
    const yMax = max === yMin ? yMin + 1 : max * 1.08;

    ctx.strokeStyle = "rgba(148,163,184,.22)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = y1 - (y1 - y0) * i / 4;
      ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
      ctx.fillStyle = "#73839e";
      ctx.font = `${9 * dpr}px system-ui`;
      ctx.fillText(num65(yMin + (yMax - yMin) * i / 4, 1), 5 * dpr, y + 3 * dpr);
    }

    const xAt = i => rows.length === 1 ? (x0 + x1) / 2 : x0 + (x1 - x0) * i / (rows.length - 1);
    const yAt = v => y1 - (y1 - y0) * ((v - yMin) / Math.max(1e-9, yMax - yMin));

    ctx.strokeStyle = "#60a5fa";
    ctx.lineWidth = 2 * dpr;
    ctx.beginPath();
    let started = false;
    rows.forEach((r, i) => {
      const v = Number(r.value);
      if (!Number.isFinite(v)) return;
      const x = xAt(i), y = yAt(v);
      if (!started) { ctx.moveTo(x, y); started = true; }
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.fillStyle = "#94a3b8";
    ctx.font = `${9 * dpr}px system-ui`;
    const first = rows[0]?.time || "--";
    const last = rows[rows.length - 1]?.time || "--";
    ctx.fillText(first.slice(11, 16), x0, h - 7 * dpr);
    ctx.fillText(last.slice(11, 16), x1 - 38 * dpr, h - 7 * dpr);
  }

  function renderPointDopGraphV65() {
    ensurePanelV65();
    const status = q65("swiftPointDopSelectedV65");
    if (!selectedPointV65) {
      if (status) status.textContent = "地点: 未選択。地図をクリックしてください。";
      drawSeriesV65({ rows: [] });
      return;
    }

    try {
      const metric = q65("swiftPointDopMetricV65")?.value || "pdoptec";
      const stepMin = Number(q65("swiftPointDopStepV65")?.value || 5);
      const series = window.swiftBuildPointDopSeries?.(selectedPointV65.lat, selectedPointV65.lon, { metric, stepMin });
      lastSeriesV65 = series;
      window.swiftPointDopLastSeriesV65 = series;
      if (status) {
        status.textContent = `地点: click lat=${selectedPointV65.lat.toFixed(3)}, lon=${selectedPointV65.lon.toFixed(3)} / grid lat=${num65(series.cell.lat)}, lon=${num65(series.cell.lon)} / GNSS使用=${series.gnss_active_selected}`;
      }
      drawSeriesV65(series);
    } catch (e) {
      console.error(e);
      if (status) status.textContent = "グラフ作成失敗: " + e.message;
      drawSeriesV65({ rows: [] });
    }
  }

  function observePointInfoV65() {
    const pre = q65("pointInfo");
    if (!pre) return;
    const update = () => {
      const p = parseLatLonFromPointInfoV65(pre.textContent || "");
      if (p && Number.isFinite(p.lat) && Number.isFinite(p.lon)) {
        selectedPointV65 = p;
        const el = q65("swiftPointDopSelectedV65");
        if (el) el.textContent = `地点: lat=${p.lat.toFixed(3)}, lon=${p.lon.toFixed(3)} / グラフ更新を押してください`;
        // Light auto refresh only if a previous graph exists.
        if (lastSeriesV65) setTimeout(renderPointDopGraphV65, 0);
      }
    };
    update();
    const mo = new MutationObserver(update);
    mo.observe(pre, { childList: true, characterData: true, subtree: true });
  }

  function parseYumaHealthV65(text) {
    const map = {};
    const blocks = String(text || "").split(/\n\s*\n/);
    for (const b of blocks) {
      const id = (b.match(/(?:ID|PRN)\s*:\s*(\d+)/i) || [])[1];
      const health = (b.match(/Health\s*:\s*([0-9]+)/i) || [])[1];
      if (id && health != null) map[String(Number(id)).padStart(2, "0")] = Number(health);
    }
    return map;
  }

  async function loadGpsYumaHealthV65() {
    const el = q65("swiftAlmanacStatusV65");
    if (el) el.textContent = "GPS Yuma almanac health取得中…";
    const urls = [
      "https://celestrak.org/GPS/almanac/Yuma/current.al3",
      "https://celestrak.org/GPS/almanac/Yuma/current.txt",
      "https://celestrak.org/GPS/almanac/Yuma/current.alm"
    ];
    let text = null;
    let used = null;
    for (const url of urls) {
      try {
        const res = await fetch(url, { cache: "no-store" });
        if (res.ok) {
          text = await res.text();
          used = url;
          break;
        }
      } catch {}
    }
    if (!text) {
      if (el) el.textContent = "GPS Almanac health取得失敗。CelesTrak Yuma current URLにアクセスできません。";
      return;
    }
    const map = parseYumaHealthV65(text);
    const n = Object.keys(map).length;
    if (!n) {
      if (el) el.textContent = "GPS Almanacは取得できたがHealth値を解析できませんでした。";
      return;
    }
    const result = window.swiftApplyGnssPrnHealthMap?.(map) || { applied: 0, inactive: 0 };
    if (el) el.textContent = `GPS Almanac health反映: PRN=${n}, 適用=${result.applied}, inactive=${result.inactive} / ${used}`;
    setTimeout(renderPointDopGraphV65, 0);
  }

  function bootV65() {
    injectStyleV65();
    for (const delay of [600, 1200, 2200]) {
      setTimeout(() => {
        ensurePanelV65();
        observePointInfoV65();
      }, delay);
    }
  }

  window.swiftRenderPointDopGraphV65 = renderPointDopGraphV65;
  window.swiftLoadGpsYumaHealthV65 = loadGpsYumaHealthV65;
  readyV65(bootV65);
})();


/* =========================================================
 * SWIFT-TEC v6.6 GNSS visible panel + saved almanac + Excel export
 * ========================================================= */
(function () {
  const GNSS_KEYS_V66 = [
    ["gps", "GPS"],
    ["qzss", "QZSS"],
    ["galileo", "Galileo"],
    ["glonass", "GLONASS"],
    ["beidou", "BeiDou"],
  ];

  function readyV66(fn) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn);
    else setTimeout(fn, 0);
  }
  function q66(id) { return document.getElementById(id); }

  function injectStyleV66() {
    if (q66("swiftGnssVisibleStyleV66")) return;
    const st = document.createElement("style");
    st.id = "swiftGnssVisibleStyleV66";
    st.textContent = `
      .swift-v66-card {
        margin-top: 10px;
        border: 1px solid #1f355a;
        border-radius: 14px;
        background: rgba(7, 14, 28, .98);
        padding: 10px;
      }
      .swift-v66-title {
        color: #eaf2ff;
        font-size: 13px;
        font-weight: 900;
      }
      .swift-v66-sub {
        color: #9fb0cc;
        font-size: 10px;
        line-height: 1.38;
        margin-top: 3px;
      }
      .swift-v66-row {
        display: flex;
        align-items: center;
        gap: 7px;
        flex-wrap: wrap;
        margin-top: 8px;
      }
      .swift-v66-btn {
        border-radius: 9px;
        border: 1px solid #3b82f6;
        background: #1d4ed8;
        color: white;
        padding: 6px 9px;
        font-size: 11px;
        font-weight: 750;
        cursor: pointer;
      }
      .swift-v66-btn.secondary {
        border-color: #334155;
        background: #0f172a;
      }
      .swift-v66-btn.warn {
        border-color: #f59e0b;
        background: #92400e;
      }
      .swift-v66-check {
        display: inline-flex;
        gap: 4px;
        align-items: center;
        border: 1px solid #1f355a;
        background: #061020;
        color: #dbeafe;
        border-radius: 999px;
        padding: 4px 8px;
        font-size: 10px;
      }
      #swiftGnssVisibleListV66 {
        max-height: 210px;
        overflow: auto;
        margin-top: 8px;
        border: 1px solid #1f355a;
        border-radius: 10px;
        padding: 5px;
        background: #020617;
      }
      #swiftGnssVisibleListV66 table {
        width: 100%;
      }
      #swiftGnssVisibleListV66 th,
      #swiftGnssVisibleListV66 td {
        font-size: 9px;
        padding: 3px 4px;
      }
      #swiftGnssVisibleStatusV66 {
        color: #c8d8f2;
        font-size: 10px;
        margin-top: 6px;
        line-height: 1.35;
      }
    `;
    document.head.appendChild(st);
  }

  function ensureVisibleGnssPanelV66() {
    const side = q66("swiftAccuracySide") || document.querySelector(".sidebar");
    if (!side || q66("swiftGnssVisiblePanelV66")) return;

    const panel = document.createElement("div");
    panel.id = "swiftGnssVisiblePanelV66";
    panel.className = "swift-v66-card";
    panel.innerHTML = `
      <div class="swift-v66-title">GNSS / DOP 可視パネル</div>
      <div class="swift-v66-sub">
        TLEは <b>satellite.js の twoline2satrec + propagate</b> でSGP4伝搬しています。
        表示されない原因は、旧GNSSカードが新UIで非表示になっていたためです。
      </div>
      <div class="swift-v66-row" id="swiftGnssConstBoxV66">
        ${GNSS_KEYS_V66.map(([k, label]) => `
          <label class="swift-v66-check"><input id="gnssConstV66_${k}" type="checkbox" ${k === "gps" || k === "qzss" ? "checked" : ""}>${label}</label>
        `).join("")}
      </div>
      <div class="swift-v66-row">
        <button class="swift-v66-btn" id="swiftGnssLoadBtnV66">GNSS TLE読込 / DOP準備</button>
        <button class="swift-v66-btn secondary" onclick="window.setAllSatSelected && window.setAllSatSelected(true)">全衛星使用</button>
        <button class="swift-v66-btn secondary" onclick="window.setAllSatSelected && window.setAllSatSelected(false)">全衛星解除</button>
        <button class="swift-v66-btn secondary" onclick="window.setAllSatActive && window.setAllSatActive(true)">全Active</button>
        <button class="swift-v66-btn secondary" onclick="window.setAllSatActive && window.setAllSatActive(false)">全Inactive</button>
      </div>
      <div class="swift-v66-row">
        <button class="swift-v66-btn warn" id="swiftApplySavedAlmanacBtnV66">保存済Almanac Health反映</button>
        <button class="swift-v66-btn secondary" onclick="window.swiftLoadGpsYumaHealthV65 && window.swiftLoadGpsYumaHealthV65()">Live Yuma Health取得</button>
      </div>
      <div id="swiftGnssVisibleStatusV66">GNSS未読込</div>
      <div id="swiftGnssVisibleListV66"></div>
    `;
    side.appendChild(panel);

    q66("swiftGnssLoadBtnV66").onclick = async () => {
      const st = q66("swiftGnssVisibleStatusV66");
      if (st) st.textContent = "GNSS TLE読込中…";
      try {
        await window.loadGnssDopData?.();
        await applySavedAlmanacHealthV66(false);
      } finally {
        relocateSatelliteListV66();
        updateVisibleGnssStatusV66();
      }
    };
    q66("swiftApplySavedAlmanacBtnV66").onclick = () => applySavedAlmanacHealthV66(true);

    for (const [k] of GNSS_KEYS_V66) {
      q66(`gnssConstV66_${k}`)?.addEventListener("change", () => updateVisibleGnssStatusV66());
    }
  }

  function relocateSatelliteListV66() {
    const holder = q66("swiftGnssVisibleListV66");
    if (!holder) return;
    let list = q66("satelliteSelectionList");
    if (!list) {
      list = document.createElement("div");
      list.id = "satelliteSelectionList";
      list.innerHTML = '<div class="small">未読込。GNSS TLE読込を押してください。</div>';
    }
    if (list.parentElement !== holder) holder.appendChild(list);

    let quick = q66("gnssQuickStatus");
    if (!quick) {
      quick = document.createElement("div");
      quick.id = "gnssQuickStatus";
      quick.className = "small";
    }
    const st = q66("swiftGnssVisibleStatusV66");
    if (st && quick.parentElement !== st) st.appendChild(quick);
  }

  function updateVisibleGnssStatusV66(extra = "") {
    const st = q66("swiftGnssVisibleStatusV66");
    if (!st) return;
    const quick = q66("gnssQuickStatus")?.textContent || "";
    const chosen = GNSS_KEYS_V66
      .filter(([k]) => q66(`gnssConstV66_${k}`)?.checked)
      .map(([, label]) => label)
      .join(", ");
    st.textContent = `選択: ${chosen || "なし"} / ${quick || "未読込"}${extra ? " / " + extra : ""}`;
  }

  async function applySavedAlmanacHealthV66(showMissing) {
    const st = q66("swiftGnssVisibleStatusV66");
    try {
      const res = await fetch("data/gnss/gps_almanac_health.json", { cache: "no-store" });
      if (!res.ok) {
        if (showMissing && st) st.textContent = "保存済Almanac Healthなし。先にDaily GNSS Almanac workflowを実行してください。";
        return;
      }
      const doc = await res.json();
      const map = doc.health_by_prn || doc.health || {};
      const result = window.swiftApplyGnssPrnHealthMap?.(map) || { applied: 0, inactive: 0 };
      relocateSatelliteListV66();
      updateVisibleGnssStatusV66(`Almanac ${doc.updated_utc || ""} / applied=${result.applied}, inactive=${result.inactive}`);
    } catch (e) {
      if (st) st.textContent = "保存済Almanac Health反映失敗: " + e.message;
    }
  }

  function loadSheetJsV66() {
    return new Promise((resolve, reject) => {
      if (window.XLSX) return resolve();
      const s = document.createElement("script");
      s.src = "https://cdn.jsdelivr.net/npm/xlsx@0.18.5/dist/xlsx.full.min.js";
      s.async = true;
      s.onload = () => resolve();
      s.onerror = () => reject(new Error("SheetJSの読み込みに失敗しました。"));
      document.head.appendChild(s);
    });
  }

  function finiteOrBlank(v) {
    const n = Number(v);
    return Number.isFinite(n) ? n : "";
  }

  async function exportPointDopExcelV66() {
    // If graph was not updated yet, try to build it once.
    if (!window.swiftPointDopLastSeriesV65 && window.swiftRenderPointDopGraphV65) {
      window.swiftRenderPointDopGraphV65();
      await new Promise(r => setTimeout(r, 150));
    }
    const series = window.swiftPointDopLastSeriesV65;
    if (!series || !Array.isArray(series.rows) || !series.rows.length) {
      alert("出力するグラフデータがありません。地図をクリックして「グラフ更新」を押してください。");
      return;
    }

    await loadSheetJsV66();

    const rows = series.rows.map(r => ({
      time_utc: r.time,
      selected_lat: finiteOrBlank(r.selected_lat),
      selected_lon: finiteOrBlank(r.selected_lon),
      grid_lat: finiteOrBlank(r.lat),
      grid_lon: finiteOrBlank(r.lon),
      tec_tecu: finiteOrBlank(r.tec),
      l1_error_m: finiteOrBlank(r.l1),
      visible_sat_count: finiteOrBlank(r.count),
      gdop: finiteOrBlank(r.gdop),
      pdop: finiteOrBlank(r.pdop),
      hdop: finiteOrBlank(r.hdop),
      vdop: finiteOrBlank(r.vdop),
      tdop: finiteOrBlank(r.tdop),
      gdop_x_l1_m: finiteOrBlank(r.gdoptec),
      pdop_x_l1_m: finiteOrBlank(r.pdoptec),
      hdop_x_l1_m: finiteOrBlank(r.hdoptec),
      vdop_x_l1_m: finiteOrBlank(r.vdoptec),
      tdop_x_l1_m: finiteOrBlank(r.tdoptec),
    }));

    const vals = series.rows.map(r => Number(r.value)).filter(Number.isFinite);
    const summary = [
      ["metric", series.metric],
      ["step_min", series.step_min],
      ["start_utc", series.start_utc],
      ["end_utc", series.end_utc],
      ["selected_lat", series.rows[0]?.selected_lat],
      ["selected_lon", series.rows[0]?.selected_lon],
      ["grid_lat", series.cell?.lat],
      ["grid_lon", series.cell?.lon],
      ["gnss_total", series.gnss_total],
      ["gnss_active_selected", series.gnss_active_selected],
      ["elevation_mask_deg", series.elevation_mask_deg],
      ["min", vals.length ? Math.min(...vals) : ""],
      ["avg", vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : ""],
      ["max", vals.length ? Math.max(...vals) : ""],
    ];

    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(summary), "Summary");
    XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(rows), "Point 5min Series");

    const lat = Number(series.rows[0]?.selected_lat || 0).toFixed(2);
    const lon = Number(series.rows[0]?.selected_lon || 0).toFixed(2);
    const fname = `swifttec_point_dop_${series.metric || "metric"}_${lat}_${lon}.xlsx`;
    XLSX.writeFile(wb, fname);
  }

  function patchV65PanelExportButton() {
    const panel = q66("swiftPointDopPanelV65");
    if (!panel || q66("swiftPointDopExcelBtnV66")) return;
    const row = panel.querySelector(".swift-v65-row");
    if (!row) return;
    const btn = document.createElement("button");
    btn.id = "swiftPointDopExcelBtnV66";
    btn.className = "swift-v65-btn secondary";
    btn.type = "button";
    btn.textContent = "Excel出力";
    btn.onclick = exportPointDopExcelV66;
    row.appendChild(btn);
  }

  function patchGnssLoadWrapperV66() {
    if (window.__swiftGnssLoadWrappedV66) return;
    const old = window.loadGnssDopData;
    if (typeof old !== "function") return;
    window.__swiftGnssLoadWrappedV66 = true;
    window.loadGnssDopData = async function () {
      const out = await old.apply(this, arguments);
      relocateSatelliteListV66();
      await applySavedAlmanacHealthV66(false);
      updateVisibleGnssStatusV66();
      return out;
    };
  }

  function bootV66() {
    injectStyleV66();
    for (const delay of [300, 900, 1600, 2600]) {
      setTimeout(() => {
        ensureVisibleGnssPanelV66();
        relocateSatelliteListV66();
        patchV65PanelExportButton();
        patchGnssLoadWrapperV66();
        updateVisibleGnssStatusV66();
      }, delay);
    }
  }

  window.swiftExportPointDopExcelV66 = exportPointDopExcelV66;
  window.swiftApplySavedAlmanacHealthV66 = applySavedAlmanacHealthV66;
  readyV66(bootV66);
})();


/* =========================================================
 * SWIFT-TEC v6.7 robust GNSS loader diagnostics
 * Adds read test and improves status when TLE exists but UI does not reflect it.
 * ========================================================= */
(function () {
  function readyV67(fn) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn);
    else setTimeout(fn, 0);
  }
  function q67(id) { return document.getElementById(id); }

  function injectStyleV67() {
    if (q67("swiftGnssDebugStyleV67")) return;
    const st = document.createElement("style");
    st.id = "swiftGnssDebugStyleV67";
    st.textContent = `
      #swiftGnssDebugBoxV67 {
        margin-top: 8px;
        border: 1px solid #1f355a;
        border-radius: 10px;
        background: #020617;
        padding: 6px;
        max-height: 210px;
        overflow: auto;
        color: #c8d8f2;
        font-size: 9px;
        white-space: pre-wrap;
        line-height: 1.35;
      }
      .swift-v67-ok { color: #86efac; font-weight: 800; }
      .swift-v67-ng { color: #fecaca; font-weight: 800; }
    `;
    document.head.appendChild(st);
  }

  function ensureDebugUiV67() {
    const panel = q67("swiftGnssVisiblePanelV66");
    if (!panel || q67("swiftGnssReadTestBtnV67")) return;

    const row = document.createElement("div");
    row.className = "swift-v66-row";
    row.innerHTML = `
      <button class="swift-v66-btn secondary" id="swiftGnssReadTestBtnV67">TLE読取テスト</button>
      <button class="swift-v66-btn secondary" id="swiftGnssForceListBtnV67">衛星リスト再表示</button>
    `;
    const status = q67("swiftGnssVisibleStatusV66");
    panel.insertBefore(row, status || panel.lastChild);

    const box = document.createElement("div");
    box.id = "swiftGnssDebugBoxV67";
    box.textContent = "TLE読取テストを押すと、local/live/satellite.js/parse件数を表示します。";
    panel.appendChild(box);

    q67("swiftGnssReadTestBtnV67").onclick = runDebugV67;
    q67("swiftGnssForceListBtnV67").onclick = () => {
      try { window.renderSatelliteSelection?.(); } catch {}
      relocateListV67();
      setDebugV67("衛星リスト再表示を実行しました。");
    };
  }

  function relocateListV67() {
    const holder = q67("swiftGnssVisibleListV66");
    const list = q67("satelliteSelectionList");
    if (holder && list && list.parentElement !== holder) holder.appendChild(list);
  }

  function setDebugV67(text) {
    const box = q67("swiftGnssDebugBoxV67");
    if (box) box.textContent = text;
  }

  function formatDebugV67(d) {
    const lines = [];
    lines.push(`selected: ${Array.isArray(d.selected) ? d.selected.join(", ") : "--"}`);
    lines.push(`satellite.js before: ${d.satellite_js_loaded_before ? "OK" : "NG"}`);
    lines.push(`satellite.js after:  ${d.satellite_js_loaded_after ? "OK" : "NG"}`);
    if (d.satellite_js_error) lines.push(`satellite.js error: ${d.satellite_js_error}`);
    lines.push(`current loaded count: ${d.current_loaded_count}`);
    for (const s of d.sources || []) {
      lines.push("");
      lines.push(`[${s.label}] parsed=${s.parsed_records}`);
      for (const mode of ["local", "live"]) {
        const r = s[mode];
        if (!r) continue;
        if (r.ok) {
          lines.push(`  ${mode}: OK status=${r.status} bytes=${r.bytes} records=${r.records}`);
          if (r.first_line) lines.push(`       first=${r.first_line.slice(0, 80)}`);
        } else {
          lines.push(`  ${mode}: NG ${r.status || ""} ${r.error || ""}`);
        }
      }
    }
    if (!d.sources?.length) lines.push("GNSSが1つも選択されていません。GPSなどをチェックしてください。");
    return lines.join("\n");
  }

  async function runDebugV67() {
    setDebugV67("TLE読取テスト中…");
    try {
      const d = await window.swiftDebugGnssLoadTest?.();
      if (!d) {
        setDebugV67("swiftDebugGnssLoadTestが見つかりません。JSが古い可能性があります。Ctrl+F5してください。");
        return;
      }
      setDebugV67(formatDebugV67(d));
    } catch (e) {
      setDebugV67("TLE読取テスト失敗: " + e.message);
    }
  }

  function patchLoadButtonV67() {
    const btn = q67("swiftGnssLoadBtnV66");
    if (!btn || btn.dataset.v67Patched) return;
    btn.dataset.v67Patched = "1";
    const old = btn.onclick;
    btn.onclick = async () => {
      setDebugV67("GNSS読込中… satellite.js → TLE local/live → parse → SGP4準備");
      try {
        if (typeof old === "function") await old();
        else await window.loadGnssDopData?.();
      } catch (e) {
        setDebugV67("GNSS読込エラー: " + e.message);
      }
      setTimeout(async () => {
        relocateListV67();
        try {
          const d = await window.swiftDebugGnssLoadTest?.();
          if (d) setDebugV67(formatDebugV67(d));
        } catch {}
      }, 350);
    };
  }

  function bootV67() {
    injectStyleV67();
    for (const delay of [600, 1200, 2200, 3600]) {
      setTimeout(() => {
        ensureDebugUiV67();
        patchLoadButtonV67();
        relocateListV67();
      }, delay);
    }
  }

  window.swiftRunGnssDebugV67 = runDebugV67;
  readyV67(bootV67);
})();


/* =========================================================
 * SWIFT-TEC v6.8 heatmap color palette selector
 * UI-only: keeps scale limits, changes palette colors.
 * ========================================================= */
(function () {
  const PALETTES_V68 = {
    classic: {
      label: "Classic（従来）",
      colors: null,
    },
    turbo: {
      label: "Turbo",
      colors: ["#30123b", "#4665d9", "#37a8fa", "#1ae4b6", "#72fe5f", "#d1e834", "#fe9b2d", "#e43d30", "#7a0403"],
    },
    viridis: {
      label: "Viridis",
      colors: ["#440154", "#46327e", "#365c8d", "#277f8e", "#1fa187", "#4ac16d", "#a0da39", "#fde725"],
    },
    plasma: {
      label: "Plasma",
      colors: ["#0d0887", "#5b02a3", "#9a179b", "#cb4679", "#ed7953", "#fb9f3a", "#fdca26", "#f0f921"],
    },
    blueRed: {
      label: "Blue → White → Red",
      colors: ["#1d4ed8", "#38bdf8", "#ffffff", "#facc15", "#dc2626"],
    },
    greenRed: {
      label: "Green → Yellow → Red",
      colors: ["#16a34a", "#bef264", "#facc15", "#f97316", "#dc2626"],
    },
    gray: {
      label: "Gray scale",
      colors: ["#020617", "#334155", "#94a3b8", "#e5e7eb", "#ffffff"],
    },
    night: {
      label: "Night high contrast",
      colors: ["#000014", "#16213e", "#0f9b8e", "#f4d35e", "#ee4266", "#ffffff"],
    },
    monoBlue: {
      label: "Mono blue",
      colors: ["#020617", "#0f172a", "#1e3a8a", "#2563eb", "#38bdf8", "#e0f2fe"],
    },
    purpleGold: {
      label: "Purple → Gold",
      colors: ["#1e103d", "#4c1d95", "#7e22ce", "#c084fc", "#fde68a", "#f59e0b"],
    },
  };

  let originalValueToColorV68 = null;

  function readyV68(fn) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn);
    else setTimeout(fn, 0);
  }
  function q68(id) { return document.getElementById(id); }

  function hexToRgbV68(hex) {
    let h = String(hex || "").replace("#", "").trim();
    if (h.length === 3) h = h.split("").map(c => c + c).join("");
    const n = parseInt(h, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

  function rgbToHexV68(r, g, b) {
    const h = n => Math.max(0, Math.min(255, Math.round(n))).toString(16).padStart(2, "0");
    return `#${h(r)}${h(g)}${h(b)}`;
  }

  function lerpV68(a, b, t) { return a + (b - a) * t; }

  function samplePaletteV68(colors, t) {
    if (!colors || !colors.length) return "#ffffff";
    if (colors.length === 1) return colors[0];
    const x = Math.max(0, Math.min(1, t)) * (colors.length - 1);
    const i0 = Math.floor(x);
    const i1 = Math.min(colors.length - 1, i0 + 1);
    const f = x - i0;
    const a = hexToRgbV68(colors[i0]);
    const b = hexToRgbV68(colors[i1]);
    return rgbToHexV68(
      lerpV68(a[0], b[0], f),
      lerpV68(a[1], b[1], f),
      lerpV68(a[2], b[2], f)
    );
  }

  function getPaletteNameV68() {
    return q68("swiftHeatmapPaletteSelectV68")?.value || localStorage.getItem("swiftHeatmapPaletteV68") || "classic";
  }

  function getReverseV68() {
    const el = q68("swiftHeatmapPaletteReverseV68");
    if (el) return !!el.checked;
    return localStorage.getItem("swiftHeatmapPaletteReverseV68") === "1";
  }

  function buildScaleWithPaletteV68(scale) {
    const sorted = (scale || []).slice().sort((a, b) => Number(a.limit) - Number(b.limit));
    if (!sorted.length) return scale;

    const name = getPaletteNameV68();
    const p = PALETTES_V68[name] || PALETTES_V68.classic;
    if (!p.colors) return sorted;

    const reverse = getReverseV68();
    const colors = reverse ? p.colors.slice().reverse() : p.colors.slice();
    const n = sorted.length;

    return sorted.map((item, idx) => ({
      limit: item.limit,
      color: samplePaletteV68(colors, n <= 1 ? 0 : idx / (n - 1)),
    }));
  }

  function patchValueToColorV68() {
    if (originalValueToColorV68) return;
    if (typeof valueToColor !== "function") return;

    originalValueToColorV68 = valueToColor;
    valueToColor = function (v, scale) {
      try {
        return originalValueToColorV68(v, buildScaleWithPaletteV68(scale));
      } catch (e) {
        return originalValueToColorV68(v, scale);
      }
    };
    window.valueToColor = valueToColor;
  }

  function injectStyleV68() {
    if (q68("swiftHeatmapPaletteStyleV68")) return;
    const st = document.createElement("style");
    st.id = "swiftHeatmapPaletteStyleV68";
    st.textContent = `
      .swift-v68-card {
        margin-top: 10px;
        border: 1px solid #1f355a;
        border-radius: 14px;
        background: rgba(7, 14, 28, .98);
        padding: 10px;
      }
      .swift-v68-title {
        color: #eaf2ff;
        font-size: 13px;
        font-weight: 900;
      }
      .swift-v68-sub {
        color: #9fb0cc;
        font-size: 10px;
        line-height: 1.38;
        margin-top: 3px;
      }
      .swift-v68-row {
        display: flex;
        gap: 7px;
        align-items: center;
        flex-wrap: wrap;
        margin-top: 8px;
      }
      .swift-v68-select {
        background: #061020;
        border: 1px solid #2a4774;
        color: #eaf2ff;
        border-radius: 9px;
        padding: 6px 8px;
        font-size: 11px;
        min-width: 170px;
      }
      .swift-v68-check {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        color: #dbeafe;
        font-size: 10px;
      }
      .swift-v68-preview {
        display: flex;
        width: 100%;
        height: 14px;
        border: 1px solid #1f355a;
        border-radius: 999px;
        overflow: hidden;
        background: #020617;
      }
      .swift-v68-preview span {
        flex: 1 1 auto;
      }
    `;
    document.head.appendChild(st);
  }

  function updatePreviewV68() {
    const box = q68("swiftHeatmapPalettePreviewV68");
    if (!box) return;
    const name = getPaletteNameV68();
    const p = PALETTES_V68[name] || PALETTES_V68.classic;
    const reverse = getReverseV68();
    const colors = p.colors ? (reverse ? p.colors.slice().reverse() : p.colors.slice()) : ["#2f55ff", "#46d9ff", "#fff176", "#ff2f2f"];
    box.innerHTML = colors.map(c => `<span style="background:${c}"></span>`).join("");
  }

  function refreshMapV68() {
    try { if (typeof updateLegend === "function") updateLegend(); } catch {}
    try { if (typeof requestDraw === "function") requestDraw(); } catch {}
    setTimeout(() => {
      try { if (typeof updateLegend === "function") updateLegend(); } catch {}
      try { if (typeof requestDraw === "function") requestDraw(); } catch {}
    }, 120);
  }

  function onChangeV68() {
    const sel = q68("swiftHeatmapPaletteSelectV68");
    const rev = q68("swiftHeatmapPaletteReverseV68");
    if (sel) localStorage.setItem("swiftHeatmapPaletteV68", sel.value);
    if (rev) localStorage.setItem("swiftHeatmapPaletteReverseV68", rev.checked ? "1" : "0");
    updatePreviewV68();
    refreshMapV68();
  }

  function ensurePaletteUiV68() {
    const side = q68("swiftAccuracySide") || document.querySelector(".sidebar");
    if (!side || q68("swiftHeatmapPalettePanelV68")) return;

    const panel = document.createElement("div");
    panel.id = "swiftHeatmapPalettePanelV68";
    panel.className = "swift-v68-card";
    panel.innerHTML = `
      <div class="swift-v68-title">ヒートマップ色設定</div>
      <div class="swift-v68-sub">TEC / DOP / TEC×DOPの色だけ変更します。閾値や計算値は変えません。</div>
      <div class="swift-v68-row">
        <select id="swiftHeatmapPaletteSelectV68" class="swift-v68-select">
          ${Object.entries(PALETTES_V68).map(([k, p]) => `<option value="${k}">${p.label}</option>`).join("")}
        </select>
        <label class="swift-v68-check"><input id="swiftHeatmapPaletteReverseV68" type="checkbox"> 反転</label>
      </div>
      <div class="swift-v68-row">
        <div id="swiftHeatmapPalettePreviewV68" class="swift-v68-preview"></div>
      </div>
    `;

    const after = q68("swiftGnssVisiblePanelV66") || q68("swiftPointDopPanelV65") || q68("swiftV57ModelRuleNote");
    if (after && after.parentElement) after.insertAdjacentElement("afterend", panel);
    else side.appendChild(panel);

    const saved = localStorage.getItem("swiftHeatmapPaletteV68") || "classic";
    const rev = localStorage.getItem("swiftHeatmapPaletteReverseV68") === "1";
    const sel = q68("swiftHeatmapPaletteSelectV68");
    if (sel && PALETTES_V68[saved]) sel.value = saved;
    const cb = q68("swiftHeatmapPaletteReverseV68");
    if (cb) cb.checked = rev;

    sel?.addEventListener("change", onChangeV68);
    cb?.addEventListener("change", onChangeV68);
    updatePreviewV68();
  }

  function addDockPaletteV68() {
    const dock = q68("swiftTimelineDockV64");
    if (!dock || q68("swiftDockPaletteSelectV68")) return;
    const row = dock.querySelector(".swift-v64-timeline-row:last-child") || dock;
    const label = document.createElement("span");
    label.className = "swift-v64-muted";
    label.textContent = "色";
    const sel = document.createElement("select");
    sel.id = "swiftDockPaletteSelectV68";
    for (const [k, p] of Object.entries(PALETTES_V68)) {
      const o = document.createElement("option");
      o.value = k;
      o.textContent = p.label.replace("（従来）", "");
      sel.appendChild(o);
    }
    sel.value = getPaletteNameV68();
    sel.onchange = () => {
      const main = q68("swiftHeatmapPaletteSelectV68");
      if (main) main.value = sel.value;
      localStorage.setItem("swiftHeatmapPaletteV68", sel.value);
      updatePreviewV68();
      refreshMapV68();
    };
    row.appendChild(label);
    row.appendChild(sel);
  }

  function bootV68() {
    patchValueToColorV68();
    injectStyleV68();
    for (const delay of [300, 900, 1600, 2600]) {
      setTimeout(() => {
        patchValueToColorV68();
        ensurePaletteUiV68();
        addDockPaletteV68();
      }, delay);
    }
  }

  window.swiftRefreshHeatmapPaletteV68 = onChangeV68;
  readyV68(bootV68);
})();

