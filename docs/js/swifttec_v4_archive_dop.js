/* SWIFT-TEC v4 add-on: 30-day NOAA TEC archive + selectable replay + multi-GNSS DOP/TEC.
   Load this after the original SWIFT-TEC script. */
(function () {
  const TEC_INDEX_URL = "data/tec/index.json";
  const TEC_BASE_URL = "data/tec/";
  const SATELLITE_JS_URL = "https://unpkg.com/satellite.js/dist/satellite.min.js";

  const GNSS_SOURCES = {
    gps:     { label: "GPS",     url: "data/gnss/gps_latest.tle",     checked: true  },
    galileo: { label: "Galileo", url: "data/gnss/galileo_latest.tle", checked: false },
    glonass: { label: "GLONASS", url: "data/gnss/glonass_latest.tle", checked: false },
    beidou:  { label: "BeiDou",  url: "data/gnss/beidou_latest.tle",  checked: false },
    qzss:    { label: "QZSS",    url: "data/gnss/qzss_latest.tle",    checked: true  },
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

  // data/tecに保存されるTECはNOAA 30分値。UIではTEC系30分、DOP系10分へ展開する。
  const TEC_REPLAY_STEP_MIN = 30;
  const DOP_REPLAY_STEP_MIN = 10;
  const TEC_FORECAST_HOURS = 72;
  const DEFAULT_TEC_FORECAST_BASE_DAYS = 7;
  let rawDisplayFrames = [];
  let activeTimelineStepMin = TEC_REPLAY_STEP_MIN;
  let tecSmoothCache = new Map();
  let selectionVersion = 0;

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

    // v4.5: 旧版(v3)の見え方を踏襲するため、TEC値の空間平滑化・再スケールは行わない。
    // data/tecの30分値を時間方向だけ線形補間し、格子値はそのまま描画/計算に使う。
    return grid;
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

  function currentTecGrid() {
    const t = gForecastTimes[currentStepIndex];
    if (!(t instanceof Date) || isNaN(t.getTime())) return null;
    return interpolateGridAtTime(t);
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
    return new Promise((resolve, reject) => {
      if (window.satellite) return resolve();
      const s = document.createElement("script");
      s.src = url;
      s.async = true;
      s.onload = resolve;
      s.onerror = () => reject(new Error("satellite.js の読み込みに失敗: " + url));
      document.head.appendChild(s);
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
      const cb = document.getElementById(`gnssConst_${key}`);
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
      setV4Status("GNSS TLEとsatellite.jsを読み込み中…");
      await loadScriptOnce(SATELLITE_JS_URL);

      const selectedConst = selectedConstellationsFromUi();
      if (!selectedConst.length) throw new Error("GNSSコンステレーションが未選択です。GPSなどを選択してください。");

      const previous = new Map(gnssSatList.map(s => [s.id, { selected: s.selected, active: s.active }]));
      const loaded = [];
      const failed = [];

      for (const key of selectedConst) {
        const src = GNSS_SOURCES[key];
        try {
          const tle = await fetchTleIfExists(src.url);
          if (!tle) {
            failed.push(`${src.label}: not found`);
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

  function bootAddon() {
    installOverrides();
    addModeOptions();
    ensureGnssUi();
    loadTecArchiveIndex(false).catch(() => {
      const info = document.getElementById("archiveIndexInfo");
      if (info) info.textContent = "TEC履歴index未作成。GitHub Actionsを実行すると表示されます。";
    });
  }

  window.loadTecArchiveIndex = loadTecArchiveIndex;
  window.loadTecArchiveRange = loadTecArchiveRange;
  window.loadTecArchivePlusCurrentForecast = loadTecArchivePlusCurrentForecast;
  window.loadTecDataDriven3DayForecast = loadTecDataDriven3DayForecast;
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

  document.addEventListener("DOMContentLoaded", bootAddon);
})();
