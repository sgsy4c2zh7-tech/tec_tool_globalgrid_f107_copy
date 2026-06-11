/* SWIFT-TEC v4 add-on: 30-day NOAA TEC archive + selectable replay + GPS PDOP x TEC.
   Load this after the original SWIFT-TEC script. */
(function () {
  const TEC_INDEX_URL = "data/tec/index.json";
  const TEC_BASE_URL = "data/tec/";
  const GPS_TLE_URL = "data/gnss/gps_latest.tle";
  const SATELLITE_JS_URL = "https://unpkg.com/satellite.js/dist/satellite.min.js";

  let archiveIndex = null;
  let gpsSatRecs = [];
  let gpsLoaded = false;
  let dopFrameCache = new Map();
  let originalDrawTecOverlay = null;
  let originalUpdateLegend = null;
  let originalSampleAtLatLon = null;

  const dopColorScale = [
    { limit: 2, color: "#00ff00" },
    { limit: 4, color: "#ffff00" },
    { limit: 8, color: "#ff9900" },
    { limit: 16, color: "#ff0000" },
  ];

  const dopTecColorScale = [
    { limit: 5, color: "#00ff00" },
    { limit: 10, color: "#ffff00" },
    { limit: 20, color: "#ff9900" },
    { limit: 40, color: "#ff0000" },
  ];

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

  function setDisplayedFrames(frames, sourceLabel) {
    if (!frames.length) throw new Error("表示できるTECフレームがありません。期間を変えてください。");
    const meta = frames[0].gridMeta;
    gGrid = { latArr: meta.latArr, lonArr: meta.lonArr, nLat: meta.nLat, nLon: meta.nLon };
    gForecastFrames = frames.map(f => f.grid);
    gForecastTimes = frames.map(f => f.time);
    gForecastStart = gForecastTimes[0];
    currentStepIndex = 0;

    const slider = document.getElementById("timeSlider");
    if (slider) {
      slider.min = "0";
      slider.max = String(Math.max(0, gForecastTimes.length - 1));
      slider.value = "0";
    }
    const t = gForecastTimes[0];
    const label = document.getElementById("timeLabel");
    const utc = document.getElementById("utcLabel");
    if (label) label.textContent = `frame 1 / ${gForecastTimes.length}`;
    if (utc) utc.textContent = `UTC: ${isoNoMs(t)}`;
    if (typeof setOverlayTime === "function") setOverlayTime(t);
    if (typeof initMapIfNeeded === "function") initMapIfNeeded();
    if (typeof updateLegend === "function") updateLegend();
    if (typeof requestDraw === "function") requestDraw();
    if (typeof updateKpLabels === "function") updateKpLabels();
    setV4Status(`${sourceLabel}: ${gForecastTimes.length}枚を表示しました。${isoNoMs(gForecastTimes[0])} 〜 ${isoNoMs(gForecastTimes[gForecastTimes.length - 1])}`);
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
          if (t instanceof Date && !isNaN(t.getTime()) && t.getTime() > lastHist) {
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

  function dynamicOnSliderChange() {
    const slider = document.getElementById("timeSlider");
    const maxStep = Math.max(0, (gForecastTimes && gForecastTimes.length ? gForecastTimes.length - 1 : N_STEPS));
    const v = parseInt(slider.value, 10) || 0;
    currentStepIndex = clamp(v, 0, maxStep);
    const t = gForecastTimes[currentStepIndex] || gForecastStart;
    const hours = gForecastStart && t ? (t.getTime() - gForecastStart.getTime()) / 3600000 : (currentStepIndex * DT_MINUTES) / 60;
    const label = document.getElementById("timeLabel");
    const utc = document.getElementById("utcLabel");
    if (label) label.textContent = gForecastTimes.length ? `frame ${currentStepIndex + 1} / ${gForecastTimes.length}  (t=${hours.toFixed(1)}h)` : `t = ${hours.toFixed(1)} h`;
    if (utc) utc.textContent = `UTC: ${(t ? isoNoMs(t) : "--")}`;
    if (typeof setOverlayTime === "function") setOverlayTime(t);
    if (typeof updateKpLabels === "function") updateKpLabels();
    if (typeof requestDraw === "function") requestDraw();
  }

  function playArchiveMovie() {
    const speedSel = document.getElementById("movieSpeedSelect");
    const speed = Math.max(1, parseFloat(speedSel?.value || "1") || 1);
    const slider = document.getElementById("timeSlider");
    if (!slider) return;
    stopArchiveMovie();
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
    for (let i = 0; i < lines.length - 1; i++) {
      if (lines[i].startsWith("1 ") && lines[i + 1].startsWith("2 ")) {
        out.push({ name: `SAT-${out.length + 1}`, l1: lines[i], l2: lines[i + 1] });
        i += 1;
      } else if (i + 2 < lines.length && lines[i + 1].startsWith("1 ") && lines[i + 2].startsWith("2 ")) {
        out.push({ name: lines[i], l1: lines[i + 1], l2: lines[i + 2] });
        i += 2;
      }
    }
    return out;
  }

  async function loadGpsDopData() {
    try {
      setV4Status("GPS TLEとsatellite.jsを読み込み中…");
      await loadScriptOnce(SATELLITE_JS_URL);
      const res = await fetch(GPS_TLE_URL, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${GPS_TLE_URL}`);
      const tle = await res.text();
      const records = parseTleText(tle);
      gpsSatRecs = records.map(r => ({ name: r.name, satrec: satellite.twoline2satrec(r.l1, r.l2) }));
      gpsLoaded = gpsSatRecs.length > 0;
      dopFrameCache.clear();
      setV4Status(`GPS DOP準備OK: ${gpsSatRecs.length}機。表示モードをDOP/TEC×DOPに切替できます。`);
      if (typeof requestDraw === "function") requestDraw();
    } catch (e) {
      console.error(e);
      gpsLoaded = false;
      setV4Status("GPS DOP準備失敗: " + e.message);
    }
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
      for (let c = 0; c < 2 * n; c++) a[col][c] /= div;
      for (let r = 0; r < n; r++) {
        if (r === col) continue;
        const f = a[r][col];
        for (let c = 0; c < 2 * n; c++) a[r][c] -= f * a[col][c];
      }
    }
    return a.map(row => row.slice(n));
  }

  function propagatedGpsEcf(time) {
    if (!gpsLoaded || !window.satellite) return [];
    const gmst = satellite.gstime(time);
    const sats = [];
    for (const s of gpsSatRecs) {
      const pv = satellite.propagate(s.satrec, time);
      if (!pv || !pv.position) continue;
      const ecf = satellite.eciToEcf(pv.position, gmst);
      if (ecf && isFinite(ecf.x) && isFinite(ecf.y) && isFinite(ecf.z)) sats.push({ name: s.name, ecf });
    }
    return sats;
  }

  function pdopAt(latDeg, lonDeg, sats, elevMaskDeg = 10) {
    if (!sats || sats.length < 4) return NaN;
    const lat = latDeg * Math.PI / 180;
    const lon = lonDeg * Math.PI / 180;
    const obs = geodeticToEcfKm(lat, lon, 0);
    const up = [Math.cos(lat) * Math.cos(lon), Math.cos(lat) * Math.sin(lon), Math.sin(lat)];
    const elevMask = elevMaskDeg * Math.PI / 180;
    const rows = [];

    for (const sat of sats) {
      const rx = sat.ecf.x - obs.x;
      const ry = sat.ecf.y - obs.y;
      const rz = sat.ecf.z - obs.z;
      const r = Math.sqrt(rx * rx + ry * ry + rz * rz);
      if (!isFinite(r) || r <= 0) continue;
      const ux = rx / r, uy = ry / r, uz = rz / r;
      const elev = Math.asin(ux * up[0] + uy * up[1] + uz * up[2]);
      if (elev < elevMask) continue;
      rows.push([-ux, -uy, -uz, 1]);
    }
    if (rows.length < 4) return NaN;

    const ata = Array.from({ length: 4 }, () => Array(4).fill(0));
    for (const row of rows) {
      for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) ata[i][j] += row[i] * row[j];
    }
    const q = invert4(ata);
    if (!q) return NaN;
    const pdop = Math.sqrt(Math.max(0, q[0][0] + q[1][1] + q[2][2]));
    return isFinite(pdop) ? pdop : NaN;
  }

  function getDopFrame(stepIndex) {
    if (!gGrid || !gForecastTimes.length || !gpsLoaded) return null;
    const t = gForecastTimes[stepIndex];
    if (!(t instanceof Date) || isNaN(t.getTime())) return null;
    const key = t.toISOString();
    if (dopFrameCache.has(key)) return dopFrameCache.get(key);

    const sats = propagatedGpsEcf(t);
    const frame = Array.from({ length: gGrid.nLat }, () => Array(gGrid.nLon).fill(NaN));
    for (let i = 0; i < gGrid.nLat; i++) {
      const lat = gGrid.latArr[i];
      for (let j = 0; j < gGrid.nLon; j++) frame[i][j] = pdopAt(lat, gGrid.lonArr[j], sats, 10);
    }
    dopFrameCache.set(key, frame);
    return frame;
  }

  function modeValue(baseTec, i, j, cfg) {
    if (mapMode === "gps") return baseTec * cfg.kL1;
    if (mapMode === "dop") {
      const df = getDopFrame(currentStepIndex);
      return df ? df[i][j] : NaN;
    }
    if (mapMode === "doptec") {
      const df = getDopFrame(currentStepIndex);
      const pdop = df ? df[i][j] : NaN;
      return isFinite(pdop) ? pdop * baseTec * cfg.kL1 : NaN;
    }
    return baseTec;
  }

  function scaleForMode() {
    if (mapMode === "gps") return gGpsColorScale;
    if (mapMode === "dop") return dopColorScale;
    if (mapMode === "doptec") return dopTecColorScale;
    return gTecColorScale;
  }

  function titleForMode() {
    if (mapMode === "gps") return "GPS L1 [m]";
    if (mapMode === "dop") return "GPS PDOP";
    if (mapMode === "doptec") return "PDOP×L1 [m]";
    return "TEC [TECU]";
  }

  function installOverrides() {
    if (typeof onSliderChange === "function") onSliderChange = dynamicOnSliderChange;

    if (typeof drawTecOverlay === "function" && !originalDrawTecOverlay) originalDrawTecOverlay = drawTecOverlay;
    drawTecOverlay = function () {
      if (!map || !tecCanvas || !tecCtx) return;
      if (!gGrid || !gForecastFrames.length) return;
      const w = tecCanvas.width, h = tecCanvas.height;
      tecCtx.clearRect(0, 0, w, h);
      const frame = gForecastFrames[currentStepIndex];
      if (!frame) return;
      const cfg = getConfigFromUI();
      const scale = scaleForMode();
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
          const tec = isFinite(frame[i][j]) ? frame[i][j] : 0;
          const val = modeValue(tec, i, j, cfg);
          if (!isFinite(val)) continue;
          tecCtx.globalAlpha = tecAlpha;
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
        div.innerHTML = `
          <div class="tec-legend-title">${title}</div>
          <canvas id="legendBar" width="18" height="130" style="display:block;margin:0 auto 4px auto;border:1px solid #444;border-radius:4px;"></canvas>
          <div class="tec-legend-labels"><span>0</span><span>${scale[scale.length - 1].limit}</span></div>
        `;
        setTimeout(() => {
          const c = document.getElementById("legendBar");
          if (!c) return;
          const ctx = c.getContext("2d");
          for (let y = 0; y < c.height; y++) {
            const v = scale[scale.length - 1].limit * (1 - y / c.height);
            ctx.fillStyle = valueToColor(v, scale);
            ctx.fillRect(0, y, c.width, 1);
          }
        }, 0);
        return div;
      };
      tecLegendControl.addTo(map);
    };

    if (typeof sampleAtLatLon === "function" && !originalSampleAtLatLon) originalSampleAtLatLon = sampleAtLatLon;
    sampleAtLatLon = function (lat, lon) {
      if (!gGrid || !gForecastFrames.length) return "未計算です。";
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
      const tec = gForecastFrames[currentStepIndex]?.[bestI]?.[bestJ];
      const cfg = getConfigFromUI();
      const gps = (isFinite(tec) ? tec : 0) * cfg.kL1;
      const df = getDopFrame(currentStepIndex);
      const pdop = df ? df[bestI][bestJ] : NaN;
      const doptec = isFinite(pdop) ? pdop * gps : NaN;
      const t = gForecastTimes[currentStepIndex];
      return [
        `Clicked: lat=${lat.toFixed(3)}, lon=${lon.toFixed(3)}`,
        `Nearest Grid: lat=${gGrid.latArr[bestI].toFixed(2)}, lon=${gGrid.lonArr[bestJ].toFixed(2)}`,
        `Time: ${(t ? isoNoMs(t) : "--")}`,
        `TEC: ${(isFinite(tec) ? tec.toFixed(2) : "NaN")} TECU`,
        `GPS L1 error: ${gps.toFixed(2)} m`,
        `GPS PDOP: ${isFinite(pdop) ? pdop.toFixed(2) : "--"}`,
        `PDOP × L1 error: ${isFinite(doptec) ? doptec.toFixed(2) : "--"} m`,
      ].join("\n");
    };
  }

  function addModeOptions() {
    const sel = document.getElementById("mapModeSelect");
    if (!sel) return;
    const opts = [
      ["dop", "GPS PDOP"],
      ["doptec", "DOP × TEC 測位誤差 [m]"],
    ];
    for (const [value, label] of opts) {
      if ([...sel.options].some(o => o.value === value)) continue;
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = label;
      sel.appendChild(opt);
    }
  }

  function bootAddon() {
    installOverrides();
    addModeOptions();
    loadTecArchiveIndex(false).catch(() => {
      const info = document.getElementById("archiveIndexInfo");
      if (info) info.textContent = "TEC履歴index未作成。GitHub Actionsを実行すると表示されます。";
    });
  }

  window.loadTecArchiveIndex = loadTecArchiveIndex;
  window.loadTecArchiveRange = loadTecArchiveRange;
  window.loadTecArchivePlusCurrentForecast = loadTecArchivePlusCurrentForecast;
  window.playArchiveMovie = playArchiveMovie;
  window.stopArchiveMovie = stopArchiveMovie;
  window.loadGpsDopData = loadGpsDopData;

  document.addEventListener("DOMContentLoaded", bootAddon);
})();
