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

  // NOAAから保存するTECは2時間値。DOP / DOP×TECモードだけ、UI上で10分刻みへ展開する。
  const TEC_ARCHIVE_STEP_MIN = 120;
  const DOP_REPLAY_STEP_MIN = 10;
  let rawDisplayFrames = [];
  let activeTimelineStepMin = TEC_ARCHIVE_STEP_MIN;
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
    return `TEC系: NOAA 2時間値`;
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

    if (isDopLikeMode() && rawDisplayFrames.length >= 2) {
      activeTimelineStepMin = DOP_REPLAY_STEP_MIN;
      gForecastTimes = buildUniformTimeline(rawDisplayFrames, DOP_REPLAY_STEP_MIN);
      // 既存処理がlengthを見るため、実グリッドはlazy補間し、ここはダミー配列にする。
      gForecastFrames = new Array(gForecastTimes.length).fill(null);
    } else {
      activeTimelineStepMin = TEC_ARCHIVE_STEP_MIN;
      gForecastTimes = rawDisplayFrames.map(f => f.time);
      gForecastFrames = rawDisplayFrames.map(f => f.grid);
    }

    if (resetIndex) currentStepIndex = 0;
    currentStepIndex = c(currentStepIndex, 0, Math.max(0, gForecastTimes.length - 1));
    gForecastStart = gForecastTimes[0] || null;
    setSliderForTimeline();
    dynamicOnSliderChange();
  }

  function interpolateGridAtTime(t) {
    if (!rawDisplayFrames.length) return null;
    if (!isDopLikeMode()) {
      return rawDisplayFrames[currentStepIndex]?.grid || rawDisplayFrames[0].grid;
    }
    const key = t.toISOString();
    if (tecInterpCache.has(key)) return tecInterpCache.get(key);

    const targetMs = t.getTime();
    if (targetMs <= rawDisplayFrames[0].time.getTime()) return rawDisplayFrames[0].grid;
    const last = rawDisplayFrames[rawDisplayFrames.length - 1];
    if (targetMs >= last.time.getTime()) return last.grid;

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
    if (w <= 1e-9) return lo.grid;
    if (w >= 1 - 1e-9) return hi.grid;

    const nLat = gGrid.nLat, nLon = gGrid.nLon;
    const out = Array.from({ length: nLat }, () => Array(nLon).fill(0));
    for (let i = 0; i < nLat; i++) {
      const rowA = lo.grid[i] || [];
      const rowB = hi.grid[i] || [];
      const rowO = out[i];
      for (let j = 0; j < nLon; j++) {
        const a = Number(rowA[j]);
        const b = Number(rowB[j]);
        if (isFinite(a) && isFinite(b)) rowO[j] = a * (1 - w) + b * w;
        else if (isFinite(a)) rowO[j] = a;
        else if (isFinite(b)) rowO[j] = b;
        else rowO[j] = NaN;
      }
    }
    tecInterpCache.set(key, out);
    // 長時間再生でメモリが増えすぎないように簡易上限。
    if (tecInterpCache.size > 24) {
      const firstKey = tecInterpCache.keys().next().value;
      tecInterpCache.delete(firstKey);
    }
    return out;
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
    currentStepIndex = 0;

    if (typeof initMapIfNeeded === "function") initMapIfNeeded();
    if (typeof updateLegend === "function") updateLegend();
    applyTimelineForCurrentMode(true);
    if (typeof updateKpLabels === "function") updateKpLabels();
    setV4Status(`${sourceLabel}: NOAA 2時間TEC ${rawDisplayFrames.length}枚を読み込み。表示=${getTimelineLabel()} / ${isoNoMs(rawDisplayFrames[0].time)} 〜 ${isoNoMs(rawDisplayFrames[rawDisplayFrames.length - 1].time)}`);
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
      drawSmoothHeatmap(frame, cfg, scale);
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
        const maxLabel = scale[scale.length - 1].limit;
        div.innerHTML = `
          <div class="tec-legend-title">${title}</div>
          <canvas id="legendBar" width="18" height="130" style="display:block;margin:0 auto 4px auto;border:1px solid #444;border-radius:4px;"></canvas>
          <div class="tec-legend-labels"><span>0</span><span>${maxLabel}</span></div>
        `;
        setTimeout(() => {
          const canvas = document.getElementById("legendBar");
          if (!canvas) return;
          const ctx = canvas.getContext("2d");
          for (let y = 0; y < canvas.height; y++) {
            const v = scale[scale.length - 1].limit * (1 - y / canvas.height);
            ctx.fillStyle = valueToColor(v, scale);
            ctx.fillRect(0, y, canvas.width, 1);
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
