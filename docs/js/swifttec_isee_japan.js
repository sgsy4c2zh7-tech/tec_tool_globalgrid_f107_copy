/* SWIFT-TEC v8.10 ISEE Japan High-Res VTEC + 10-day Base forecast */
(function () {
  "use strict";

  const TEC_INDEX_URL = "data/isee_tec/index.json";
  const TEC_BASE_URL = "data/isee_tec/";
  const BASE_INDEX_URL = "data/isee_base/index.json";
  const BASE_ROOT_URL = "data/isee_base/";
  const JAPAN_GRID_COEFF_URL = "data/ai/isee_japan/kp_grid_coefficients.json";

  const FORECAST_STEP_MIN = 30;
  const FORECAST_HOURS = 96;

  window.swiftIseeFrames = window.swiftIseeFrames || [];
  window.swiftIseeTimes = window.swiftIseeTimes || [];

  const baseSlotCache = new Map();
  let baseIndexCache = null;
  let gridCoeffCache = null;

  function status(msg) {
    const el = document.getElementById("swiftV52Status") || document.getElementById("iseeTecStatus");
    if (el) el.textContent = msg;
    try { window.logInfo?.(msg); } catch {}
  }

  async function getJson(url) {
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${url}`);
    return await r.json();
  }

  function normalizeFrame(obj, entry) {
    const latArr = obj.lat_arr || obj.latArr || [];
    const lonArr = obj.lon_arr || obj.lonArr || [];
    const grid = obj.grid || obj.tec || [];
    const t = new Date(obj.time_utc || entry?.time_utc || 0);
    if (!latArr.length || !lonArr.length || !grid.length || isNaN(t.getTime())) {
      throw new Error("ISEE JSONの格子/時刻形式が不正です");
    }
    return {
      validTime: t,
      time: t,
      latArr,
      lonArr,
      nLat: latArr.length,
      nLon: lonArr.length,
      grid,
      quantity: obj.quantity || "VTEC",
      units: obj.units || "TECU",
      sourceFile: obj.source_file || entry?.file || "ISEE",
    };
  }

  async function loadLatest(showOnMap = false) {
    status("ISEE Japan High-Res VTEC [TECU]（30日保存）を読込中…");
    const idx = await getJson(TEC_INDEX_URL);
    const entries = Array.isArray(idx.frames) ? idx.frames.slice() : [];
    const latency = Number(idx.source_latency_hours_at_fetch);
    const latestSource = idx.latest_source_hour_utc || "--";
    entries.sort((a,b) => String(a.time_utc).localeCompare(String(b.time_utc)));
    if (!entries.length) throw new Error("data/isee_tec/index.json にframeがありません");

    // Latest 24h of the newest actually-published ISEE product for observation/replay.
    const latestMs = new Date(entries[entries.length - 1].time_utc).getTime();
    const selected = entries.filter(e => {
      const ms = new Date(e.time_utc).getTime();
      return Number.isFinite(ms) && ms >= latestMs - 24 * 3600 * 1000;
    }).slice(-320);

    const loaded = [];
    for (const e of selected) {
      loaded.push(normalizeFrame(await getJson(TEC_BASE_URL + e.file), e));
    }
    if (!loaded.length) throw new Error("ISEEフレームを読み込めませんでした");

    window.swiftIseeFrames = loaded;
    window.swiftIseeTimes = loaded.map(f => f.validTime);

    const legacy = document.getElementById("tecSourceSelect");
    if (legacy) legacy.value = "isee";
    const compact = document.getElementById("swiftV52TecSource");
    if (compact) compact.value = "isee_japan_highres";

    const first = loaded[0], last = loaded[loaded.length - 1];
    status(
      `ISEE VTEC読込OK: ${loaded.length}枚 / ${first.nLat}×${first.nLon} / VTEC [TECU] / ` +
      `最新source=${latestSource}${Number.isFinite(latency) ? ` / ISEE遅延≈${latency.toFixed(1)}h` : ""}`
    );

    if (showOnMap) showLatestOnMap();
    return loaded;
  }

  function showLatestOnMap() {
    const loaded = window.swiftIseeFrames || [];
    if (!loaded.length) {
      loadLatest(true).catch(e => status("ISEE日本表示失敗: " + e.message));
      return;
    }
    const f = loaded[loaded.length - 1];
    try {
      gGrid = { latArr:f.latArr, lonArr:f.lonArr, nLat:f.nLat, nLon:f.nLon };
      gForecastFrames = loaded.map(x => x.grid);
      gForecastTimes = loaded.map(x => x.validTime);
      gForecastStart = gForecastTimes[0];
      currentStepIndex = Math.max(0, gForecastTimes.length - 1);

      const slider = document.getElementById("timeSlider");
      if (slider) {
        slider.min = "0";
        slider.max = String(Math.max(0, gForecastTimes.length - 1));
        slider.value = String(currentStepIndex);
      }
      try { window.swiftClearIsee10DayBaseForecastMode?.(); } catch {}
      if (typeof initMapIfNeeded === "function") initMapIfNeeded();
      if (typeof gMap !== "undefined" && gMap?.fitBounds) {
        gMap.fitBounds([[24,122],[46,150]], { padding:[8,8] });
      }
      if (typeof dynamicOnSliderChange === "function") dynamicOnSliderChange();
      else {
        if (typeof updateLegend === "function") updateLegend();
        if (typeof requestDraw === "function") requestDraw();
      }
    } catch (e) {
      console.error(e);
      status("ISEE日本表示に失敗: " + e.message);
    }
  }

  function minuteOfDayFromHHMM(hhmm) {
    const s = String(hhmm || "").padStart(4, "0");
    const h = Number(s.slice(0,2)), m = Number(s.slice(2,4));
    return h * 60 + m;
  }

  function nearestBaseSlot(slots, t) {
    const target = t.getUTCHours() * 60 + t.getUTCMinutes();
    let best = null, bestD = Infinity;
    for (const s of slots || []) {
      const mm = minuteOfDayFromHHMM(s.slot_utc_hhmm);
      let d = Math.abs(mm - target);
      d = Math.min(d, 1440 - d);
      if (d < bestD) { bestD = d; best = s; }
    }
    return best;
  }

  async function loadBaseIndex() {
    baseIndexCache = await getJson(BASE_INDEX_URL);
    if (!Array.isArray(baseIndexCache.slots) || !baseIndexCache.slots.length) {
      throw new Error("docs/data/isee_base/index.json に10日Base VTECがありません。Update ISEE Japan VTEC and AIを先に実行してください。");
    }
    return baseIndexCache;
  }

  async function loadBaseSlot(entry) {
    const key = entry.file || `${entry.slot_utc_hhmm}.json`;
    if (baseSlotCache.has(key)) return baseSlotCache.get(key);
    const doc = await getJson(BASE_ROOT_URL + key);
    if (doc.quantity !== "Base VTEC" && !String(doc.quantity || "").includes("VTEC")) {
      console.warn("Unexpected Base quantity:", doc.quantity);
    }
    baseSlotCache.set(key, doc);
    return doc;
  }

  async function loadGridCoefficients() {
    gridCoeffCache = await getJson(JAPAN_GRID_COEFF_URL);
    return gridCoeffCache;
  }

  function monthGrid(coeffDoc, month) {
    const root = coeffDoc?.coefficients_grid || coeffDoc?.grid_coefficients || {};
    return root[String(month)] || root[month] || null;
  }

  function coeffAt(mg, i, j) {
    if (!mg) return null;
    const n = Number(mg.sample_count?.[i]?.[j] ?? mg.count?.[i]?.[j] ?? 0);
    if (!Number.isFinite(n) || n < 4) return null;
    return {
      k0: Number(mg.k0?.[i]?.[j] || 0),
      k1: Number(mg.k1?.[i]?.[j] || 0),
      k2: Number(mg.k2?.[i]?.[j] || 0),
      k3: Number(mg.k3?.[i]?.[j] || 0),
      n,
    };
  }

  function F(cf, kp) {
    if (!cf) return 0;
    const x = (Number.isFinite(kp) ? kp : 3.0) - 3.0;
    return cf.k0 + cf.k1*x + cf.k2*x*x + cf.k3*x*x*x;
  }

  function roundedNowUtc30() {
    const t = new Date();
    t.setUTCSeconds(0,0);
    t.setUTCMinutes(Math.floor(t.getUTCMinutes()/30)*30);
    return t;
  }

  function makeForecastKpSeries(startUtc) {
    const txt = (document.getElementById("noaaKpText")?.value || "").trim();
    const nSteps = Math.round(FORECAST_HOURS * 60 / FORECAST_STEP_MIN);

    if (typeof buildForecastKpSeriesFrom3DayText === "function" && txt) {
      const mode = (typeof gKpInputMode !== "undefined" ? gKpInputMode : "auto");
      const rows = buildForecastKpSeriesFrom3DayText(txt, startUtc, nSteps, FORECAST_STEP_MIN, mode);
      if (Array.isArray(rows) && rows.length) return rows;
    }

    // Fallback only if NOAA text is unavailable.
    return Array.from({length:nSteps+1}, (_,i) => ({
      t: new Date(startUtc.getTime() + i*FORECAST_STEP_MIN*60000),
      kp: 3.0,
    }));
  }

  function aiEnabled() {
    const compact = document.getElementById("swiftV52AiEnabled");
    if (compact) return !!compact.checked;
    const legacy = document.getElementById("kpAiCorrectionEnabled");
    return legacy ? !!legacy.checked : true;
  }

  async function run10DayBaseForecast() {
    status("ISEE 10日Base VTEC・Japan AI係数を読み込み中…");

    const [baseIndex, coeffDoc] = await Promise.all([
      loadBaseIndex(),
      loadGridCoefficients(),
    ]);

    const slots = baseIndex.slots || [];
    const startUtc = roundedNowUtc30();
    const kpSeries = makeForecastKpSeries(startUtc);
    const nSteps = Math.round(FORECAST_HOURS * 60 / FORECAST_STEP_MIN);

    const frames = [];
    let gridMeta = null;
    let minDaysUsed = Infinity;
    let maxDaysUsed = 0;
    const useAi = aiEnabled();

    for (let s=0; s<=nSteps; s++) {
      const t = new Date(startUtc.getTime() + s*FORECAST_STEP_MIN*60000);
      const slotEntry = nearestBaseSlot(slots, t);
      if (!slotEntry) throw new Error("10日BaseのUTC時刻slotが見つかりません。");

      const baseDoc = await loadBaseSlot(slotEntry);
      const baseGrid = baseDoc.grid || [];
      const latArr = baseDoc.lat_arr || [];
      const lonArr = baseDoc.lon_arr || [];
      if (!latArr.length || !lonArr.length || !baseGrid.length) {
        throw new Error(`Base VTEC格子が不正: ${slotEntry.file}`);
      }
      if (!gridMeta) {
        gridMeta = { latArr, lonArr, nLat:latArr.length, nLon:lonArr.length };
      }

      const daysUsed = Number(baseDoc.days_used || slotEntry.days_used || 0);
      if (Number.isFinite(daysUsed)) {
        minDaysUsed = Math.min(minDaysUsed, daysUsed);
        maxDaysUsed = Math.max(maxDaysUsed, daysUsed);
      }

      const kpF = Number(kpSeries[s]?.kp);
      const mg = monthGrid(coeffDoc, t.getUTCMonth()+1);
      const outGrid = Array.from({length:gridMeta.nLat}, () => Array(gridMeta.nLon).fill(null));

      for (let i=0; i<gridMeta.nLat; i++) {
        for (let j=0; j<gridMeta.nLon; j++) {
          const b = Number(baseGrid?.[i]?.[j]);
          if (!Number.isFinite(b)) {
            outGrid[i][j] = null;
            continue;
          }
          const cf = coeffAt(mg, i, j);
          const kpTerm = useAi ? F(cf, kpF) : 0;
          const vtec = Math.max(0, Math.min(300, b + kpTerm));
          outGrid[i][j] = Number.isFinite(vtec) ? vtec : null;
        }
      }

      frames.push({
        time:t,
        grid:outGrid,
        gridMeta,
        sourceFile:`ISEE Base10 ${slotEntry.slot_utc_hhmm}Z`,
      });
    }

    if (!frames.length) throw new Error("ISEE 10日Base予報フレームを生成できませんでした。");
    if (typeof window.swiftInstallIsee10DayBaseForecast !== "function") {
      throw new Error("swifttec_v4_archive_dop.js がv8.10ではありません。");
    }

    window.swiftInstallIsee10DayBaseForecast(frames, kpSeries, {gridMeta});

    const daysTxt = Number.isFinite(minDaysUsed)
      ? (minDaysUsed === maxDaysUsed ? `${minDaysUsed}日` : `${minDaysUsed}〜${maxDaysUsed}日`)
      : "--";
    status(
      `Japan予報OK: ISEE同一UTC時刻 ${daysTxt} Base / ` +
      `${useAi ? "予報Kp + Japan格子AI" : "AI補正OFF"} / ` +
      `+0〜+4日 / VTEC [TECU]`
    );
    return frames;
  }

  window.swiftIseeLoadLatest = loadLatest;
  window.swiftIseeShowJapan = showLatestOnMap;
  window.swiftIseeRun10DayBaseForecast = run10DayBaseForecast;
})();
