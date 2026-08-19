/* SWIFT-TEC v8.10 ISEE Japan High-Res VTEC + 10-day Base forecast */
(function () {
  "use strict";

  const TEC_INDEX_URL = "data/isee_tec/index.json";
  const TEC_BASE_URL = "data/isee_tec/";
  const MEAN_INDEX_URL = "data/isee_mean/index.json";
  const MEAN_ROOT_URL = "data/isee_mean/";
  const JAPAN_GRID_COEFF_URL = "data/ai/isee_japan/kp_grid_coefficients.json";

  const FORECAST_STEP_MIN = 30;
  const FORECAST_HOURS = 96;

  window.swiftIseeFrames = window.swiftIseeFrames || [];
  window.swiftIseeTimes = window.swiftIseeTimes || [];

  const meanSlotCache = new Map();
  let meanIndexCache = null;
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

  function resolveBaseSlot(slots, t) {
    const target = t.getUTCHours() * 60 + t.getUTCMinutes();
    let exact = null;
    let best = null, bestD = Infinity;

    for (const s of slots || []) {
      const mm = minuteOfDayFromHHMM(s.slot_utc_hhmm);
      if (mm === target) exact = s;
      let d = Math.abs(mm - target);
      d = Math.min(d, 1440 - d);
      if (d < bestD) { bestD = d; best = s; }
    }

    if (exact) return { entry: exact, diffMin: 0 };

    // ISEE Base is normally 5-minute slots.  A 30-minute forecast target
    // should therefore have an exact slot.  Permit only a small gap instead
    // of silently mapping many hours to the same 23:xx frame.
    if (best && bestD <= 10) return { entry: best, diffMin: bestD };
    return { entry: null, diffMin: bestD };
  }

  async function loadMeanIndex() {
    meanIndexCache = await getJson(MEAN_INDEX_URL);
    if (!Array.isArray(meanIndexCache.slots) || !meanIndexCache.slots.length) {
      throw new Error("docs/data/isee_mean/index.json にISEE時間別平均VTECがありません。Update ISEE Japan VTEC and AIを先に実行してください。");
    }
    return meanIndexCache;
  }

  async function loadMeanSlot(entry) {
    const key = entry.file || `${entry.slot_utc_hhmm}.json`;
    if (meanSlotCache.has(key)) return meanSlotCache.get(key);
    const doc = await getJson(MEAN_ROOT_URL + key);
    if (!String(doc.quantity || "").includes("VTEC")) {
      console.warn("Unexpected ISEE mean quantity:", doc.quantity);
    }
    meanSlotCache.set(key, doc);
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
    const k0 = Number(mg.k0?.[i]?.[j] ?? 0);
    const k1 = Number(mg.k1?.[i]?.[j] ?? 0);
    const k2 = Number(mg.k2?.[i]?.[j] ?? 0);
    const k3 = Number(mg.k3?.[i]?.[j] ?? 0);

    // v8.11:
    // Persisted coefficients remain valid even if the latest training run has
    // a small current-run sample_count. The old v8.10 n<4 gate could turn the
    // entire Kp correction into zero and make every forecast day repeat.
    if (![k0,k1,k2,k3].every(Number.isFinite)) return null;
    return { k0, k1, k2, k3, n: Number.isFinite(n) ? n : 0 };
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

    const parser =
      (typeof window.buildForecastKpSeriesFrom3DayText === "function" && window.buildForecastKpSeriesFrom3DayText) ||
      (typeof buildForecastKpSeriesFrom3DayText === "function" && buildForecastKpSeriesFrom3DayText);

    if (parser && txt) {
      const mode = (typeof gKpInputMode !== "undefined" ? gKpInputMode : "auto");
      const rows = parser(txt, startUtc, nSteps, FORECAST_STEP_MIN, mode);
      if (Array.isArray(rows) && rows.length >= 2) {
        return rows.map((r,i) => ({
          t: r.t instanceof Date ? r.t : new Date(r.t || r.time || startUtc.getTime() + i*FORECAST_STEP_MIN*60000),
          kp: Number(r.kp),
        }));
      }
    }

    // Second fallback: use an already-built forecast Kp series from SWIFT-TEC.
    try {
      if (Array.isArray(gKpSeries) && gKpSeries.length) {
        const valid = gKpSeries.map(r => ({
          t: r.t instanceof Date ? r.t : new Date(r.t || r.time || r.time_utc || 0),
          kp: Number(r.kp ?? r.Kp ?? r.value),
        })).filter(r => !isNaN(r.t.getTime()) && Number.isFinite(r.kp));

        if (valid.length) {
          const out = [];
          for (let i=0; i<=nSteps; i++) {
            const t = new Date(startUtc.getTime() + i*FORECAST_STEP_MIN*60000);
            let best = valid[0], bd = Infinity;
            for (const r of valid) {
              const d = Math.abs(r.t.getTime() - t.getTime());
              if (d < bd) { bd = d; best = r; }
            }
            out.push({t, kp:best.kp});
          }
          return out;
        }
      }
    } catch {}

    // Never silently use constant Kp=3. That hides a broken Kp input and makes
    // the Japan forecast look frozen.
    throw new Error(
      "予報Kpを生成できません。『予報時にNOAA 3-Day Kpを自動取得』をONにして、もう一度予報実行してください。"
    );
  }

  function aiEnabled() {
    const compact = document.getElementById("swiftV52AiEnabled");
    if (compact) return !!compact.checked;
    const legacy = document.getElementById("kpAiCorrectionEnabled");
    return legacy ? !!legacy.checked : true;
  }

  function correctionClipTecU() {
    const a = Number(document.getElementById("swiftV52AiClip")?.value);
    const b = Number(document.getElementById("kpAiCorrectionClip")?.value);
    const v = Number.isFinite(a) ? a : (Number.isFinite(b) ? b : 20);
    return Math.max(1, Math.min(60, v));
  }

  function gridStats(grid) {
    let min=Infinity, max=-Infinity, sum=0, n=0;
    for (const row of grid || []) {
      for (const v0 of row || []) {
        const v=Number(v0);
        if (!Number.isFinite(v)) continue;
        min=Math.min(min,v); max=Math.max(max,v); sum+=v; n++;
      }
    }
    return {min:n?min:NaN,max:n?max:NaN,mean:n?sum/n:NaN,n};
  }

  function validateBaseSlotCoverage(slots) {
    const unique = new Set((slots || []).map(x => String(x.slot_utc_hhmm || "").padStart(4,"0")));
    const count = unique.size;

    // Full ISEE native day = 288 five-minute slots.
    // For our 30-minute forecast, at least the 48 half-hour points must exist.
    let halfHourCount = 0;
    for (let h=0; h<24; h++) {
      for (const m of (0,30)) {
        const key = String(h).padStart(2,"0") + String(m).padStart(2,"0");
        if (unique.has(key)) halfHourCount++;
      }
    }

    return { count, halfHourCount };
  }

  function maxAbsGridDiff(a, b) {
    let mx = 0, n = 0, sum = 0;
    const nr = Math.min(a?.length || 0, b?.length || 0);
    for (let i=0; i<nr; i++) {
      const nc = Math.min(a?.[i]?.length || 0, b?.[i]?.length || 0);
      for (let j=0; j<nc; j++) {
        const x = Number(a[i][j]), y = Number(b[i][j]);
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        const d = Math.abs(x-y);
        mx = Math.max(mx,d); sum += d; n++;
      }
    }
    return { max: mx, mean: n ? sum/n : NaN, n };
  }

  async function runIseeMeanForecast() {
    status("ISEE時間別平均VTEC・Japan AI係数を読み込み中…");

    const [meanIndex, coeffDoc] = await Promise.all([
      loadMeanIndex(),
      loadGridCoefficients(),
    ]);

    const slots = meanIndex.slots || [];
    const coverage = validateBaseSlotCoverage(slots);

    if (coverage.halfHourCount < 40) {
      throw new Error(
        `ISEE時間別平均の時刻スロット不足: ${coverage.count}/288 (5分), ` +
        `${coverage.halfHourCount}/48 (30分)。` +
        `現在の時間別平均が1時間分などしか無いため、同じ格子を別時刻へ使い回してヒートマップが止まって見えます。` +
        `Update ISEE Japan VTEC and AI を実行して、少なくとも24時間分のISEE frameを取得してください。`
      );
    }

    const startUtc = roundedNowUtc30();
    const kpSeries = makeForecastKpSeries(startUtc);
    const nSteps = Math.round(FORECAST_HOURS * 60 / FORECAST_STEP_MIN);

    const frames = [];
    let gridMeta = null;
    let minDaysUsed = Infinity;
    let maxDaysUsed = 0;
    const useAi = aiEnabled();
    const clip = correctionClipTecU();
    const kpVals = kpSeries.map(r => Number(r.kp)).filter(Number.isFinite);
    const kpMin = kpVals.length ? Math.min(...kpVals) : NaN;
    const kpMax = kpVals.length ? Math.max(...kpVals) : NaN;
    let activeCoeffCells = 0;
    let coeffCellsChecked = false;
    let minAppliedDelta = Infinity;
    let maxAppliedDelta = -Infinity;

    for (let s=0; s<=nSteps; s++) {
      const t = new Date(startUtc.getTime() + s*FORECAST_STEP_MIN*60000);
      const resolved = resolveBaseSlot(slots, t);
      const slotEntry = resolved.entry;
      if (!slotEntry) {
        throw new Error(
          `ISEE時間別平均のUTC ${String(t.getUTCHours()).padStart(2,"0")}:` +
          `${String(t.getUTCMinutes()).padStart(2,"0")} slotがありません。` +
          `最寄りでも${Number.isFinite(resolved.diffMin) ? resolved.diffMin.toFixed(0) : "--"}分離れています。`
        );
      }

      const meanDoc = await loadMeanSlot(slotEntry);
      const meanGrid = meanDoc.grid || [];
      const latArr = meanDoc.lat_arr || [];
      const lonArr = meanDoc.lon_arr || [];
      if (!latArr.length || !lonArr.length || !meanGrid.length) {
        throw new Error(`ISEE平均VTEC格子が不正: ${slotEntry.file}`);
      }
      if (!gridMeta) {
        gridMeta = { latArr, lonArr, nLat:latArr.length, nLon:lonArr.length };
      }

      const daysUsed = Number(meanDoc.days_used || slotEntry.days_used || 0);
      if (Number.isFinite(daysUsed)) {
        minDaysUsed = Math.min(minDaysUsed, daysUsed);
        maxDaysUsed = Math.max(maxDaysUsed, daysUsed);
      }

      // Display-only KpB:
      // weighted mean of the historical Kp values that were removed from the
      // 10-day Base VTEC. It is NOT re-used in the forecast equation.
      const baseKpInfo = {kp:NaN,n:0}; // ISEE mode has no Base Kp

      const kpF = Number(kpSeries[s]?.kp);
      const mg = monthGrid(coeffDoc, t.getUTCMonth()+1);
      const outGrid = Array.from({length:gridMeta.nLat}, () => Array(gridMeta.nLon).fill(null));

      for (let i=0; i<gridMeta.nLat; i++) {
        for (let j=0; j<gridMeta.nLon; j++) {
          const b = Number(meanGrid?.[i]?.[j]);
          if (!Number.isFinite(b)) {
            outGrid[i][j] = null;
            continue;
          }
          const cf = coeffAt(mg, i, j);
          if (!coeffCellsChecked && cf && (Math.abs(cf.k0)+Math.abs(cf.k1)+Math.abs(cf.k2)+Math.abs(cf.k3) > 1e-10)) {
            activeCoeffCells++;
          }

          let kpTerm = useAi ? F(cf, kpF) : 0;
          if (!Number.isFinite(kpTerm)) kpTerm = 0;
          kpTerm = Math.max(-clip, Math.min(clip, kpTerm));
          minAppliedDelta = Math.min(minAppliedDelta, kpTerm);
          maxAppliedDelta = Math.max(maxAppliedDelta, kpTerm);

          const vtec = Math.max(0, Math.min(300, b + kpTerm));
          outGrid[i][j] = Number.isFinite(vtec) ? vtec : null;
        }
      }

      if (!coeffCellsChecked) {
        activeCoeffCells = 0;
        for (let ii=0; ii<gridMeta.nLat; ii++) {
          for (let jj=0; jj<gridMeta.nLon; jj++) {
            const cfx = coeffAt(mg, ii, jj);
            if (cfx && (Math.abs(cfx.k0)+Math.abs(cfx.k1)+Math.abs(cfx.k2)+Math.abs(cfx.k3) > 1e-10)) {
              activeCoeffCells++;
            }
          }
        }
        coeffCellsChecked = true;
      }

      frames.push({
        time:t,
        grid:outGrid,
        gridMeta,
        sourceFile:`ISEE Mean ${slotEntry.slot_utc_hhmm}Z`,
        baseKpDisplay: baseKpInfo.kp,
        baseDaysUsed: daysUsed,
        baseSlotUtc: String(slotEntry.slot_utc_hhmm || ""),
      });
    }

    if (!frames.length) throw new Error("ISEE 10日Base予報フレームを生成できませんでした。");

    // Verify that the produced map itself changes over time.
    // Compare +0h, +6h, +12h and +24h when available.
    const probeIdx = [0, 12, 24, 48].filter(i => i < frames.length);
    let variationMax = 0;
    let variationMeanMax = 0;
    for (let k=1; k<probeIdx.length; k++) {
      const d = maxAbsGridDiff(frames[probeIdx[0]].grid, frames[probeIdx[k]].grid);
      if (Number.isFinite(d.max)) variationMax = Math.max(variationMax, d.max);
      if (Number.isFinite(d.mean)) variationMeanMax = Math.max(variationMeanMax, d.mean);
    }

    if (typeof window.swiftInstallIsee10DayBaseForecast !== "function") {
      throw new Error("swifttec_v4_archive_dop.js がv8.10ではありません。");
    }

    window.swiftInstallIsee10DayBaseForecast(frames, kpSeries, {gridMeta});

    const daysTxt = Number.isFinite(minDaysUsed)
      ? (minDaysUsed === maxDaysUsed ? `${minDaysUsed}日` : `${minDaysUsed}〜${maxDaysUsed}日`)
      : "--";
    const deltaSpan = (Number.isFinite(minAppliedDelta) && Number.isFinite(maxAppliedDelta))
      ? `${minAppliedDelta.toFixed(2)}〜${maxAppliedDelta.toFixed(2)} TECU`
      : "--";

    const kpTxt = (Number.isFinite(kpMin) && Number.isFinite(kpMax))
      ? `${kpMin.toFixed(2)}〜${kpMax.toFixed(2)}`
      : "--";

    if (useAi && activeCoeffCells === 0) {
      status(
        `⚠ Japan予報は生成しましたが、Japan AI係数が全格子0です。` +
        `KpF=${kpTxt} / ISEE平均=${daysTxt} / 平均時刻=${coverage.halfHourCount}/48 / ` +
        `地図変化 max=${variationMax.toFixed(2)} TECU。`
      );
    } else {
      status(
        `Japan予報OK: ISEE平均=${daysTxt} / KpF=${kpTxt} / ` +
        `AI有効格子=${activeCoeffCells} / Kp補正=${deltaSpan} / ` +
        `平均時刻=${coverage.halfHourCount}/48 / ` +
        `地図変化 max=${variationMax.toFixed(2)} TECU, mean=${variationMeanMax.toFixed(2)} / ` +
        `+0〜+4日 / VTEC [TECU]`
      );
    }
    return frames;
  }

  window.swiftIseeLoadLatest = loadLatest;
  window.swiftIseeShowJapan = showLatestOnMap;
  window.swiftIseeRunMeanForecast = runIseeMeanForecast;
  window.swiftIseeRunMeanForecast = runIseeMeanForecast;
  window.swiftIseeRun10DayBaseForecast = runIseeMeanForecast; // backward-compatible alias // backward-compatible alias
})();
