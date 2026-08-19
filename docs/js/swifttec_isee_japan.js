/* SWIFT-TEC v8.7 ISEE Japan High-Res TEC add-on */
(function () {
  "use strict";

  const INDEX_URL = "data/isee_tec/index.json";
  const BASE_URL = "data/isee_tec/";
  window.swiftIseeFrames = window.swiftIseeFrames || [];
  window.swiftIseeTimes = window.swiftIseeTimes || [];

  function setStatus(msg) {
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
      sourceFile: obj.source_file || entry?.file || "ISEE",
    };
  }

  async function loadLatest(showOnMap = false) {
    try {
      setStatus("ISEE Japan High-Res TECを読込中…");
      const idx = await getJson(INDEX_URL);
      const entries = Array.isArray(idx.frames) ? idx.frames.slice() : [];
      entries.sort((a,b) => String(a.time_utc).localeCompare(String(b.time_utc)));
      if (!entries.length) throw new Error("data/isee_tec/index.json にframeがありません");

      // Load latest ~24h when available. This is enough for Base/forecast input
      // while the map itself only renders the selected frame.
      const latestMs = new Date(entries[entries.length - 1].time_utc).getTime();
      const selected = entries.filter(e => {
        const ms = new Date(e.time_utc).getTime();
        return Number.isFinite(ms) && ms >= latestMs - 24 * 3600 * 1000;
      }).slice(-320);

      const loaded = [];
      for (const e of selected) {
        loaded.push(normalizeFrame(await getJson(BASE_URL + e.file), e));
      }
      if (!loaded.length) throw new Error("ISEEフレームを読み込めませんでした");

      window.swiftIseeFrames = loaded;
      window.swiftIseeTimes = loaded.map(f => f.validTime);

      const legacy = document.getElementById("tecSourceSelect");
      if (legacy) legacy.value = "isee";

      const compact = document.getElementById("swiftV52TecSource");
      if (compact) compact.value = "isee_japan_highres";

      const first = loaded[0], last = loaded[loaded.length - 1];
      setStatus(`ISEE読込OK: ${loaded.length}枚 / ${first.nLat}×${first.nLon} / ${first.validTime.toISOString()} ～ ${last.validTime.toISOString()}`);

      if (showOnMap) showLatestOnMap();
      return loaded;
    } catch (e) {
      console.error(e);
      setStatus("ISEE読込失敗: " + e.message);
      throw e;
    }
  }

  function showLatestOnMap() {
    const loaded = window.swiftIseeFrames || [];
    if (!loaded.length) {
      loadLatest(true);
      return;
    }
    const f = loaded[loaded.length - 1];
    try {
      window.gGrid = { latArr:f.latArr, lonArr:f.lonArr, nLat:f.nLat, nLon:f.nLon };
    } catch {}
    try {
      gGrid = { latArr:f.latArr, lonArr:f.lonArr, nLat:f.nLat, nLon:f.nLon };
      gForecastFrames = loaded.map(x => x.grid);
      gForecastTimes = loaded.map(x => x.validTime);
      gForecastStart = gForecastTimes[0] || null;
      currentStepIndex = Math.max(0, gForecastTimes.length - 1);

      const slider = document.getElementById("timeSlider");
      if (slider) {
        slider.min = "0";
        slider.max = String(Math.max(0, gForecastTimes.length - 1));
        slider.value = String(currentStepIndex);
      }

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
      setStatus("ISEE日本表示に失敗: " + e.message);
    }
  }

  window.swiftIseeLoadLatest = loadLatest;
  window.swiftIseeShowJapan = showLatestOnMap;
})();
