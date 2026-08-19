/* SWIFT-TEC ISEE Japan High-Res TEC add-on */
(function () {
  "use strict";

  const INDEX_URL = "data/isee_tec/index.json";
  const BASE_URL = "data/isee_tec/";
  window.swiftIseeFrames = window.swiftIseeFrames || [];
  window.swiftIseeTimes = window.swiftIseeTimes || [];

  function status(msg) {
    const el = document.getElementById("iseeTecStatus");
    if (el) el.textContent = msg;
    if (typeof logInfo === "function") logInfo(msg);
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

  async function loadLatest(showOnMap) {
    try {
      status("ISEE Japan High-Res TECを読込中...");
      const idx = await getJson(INDEX_URL);
      const entries = Array.isArray(idx.frames) ? idx.frames.slice() : [];
      entries.sort((a,b) => String(a.time_utc).localeCompare(String(b.time_utc)));
      if (!entries.length) throw new Error("data/isee_tec/index.json にframeがありません");

      // Keep latest 1 hour / max 13 frames. 5-min products are expected.
      const latestMs = new Date(entries[entries.length - 1].time_utc).getTime();
      const selected = entries.filter(e => {
        const ms = new Date(e.time_utc).getTime();
        return isFinite(ms) && ms >= latestMs - 60 * 60 * 1000;
      }).slice(-13);

      const loaded = [];
      for (const e of selected) {
        loaded.push(normalizeFrame(await getJson(BASE_URL + e.file), e));
      }
      window.swiftIseeFrames = loaded;
      window.swiftIseeTimes = loaded.map(f => f.validTime);

      const first = loaded[0], last = loaded[loaded.length - 1];
      status(`ISEE読込OK: ${loaded.length}枚 / ${first.nLat}×${first.nLon} / ${first.validTime.toISOString()} ～ ${last.validTime.toISOString()}`);

      const sel = document.getElementById("tecSourceSelect");
      if (sel) sel.value = "isee";
      if (typeof fillForecastStartCandidates === "function") fillForecastStartCandidates();

      if (showOnMap) {
        gGrid = { latArr:first.latArr, lonArr:first.lonArr, nLat:first.nLat, nLon:first.nLon };
        gForecastFrames = loaded.map(f => f.grid);
        gForecastTimes = loaded.map(f => f.validTime);
        gForecastStart = gForecastTimes[0];
        currentStepIndex = 0;
        const slider = document.getElementById("timeSlider");
        if (slider) {
          slider.min = "0";
          slider.max = String(Math.max(0, gForecastTimes.length - 1));
          slider.value = "0";
        }
        if (typeof initMapIfNeeded === "function") initMapIfNeeded();
        if (typeof gMap !== "undefined" && gMap && typeof gMap.fitBounds === "function") {
          gMap.fitBounds([[24,122],[46,150]], { padding:[8,8] });
        }
        if (typeof updateLegend === "function") updateLegend();
        if (typeof requestDraw === "function") requestDraw();
      }
    } catch (e) {
      console.error(e);
      status("ISEE読込失敗: " + e.message);
    }
  }

  window.swiftIseeLoadLatest = loadLatest;
})();
