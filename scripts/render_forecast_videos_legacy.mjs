#!/usr/bin/env node
/**
 * SWIFT-TEC legacy forecast video renderer.
 *
 * Produces:
 *   NOAA Global: L1 + VDOP×L1 vertical error
 *   ISEE Japan: L1 + VDOP×L1 vertical error
 *   NOAA Japan: L1 + HDOP×L1 horizontal error + VDOP×L1 vertical error
 *   docs/data/videos_legacy/index.json
 *
 * Retention:
 *   2 UTC calendar days total.
 */

import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { spawnSync } from "node:child_process";
import { chromium } from "playwright";

const ROOT = process.cwd();
const DOCS = path.join(ROOT, "docs");
const VIDEO_ROOT = path.join(DOCS, "data", "videos_legacy");
const LATEST_DIR = path.join(VIDEO_ROOT, "latest");
const ARCHIVE_DIR = path.join(VIDEO_ROOT, "archive");

const BASE_URL = process.env.SWIFTTEC_RENDER_URL || "http://127.0.0.1:8000/";
const MAX_CAPTURE_FRAMES = Math.max(24, Number(process.env.SWIFTTEC_VIDEO_MAX_FRAMES || 600));
const FRAME_DELAY_MS = Math.max(30, Number(process.env.SWIFTTEC_VIDEO_FRAME_DELAY_MS || 80));
const VIDEO_FPS = Math.max(1, Number(process.env.SWIFTTEC_VIDEO_FPS || 10));
const KEEP_DAYS = Math.max(1, Number(process.env.SWIFTTEC_VIDEO_KEEP_DAYS || 2));

const VIDEO_VIEWPORT = {
  width: Math.max(1280, Number(process.env.SWIFTTEC_VIDEO_WIDTH || 1700)),
  height: Math.max(720, Number(process.env.SWIFTTEC_VIDEO_HEIGHT || 900)),
};

// User-requested visual style from the current SWIFT-TEC UI.
const HEATMAP_STYLE = {
  alpha: 0.40,
  palette: "classic",
  reverse: false,
  colors: ["#0066ff", "#00e5e5", "#ff9f0a", "#ff0000"],
  // Legacy heatmap breakpoints unified to 5 / 10 / 20 / 30 m.
  gpsLimits: [5, 10, 20, 30],
  // DOP × L1 error scales.
  hdopTecLimits: [5, 10, 20, 30],
  vdopTecLimits: [5, 10, 20, 30],
};

// Video-only map camera.
// Global: Greenwich-centered world so Americas are on the LEFT and Japan/Asia on the RIGHT.
// ISEE: tighter Japan focus than the full 24–46N / 122–150E source grid.
const VIDEO_CAMERA = {
  global: {
    center: [12.0, 0.0],
    zoom: 2,
  },

  // Existing ISEE Japan camera. Do not change.
  japan: {
    bounds: [[27.0, 125.0], [46.5, 148.0]],
    padding: [18, 18],
  },

  // NOAA Japan:
  // 1) First calculate EXACTLY the same zoom as ISEE Japan
  //    by fitting the same reference bounds with the same padding.
  // 2) Then keep that zoom and move only the center 1.0 degree south.
  noaaJapanSameZoomSouth1: {
    referenceBounds: [[27.0, 125.0], [46.5, 148.0]],
    referencePadding: [18, 18],
    shiftLat: -1.0,
    shiftLng: 0.0,
  },
};

const MOVIE_TYPES = [
  {
    key: "l1",
    mapMode: "gps",
    label: "GNSS L1 ionospheric error",
    suffix: "l1_error",
    needsGpsDop: false,
  },
  {
    key: "horizontal",
    mapMode: "hdoptec",
    label: "GPS HDOP × L1 horizontal error",
    suffix: "horizontal_error",
    needsGpsDop: true,
    // Add horizontal error only to the NOAA Japan target.
    onlyTargets: ["noaa_japan"],
  },
  {
    key: "vertical",
    mapMode: "vdoptec",
    label: "GPS VDOP × L1 vertical error",
    suffix: "vertical_error",
    needsGpsDop: true,
  },
];

const RENDER_TARGETS = [
  {
    key: "noaa_global",
    source: "noaa",
    label: "NOAA / Global",
    cameraKey: "global",
    filePrefix: "noaa",
  },
  {
    key: "isee_japan",
    source: "isee",
    label: "ISEE Japan",
    cameraKey: "japan",
    filePrefix: "isee",
  },
  {
    key: "noaa_japan",
    source: "noaa",
    label: "NOAA Japan",
    cameraKey: "noaaJapanSameZoomSouth1",
    filePrefix: "noaa_japan",
  },
];

function pad2(n) {
  return String(n).padStart(2, "0");
}

function utcDayString(d = new Date()) {
  return `${d.getUTCFullYear()}-${pad2(d.getUTCMonth() + 1)}-${pad2(d.getUTCDate())}`;
}

function parsePagePassword() {
  if (process.env.SWIFTTEC_PAGE_PASSWORD) return process.env.SWIFTTEC_PAGE_PASSWORD;
  try {
    const html = fs.readFileSync(path.join(DOCS, "index.html"), "utf8");
    const m = html.match(/const\s+PASSWORD\s*=\s*["']([^"']+)["']/);
    return m?.[1] || "";
  } catch {
    return "";
  }
}

function ensureDirs() {
  fs.mkdirSync(LATEST_DIR, { recursive: true });
  fs.mkdirSync(ARCHIVE_DIR, { recursive: true });
}

function pruneArchives(now = new Date()) {
  ensureDirs();
  const cutoff = new Date(Date.UTC(
    now.getUTCFullYear(),
    now.getUTCMonth(),
    now.getUTCDate() - (KEEP_DAYS - 1),
    0, 0, 0, 0
  ));

  for (const name of fs.readdirSync(ARCHIVE_DIR)) {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(name)) continue;
    const d = new Date(`${name}T00:00:00Z`);
    if (!Number.isFinite(d.getTime())) continue;
    if (d < cutoff) {
      fs.rmSync(path.join(ARCHIVE_DIR, name), { recursive: true, force: true });
      console.log(`Pruned old video archive: ${name}`);
    }
  }
}

function ffmpegEncode(frameDir, outputPath) {
  const args = [
    "-y",
    "-hide_banner",
    "-loglevel", "warning",
    "-framerate", String(VIDEO_FPS),
    "-i", path.join(frameDir, "%05d.png"),
    "-vf", "scale=1280:-2:flags=lanczos,format=yuv420p",
    "-c:v", "libx264",
    "-preset", "medium",
    "-crf", "24",
    "-movflags", "+faststart",
    outputPath,
  ];

  const r = spawnSync("ffmpeg", args, { cwd: ROOT, stdio: "inherit" });
  if (r.status !== 0) throw new Error(`ffmpeg failed (${r.status}) for ${outputPath}`);
}

async function setForecastSource(page, source) {
  const wanted = source === "isee" ? "isee_japan_highres" : "archive_data_30m";

  await page.waitForFunction(() => !!document.getElementById("swiftV52TecSource"), null, {
    timeout: 60000,
  });

  const result = await page.evaluate(({ wantedValue, sourceName }) => {
    const sel = document.getElementById("swiftV52TecSource");
    if (!sel) return { ok: false, reason: "swiftV52TecSource missing" };

    const options = [...sel.options];
    let opt = options.find(o => o.value === wantedValue);

    if (!opt) {
      opt = options.find(o => {
        const t = String(o.textContent || "").toLowerCase();
        return sourceName === "isee"
          ? t.includes("isee")
          : (t.includes("noaa") || t.includes("global"));
      });
    }

    if (!opt) {
      return {
        ok: false,
        reason: `source option missing: ${wantedValue}`,
        options: options.map(o => ({ value: o.value, text: o.textContent })),
      };
    }

    sel.value = opt.value;
    sel.dispatchEvent(new Event("change", { bubbles: true }));
    return { ok: true, value: opt.value, text: opt.textContent };
  }, { wantedValue: wanted, sourceName: source });

  if (!result.ok) throw new Error(JSON.stringify(result));
  console.log(`${source}: selected ${result.value} / ${result.text}`);
  await page.waitForTimeout(700);

  if (source === "isee") {
    try {
      const b = page.getByRole("button", { name: /日本表示/ }).first();
      if (await b.count()) {
        await b.click({ timeout: 3000 });
        await page.waitForTimeout(500);
      }
    } catch {}
  }
}

async function runForecast(page, source) {
  console.log(`${source}: starting forecast`);

  const result = await page.evaluate(async () => {
    if (typeof window.swiftV52RunForecast === "function") {
      try {
        const x = await window.swiftV52RunForecast();
        return { mode: "api", result: x ?? null };
      } catch (e) {
        return { mode: "api", error: String(e?.stack || e?.message || e) };
      }
    }
    return { mode: "button" };
  });

  if (result?.error) {
    throw new Error(`${source}: forecast API failed: ${result.error}`);
  }

  if (result?.mode === "button") {
    const button = page.getByRole("button", { name: /^予報実行$/ }).first();
    await button.click({ timeout: 20000 });
  }

  await page.waitForFunction(() => {
    const status = String(document.getElementById("swiftV52Status")?.textContent || "");
    if (/予報失敗|失敗:|error/i.test(status)) throw new Error(status);

    const slider = document.getElementById("timeSlider");
    if (!slider) return false;
    const max = Number(slider.max || 0);
    return Number.isFinite(max) && max >= 2;
  }, null, { timeout: 180000 });

  await page.waitForTimeout(1200);

  const state = await page.evaluate(() => ({
    status: document.getElementById("swiftV52Status")?.textContent || "",
    sliderMin: Number(document.getElementById("timeSlider")?.min || 0),
    sliderMax: Number(document.getElementById("timeSlider")?.max || 0),
  }));

  console.log(`${source}: forecast ready`, state);
  return state;
}

async function ensureVideoOverlay(page, target, movieType) {
  await page.evaluate(({ targetLabel, movieLabel }) => {
    const mapEl = document.getElementById("tecMap");
    if (!mapEl) return;

    let el = document.getElementById("swiftVideoRenderStamp");
    if (!el) {
      el = document.createElement("div");
      el.id = "swiftVideoRenderStamp";
      Object.assign(el.style, {
        position: "absolute",
        left: "12px",
        bottom: "12px",
        zIndex: "99999",
        background: "rgba(2,6,23,.82)",
        color: "#fff",
        border: "1px solid rgba(147,197,253,.75)",
        borderRadius: "10px",
        padding: "7px 10px",
        font: "700 13px/1.35 system-ui,-apple-system,sans-serif",
        whiteSpace: "pre-line",
        pointerEvents: "none",
        boxShadow: "0 5px 18px rgba(0,0,0,.35)",
      });
      mapEl.appendChild(el);
    }

    el.dataset.targetLabel = targetLabel;
    el.dataset.movieLabel = movieLabel;
  }, {
    targetLabel: target.label,
    movieLabel: movieType.label,
  });
}

async function moveSliderAndStamp(page, index, target, movieType) {
  await page.evaluate(({ idx, sourceName, targetLabel, movieLabel }) => {
    const slider = document.getElementById("timeSlider");
    if (!slider) throw new Error("timeSlider missing");

    slider.value = String(idx);
    slider.dispatchEvent(new Event("input", { bubbles: true }));
    slider.dispatchEvent(new Event("change", { bubbles: true }));

    const dockSlider = document.getElementById("swiftTimelineSliderV64");
    if (dockSlider) dockSlider.value = String(idx);

    const stamp = document.getElementById("swiftVideoRenderStamp");
    if (stamp) {
      const t =
        document.getElementById("timeOverlay")?.textContent?.trim() ||
        document.getElementById("timeLabel")?.textContent?.trim() ||
        document.getElementById("currentTimeLabel")?.textContent?.trim() ||
        `frame ${idx}`;

      const kpF = document.getElementById("swiftV56KpF")?.textContent?.trim() || "--";
      const kpB =
        document.getElementById("swiftV56KpB")?.textContent?.trim() ||
        (sourceName === "isee" ? "なし" : "--");

      stamp.textContent =
        `${targetLabel} / ${movieLabel}\n${t}\nKpF ${kpF} / KpB ${kpB}`;
    }
  }, {
    idx: index,
    sourceName: target.source,
    targetLabel: target.label,
    movieLabel: movieType.label,
  });

  await page.waitForTimeout(FRAME_DELAY_MS);
}

async function configureGpsOnlyDop(page) {
  // Wait for the visible GNSS panel when possible. The core old checkboxes are
  // also set as a fallback.
  await page.waitForTimeout(500);

  const result = await page.evaluate(async () => {
    const keys = ["gps", "galileo", "glonass", "beidou", "qzss"];

    for (const key of keys) {
      const wanted = key === "gps";
      const visible = document.getElementById(`gnssConstV66_${key}`);
      const legacy = document.getElementById(`gnssConst_${key}`);

      if (visible) {
        visible.checked = wanted;
        visible.dispatchEvent(new Event("change", { bubbles: true }));
      }
      if (legacy) {
        legacy.checked = wanted;
        legacy.dispatchEvent(new Event("change", { bubbles: true }));
      }
    }

    if (typeof window.loadGnssDopData !== "function") {
      return { ok: false, reason: "loadGnssDopData missing" };
    }

    await window.loadGnssDopData();

    // Apply saved GPS Almanac health if available. Unhealthy GPS satellites
    // remain inactive and are not forced back on.
    try {
      await window.swiftApplySavedAlmanacHealthV66?.(false);
    } catch {}

    const status =
      document.getElementById("gnssQuickStatus")?.textContent ||
      document.getElementById("swiftGnssVisibleStatusV66")?.textContent ||
      "";

    return { ok: true, status };
  });

  if (!result.ok) throw new Error(`GPS DOP setup failed: ${result.reason}`);
  console.log(`GPS-only DOP ready: ${result.status}`);
  await page.waitForTimeout(800);
}

async function applyRequestedVisualStyle(page, movieType) {
  const limits =
    movieType.mapMode === "hdoptec"
      ? HEATMAP_STYLE.hdopTecLimits
      : movieType.mapMode === "vdoptec"
        ? HEATMAP_STYLE.vdopTecLimits
        : HEATMAP_STYLE.gpsLimits;

  await page.evaluate(({ mapMode, style, limits }) => {
    // 1) Heatmap opacity
    const alpha = document.getElementById("tecAlpha");
    if (alpha) {
      alpha.value = String(style.alpha);
      alpha.dispatchEvent(new Event("input", { bubbles: true }));
      alpha.dispatchEvent(new Event("change", { bubbles: true }));
      try { window.onTecAlphaChange?.(); } catch {}
    }

    // 2) Classic palette / no reverse (legacy selectors + v7.4 unified editor)
    try {
      localStorage.setItem("swiftHeatmapPaletteV68", style.palette);
      localStorage.setItem("swiftHeatmapPaletteReverseV68", style.reverse ? "1" : "0");

      const store = JSON.parse(
        localStorage.getItem("swiftUnifiedHeatmapScaleV74") || "{}"
      );

      const group = ["hdoptec", "vdoptec"].includes(mapMode) ? "doptec" : "gps";
      store[group] = {
        limits: [...limits],
        colors: [...style.colors],
        label: group === "gps"
          ? "L1電離圏誤差 [m]"
          : "DOP × L1誤差 [m]",
        unit: "m",
      };
      localStorage.setItem("swiftUnifiedHeatmapScaleV74", JSON.stringify(store));
    } catch {}

    const pal = document.getElementById("swiftUnifiedPaletteV74");
    if (pal) pal.value = "classic";

    const reverse = document.getElementById("swiftUnifiedReverseV74");
    if (reverse) reverse.checked = false;

    const legacyPal = document.getElementById("swiftHeatmapPaletteSelectV68");
    if (legacyPal) legacyPal.value = "classic";

    const legacyRev = document.getElementById("swiftHeatmapPaletteReverseV68");
    if (legacyRev) legacyRev.checked = false;

    const groupSel = document.getElementById("swiftUnifiedGroupV74");
    if (groupSel) groupSel.value = ["hdoptec", "vdoptec"].includes(mapMode) ? "doptec" : "gps";

    for (let i = 0; i < 4; i++) {
      const c = document.getElementById(`swiftUnifiedColor${i + 1}V74`);
      const n = document.getElementById(`swiftUnifiedLimit${i + 1}V74`);
      if (c) c.value = style.colors[i];
      if (n) n.value = String(limits[i]);
    }

    try { window.swiftApplyUnifiedHeatmapScaleV74?.(); } catch {}

    // 3) Requested map metric
    const mode = document.getElementById("mapModeSelect");
    if (!mode) throw new Error("mapModeSelect missing");

    const hasMode = [...mode.options].some(o => o.value === mapMode);
    if (!hasMode) {
      throw new Error(
        `map mode ${mapMode} missing: ` +
        [...mode.options].map(o => o.value).join(",")
      );
    }

    mode.value = mapMode;
    mode.dispatchEvent(new Event("change", { bubbles: true }));
    try { window.changeMapMode?.(); } catch {}

    const dockMode = document.getElementById("swiftV64MapModeSelect");
    if (dockMode && [...dockMode.options].some(o => o.value === mapMode)) {
      dockMode.value = mapMode;
    }

    // 4) Redraw legend and map
    try { window.swiftRefreshHeatmapPaletteV68?.(); } catch {}
    try { window.updateLegend?.(); } catch {}
    try { window.requestDraw?.(); } catch {}
  }, {
    mapMode: movieType.mapMode,
    style: HEATMAP_STYLE,
    limits,
  });

  await page.waitForTimeout(700);
}

async function applyLegacyCamera(page, target) {
  const result = await page.evaluate(({ targetConfig, cameraConfig }) => {
    try {
      if (typeof map === "undefined" || !map?.setView || !map?.invalidateSize) {
        return { ok: false, reason: "Leaflet map variable is unavailable" };
      }

      map.invalidateSize({ pan: false });

      const camera = cameraConfig[targetConfig.cameraKey];
      if (!camera) {
        return { ok: false, reason: `camera ${targetConfig.cameraKey} missing` };
      }

      if (Array.isArray(camera.referenceBounds)) {
        // NOAA Japan special camera:
        // derive the zoom from the exact same bounds/padding as ISEE Japan,
        // then move only the center. This guarantees the same Leaflet zoom.
        map.fitBounds(camera.referenceBounds, {
          padding: camera.referencePadding || [10, 10],
          animate: false,
        });

        const refCenter = map.getCenter();
        const refZoom = map.getZoom();

        map.setView(
          [
            refCenter.lat + Number(camera.shiftLat || 0),
            refCenter.lng + Number(camera.shiftLng || 0),
          ],
          refZoom,
          { animate: false }
        );
      } else if (Array.isArray(camera.center) && Number.isFinite(camera.zoom)) {
        map.setView(camera.center, camera.zoom, { animate: false });
      } else if (camera.bounds) {
        map.fitBounds(camera.bounds, {
          padding: camera.padding || [10, 10],
          animate: false,
        });
      } else {
        return { ok: false, reason: `camera ${targetConfig.cameraKey} has no valid view settings` };
      }

      map.invalidateSize({ pan: false });

      const c = map.getCenter();
      return {
        ok: true,
        target: targetConfig.key,
        center: [Number(c.lat.toFixed(3)), Number(c.lng.toFixed(3))],
        zoom: map.getZoom(),
      };
    } catch (e) {
      return { ok: false, reason: String(e?.stack || e?.message || e) };
    }
  }, {
    targetConfig: target,
    cameraConfig: VIDEO_CAMERA,
  });

  if (!result.ok) {
    throw new Error(`${target.key}: map camera failed: ${result.reason}`);
  }

  console.log(`${target.key}: video camera`, result);

  await page.evaluate(() => {
    try { window.dispatchEvent(new Event("resize")); } catch {}
    try { window.requestDraw?.(); } catch {}
    try { window.swiftResetHeatmapCacheV830?.(); } catch {}
  });

  await page.waitForTimeout(700);
}

async function enterRequestedMapView(page, target) {
  await page.evaluate(() => {
    document.body.classList.remove("swift-map-fs-on");

    if (typeof window.swiftEnterMapFocusMode === "function") {
      window.swiftEnterMapFocusMode();
    } else {
      document.documentElement.classList.add("swift-map-focus");
    }

    try { window.dispatchEvent(new Event("resize")); } catch {}
  });

  await page.waitForTimeout(450);
  await applyLegacyCamera(page, target);
}

async function reapplyVideoCamera(page, target) {
  await applyLegacyCamera(page, target);
}


function captureIndices(min, max) {
  const total = Math.max(0, max - min + 1);
  if (total <= 0) return [];
  const stride = Math.max(1, Math.ceil(total / MAX_CAPTURE_FRAMES));
  const out = [];
  for (let i = min; i <= max; i += stride) out.push(i);
  if (out[out.length - 1] !== max) out.push(max);
  return out;
}

async function renderOneTarget(browser, target, archiveDayDir, password) {
  const source = target.source;

  const page = await browser.newPage({
    viewport: VIDEO_VIEWPORT,
    deviceScaleFactor: 1,
  });

  page.setDefaultTimeout(60000);

  page.on("console", msg => {
    const type = msg.type();
    if (type === "error" || type === "warning") {
      console.log(`[browser ${target.key} ${type}] ${msg.text()}`);
    }
  });

  page.on("pageerror", err => {
    console.log(`[browser ${target.key} pageerror] ${err.stack || err.message}`);
  });

  page.on("dialog", async dialog => {
    try {
      if (dialog.type() === "prompt") await dialog.accept(password || "");
      else await dialog.accept();
    } catch {}
  });

  try {
    await page.goto(BASE_URL, {
      waitUntil: "domcontentloaded",
      timeout: 120000,
    });

    await page.waitForSelector("#tecMap", { timeout: 90000 });
    await page.waitForTimeout(1400);

    await setForecastSource(page, source);
    const forecastState = await runForecast(page, source);

    await configureGpsOnlyDop(page);
    await enterRequestedMapView(page, target);

    const outputs = [];

    for (const movieType of MOVIE_TYPES) {
      if (
        Array.isArray(movieType.onlyTargets) &&
        !movieType.onlyTargets.includes(target.key)
      ) {
        continue;
      }

      console.log(`${target.key}/${movieType.key}: preparing visual style`);

      await applyRequestedVisualStyle(page, movieType);
      await reapplyVideoCamera(page, target);
      await ensureVideoOverlay(page, target, movieType);

      const slider = page.locator("#timeSlider");
      const bounds = await slider.evaluate(el => ({
        min: Number(el.min || 0),
        max: Number(el.max || 0),
      }));

      const indices = captureIndices(bounds.min, bounds.max);
      if (indices.length < 2) {
        throw new Error(`${target.key}/${movieType.key}: too few frames (${indices.length})`);
      }

      const tmp = fs.mkdtempSync(
        path.join(os.tmpdir(), `swifttec-${target.key}-${movieType.key}-`)
      );

      let frameNo = 0;
      for (const idx of indices) {
        await moveSliderAndStamp(page, idx, target, movieType);

        const framePath = path.join(
          tmp,
          `${String(frameNo).padStart(5, "0")}.png`
        );

        await page.screenshot({
          path: framePath,
          type: "png",
          fullPage: false,
        });

        frameNo++;
      }

      const fileName = `${target.filePrefix}_${movieType.suffix}.mp4`;
      const archiveOutput = path.join(archiveDayDir, fileName);
      ffmpegEncode(tmp, archiveOutput);
      fs.rmSync(tmp, { recursive: true, force: true });

      const latestOutput = path.join(LATEST_DIR, fileName);
      fs.copyFileSync(archiveOutput, latestOutput);

      outputs.push({
        target: target.key,
        source,
        camera: target.cameraKey,
        label: target.label,
        movie: movieType.key,
        metric: movieType.label,
        map_mode: movieType.mapMode,
        file: path.relative(DOCS, archiveOutput).replaceAll("\\", "/"),
        latest: path.relative(DOCS, latestOutput).replaceAll("\\", "/"),
        frames: indices.length,
        slider_min: bounds.min,
        slider_max: bounds.max,
        heatmap_alpha: HEATMAP_STYLE.alpha,
        palette: HEATMAP_STYLE.palette,
        thresholds_m:
          movieType.mapMode === "hdoptec"
            ? HEATMAP_STYLE.hdopTecLimits
            : movieType.mapMode === "vdoptec"
              ? HEATMAP_STYLE.vdopTecLimits
              : HEATMAP_STYLE.gpsLimits,
        colors: HEATMAP_STYLE.colors,
        status: forecastState.status || "",
        bytes: fs.statSync(archiveOutput).size,
      });
    }

    return outputs;
  } finally {
    await page.close();
  }
}

function buildIndex(results) {
  const fileFor = (day, prefix, suffix) => {
    const p = path.join(ARCHIVE_DIR, day, `${prefix}_${suffix}.mp4`);
    return fs.existsSync(p)
      ? `data/videos_legacy/archive/${day}/${prefix}_${suffix}.mp4`
      : null;
  };

  const days = fs.readdirSync(ARCHIVE_DIR)
    .filter(name => /^\d{4}-\d{2}-\d{2}$/.test(name))
    .sort()
    .reverse()
    .slice(0, KEEP_DAYS)
    .map(day => ({
      day_utc: day,
      noaa_l1_error: fileFor(day, "noaa", "l1_error"),
      noaa_vertical_error: fileFor(day, "noaa", "vertical_error"),
      isee_l1_error: fileFor(day, "isee", "l1_error"),
      isee_vertical_error: fileFor(day, "isee", "vertical_error"),
      noaa_japan_l1_error: fileFor(day, "noaa_japan", "l1_error"),
      noaa_japan_horizontal_error: fileFor(day, "noaa_japan", "horizontal_error"),
      noaa_japan_vertical_error: fileFor(day, "noaa_japan", "vertical_error"),
    }));

  const doc = {
    version: "swifttec-forecast-video-legacy-v6-noaa-japan-same-zoom-south1",
    updated_utc: new Date().toISOString(),
    keep_days: KEEP_DAYS,
    visual: {
      viewport: VIDEO_VIEWPORT,
      heatmap_alpha: HEATMAP_STYLE.alpha,
      palette: HEATMAP_STYLE.palette,
      reverse: HEATMAP_STYLE.reverse,
      colors: HEATMAP_STYLE.colors,
      gps_limits_m: HEATMAP_STYLE.gpsLimits,
      horizontal_limits_m: HEATMAP_STYLE.hdopTecLimits,
      vertical_limits_m: HEATMAP_STYLE.vdopTecLimits,
      gps_constellation_only: true,
      horizontal_metric: "GPS HDOP × L1 horizontal error",
      vertical_metric: "GPS VDOP × L1 vertical error",
      global_camera: VIDEO_CAMERA.global,
      japan_camera: VIDEO_CAMERA.japan,
      noaa_japan_camera: VIDEO_CAMERA.noaaJapanSameZoomSouth1,
    },
    latest: {
      noaa_l1_error: "data/videos_legacy/latest/noaa_l1_error.mp4",
      noaa_vertical_error: "data/videos_legacy/latest/noaa_vertical_error.mp4",
      isee_l1_error: "data/videos_legacy/latest/isee_l1_error.mp4",
      isee_vertical_error: "data/videos_legacy/latest/isee_vertical_error.mp4",
      noaa_japan_l1_error: "data/videos_legacy/latest/noaa_japan_l1_error.mp4",
      noaa_japan_horizontal_error: "data/videos_legacy/latest/noaa_japan_horizontal_error.mp4",
      noaa_japan_vertical_error: "data/videos_legacy/latest/noaa_japan_vertical_error.mp4",
    },
    current_run: results,
    archive: days,
  };

  fs.writeFileSync(
    path.join(VIDEO_ROOT, "index.json"),
    JSON.stringify(doc, null, 2) + "\n",
    "utf8"
  );
}

async function main() {
  ensureDirs();
  pruneArchives();

  const today = utcDayString();
  const archiveDayDir = path.join(ARCHIVE_DIR, today);
  fs.mkdirSync(archiveDayDir, { recursive: true });

  const password = parsePagePassword();

  const browser = await chromium.launch({
    headless: true,
    args: ["--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu"],
  });

  const results = [];
  try {
    for (const target of RENDER_TARGETS) {
      results.push(
        ...(await renderOneTarget(browser, target, archiveDayDir, password))
      );
    }
  } finally {
    await browser.close();
  }

  pruneArchives();
  buildIndex(results);

  console.log("Video generation complete:");
  console.log(JSON.stringify(results, null, 2));
}

main().catch(err => {
  console.error(err?.stack || err);
  process.exit(1);
});
