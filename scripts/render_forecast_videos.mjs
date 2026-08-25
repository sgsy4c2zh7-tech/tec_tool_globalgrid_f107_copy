#!/usr/bin/env node
/**
 * SWIFT-TEC scheduled forecast video renderer — multi-region edition.
 *
 * Runs at 09:00 and 14:00 JST.
 * Retention: one JST calendar day. Both runs are kept.
 *
 * Each configured view generates:
 *   - TEC-based GNSS L1 ionospheric error [m]
 *   - GPS-only VDOP × L1 vertical error [m]
 */

import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { spawnSync } from "node:child_process";
import { chromium } from "playwright";

const ROOT = process.cwd();
const DOCS = path.join(ROOT, "docs");
const VIDEO_ROOT = path.join(DOCS, "data", "videos");
const LATEST_DIR = path.join(VIDEO_ROOT, "latest");
const ARCHIVE_DIR = path.join(VIDEO_ROOT, "archive");

const BASE_URL = process.env.SWIFTTEC_RENDER_URL || "http://127.0.0.1:8000/";
const MAX_CAPTURE_FRAMES = Math.max(24, Number(process.env.SWIFTTEC_VIDEO_MAX_FRAMES || 600));
const FRAME_DELAY_MS = Math.max(30, Number(process.env.SWIFTTEC_VIDEO_FRAME_DELAY_MS || 80));
const VIDEO_FPS = Math.max(1, Number(process.env.SWIFTTEC_VIDEO_FPS || 10));
const KEEP_DAYS = Math.max(1, Number(process.env.SWIFTTEC_VIDEO_KEEP_DAYS || 1));

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
  // The screenshot uses 3 / 5 / 10 / 20 m for L1.
  gpsLimits: [3, 5, 10, 20],
  // Keep the same visual scale for the vertical-error movie as requested.
  vdopTecLimits: [3, 5, 10, 20],
};

// Video camera presets.
// Existing views plus the three newly requested regional views.
const VIDEO_VIEWS = [
  {
    key: "global",
    sources: ["noaa"],
    label: "Global",
    camera: {
      type: "setView",
      center: [12.0, 0.0],
      zoom: 2,
    },
    thresholds: [3, 5, 10, 20],
  },
  {
    key: "japan",
    sources: ["isee"],
    label: "Japan",
    camera: {
      type: "fitBounds",
      bounds: [[27.0, 125.0], [46.5, 148.0]],
      padding: [18, 18],
    },
    thresholds: [3, 5, 10, 20],
  },

  // Added view 1: NOAA, roughly the first screenshot.
  {
    key: "east_asia",
    sources: ["noaa"],
    label: "East Asia Wide",
    camera: {
      type: "fitBounds",
      bounds: [[-10.0, 70.0], [60.0, 165.0]],
      padding: [12, 12],
    },
    thresholds: [3, 5, 10, 20],
  },

  // Added view 2: NOAA and ISEE, roughly the second screenshot.
  {
    key: "japan_close",
    sources: ["noaa", "isee"],
    label: "Japan Close",
    camera: {
      type: "fitBounds",
      bounds: [[22.0, 118.0], [47.0, 151.0]],
      padding: [10, 10],
    },
    thresholds: [3, 5, 10, 20],
  },

  // Added view 3: NOAA, roughly the third screenshot.
  // Same colors and opacity. Only breakpoints change to 5/10/20/30.
  {
    key: "philippines",
    sources: ["noaa"],
    label: "Philippines / South China Sea",
    camera: {
      type: "fitBounds",
      bounds: [[-11.0, 103.0], [29.0, 146.0]],
      padding: [10, 10],
    },
    thresholds: [5, 10, 20, 30],
  },
];

const MOVIE_TYPES = [
  {
    key: "l1",
    mapMode: "gps",
    label: "GNSS L1 ionospheric error",
    suffix: "l1_error",
    needsGpsDop: false,
  },
  {
    key: "vertical",
    mapMode: "vdoptec",
    label: "GPS VDOP × L1 vertical error",
    suffix: "vertical_error",
    needsGpsDop: true,
  },
];

function pad2(n) {
  return String(n).padStart(2, "0");
}

function utcDayString(d = new Date()) {
  return `${d.getUTCFullYear()}-${pad2(d.getUTCMonth() + 1)}-${pad2(d.getUTCDate())}`;
}

function jstParts(d = new Date()) {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Tokyo",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  }).formatToParts(d);

  const v = {};
  for (const p of parts) {
    if (p.type !== "literal") v[p.type] = p.value;
  }

  return {
    day: `${v.year}-${v.month}-${v.day}`,
    hhmm: `${v.hour}${v.minute}`,
  };
}

function currentJstRunSlot(d = new Date()) {
  const p = jstParts(d);
  const hh = Number(p.hhmm.slice(0, 2));
  const mm = Number(p.hhmm.slice(2, 4));
  const total = hh * 60 + mm;

  // GitHub cron can start a few minutes late.
  if (Math.abs(total - 9 * 60) <= 30) return { day: p.day, slot: "0900" };
  if (Math.abs(total - 14 * 60) <= 30) return { day: p.day, slot: "1400" };

  // Manual workflow_dispatch: keep its real JST time.
  return { day: p.day, slot: p.hhmm };
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

  // Retention is counted in JST calendar days.
  // KEEP_DAYS=1 means keep today's folder, including BOTH 0900 and 1400.
  const today = jstParts(now).day;
  const [y, m, d] = today.split("-").map(Number);

  const keep = new Set();
  for (let i = 0; i < KEEP_DAYS; i++) {
    const x = new Date(Date.UTC(y, m - 1, d - i, 12, 0, 0));
    keep.add(utcDayString(x));
  }

  for (const name of fs.readdirSync(ARCHIVE_DIR)) {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(name)) continue;

    if (!keep.has(name)) {
      fs.rmSync(
        path.join(ARCHIVE_DIR, name),
        { recursive: true, force: true }
      );
      console.log(`Pruned old JST video archive: ${name}`);
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

async function ensureVideoOverlay(page, source, movieType, view) {
  await page.evaluate(({ sourceName, movieLabel, viewLabel }) => {
    const map = document.getElementById("tecMap");
    if (!map) return;

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
      map.appendChild(el);
    }

    el.dataset.source = sourceName;
    el.dataset.movieLabel = movieLabel;
    el.dataset.viewLabel = viewLabel;
  }, {
    sourceName: source,
    movieLabel: movieType.label,
    viewLabel: view.label,
  });
}

async function moveSliderAndStamp(page, index, source, movieType, view) {
  await page.evaluate(({ idx, sourceName, movieLabel, viewLabel }) => {
    const slider = document.getElementById("timeSlider");
    if (!slider) throw new Error("timeSlider missing");

    slider.value = String(idx);
    slider.dispatchEvent(new Event("input", { bubbles: true }));
    slider.dispatchEvent(new Event("change", { bubbles: true }));

    // v6.4 dock keeps its own slider; sync it for a screenshot matching the UI.
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

      const label = sourceName === "isee" ? "ISEE Japan" : "NOAA / Global";
      stamp.textContent =
        `${label} / ${viewLabel} / ${movieLabel}\n${t}\nKpF ${kpF} / KpB ${kpB}`;
    }
  }, {
    idx: index,
    sourceName: source,
    movieLabel: movieType.label,
    viewLabel: view.label,
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

async function applyRequestedVisualStyle(page, movieType, view) {
  const limits = [...view.thresholds];

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

      const group = mapMode === "vdoptec" ? "doptec" : "gps";
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
    if (groupSel) groupSel.value = mapMode === "vdoptec" ? "doptec" : "gps";

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

async function applyLeafletVideoCamera(page, view) {
  const result = await page.evaluate(({ viewKey, camera }) => {
    try {
      if (typeof map === "undefined" || !map?.setView || !map?.invalidateSize) {
        return {
          ok: false,
          reason: "Leaflet map variable is unavailable",
        };
      }

      map.invalidateSize({ pan: false });

      if (camera.type === "fitBounds") {
        map.fitBounds(camera.bounds, {
          padding: camera.padding || [10, 10],
          animate: false,
        });
      } else {
        map.setView(
          camera.center,
          camera.zoom,
          { animate: false }
        );
      }

      map.invalidateSize({ pan: false });

      const center = map.getCenter();
      return {
        ok: true,
        view: viewKey,
        center: [
          Number(center.lat.toFixed(3)),
          Number(center.lng.toFixed(3)),
        ],
        zoom: map.getZoom(),
      };
    } catch (e) {
      return {
        ok: false,
        reason: String(e?.stack || e?.message || e),
      };
    }
  }, {
    viewKey: view.key,
    camera: view.camera,
  });

  if (!result.ok) {
    throw new Error(`${view.key}: map camera failed: ${result.reason}`);
  }

  console.log(`video camera ${view.key}:`, result);

  await page.evaluate(() => {
    try { window.dispatchEvent(new Event("resize")); } catch {}
    try { window.requestDraw?.(); } catch {}
    try { window.swiftResetHeatmapCacheV830?.(); } catch {}
  });

  await page.waitForTimeout(550);
}

async function enterRequestedMapView(page, view) {
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
  await applyLeafletVideoCamera(page, view);
}

async function reapplyVideoCamera(page, view) {
  await applyLeafletVideoCamera(page, view);
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

async function renderOneSource(browser, source, archiveDayDir, password) {
  const page = await browser.newPage({
    viewport: VIDEO_VIEWPORT,
    deviceScaleFactor: 1,
  });

  page.setDefaultTimeout(60000);

  page.on("console", msg => {
    const type = msg.type();
    if (type === "error" || type === "warning") {
      console.log(`[browser ${source} ${type}] ${msg.text()}`);
    }
  });

  page.on("pageerror", err => {
    console.log(`[browser ${source} pageerror] ${err.stack || err.message}`);
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

    // GPS-only constellation for VDOP × L1 vertical-error movies.
    await configureGpsOnlyDop(page);

    const sourceViews = VIDEO_VIEWS.filter(v => v.sources.includes(source));
    if (!sourceViews.length) {
      throw new Error(`${source}: no configured video views`);
    }

    const outputs = [];
    let focusModeEntered = false;

    for (const view of sourceViews) {
      if (!focusModeEntered) {
        await enterRequestedMapView(page, view);
        focusModeEntered = true;
      } else {
        await reapplyVideoCamera(page, view);
      }

      for (const movieType of MOVIE_TYPES) {
        console.log(`${source}/${view.key}/${movieType.key}: prepare`);

        await applyRequestedVisualStyle(page, movieType, view);
        await reapplyVideoCamera(page, view);
        await ensureVideoOverlay(page, source, movieType, view);

        const slider = page.locator("#timeSlider");
        const bounds = await slider.evaluate(el => ({
          min: Number(el.min || 0),
          max: Number(el.max || 0),
        }));

        const indices = captureIndices(bounds.min, bounds.max);
        if (indices.length < 2) {
          throw new Error(
            `${source}/${view.key}/${movieType.key}: too few frames ` +
            `(${indices.length})`
          );
        }

        const tmp = fs.mkdtempSync(
          path.join(
            os.tmpdir(),
            `swifttec-${source}-${view.key}-${movieType.key}-`
          )
        );

        console.log(
          `${source}/${view.key}/${movieType.key}: ` +
          `${indices.length} frames`
        );

        let frameNo = 0;
        for (const idx of indices) {
          await moveSliderAndStamp(page, idx, source, movieType, view);

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

        const fileName =
          `${source}_${view.key}_${movieType.suffix}.mp4`;

        const archiveOutput = path.join(archiveDayDir, fileName);
        ffmpegEncode(tmp, archiveOutput);
        fs.rmSync(tmp, { recursive: true, force: true });

        const latestOutput = path.join(LATEST_DIR, fileName);
        fs.copyFileSync(archiveOutput, latestOutput);

        outputs.push({
          source,
          view: view.key,
          view_label: view.label,
          camera: view.camera,
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
          thresholds_m: [...view.thresholds],
          colors: HEATMAP_STYLE.colors,
          gps_only_for_vertical_error: movieType.key === "vertical",
          status: forecastState.status || "",
          bytes: fs.statSync(archiveOutput).size,
        });
      }
    }

    return outputs;
  } finally {
    await page.close();
  }
}


function buildIndex(results) {
  const run = currentJstRunSlot();
  const dayDir = path.join(ARCHIVE_DIR, run.day);

  const archiveRunsToday = [];

  if (fs.existsSync(dayDir)) {
    for (const slot of fs.readdirSync(dayDir).sort()) {
      const slotDir = path.join(dayDir, slot);
      if (!fs.statSync(slotDir).isDirectory()) continue;

      const files = fs.readdirSync(slotDir)
        .filter(name => name.endsWith(".mp4"))
        .sort()
        .map(name =>
          `data/videos/archive/${run.day}/${slot}/${name}`
        );

      archiveRunsToday.push({
        day_jst: run.day,
        slot_jst: slot,
        files,
      });
    }
  }

  const latest = {};
  for (const view of VIDEO_VIEWS) {
    for (const source of view.sources) {
      for (const movieType of MOVIE_TYPES) {
        const fileName =
          `${source}_${view.key}_${movieType.suffix}.mp4`;

        latest[`${source}_${view.key}_${movieType.key}`] =
          `data/videos/latest/${fileName}`;
      }
    }
  }

  const doc = {
    version: "swifttec-forecast-video-v3-multiregion",
    updated_utc: new Date().toISOString(),
    schedule_jst: ["09:00", "14:00"],
    keep_days_jst: KEEP_DAYS,
    current_jst_day: run.day,
    current_run_slot_jst: run.slot,

    visual: {
      viewport: VIDEO_VIEWPORT,
      heatmap_alpha: HEATMAP_STYLE.alpha,
      palette: HEATMAP_STYLE.palette,
      reverse: HEATMAP_STYLE.reverse,
      colors: HEATMAP_STYLE.colors,
      gps_constellation_only_for_vertical_error: true,
      vertical_metric: "GPS VDOP × L1 ionospheric error",
      views: VIDEO_VIEWS,
    },

    latest,
    current_run: results,
    archive_runs_today: archiveRunsToday,
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

  const run = currentJstRunSlot();
  const archiveDayDir = path.join(
    ARCHIVE_DIR,
    run.day,
    run.slot
  );
  fs.mkdirSync(archiveDayDir, { recursive: true });

  console.log(
    `JST video run: ${run.day} ${run.slot}; retention=${KEEP_DAYS} day(s)`
  );

  const password = parsePagePassword();

  const browser = await chromium.launch({
    headless: true,
    args: ["--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu"],
  });

  const results = [];
  try {
    results.push(
      ...(await renderOneSource(browser, "noaa", archiveDayDir, password))
    );
    results.push(
      ...(await renderOneSource(browser, "isee", archiveDayDir, password))
    );
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
