#!/usr/bin/env node
/**
 * SWIFT-TEC daily forecast video renderer.
 *
 * Produces:
 *   docs/data/videos/latest/noaa_l1_error.mp4
 *   docs/data/videos/latest/noaa_vertical_error.mp4
 *   docs/data/videos/latest/isee_l1_error.mp4
 *   docs/data/videos/latest/isee_vertical_error.mp4
 *   docs/data/videos/archive/YYYY-MM-DD/<same four files>
 *   docs/data/videos/index.json
 *
 * Retention:
 *   current UTC day + previous 2 UTC days = 3 days total.
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
const KEEP_DAYS = Math.max(1, Number(process.env.SWIFTTEC_VIDEO_KEEP_DAYS || 3));

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

async function ensureVideoOverlay(page, source, movieType) {
  await page.evaluate(({ sourceName, movieLabel }) => {
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
  }, { sourceName: source, movieLabel: movieType.label });
}

async function moveSliderAndStamp(page, index, source, movieType) {
  await page.evaluate(({ idx, sourceName, movieLabel }) => {
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
        `${label} / ${movieLabel}\n${t}\nKpF ${kpF} / KpB ${kpB}`;
    }
  }, {
    idx: index,
    sourceName: source,
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
  const limits = movieType.mapMode === "vdoptec"
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

async function enterRequestedMapView(page, source) {
  // ISEE forecast already fits Japan bounds in its forecast installer.
  // NOAA remains in global view. Then use the existing "地図だけ拡大" mode,
  // which matches the supplied screenshots: full viewport map + bottom dock.
  await page.evaluate((sourceName) => {
    // Ensure any forced-fullscreen mode is off; map-focus is the requested UI.
    document.body.classList.remove("swift-map-fs-on");

    if (typeof window.swiftEnterMapFocusMode === "function") {
      window.swiftEnterMapFocusMode();
    } else {
      document.documentElement.classList.add("swift-map-focus");
    }

    // If the ISEE Japan button/fitBounds has not run yet, trigger the public
    // loader's focus path when available without re-running forecast.
    if (sourceName === "isee") {
      try {
        if (typeof gMap !== "undefined" && gMap?.fitBounds) {
          gMap.fitBounds([[24, 122], [46, 150]], { padding: [8, 8] });
        }
      } catch {}
    }

    try { window.dispatchEvent(new Event("resize")); } catch {}
    try { window.requestDraw?.(); } catch {}
  }, source);

  await page.waitForTimeout(1200);
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

    // GPS-only constellation is used for the vertical DOP×L1 movie.
    // Loading it once also makes DOP status available in the dock.
    await configureGpsOnlyDop(page);

    await enterRequestedMapView(page, source);

    const outputs = [];

    for (const movieType of MOVIE_TYPES) {
      console.log(`${source}/${movieType.key}: preparing visual style`);

      await applyRequestedVisualStyle(page, movieType);
      await ensureVideoOverlay(page, source, movieType);

      const slider = page.locator("#timeSlider");
      const bounds = await slider.evaluate(el => ({
        min: Number(el.min || 0),
        max: Number(el.max || 0),
      }));

      const indices = captureIndices(bounds.min, bounds.max);
      if (indices.length < 2) {
        throw new Error(`${source}/${movieType.key}: too few frames (${indices.length})`);
      }

      const tmp = fs.mkdtempSync(
        path.join(os.tmpdir(), `swifttec-${source}-${movieType.key}-`)
      );

      console.log(
        `${source}/${movieType.key}: capture ${indices.length} frames ` +
        `(slider ${bounds.min}..${bounds.max}, viewport ` +
        `${VIDEO_VIEWPORT.width}x${VIDEO_VIEWPORT.height})`
      );

      let frameNo = 0;
      for (const idx of indices) {
        await moveSliderAndStamp(page, idx, source, movieType);

        const framePath = path.join(
          tmp,
          `${String(frameNo).padStart(5, "0")}.png`
        );

        // Capture the full map-focus viewport, including the large UTC label,
        // legend and bottom timeline dock shown in the user's screenshots.
        await page.screenshot({
          path: framePath,
          type: "png",
          fullPage: false,
        });

        frameNo++;
      }

      const fileName = `${source}_${movieType.suffix}.mp4`;
      const archiveOutput = path.join(archiveDayDir, fileName);
      ffmpegEncode(tmp, archiveOutput);
      fs.rmSync(tmp, { recursive: true, force: true });

      const latestOutput = path.join(LATEST_DIR, fileName);
      fs.copyFileSync(archiveOutput, latestOutput);

      outputs.push({
        source,
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
          movieType.mapMode === "vdoptec"
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
  const fileFor = (day, source, suffix) => {
    const p = path.join(ARCHIVE_DIR, day, `${source}_${suffix}.mp4`);
    return fs.existsSync(p)
      ? `data/videos/archive/${day}/${source}_${suffix}.mp4`
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
    }));

  const doc = {
    version: "swifttec-forecast-video-v2",
    updated_utc: new Date().toISOString(),
    keep_days: KEEP_DAYS,
    visual: {
      viewport: VIDEO_VIEWPORT,
      heatmap_alpha: HEATMAP_STYLE.alpha,
      palette: HEATMAP_STYLE.palette,
      reverse: HEATMAP_STYLE.reverse,
      colors: HEATMAP_STYLE.colors,
      gps_limits_m: HEATMAP_STYLE.gpsLimits,
      vertical_limits_m: HEATMAP_STYLE.vdopTecLimits,
      gps_constellation_only: true,
      vertical_metric: "GPS VDOP × L1 ionospheric error",
    },
    latest: {
      noaa_l1_error: "data/videos/latest/noaa_l1_error.mp4",
      noaa_vertical_error: "data/videos/latest/noaa_vertical_error.mp4",
      isee_l1_error: "data/videos/latest/isee_l1_error.mp4",
      isee_vertical_error: "data/videos/latest/isee_vertical_error.mp4",
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
  assertNoMultiArgEvaluateV833();
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
