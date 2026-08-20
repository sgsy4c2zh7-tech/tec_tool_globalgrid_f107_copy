#!/usr/bin/env node
/**
 * SWIFT-TEC daily forecast video renderer.
 *
 * Produces:
 *   docs/data/videos/latest/noaa_forecast.mp4
 *   docs/data/videos/latest/isee_forecast.mp4
 *   docs/data/videos/archive/YYYY-MM-DD/noaa_forecast.mp4
 *   docs/data/videos/archive/YYYY-MM-DD/isee_forecast.mp4
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
const MAX_CAPTURE_FRAMES = Math.max(24, Number(process.env.SWIFTTEC_VIDEO_MAX_FRAMES || 240));
const FRAME_DELAY_MS = Math.max(40, Number(process.env.SWIFTTEC_VIDEO_FRAME_DELAY_MS || 120));
const VIDEO_FPS = Math.max(1, Number(process.env.SWIFTTEC_VIDEO_FPS || 6.6667));
const KEEP_DAYS = Math.max(1, Number(process.env.SWIFTTEC_VIDEO_KEEP_DAYS || 3));

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

async function ensureVideoOverlay(page, source) {
  await page.evaluate((sourceName) => {
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
  }, source);
}

async function moveSliderAndStamp(page, index, source) {
  await page.evaluate(({ idx, sourceName }) => {
    const slider = document.getElementById("timeSlider");
    if (!slider) throw new Error("timeSlider missing");

    slider.value = String(idx);
    slider.dispatchEvent(new Event("input", { bubbles: true }));
    slider.dispatchEvent(new Event("change", { bubbles: true }));

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
      stamp.textContent = `${label}\n${t}\nKpF ${kpF} / KpB ${kpB}`;
    }
  }, { idx: index, sourceName: source });

  await page.waitForTimeout(FRAME_DELAY_MS);
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

async function renderOne(browser, source, archiveDayDir, password) {
  const page = await browser.newPage({
    viewport: { width: 1440, height: 900 },
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
    await page.goto(BASE_URL, { waitUntil: "domcontentloaded", timeout: 120000 });
    await page.waitForSelector("#tecMap", { timeout: 90000 });
    await page.waitForTimeout(1200);

    await setForecastSource(page, source);
    const forecastState = await runForecast(page, source);
    await ensureVideoOverlay(page, source);

    const slider = page.locator("#timeSlider");
    const bounds = await slider.evaluate(el => ({
      min: Number(el.min || 0),
      max: Number(el.max || 0),
    }));

    const indices = captureIndices(bounds.min, bounds.max);
    if (indices.length < 2) throw new Error(`${source}: too few frames (${indices.length})`);

    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), `swifttec-${source}-`));
    const target = page.locator("#tecMap");

    console.log(`${source}: capture ${indices.length} frames (slider ${bounds.min}..${bounds.max})`);

    let frameNo = 0;
    for (const idx of indices) {
      await moveSliderAndStamp(page, idx, source);
      const framePath = path.join(tmp, `${String(frameNo).padStart(5, "0")}.png`);
      await target.screenshot({ path: framePath, type: "png" });
      frameNo++;
    }

    const archiveOutput = path.join(
      archiveDayDir,
      source === "isee" ? "isee_forecast.mp4" : "noaa_forecast.mp4"
    );

    ffmpegEncode(tmp, archiveOutput);
    fs.rmSync(tmp, { recursive: true, force: true });

    const latestOutput = path.join(
      LATEST_DIR,
      source === "isee" ? "isee_forecast.mp4" : "noaa_forecast.mp4"
    );
    fs.copyFileSync(archiveOutput, latestOutput);

    return {
      source,
      file: path.relative(DOCS, archiveOutput).replaceAll("\\", "/"),
      latest: path.relative(DOCS, latestOutput).replaceAll("\\", "/"),
      frames: indices.length,
      slider_min: bounds.min,
      slider_max: bounds.max,
      status: forecastState.status || "",
      bytes: fs.statSync(archiveOutput).size,
    };
  } finally {
    await page.close();
  }
}

function buildIndex(results) {
  const days = fs.readdirSync(ARCHIVE_DIR)
    .filter(name => /^\d{4}-\d{2}-\d{2}$/.test(name))
    .sort()
    .reverse()
    .slice(0, KEEP_DAYS)
    .map(day => ({
      day_utc: day,
      noaa: fs.existsSync(path.join(ARCHIVE_DIR, day, "noaa_forecast.mp4"))
        ? `data/videos/archive/${day}/noaa_forecast.mp4`
        : null,
      isee: fs.existsSync(path.join(ARCHIVE_DIR, day, "isee_forecast.mp4"))
        ? `data/videos/archive/${day}/isee_forecast.mp4`
        : null,
    }));

  const doc = {
    version: "swifttec-forecast-video-v1",
    updated_utc: new Date().toISOString(),
    keep_days: KEEP_DAYS,
    latest: {
      noaa: "data/videos/latest/noaa_forecast.mp4",
      isee: "data/videos/latest/isee_forecast.mp4",
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

function assertNoMultiArgEvaluateV833() {
  // Playwright page.evaluate supports exactly one serializable argument.
  // This catches accidental reintroduction of the v8.32 bug in simple cases.
  const src = fs.readFileSync(new URL(import.meta.url), "utf8");
  const suspicious = [
    /\},\s*wanted\s*,\s*source\s*\)/,
    /\},\s*index\s*,\s*source\s*\)/,
  ];
  for (const re of suspicious) {
    if (re.test(src)) throw new Error(`v8.33 self-check failed: ${re}`);
  }
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
    results.push(await renderOne(browser, "noaa", archiveDayDir, password));
    results.push(await renderOne(browser, "isee", archiveDayDir, password));
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
