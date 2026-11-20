import { fmt } from "./ui.js";

const COLORS = {target: "#7ce2d8", comparison: "#efba75", check: "#aaa3ff"};

function context(canvas) {
  const bounds = canvas.getBoundingClientRect();
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.round(bounds.width * dpr);
  canvas.height = Math.round(bounds.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  return {ctx, width: bounds.width, height: bounds.height};
}

function chartFrame(canvas, xBounds, yBounds, xText, yText, invertY) {
  const { ctx, width, height } = context(canvas);
  const left = 62, right = Math.max(80, width - 17), top = 15, bottom = Math.max(25, height - 29);
  const [minX, maxX] = xBounds;
  const [minY, maxY] = yBounds;
  const cx = value => left + (value - minX) / Math.max(maxX - minX, 1e-6) * (right - left);
  const cy = value => {
    const t = (value - minY) / Math.max(maxY - minY, 1e-6);
    return invertY ? top + t * (bottom - top) : bottom - t * (bottom - top);
  };
  ctx.font = "11px Consolas, monospace";
  ctx.strokeStyle = "#293346";
  ctx.fillStyle = "#a4b4d2";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let step = 0; step <= 4; step += 1) {
    const level = minY + (maxY - minY) * step / 4;
    const yy = cy(level);
    ctx.beginPath();
    ctx.moveTo(left, yy);
    ctx.lineTo(right, yy);
    ctx.stroke();
    ctx.fillText(yText(level), left - 7, yy);
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let step = 0; step <= 4; step += 1) {
    const value = minX + (maxX - minX) * step / 4;
    const xx = cx(value);
    ctx.strokeStyle = "#242d40";
    ctx.beginPath();
    ctx.moveTo(xx, top);
    ctx.lineTo(xx, bottom);
    ctx.stroke();
    ctx.fillText(xText(value), xx, bottom + 6);
  }
  return {ctx, cx, cy, top, bottom};
}

function range(values, margin = 0.04) {
  const good = values.filter(value => typeof value === "number" && Number.isFinite(value));
  if (!good.length) return [0, 1];
  const low = Math.min(...good), high = Math.max(...good);
  const pad = Math.max(margin, (high - low) * 0.08);
  return [low - pad, high + pad];
}

function noData(canvas, label) {
  const {ctx, width, height} = context(canvas);
  ctx.fillStyle = "#9caacf";
  ctx.font = "13px system-ui";
  ctx.textAlign = "center";
  ctx.fillText(label, width / 2, height / 2);
}

export function curveChart(canvas, curve, activeId) {
  const samples = curve.rows;
  const measures = samples.flatMap(row => [row.target, row.check]).filter(row => row?.state === "measured");
  if (!measures.length) {
    noData(canvas, "没有同时可用的目标／比较观测");
    return;
  }
  const timestamps = samples.map(row => row.time_seconds);
  const y = range(measures.flatMap(row => [row.magnitude - row.error, row.magnitude + row.error]));
  const x0 = Math.min(...timestamps);
  const x1 = Math.max(x0 + 1, ...timestamps);
  const tickTime = value => new Date(value * 1000).toISOString().slice(11, 16);
  const {ctx, cx, cy, top, bottom} = chartFrame(canvas, [x0, x1], y, tickTime, val => fmt(val, 3), true);
  const current = samples.find(row => row.frame === activeId);
  if (current) {
    ctx.strokeStyle = "#5b6d8d";
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(cx(current.time_seconds), top);
    ctx.lineTo(cx(current.time_seconds), bottom);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  for (const role of ["target", "check"]) {
    ctx.strokeStyle = COLORS[role];
    ctx.fillStyle = COLORS[role];
    let previous = null;
    for (const row of samples) {
      const measurement = row[role];
      if (measurement?.state !== "measured") {
        previous = null;
        continue;
      }
      const x = cx(row.time_seconds), y = cy(measurement.magnitude);
      const errorLow = cy(measurement.magnitude - measurement.error);
      const errorHigh = cy(measurement.magnitude + measurement.error);
      if (previous !== null) {
        ctx.beginPath();
        ctx.moveTo(previous[0], previous[1]);
        ctx.lineTo(x, y);
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.moveTo(x, errorLow);
      ctx.lineTo(x, errorHigh);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(x, y, row.frame === activeId ? 4 : 2.8, 0, Math.PI * 2);
      ctx.fill();
      previous = [x, y];
    }
  }
}

const NOISE_STATE_COLORS = {
  stable: "#7ce2d8",
  deviation_confirmed: "#ef8f75",
  insufficient_independent: "#efba75",
  saturated_present: "#d491e3",
  no_valid_tiles: "#8b97ad",
  low_signal: "#9caacf",
  no_model: "#efba75",
};

export function noiseCurveChart(canvas, report) {
  const levels = report.levels.filter(level => level.noise_adu !== null);
  if (!levels.length) {
    noData(canvas, "没有可绘制的亮度档");
    return;
  }
  const xValues = levels.map(level => level.median_net_adu);
  const curve = report.model_curve || [];
  if (curve.length) xValues.push(...curve.map(point => point.net_level_adu));
  const yValues = [...levels.map(level => level.noise_adu),
    ...levels.map(level => level.noise_sigma ? level.noise_adu + level.noise_sigma : level.noise_adu),
    ...curve.map(point => point.expected_noise_adu)];
  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  const x0 = Math.max(0, xMin - (xMax - xMin || xMax) * 0.06);
  const x1 = xMax + (xMax - xMin || xMax) * 0.06;
  const y0 = Math.max(0, Math.min(...yValues) * 0.9);
  const y1 = Math.max(...yValues) * 1.05;
  const {ctx, cx, cy, top, bottom} = chartFrame(canvas, [x0, x1], [y0, y1],
    value => fmt(value, 0), value => fmt(value, 2));
  const window = report.stable_range;
  if (window?.lower_adu !== null && window?.upper_adu !== null && window.lower_adu !== undefined) {
    ctx.fillStyle = "rgba(124, 226, 216, 0.10)";
    ctx.fillRect(cx(window.lower_adu), top, Math.max(2, cx(window.upper_adu) - cx(window.lower_adu)), bottom - top);
    ctx.strokeStyle = "rgba(124, 226, 216, 0.55)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(cx(window.upper_adu), top);
    ctx.lineTo(cx(window.upper_adu), bottom);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  if (curve.length) {
    ctx.strokeStyle = "#b7cbff";
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    curve.forEach((point, index) => {
      const x = cx(point.net_level_adu), y = cy(point.expected_noise_adu);
      index ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }
  for (const level of levels) {
    const x = cx(level.median_net_adu), y = cy(level.noise_adu);
    const color = NOISE_STATE_COLORS[level.state] || "#9caacf";
    if (level.noise_sigma) {
      ctx.strokeStyle = color;
      ctx.beginPath();
      ctx.moveTo(x, cy(level.noise_adu - level.noise_sigma));
      ctx.lineTo(x, cy(level.noise_adu + level.noise_sigma));
      ctx.stroke();
    }
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, level.state === "stable" ? 4 : 3.4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#a4b4d2";
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText(String(level.index + 1), x, y - 7);
  }
}

export function histogramChart(canvas, result) {
  if (!result?.counts.length) {
    noData(canvas, "无有效像素");
    return;
  }
  const {counts, edges} = result;
  const {ctx, cx, cy} = chartFrame(canvas, [edges[0], edges.at(-1)], [0, Math.max(...counts)], v => fmt(v, 0), v => fmt(v, 0));
  ctx.fillStyle = "#73c6c8";
  counts.forEach((count, index) => {
    const x = cx(edges[index]), width = Math.max(1, cx(edges[index + 1]) - x - 1);
    ctx.fillRect(x, cy(count), width, cy(0) - cy(count));
  });
}

export function profileChart(canvas, result) {
  if (!result?.rows.some(row => row.value !== null)) {
    noData(canvas, "线上无有效像素");
    return;
  }
  const values = range(result.rows.map(row => row.value), 0.1);
  const {ctx, cx, cy} = chartFrame(canvas, [0, result.length], values, v => fmt(v, 0), v => fmt(v, 0));
  ctx.strokeStyle = COLORS.target;
  let connected = false;
  ctx.beginPath();
  for (const row of result.rows) {
    if (row.value === null) {
      connected = false;
      continue;
    }
    if (connected) ctx.lineTo(cx(row.distance), cy(row.value));
    else ctx.moveTo(cx(row.distance), cy(row.value));
    connected = true;
  }
  ctx.stroke();
}

export function drawRegions(canvas, doc, frame, stream, stars, toolPoints) {
  const {ctx, width, height} = context(canvas);
  const {width: w, height: h} = doc.dimensions;
  const sx = width / w, sy = height / h;
  const px = x => (x + .5) * sx, py = y => (y + .5) * sy;
  const off = frame.id === doc.reference ? {dx: 0, dy: 0} : doc.alignments[frame.id];
  const native = stream !== "aligned";
  ctx.lineWidth = 1.25;
  ctx.font = "10px Consolas, sans-serif";
  for (const mask of doc.masks.filter(region => !region.frame || region.frame === frame.id)) {
    if (!native && !off) continue;
    const dx = native ? 0 : -off.dx;
    const dy = native ? 0 : -off.dy;
    ctx.fillStyle = "#ac63bb44";
    ctx.strokeStyle = "#d491e3";
    ctx.fillRect(px(mask.left - .5 + dx), py(mask.top - .5 + dy), (mask.right - mask.left) * sx, (mask.bottom - mask.top) * sy);
    ctx.strokeRect(px(mask.left - .5 + dx), py(mask.top - .5 + dy), (mask.right - mask.left) * sx, (mask.bottom - mask.top) * sy);
  }
  if (frame.kind === "science" && (!native || off)) {
    for (const source of doc.apertures) {
      const x = source.x + (native ? off.dx : 0), y = source.y + (native ? off.dy : 0);
      const radius = Math.max(3, doc.photometry.radius * sx);
      const color = COLORS[source.role];
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(px(x), py(y), radius, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.strokeStyle = color + "90";
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.arc(px(x), py(y), doc.photometry.sky_outer * sx, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillText(source.label.split(" ")[0], px(x) + radius + 4, py(y) - radius - 2);
    }
  }
  for (const star of stars || []) {
    if (star.state !== "measured") continue;
    ctx.strokeStyle = "#b7cbff85";
    ctx.beginPath();
    ctx.rect(px(star.x) - 3, py(star.y) - 3, 6, 6);
    ctx.stroke();
  }
  ctx.fillStyle = "#f5c682";
  ctx.strokeStyle = "#f5c682";
  for (const point of toolPoints) {
    ctx.beginPath();
    ctx.arc(px(point.x), py(point.y), 3, 0, 2 * Math.PI);
    ctx.fill();
  }
  if (toolPoints.length === 2) {
    ctx.beginPath();
    ctx.moveTo(px(toolPoints[0].x), py(toolPoints[0].y));
    ctx.lineTo(px(toolPoints[1].x), py(toolPoints[1].y));
    ctx.stroke();
  }
}
