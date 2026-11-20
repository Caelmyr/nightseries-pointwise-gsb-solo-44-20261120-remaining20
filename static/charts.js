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

function drawTrack(ctx, px, py, frameId, toX, toY, track) {
  const result = track.result;
  ctx.font = "10px Consolas, sans-serif";
  if (result) {
    ctx.lineWidth = 1;
    ctx.strokeStyle = "#efba7566";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    result.rows.forEach((row, index) => {
      const x = px(toX(row.pred_x)), y = py(toY(row.pred_y));
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
    const row = result.rows.find(item => item.frame === frameId);
    if (row) {
      for (const cand of row.candidates) {
        if (cand.ref_x == null || cand.ref_y == null) continue;
        ctx.strokeStyle = cand.stationary === "yes" ? "#8d99ae" : "#aaa3ff";
        ctx.strokeRect(px(toX(cand.ref_x)) - 3.5, py(toY(cand.ref_y)) - 3.5, 7, 7);
      }
      const x = px(toX(row.pred_x)), y = py(toY(row.pred_y));
      ctx.strokeStyle = "#efba75";
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.arc(x, y, 7, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(x - 10, y); ctx.lineTo(x - 4, y); ctx.moveTo(x + 4, y); ctx.lineTo(x + 10, y);
      ctx.moveTo(x, y - 10); ctx.lineTo(x, y - 4); ctx.moveTo(x, y + 4); ctx.lineTo(x, y + 10);
      ctx.stroke();
      ctx.fillStyle = "#efba75";
      ctx.fillText("预测", x + 9, y - 9);
      if (row.measured) {
        const mx = px(toX(row.measured.ref_x)), my = py(toY(row.measured.ref_y));
        ctx.strokeStyle = "#7ce2d8";
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.arc(mx, my, 7, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.lineWidth = 1.25;
        ctx.fillStyle = "#7ce2d8";
        ctx.fillText("实测", mx + 9, my + 15);
      }
    }
  }
  for (const seed of track.seeds || []) {
    if (seed.frame !== frameId) continue;
    const x = px(toX(seed.x)), y = py(toY(seed.y));
    ctx.strokeStyle = "#ffffff";
    ctx.beginPath();
    ctx.moveTo(x, y - 6);
    ctx.lineTo(x + 6, y);
    ctx.lineTo(x, y + 6);
    ctx.lineTo(x - 6, y);
    ctx.closePath();
    ctx.stroke();
    ctx.fillStyle = "#ffffff";
    ctx.fillText("种子", x + 8, y - 8);
  }
}

export function drawRegions(canvas, doc, frame, stream, stars, toolPoints, track) {
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
  if (track && frame.kind === "science" && (!native || off)) {
    const toX = x => native ? x + off.dx : x;
    const toY = y => native ? y + off.dy : y;
    drawTrack(ctx, px, py, frame.id, toX, toY, track);
  }
}
