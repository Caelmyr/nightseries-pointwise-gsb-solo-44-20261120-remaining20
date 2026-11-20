import { fmt } from "./ui.js";

const SIDE_COLORS = {A: "#7ce2d8", B: "#efba75"};
const MUTED = "#96a2ba";
const REJECTED = "#9a76a8";
const GRID = "#293346";
const TEXT = "#a4b4d2";

function context(canvas) {
  const bounds = canvas.getBoundingClientRect();
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.round(bounds.width * dpr);
  canvas.height = Math.round(bounds.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  return {ctx, width: bounds.width, height: bounds.height};
}

function noData(canvas, label) {
  const {ctx, width, height} = context(canvas);
  ctx.fillStyle = "#9caacf";
  ctx.font = "13px system-ui";
  ctx.textAlign = "center";
  ctx.fillText(label, width / 2, height / 2);
}

export function shellTraceChart(canvas, trace, side) {
  if (!trace?.points?.length) {
    noData(canvas, "剖面轨迹不可用");
    return;
  }
  const {ctx, width, height} = context(canvas);
  const left = 54, right = Math.max(60, width - 12), top = 14, bottom = Math.max(22, height - 26);
  const radii = trace.points.map(point => point.r);
  const levels = trace.points.map(point => point.v).filter(value => value !== null);
  const minX = Math.min(...radii), maxX = Math.max(...radii);
  let minY = Math.min(...levels), maxY = Math.max(...levels);
  if (!(maxY > minY)) {
    maxY = minY + 1;
  }
  const margin = (maxY - minY) * 0.08;
  minY -= margin;
  maxY += margin;
  const cx = value => left + (value - minX) / Math.max(maxX - minX, 1e-6) * (right - left);
  const cy = value => bottom - (value - minY) / Math.max(maxY - minY, 1e-6) * (bottom - top);
  ctx.font = "11px Consolas, monospace";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  ctx.strokeStyle = GRID;
  ctx.fillStyle = TEXT;
  for (let step = 0; step <= 3; step += 1) {
    const level = minY + (maxY - minY) * step / 3;
    ctx.beginPath();
    ctx.moveTo(left, cy(level));
    ctx.lineTo(right, cy(level));
    ctx.stroke();
    ctx.fillText(fmt(level, 0), left - 6, cy(level));
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let step = 0; step <= 4; step += 1) {
    const value = minX + (maxX - minX) * step / 4;
    ctx.fillText(fmt(value, 1), cx(value), bottom + 6);
  }
  const color = SIDE_COLORS[side] || SIDE_COLORS.A;
  for (const [radius, style] of [[trace.inner_edge, "in"], [trace.outer_edge, "out"]]) {
    if (radius == null || radius < minX || radius > maxX) continue;
    ctx.strokeStyle = color;
    ctx.globalAlpha = style === "in" ? 0.9 : 0.55;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(cx(radius), top);
    ctx.lineTo(cx(radius), bottom);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    ctx.fillStyle = color;
    ctx.textBaseline = "top";
    ctx.fillText(`${style === "in" ? "内缘" : "外缘"} ${fmt(radius, 1)}`, cx(radius) + 3, top + 1);
  }
  ctx.strokeStyle = trace.state === "rejected" ? REJECTED : color;
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  let connected = false;
  for (const point of trace.points) {
    if (point.v === null) {
      connected = false;
      continue;
    }
    if (connected) ctx.lineTo(cx(point.r), cy(point.v));
    else ctx.moveTo(cx(point.r), cy(point.v));
    connected = true;
  }
  ctx.stroke();
  ctx.fillStyle = MUTED;
  ctx.textAlign = "left";
  ctx.textBaseline = "bottom";
  ctx.fillText(`半径 px · 角度 ${fmt(trace.angle, 1)}°`, left, bottom - 2);
}
