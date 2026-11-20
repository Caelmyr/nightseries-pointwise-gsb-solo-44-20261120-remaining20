import {
  E, action, apiPath, badge, box, check, field, fmt, input, panel, pct, pick, sources,
  stateLabel, status, table, time,
} from "./ui.js";
import { filters, selectedFrame } from "./views.js";

const numeric = (label, name, value, min, max, step = "any") => input(label, name, value, {type: "number", min, max, step});
const sourcePick = (doc, value) => pick("参考坐标孔径", "source", doc.apertures.map(row => [row.id, row.label]), value);
const bandPick = (doc, value, all = false) => pick("滤镜", "band", filters(doc, all), value);
const framePick = (doc, label, key, current, active = false) => pick(label, key, doc.frames.filter(row => !active || row.enabled).map(row => [row.id, row.filename]), current);
const rates = value => fmt(value, 3);
const stamp = value => `${time(value)} UTC`;
const grid = (heads, rows) => table(heads, rows.map(cells => ({cells})));
const coverageModes = [["detector", "探测器画幅"], ["valid", "校准有效像素"], ["unsaturated", "校准且未饱和"]];
const figures = figures => figures.filter(([url]) => url).map(([url, label]) =>
  `<figure class="wide-preview"><img src="${E(url)}" alt="${E(label)}"><figcaption>${E(label)}</figcaption></figure>`).join("");
const flag = code => status(stateLabel(code), ["measured", "inside", "consistent", "ok", "used"].includes(code));
const sourceFits = [["raw", "原始像素"], ["calibrated", "校准后"]];
const overview = (stats, keys) => `<div class="meta-list">${keys.map(([name, title, render]) => `<div><span>${E(title)}</span><b>${render ? render(stats[name]) : E(stats[name] ?? "—")}</b></div>`).join("")}</div>`;
const imageBlock = (url, title) => url ? figures([[url, title]]) : "";
const downloadButton = (label, format, filename) => action(label, "tool-export", {style: "tiny ghost", attrs: `data-format="${format}" data-name="${filename}"`});
const runButton = label => `<button type="submit" class="button primary">${E(label)}</button>`;

function filtered(doc, state) {
  return state.band === undefined ? filters(doc)[0]?.[0] || "" : state.band;
}

function timelineResult(result) {
  const {data} = result;
  return `${overview(data, [["span_seconds", "总时间跨度 s", rates], ["covered_seconds", "实际曝光覆盖 s", rates],
    ["duty_cycle", "观测占空比", pct]])}${grid(["滤镜", "曝光数", "曝光和 s", "中位间距 s", "记录开始", "记录结束"],
    data.by_band.map(row => [E(row.band || "—"), row.count, rates(row.exposure_seconds), rates(row.median_cadence), stamp(row.first), stamp(row.last)]))}
    <hr><h3>曝光间隔</h3>${data.gaps.length
      ? grid(["前帧结束", "后帧开始", "间隔秒", "提示"], data.gaps.map(row =>
        [stamp(row.from), stamp(row.to), rates(row.seconds), flag(row.large ? "长间隔" : "正常")])) : box("无间隔。")}
    ${data.overlaps.length ? `<p class="help">${data.overlaps.length} 个曝光间隔重叠，建议核对原始时钟日志。</p>` : ""}`;
}

function skyResult(result) {
  const data = result.data;
  if (!data.field) return box("当前参考曝光缺少天球解。可导入带有 TAN 坐标的曝光或用定标星拟合。");
  const {field: sky} = data;
  return `${overview(sky, [["center", "参考中心 RA / Dec °", point => fmt(point.ra, 6) + " / " + fmt(point.dec, 6)],
    ["width_arcmin", "图宽 角分", rates], ["height_arcmin", "图高 角分", rates],
    ["solution", "天球系 / 主尺度", v => E(v.system) + " / " + fmt(v.major_scale_arcsec, 4) + "″/px"]])}
    ${data.point ? `<p class="help">像素 (${fmt(data.point.x, 2)}, ${fmt(data.point.y, 2)}) 对应 ${fmt(data.point.ra, 6)}° / ${fmt(data.point.dec, 6)}°</p>` : ""}
    ${data.plate ? `<h3>定标残差</h3>${grid(["定标星", "加入拟合", "残差角秒"], data.plate.anchors.map(row =>
      [E(row.label), row.included ? "是" : "否", rates(row.residual_arcsec)]))}` : ""}
    ${data.rows.length ? `<h3>本地星表</h3>${grid(["名称", "RA °", "Dec °", "X", "Y", "覆盖"], data.rows.map(row =>
      [E(row.label), fmt(row.ra, 6), fmt(row.dec, 6), fmt(row.x), fmt(row.y), flag(row.state)]))}` : ""}`;
}

function guideResult({data}) {
  return `${overview(data.summary, [["measured", "可测光心"], ["max_distance", "最大残差 px", rates], ["outside_limit", "超出提示阈值"]])}
    ${grid(["曝光", "中点 UTC", "导星状态", "原生 ΔX", "原生 ΔY", "残余 ΔX", "残余 ΔY", "信噪比"], data.rows.map(row =>
      [E(row.filename), stamp(row.timestamp), flag(row.state), rates(row.native_dx), rates(row.native_dy),
        rates(row.residual_dx), rates(row.residual_dy), fmt(row.snr, 1)]))}`;
}

function detectorResult(result) {
  const {data, images} = result;
  return `${overview(data, [["flagged_pixels", "候选像素"], ["total_regions", "合并坏区"], ["available_fraction", "有足够校准支持", pct]])}
    ${imageBlock(images.image, "坏点分布与候选框")}
    ${data.regions.length ? grid(["选择", "像素", "支持 / 可用", "面积", "严重度"], data.regions.map(row => [
      `<input type="checkbox" data-detector-region="${E(row.id)}" aria-label="选择 ${row.x},${row.y}">`,
      row.x + ", " + row.y, row.votes + " / " + row.support, row.pixels, fmt(row.score, 2),
    ])) + action("将选中区域加入掩膜", "install-badmask", {style: "tiny"}) : box("当前没有达到最低重复支持的候选。")}`;
}

function plateAuditResult({data, images}) {
  const summary = overview(data, [["tested", "已核对曝光"], ["mismatches", "超差曝光"], ["tolerance", "角距容差 ″", rates]]);
  return summary + imageBlock(images.image, "本次最大偏差曝光 · 天球控制点") +
    grid(["帧", "状态", "参与检点", "中位 ″", "最大 ″", "均方根 ″"], data.rows.map(row =>
      [E(row.filename), flag(row.state), row.used, rates(row.median_error), rates(row.maximum_error), rates(row.rms_error)]));
}

function backgroundResult({data, images}) {
  const summary = overview(data, [["valid_tiles", data.sample === "anchors" ? "可用空天样本" : "拟合背景块"],
    ["range", "拟合场跨度 ADU", rates], ["fit_rms_adu", "拟合误差 ADU", rates],
    ...(data.sample === "anchors" ? [["supported_fraction", "有依据扣除范围", pct],
      ["conflict_fraction", "样本趋势冲突", pct], ["retained_fraction", "保留原像素", pct]] : [])]);
  const anchorRows = data.sample === "anchors" ? grid(["空天样本", "范围", "有效像素", "支持", "背景 ADU", "拟合误差 ADU", "状态"],
    data.anchors.map(row => [E(row.label),
      `${row.bounds[0]},${row.bounds[1]} → ${row.bounds[2]},${row.bounds[3]}`,
      row.samples, pct(row.support), rates(row.level), rates(row.fit_rms),
      flag(row.state) + (row.reason ? ` · ${E(row.reason)}` : "")])) : "";
  const tileRows = data.tiles?.length ? grid(["格", "背景 ADU", "支持", "拟合残差 ADU"], data.tiles.map(row =>
    [row.column + ", " + row.row, rates(row.level), pct(row.support), rates(row.residual)])) : "";
  const supportCaption = data.sample === "anchors"
    ? "青色：有样本依据的扣除范围（越亮表示邻近样本越多）· 橙色：两块空天趋势冲突，保留原值 · 深紫：距空天太远或无有效像素，保留原值 · 细框为所用空天样本"
    : "青色：自动网格覆盖的有效像素 · 深紫：坏区与饱和像素";
  return `${summary}
    <div class="two-previews">${figures([[images.sky, "局部平滑天空背景"], [images.residual, "保留天体的残差图"]])}</div>
    ${images.support ? figures([[images.support, supportCaption]]) : ""}
    ${anchorRows}${tileRows}`;
}

function footprintResult({data, images}) {
  return `${overview(data, [["used", "已用帧数"], ["adequate", "符合覆盖比例", pct],
   ["best", "最大可用内矩形", row => row ? `${row.left},${row.top} → ${row.right},${row.bottom}` : "—"]])}
   ${imageBlock(images.image, "亮度为支持数，金框为最大可用内矩形")}
   ${grid(["科学曝光", "状态", "覆盖比例"], data.rows.map(row => [E(row.filename), flag(row.state), pct(row.fraction)]))}`;
}

function gainResult({data}) {
  const pairs = [...data.bias_pairs.map(row => ["偏置", row]), ...data.flat_pairs.map(row => ["平场", row])];
  return `${overview(data, [["gain_e_per_adu", "增益 e⁻/ADU", rates], ["gain_pair_spread", "配对增益离散", rates],
    ["read_noise_adu", "读出噪声 ADU", rates], ["read_noise_e", "读出噪声 e⁻", rates], ["tile_count", "每对取样格"]])}
    ${grid(["类型", "第一张", "第二张", "有效块", "估计值", "离散"], pairs.map(([type, row]) =>
      [type, E(row.first), E(row.second), row.patches ?? "—", rates(row.median), rates(row.sigma)]))}`;
}

function growthResult({data}) {
  return `${overview(data, [["recommended_plateau_radius", "累计光度进入平台 px", rates], ["max_snr_radius", "最高中位信噪比 px", rates],
    ["min_frames", "最少有效曝光"]])}
    ${grid(["半径 px", "覆盖帧数", "累计比例", "跨帧离散", "中位信噪比", "孔径覆盖"], data.radii.map(row =>
      [rates(row.radius), row.count, pct(row.fraction), rates(row.fraction_spread), rates(row.median_snr), pct(row.median_overlap)]))}`;
}

function pairResult({data, images}) {
  return `${overview(data, [["slope", "B / A 强度比例", rates], ["offset", "B 端背景常量 ADU/s", rates],
    ["support", "公共覆盖", pct], ["included", "纳入拟合样本"], ["rejected", "剪裁样本"]])}
    ${imageBlock(images.image, "减去一般强度转移后的配对残差")}
    ${grid(["格", "公共有效率", "中位残差 ADU/s", "离散"], data.tiles.map(row =>
      [row.column + ", " + row.row, pct(row.support), rates(row.median), rates(row.sigma)]))}`;
}

function surfaceResult({data, images}) {
  return `${overview(data, [["surface_summary", "中位表面亮度 ADU/s/px", v => rates(v.median)],
    ["net_summary", "中位区域净亮度 ADU/s", v => rates(v.median)]])}
    ${imageBlock(images.image, "参考帧中的目标和天空采样区")}
    ${grid(["曝光", "状态", "目标覆盖", "天空覆盖", "净 ADU/s", "ADU/s/px", "估计离散"], data.rows.map(row =>
      [E(row.filename), flag(row.state), pct(row.target_fraction), pct(row.sky_fraction), rates(row.net_rate),
        rates(row.surface_rate), rates(row.uncertainty)]))}`;
}

function contactResult({data, images}) {
  return `${imageBlock(images.image, `全幅接触册 · ${data.displayed} / ${data.count} 张`)}
    ${grid(["曝光", "状态", "显示下界", "显示上界"], data.tiles.map(row =>
      [E(row.filename), flag(row.state), rates(row.bounds?.[0]), rates(row.bounds?.[1])]))}`;
}

function cutoutResult({data}) {
  return grid(["曝光", "状态", "原生左上 X", "原生左上 Y", "覆盖"], data.frames.map(row =>
    [E(row.filename), flag(row.state), row.origin_x ?? "—", row.origin_y ?? "—", pct(row.valid_fraction)]));
}

function surfaceFields(doc, current) {
  const w = doc.dimensions.width, h = doc.dimensions.height;
  const groups = [
    ["target", "目标", [w * .3, h * .3, w * .55, h * .55]],
    ["sky", "天空", [w * .67, h * .65, w * .88, h * .88]],
  ];
  return groups.flatMap(([prefix, name, limits]) => ["left", "top", "right", "bottom"].map((boundary, index) =>
    numeric(`${name}${["左", "上", "右", "下"][index]}`, `${prefix}_${boundary}`,
      current[prefix]?.[boundary] ?? Math.round(limits[index]), 0, index % 2 === 0 ? w : h, 1))).join("");
}

export const definitions = {
  timeline: {
    title: "曝光时序", route: "/timeline", method: "GET", result: timelineResult, csv: "observing-coverage.csv",
    fields: (d, o) => bandPick(d, o.band ?? "all", true) + numeric("长间隔阈值 s", "gap_limit", o.gap_limit ?? 120, 1, 86400),
    toolbar: d => action("校对曝光日志", "log-edit", {style: "tiny"}) +
      `<a class="button tiny ghost" href="${apiPath(d.id, "/timeline/log")}">下载曝光日志 ↓</a>`,
  },
  sky: {
    title: "天区坐标", route: "/sky", method: "GET", result: skyResult, csv: "sky-catalog.csv",
    fields: (d, o) => numeric("参考像素 X", "x", o.x ?? Math.round(d.dimensions.width / 2), 0, d.dimensions.width - 1)
      + numeric("参考像素 Y", "y", o.y ?? Math.round(d.dimensions.height / 2), 0, d.dimensions.height - 1),
    toolbar: d => action("导入本地星表", "sky-catalog", {style: "tiny"}) +
      action("定标星拟合", "sky-plate", {style: "tiny ghost"}) +
      (d.plate ? action("移除手工定标", "sky-plate-clear", {style: "tiny danger"}) : ""),
  },
  sky_audit: {
    title: "图幅一致性", route: "/sky/audit", result: plateAuditResult, csv: "reference-plate-audit.csv", images: ["image"],
    fields: (d, o) => bandPick(d, o.band ?? "all", true) +
      numeric("每行检点数", "subdivisions", o.subdivisions ?? 5, 3, 12, 1) +
      numeric("比较容差 ″", "tolerance", o.tolerance ?? 3.5, .05, 120),
  },
  guide: {
    title: "导星光心", route: "/guide", result: guideResult, csv: "guiding-review.csv",
    fields: (d, o) => bandPick(d, o.band ?? "all", true) +
      pick("搜索中心", "base", [["alignment", "沿已登记位移"], ["fixed", "固定原图坐标"]], o.base ?? "alignment") +
      numeric("参考 X px", "x", o.x ?? d.apertures[0]?.x ?? d.dimensions.width / 2, 0, d.dimensions.width - 1) +
      numeric("参考 Y px", "y", o.y ?? d.apertures[0]?.y ?? d.dimensions.height / 2, 0, d.dimensions.height - 1) +
      numeric("搜索半径 px", "search_radius", o.search_radius ?? 11, 3, 40, 1) +
      numeric("偏离提示阈值 px", "movement_limit", o.movement_limit ?? 2, .25, 40),
  },
  detector: {
    title: "坏点监测", route: "/detector", result: detectorResult, images: ["image"],
    fields: (d, o) => pick("校准证据", "kind", [["hot", "暗场热点"], ["unstable", "偏置不稳定"], ["low_response", "平场低响应"]], o.kind ?? "hot") +
      bandPick(d, filtered(d, o)) + numeric("阈值（自动可空）", "threshold", o.threshold ?? "", .01, 10000) +
      numeric("最低重复比例", "fraction", o.fraction ?? .75, .5, 1) +
      numeric("合并外扩 px", "padding", o.padding ?? 1, 0, 4, 1),
  },
  gain: {
    title: "增益复核", route: "/gain", result: gainResult, csv: "detector-gain-check.csv",
    fields: (d, o) => bandPick(d, filtered(d, o)) + numeric("取样块宽度 px", "tile_size", o.tile_size ?? 32, 12, 96, 1) +
      numeric("平场亮度允许差", "stability", o.stability ?? .025, .001, .10),
  },
  background: {
    title: "天空背景图", route: (_, o) => `/exposures/${o.frame}/background`, result: backgroundResult, images: ["sky", "residual", "support"],
    fields: (d, o, v) => framePick(d, "曝光", "frame", o.frame ?? selectedFrame(v)?.id) +
      pick("影像层", "stream", Object.entries(sources), o.stream ?? "calibrated") +
      pick("背景样本", "sample", [["anchors", "只用我圈定的空天"], ["auto", "自动分割网格"]], o.sample ?? (d.sky_samples?.length ? "anchors" : "auto"), true) +
      pick("拟合阶次", "kind", [["plane", "平面"], ["curved", "二次面"]], o.kind ?? "plane") +
      numeric("每边网格数（自动样本）", "tiles", o.tiles ?? 8, 4, 18, 1) +
      numeric("局部照顾半径 px", "reach", o.reach ?? 64, 8, 512, 1) +
      numeric("样本冲突阈值 σ", "conflict_limit", o.conflict_limit ?? 3.2, 1.5, 8) +
      numeric("预测不确定度上限 / 样本噪声", "error_limit", o.error_limit ?? 1, .25, 4),
  },
  growth: {
    title: "孔径生长", route: "/growth", result: growthResult, csv: "aperture-growth.csv",
    fields: (d, o) => sourcePick(d, o.source) + bandPick(d, filtered(d, o)) +
      numeric("最小孔径半径", "minimum", o.minimum ?? 1.5, 1.5, 28) +
      numeric("最大孔径半径", "maximum", o.maximum ?? Math.min(7, d.photometry.sky_inner - 1), 1.5, 28) +
      numeric("目标累计光度比例", "plateau", o.plateau ?? .98, .75, .999),
  },
  pair: {
    title: "配对强度", route: "/pair-inspection", result: pairResult, images: ["image"],
    fields: (d, o) => framePick(d, "曝光 A", "a", o.a ?? d.reference) +
      framePick(d, "曝光 B", "b", o.b ?? d.frames.find(row => row.id !== d.reference)?.id) +
      numeric("明亮样本分位", "low_quantile", o.low_quantile ?? .84, .05, .99) +
      check("用星场亮区估计倍率", "star_only", o.star_only ?? true),
  },
  footprint: {
    title: "共同覆盖", route: "/footprint", result: footprintResult, images: ["image"],
    fields: (d, o) => bandPick(d, o.band ?? "all", true) +
      pick("覆盖依据", "layer", coverageModes, o.layer ?? "valid") +
      numeric("最低覆盖帧数", "minimum", o.minimum ?? Math.min(2, d.frames.filter(row => row.enabled).length), 1, 48, 1) +
      numeric("边缘留量 px", "margin", o.margin ?? 0, 0, 32, 1),
  },
  surface: {
    title: "区域光度", route: "/surface", result: surfaceResult, images: ["image"], csv: "extended-source-photometry.csv",
    fields: (d, o) => bandPick(d, filtered(d, o)) + numeric("最低区域可用比例", "fraction", o.fraction ?? .85, .5, 1) + surfaceFields(d, o),
  },
  contact: {
    title: "曝光预览册", route: "/contact", result: contactResult, images: ["image"], exportFormat: "download", suffix: "png",
    fields: (d, o) => pick("影像层", "stream", Object.entries(sources), o.stream ?? "calibrated") +
      numeric("排版列数", "columns", o.columns ?? 4, 2, 6, 1) +
      pick("版面主题", "theme", [["dark", "深色"], ["print", "浅色"]], o.theme ?? "dark") +
      check("统一亮度范围", "linked_scale", o.linked_scale ?? true) +
      check("在预览册中标出坏区", "flag_invalid", o.flag_invalid ?? false),
  },
  cutout: {
    title: "孔径取证框", route: "/cutout", result: cutoutResult, exportFormat: "download", suffix: "zip",
    fields: (d, o) => sourcePick(d, o.source) + bandPick(d, filtered(d, o)) +
      pick("取证影像层", "stream", sourceFits, o.stream ?? "calibrated") +
      numeric("输出边长 px（奇数）", "size", o.size ?? 41, 17, 161, 2),
  },
};

export function fieldTools(view) {
  const doc = view.doc;
  if (!doc.frames.length) return panel("核验工具", box("请先导入一张科学曝光。"));
  const key = view.fieldTool || "timeline";
  const definition = definitions[key];
  const result = view.toolResult?.key === key ? view.toolResult : null;
  const options = view.fieldOptions?.[key] || {};
  const selectors = `<div class="widget-controls"><label class="field">核验类型<select id="field-tool" aria-label="核验类型">
    ${Object.entries(definitions).map(([id, tool]) => `<option value="${id}" ${id === key ? "selected" : ""}>${E(tool.title)}</option>`).join("")}
    </select></label></div>`;
  const buttons = definition.toolbar ? definition.toolbar(doc) : "";
  let exports = "";
  if (result && definition.csv) exports += downloadButton("下载 CSV", "csv", definition.csv);
  if (result && definition.suffix) exports += downloadButton(definition.suffix === "png" ? "下载预览册" : "下载 FITS／PNG ZIP",
    definition.exportFormat, definition.suffix === "png" ? "exposure-contact-sheet.png" : "native-detector-cutouts.zip");
  return `<div class="column">
    ${panel(definition.title, `<form id="field-form" class="forms three">${definition.fields(doc, options, view)}<div class="actions wide">
      ${runButton("执行" + definition.title)}${buttons}</div></form>`, {controls: selectors})}
    ${panel("检查结果", result ? definition.result(result) : box("执行后在这里核对结果。"), {controls: exports})}</div>`;
}
