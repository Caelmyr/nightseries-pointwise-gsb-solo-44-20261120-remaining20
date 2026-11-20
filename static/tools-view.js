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
const NOISE_STATES = {
  stable: ["稳定", true],
  deviation_confirmed: ["偏离已证实", false],
  insufficient_independent: ["独立复测不足", false],
  saturated_present: ["档内有饱和", false],
  no_valid_tiles: ["无有效取样块", false],
  low_signal: ["信号过低", false],
  no_model: ["缺参考模型", false],
};
const noiseFlag = code => {
  const [label, good] = NOISE_STATES[code] || [code, false];
  return status(label, good);
};
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
  return `${overview(data, [["valid_tiles", "拟合背景块"], ["range", "拟合场跨度 ADU", rates], ["fit_rms_adu", "拟合误差 ADU", rates]])}
    <div class="two-previews">${figures([[images.sky, "平滑天空背景"], [images.residual, "保留天体的残差图"]])}</div>
    ${grid(["格", "背景 ADU", "支持", "拟合残差 ADU"], data.tiles.map(row =>
      [row.column + ", " + row.row, rates(row.level), pct(row.support), rates(row.residual)]))}`;
}

function footprintResult({data, images}) {
  return `${overview(data, [["used", "已用帧数"], ["adequate", "符合覆盖比例", pct],
   ["best", "最大可用内矩形", row => row ? `${row.left},${row.top} → ${row.right},${row.bottom}` : "—"]])}
   ${imageBlock(images.image, "亮度为支持数，金框为最大可用内矩形")}
   ${grid(["科学曝光", "状态", "覆盖比例"], data.rows.map(row => [E(row.filename), flag(row.state), pct(row.fraction)]))}`;
}

function gainResult({data}) {
  const allPairs = [...data.bias_pairs.map(row => ["偏置", row]), ...data.flat_pairs.map(row => ["平场", row])];
  return `${overview(data, [["gain_e_per_adu", "增益 e⁻/ADU", rates], ["gain_pair_spread", "配对增益离散", rates],
    ["read_noise_adu", "读出噪声 ADU", rates], ["read_noise_e", "读出噪声 e⁻", rates], ["tile_count", "每对取样格"]])}
    ${grid(["类型", "第一张", "第二张", "有效块", "估计值", "离散"], allPairs.map(([type, row]) =>
      [type, E(row.first), E(row.second), row.patches ?? "—", rates(row.median), rates(row.sigma)]))}`;
}

function noiseCurveResult({data}) {
  const window = data.stable_range || {};
  const windowKpis = [
    ["lower_adu", "稳定范围下限 净 ADU", v => rates(v)],
    ["upper_adu", "稳定范围上限 净 ADU", v => rates(v)],
    ["status", "上限依据", v => ({
      confirmed_ceiling: "偏离已证实", verified_top: "最高档已验证", evidence_incomplete: "亮档证据不足",
      no_stable_level: "无稳定档",
    }[v] || v)],
  ];
  const windowPanel = `<div class="meta-list">${windowKpis.map(([name, title, render]) =>
    `<div><span>${E(title)}</span><b>${window[name] == null ? "—" : E(render(window[name]))}</b></div>`).join("")}</div>
    <p class="help">${E(window.note || "—")}</p>
    ${window.evidence_incomplete_levels?.length ? `<p class="help">证据不足档：${
      window.evidence_incomplete_levels.map(row => `第 ${row.level + 1} 档（${NOISE_STATES[row.state]?.[0] || row.state}）`).join("、")
    }</p>` : ""}`;
  const levelRows = data.levels.map(level => [
    String(level.index + 1), rates(level.median_net_adu), rates(level.level_spread_adu),
    `${level.independent_frames} / ${level.frame_count}`,
    `${level.pair_count_measured}${level.pair_count_capped ? "（封顶）" : ""}`,
    rates(level.gain_median), rates(level.noise_adu), pct(level.max_saturated_fraction),
    noiseFlag(level.state),
  ]);
  const pairRows = data.levels.flatMap(level => level.pairs.map(pair => [
    String(level.index + 1), pair.state === "measured" ? status("已测量", true) : status("无有效块", false),
    E(pair.first), E(pair.second), rates(pair.net_level_adu), String(pair.tiles_kept),
    rates(pair.gain_median), rates(pair.noise_adu), E(rejectionText(pair.rejection_reasons)),
  ]));
  const frameRows = data.levels.flatMap(level => level.frames.map(frame => [
    String(level.index + 1), E(frame.filename), rates(frame.exposure), rates(frame.net_level_adu),
    rates(frame.tile_level_spread_adu), pct(frame.saturated_fraction), pct(frame.invalid_fraction),
    frame.used_in_pairs ? status("已纳入", true) : status("未纳入", false),
  ]));
  const legend = `<div class="legend-inline">${Object.entries(NOISE_STATES).map(([code, [label]]) =>
    `<span><i style="background:${noiseColor(code)};border-radius:50%"></i>${E(label)}</span>`).join("")}
    <span><i style="background:#b7cbff;height:2px;width:10px;border-radius:0"></i>光子转移参考</span>
    <span><i style="background:rgba(124,226,216,.5);width:10px;height:10px"></i>稳定范围</span></div>`;
  return `${overview(data, [["reference_gain_e_per_adu", "参考增益 e⁻/ADU", rates], ["read_noise_adu", "读出噪声 ADU", rates],
    ["bias_pair_count", "偏置对", null], ["flat_frame_count", "启用平场", null],
    ["min_independent", "独立复测最少帧数", null]])}
    <h3>稳定工作范围</h3>${windowPanel}
    <h3>噪声随亮度</h3>${legend}
    <canvas class="chart" id="noise-curve-chart" aria-label="噪声随平场净亮度变化曲线"></canvas>
    <p class="help">纵轴为单帧噪声 ADU（配对差方差减半、扣除偏置噪声底后的平方根）；点旁数字为亮度档序号。</p>
    <h3>逐亮度档（可追溯实际纳入的平场）</h3>
    ${grid(["档", "净亮度 ADU", "档内跨度", "独立／总平场", "有效配对", "增益 e⁻/ADU", "噪声 ADU", "最高饱和比例", "证据状态"], levelRows)}
    ${data.levels.some(level => level.state_reason) ? `<ul class="help compact-list">${data.levels.map(level =>
      `<li>第 ${level.index + 1} 档：${E(level.state_reason)}</li>`).join("")}</ul>` : ""}
    <h3>实际纳入的平场清单</h3>
    ${grid(["档", "平场文件", "曝光 s", "净亮度 ADU", "块间跨度", "饱和比例", "无效比例", "是否纳入配对"], frameRows)}
    ${data.excluded_frames?.length ? `<p class="help">未分档：${data.excluded_frames.map(row => E(row.filename)).join("、")}</p>` : ""}
    <h3>逐配对取证（含共用原图的所有配对与剔除原因）</h3>
    ${grid(["档", "状态", "第一张", "第二张", "净亮度 ADU", "保留块", "增益 e⁻/ADU", "噪声 ADU", "块剔除统计"], pairRows)}`;
}

function noiseColor(code) {
  return {
    stable: "#7ce2d8", deviation_confirmed: "#ef8f75", insufficient_independent: "#efba75",
    saturated_present: "#d491e3", no_valid_tiles: "#8b97ad", low_signal: "#9caacf", no_model: "#efba75",
  }[code] || "#9caacf";
}
const TILE_REJECTION_LABELS = {
  low_coverage: "覆盖不足", low_signal: "信号过低", lamp_drift: "灯源变动",
  trim_failed: "剪裁不足", floor_not_exceeded: "未超噪声底", gain_out_of_range: "增益越界",
};
const rejectionText = reasons => Object.entries(reasons || {})
  .map(([code, count]) => `${TILE_REJECTION_LABELS[code] || code}:${count}`).join(" ") || "—";

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
  noise_curve: {
    title: "噪声亮度曲线", route: "/noise-curve", result: noiseCurveResult,
    csv: "flat-noise-levels.csv", extraCsv: "flat-noise-pairs.csv",
    fields: (d, o) => {
      const flatBands = [...new Set(d.calibration_frames
        .filter(frame => frame.enabled && frame.kind === "flat")
        .map(frame => frame.band))].sort();
      const current = o.band ?? flatBands[0] ?? filters(d)[0]?.[0] ?? "";
      return pick("滤镜", "band", flatBands.map(band => [band, band || "无滤镜标签"]), current) +
        numeric("取样块宽度 px", "tile_size", o.tile_size ?? 32, 12, 96, 1) +
        numeric("亮度分档相对间隔", "level_gap", o.level_gap ?? .08, .02, .50) +
        numeric("配对亮度允许差", "stability", o.stability ?? .025, .001, .10) +
        numeric("稳定允许相对偏差", "tolerance", o.tolerance ?? .10, .03, .35) +
        numeric("独立复测最少平场数", "min_independent", o.min_independent ?? 3, 2, 8, 1) +
        numeric("稳定判定最低净 ADU", "min_net_adu", o.min_net_adu ?? 200, 10, 50000);
    },
  },
  background: {
    title: "天空背景图", route: (_, o) => `/exposures/${o.frame}/background`, result: backgroundResult, images: ["sky", "residual"],
    fields: (d, o, v) => framePick(d, "曝光", "frame", o.frame ?? selectedFrame(v)?.id) +
      pick("影像层", "stream", Object.entries(sources), o.stream ?? "calibrated") +
      pick("拟合阶次", "kind", [["plane", "平面"], ["curved", "二次面"]], o.kind ?? "plane") +
      numeric("每边网格数", "tiles", o.tiles ?? 8, 4, 18, 1),
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
  const hasCalibration = doc.calibration_frames.length > 0;
  if (!doc.frames.length) {
    if (!hasCalibration) return panel("核验工具", box("请先导入一张科学曝光。"));
    view.fieldTool = "noise_curve";
  }
  const key = view.fieldTool || "timeline";
  const definition = definitions[key];
  const result = view.toolResult?.key === key ? view.toolResult : null;
  const options = view.fieldOptions?.[key] || {};
  const selectors = `<div class="widget-controls"><label class="field">核验类型<select id="field-tool" aria-label="核验类型">
    ${Object.entries(definitions).map(([id, tool]) => `<option value="${id}" ${id === key ? "selected" : ""}>${E(tool.title)}</option>`).join("")}
    </select></label></div>`;
  const buttons = definition.toolbar ? definition.toolbar(doc) : "";
  let exports = "";
  if (result && definition.csv) exports += downloadButton("下载逐档 CSV", "level_csv", definition.csv);
  if (result && definition.extraCsv) exports += downloadButton("下载逐配对 CSV", "pair_csv", definition.extraCsv);
  if (result && definition.suffix) exports += downloadButton(definition.suffix === "png" ? "下载预览册" : "下载 FITS／PNG ZIP",
    definition.exportFormat, definition.suffix === "png" ? "exposure-contact-sheet.png" : "native-detector-cutouts.zip");
  return `<div class="column">
    ${panel(definition.title, `<form id="field-form" class="forms three">${definition.fields(doc, options, view)}<div class="actions wide">
      ${runButton("执行" + definition.title)}${buttons}</div></form>`, {controls: selectors})}
    ${panel("检查结果", result ? definition.result(result) : box("执行后在这里核对结果。"), {controls: exports})}</div>`;
}
