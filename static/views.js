import {
  E, action, apiPath, assetKinds, badge, box, check, choose, field, fmt, input, panel, pct, pick, queries,
  roles, shortDate, sources, stateLabel, status, table, time,
} from "./ui.js";

export const tabs = {
  inspect: "影像阅览", acquisition: "采集与校准", alignment: "对准与差分",
  photometry: "孔径光度", quality: "曝光质检", stack: "叠加", field: "核验工具", archives: "结果归档",
};

export function selectedFrame(view) {
  return [...view.doc.frames, ...view.doc.calibration_frames].find(frame => frame.id === view.frame);
}

export function filters(doc, includeAll = false) {
  const entries = [...new Set(doc.frames.filter(frame => frame.enabled).map(frame => frame.band))].sort();
  return [...(includeAll ? [["all", "全部滤镜"]] : []), ...entries.map(value => [value, value || "无滤镜标签"])];
}

export const currentBand = doc => filters(doc)[0]?.[0] ?? "";

export function noSeries() {
  return `<section class="empty"><span class="logo" aria-hidden="true"><i></i><i></i><i></i></span>
    <h1>观测库工作区</h1><p>打开一组曝光查看星场、校准原图，并沿时间量测天体亮度。可以导入 FITS 或灰度相机图，或先用合成星场走一遍流程。</p>
    <div class="actions">${action("打开疏散星场", "demo", {style: "primary", attrs: 'data-scenario="variable"'})}
    ${action("新建空白观测", "new", {style: "ghost"})}</div></section>`;
}

export function title(view) {
  const doc = view.doc;
  const frames = doc.frames.length;
  const enabled = doc.frames.filter(frame => frame.enabled).length;
  const aligned = Object.keys(doc.alignments).length + (doc.reference ? 1 : 0);
  const dt = doc.frames[0]?.started;
  return `<div class="titlebar"><div><h1>${E(doc.name)}</h1><div class="title-details">
    ${doc.synthetic ? badge("合成图像", "cyan") : ""}<span><b>${frames}</b> 张曝光</span><span><b>${enabled}</b> 张参与</span>
    <span><b>${aligned}</b> 张已对准</span>${dt ? `<span class="mono">${shortDate(dt)} UTC</span>` : ""}</div>
    ${doc.notes ? `<p class="meta-note">${E(doc.notes)}</p>` : ""}</div><div class="title-buttons">
    ${action("观测设置", "observation-settings", {style: "ghost"})}
    <a class="button ghost" href="${apiPath(doc.id, "/bundle")}">下载观测包 ↓</a></div></div>
    <nav class="tabs" aria-label="工作区">${Object.entries(tabs).map(([key, label]) => `<button
      class="${view.tab === key ? "selected" : ""}" type="button" data-action="tab" data-tab="${key}" ${view.tab === key ? 'aria-current="page"' : ""}>${label}</button>`).join("")}</nav>`;
}

function previewPath(view, frame) {
  const stream = frame.kind === "science" ? view.stream : "raw";
  return apiPath(view.doc.id, `/exposures/${frame.id}/image.png?` + queries({
    stream, flags: view.flags ? 1 : 0, version: view.doc.revision,
  }));
}

function selectedPreview(view, frame) {
  const doc = view.doc;
  const ready = view.stream !== "aligned" || frame.id === doc.reference || doc.alignments[frame.id];
  const stream = frame.kind === "science" ? view.stream : "raw";
  const controls = `<div class="widget-controls"><select id="stream" aria-label="影像图层" ${frame.kind === "science" ? "" : "disabled"}>
    ${choose(Object.entries(sources), stream)}</select>
    ${action(view.flags ? "隐藏坏区标色" : "显示坏区标色", "flags", {style: "ghost"})}</div>`;
  const image = ready || frame.kind !== "science"
    ? `<div class="frame-layer"><img id="frame-image" alt="${E(frame.filename)} 的星场预览" src="${previewPath(view, frame)}">
      <canvas id="overlay" aria-label="影像区域与像素取样"></canvas></div>`
    : box("当前曝光待对准。可先查看原始像素，或在对准页计算位移。");
  const tape = doc.frames.length > 0 ? `<div class="frame-tape" aria-label="曝光序列">
    ${doc.frames.map((item, index) => `<button type="button" class="${item.id === frame.id ? "active" : ""} ${item.enabled ? "" : "off"}"
      data-action="frame" data-id="${item.id}" aria-label="查看 ${E(item.filename)}" title="${E(item.filename)} · ${time(item.started)} UTC">${String(index + 1).padStart(2, "0")}</button>`).join("")}</div>` : "";
  return panel("曝光影像", `<div class="stage">${image}</div><div class="view-footer">
    <span>${E(frame.filename)} · ${doc.dimensions.width} × ${doc.dimensions.height}</span>
    <span id="cursor-readout">X — / Y —</span></div>${tape}`,
  {hint: frame.kind === "science" ? "点击图面读取像素，可取孔径圆心或采样线端点。" : "校准文件原图", controls, cls: "preview-card", bare: true});
}

function exposureDetails(view, frame) {
  const doc = view.doc;
  const off = frame.id === doc.reference ? {dx: 0, dy: 0, source: "reference"} : doc.alignments[frame.id];
  const readout = view.pixel ? `<div class="meta-list"><div><span>像素 (${view.pixel.x}, ${view.pixel.y})</span><b class="mono">
    ${view.pixel.valid ? fmt(view.pixel.value, 2) : "无效"} ADU</b></div><div><span>邻域中位</span><b>${fmt(view.pixel.nearby.median, 2)} ADU</b></div>
    <div><span>饱和</span><b>${view.pixel.saturated ? "是" : "否"}</b></div></div>` : "";
  return panel("曝光信息", `<div class="kpis"><div class="kpi"><span>曝光</span><strong>${fmt(frame.exposure, 1)}s</strong></div>
    <div class="kpi"><span>滤镜</span><strong>${E(frame.band || "—")}</strong></div>
    <div class="kpi"><span>开始 · UTC</span><strong>${time(frame.started)}</strong></div></div>
    <div class="meta-list"><div><span>参与状态</span><b>${frame.enabled ? "参与计算" : "已挂起"}</b></div>
    <div><span>文件</span><b>${E(frame.filename)}</b></div><div><span>参考位移</span><b class="mono">
    ${off ? `${fmt(off.dx)}, ${fmt(off.dy)} px` : "—"}</b></div></div>
    ${frame.notes ? `<p class="help">${E(frame.notes)}</p>` : ""}${readout}
    <div class="actions">${action("编辑曝光", "edit-exposure", {style: "tiny", attrs: `data-id="${frame.id}"`})}
    ${action("查看头信息", "header", {style: "tiny ghost"})}
    <a class="button tiny ghost" href="${apiPath(doc.id, `/exposures/${frame.id}/original`)}">下载原图 ↓</a></div>`);
}

function inspectTools(view, frame) {
  const state = {pixel: "读像素", aperture: "选孔径", profile: "像素剖面", mask: "框选坏区"};
  const native = view.stream !== "aligned";
  const coordinateNote = view.tool === "mask" ? "在原生影像选取两个对角点，坏区作用于当前曝光或整个探测器。"
    : view.tool === "profile" ? "选取两个端点，显示并可下载中间的像素值。" : "";
  const legend = `<div class="legend-inline"><span><i class="target"></i>目标</span><span><i class="comparison"></i>比较</span><span><i class="check"></i>检查</span></div>`;
  const buttons = Object.entries(state).map(([key, label]) => action(label, "tool", {
    style: `tiny ${view.tool === key ? "primary" : "ghost"}`,
    attrs: `data-tool="${key}"`,
    disabled: key === "mask" && (!native || frame.kind !== "science") || key === "aperture" && frame.kind !== "science",
  })).join("");
  const quick = action("分布直方图", "histogram", {style: "tiny ghost"}) +
    action(view.stars?.length ? "清除检星" : "显示星点", "stars", {style: "tiny ghost"}) +
    action("调整拉伸", "stretch", {style: "tiny ghost"});
  return panel("图面工具", `<div class="actions">${buttons}</div>${coordinateNote ? `<p class="help">${coordinateNote}</p>` : ""}
    <hr><div class="actions">${quick}</div>${view.chart === "histogram" ? '<canvas class="chart short" id="histogram-chart" aria-label="像素值直方图"></canvas>' : ""}
    ${view.chart === "profile" ? `<canvas class="chart short" id="profile-chart" aria-label="实测像素剖面"></canvas>
      ${action("下载剖面 CSV", "profile-csv", {style: "tiny ghost"})}` : ""}`, {controls: legend});
}

export function inspection(view) {
  const frame = selectedFrame(view);
  if (!frame) {
    return panel("曝光影像", box("导入科学曝光或点选左侧合成星场。"), {controls: action("导入曝光", "upload", {style: "primary"})});
  }
  return `<div class="workspace-grid"><div class="column">${selectedPreview(view, frame)}</div>
    <div class="column">${exposureDetails(view, frame)}${inspectTools(view, frame)}</div></div>`;
}

function exposureTable(view, frames, isCal) {
  if (!frames.length) return box(isCal ? "暂无校准参考。" : "暂无科学曝光。");
  const columns = isCal ? ["原图", "用途", "滤镜", "秒", "参与", ""] : ["原图", "UTC", "滤镜", "秒", "对准", "参与", ""];
  const rows = frames.map(frame => ({
    cls: frame.id === view.frame ? "active" : "",
    cells: [
      `<button class="button ghost tiny" type="button" data-action="open-frame" data-id="${frame.id}">${E(frame.filename)}</button>`,
      isCal ? assetKinds[frame.kind] : `<span class="mono">${time(frame.started)}</span>`,
      E(frame.band || "—"), `<span class="mono">${fmt(frame.exposure, 1)}</span>`,
      ...(isCal ? [] : [frame.id === view.doc.reference ? badge("基准", "cyan")
        : view.doc.alignments[frame.id] ? E(stateLabel(view.doc.alignments[frame.id].source)) : "—"]),
      status(frame.enabled ? "参与" : "挂起", frame.enabled),
      action("编辑", "edit-exposure", {style: "tiny ghost", attrs: `data-id="${frame.id}"`}),
    ],
  }));
  return table(columns, rows);
}

export function acquisition(view) {
  const doc = view.doc;
  const calibration = doc.calibration;
  const rows = view.calibration?.masters ?? {};
  const summaries = Object.entries(rows).filter(([key]) => key !== "dark_applications").map(([key, value]) => ({
    cells: [E(key), fmt(value.median, 3), fmt(value.sigma, 3), String(value.invalid)],
  }));
  return `<div class="workspace-grid"><div class="column">
    ${panel("科学曝光", exposureTable(view, doc.frames, false), {bare: true, controls: action("导入曝光", "upload", {style: "primary"})})}
    ${panel("校准参考", exposureTable(view, doc.calibration_frames, true), {bare: true, controls: action("导入校准帧", "upload", {attrs: 'data-kind="bias"'})})}</div>
    <div class="column">${panel("校准顺序", `<p class="help">原图保留。依次应用偏置、暗信号与同滤镜平场，坏区随测量一起传递。</p>
      <div class="meta-list"><div><span>偏置</span><b>${calibration.bias ? "开启" : "停用"}</b></div>
      <div><span>暗场</span><b>${calibration.dark ? "开启" : "停用"}</b></div>
      <div><span>平场</span><b>${calibration.flat ? "开启" : "停用"}</b></div>
      <div><span>平场最低响应</span><b>${fmt(calibration.flat_floor, 2)}</b></div></div>
      <div class="actions">${action("编辑校准", "edit-calibration", {style: "tiny"})}${action("读取主校准", "calibration", {style: "tiny ghost"})}</div>
      ${summaries.length ? `<hr>${table(["主图", "中位", "噪声", "无效像素"], summaries)}` : ""}
      ${rows.dark_applications?.length ? `<hr><div class="meta-list">${rows.dark_applications.map(row => `<div><span>${fmt(row.exposure, 1)}s · ${row.files.length} 张</span><b>${E(row.method || "未应用暗场")}</b></div>`).join("")}</div>` : ""}`)}
    ${regionPanel(view, "masks")}</div></div>`;
}

function regionPanel(view, key) {
  const document = view.doc;
  const aperture = key === "apertures";
  const label = aperture ? "孔径列表" : "探测器坏区";
  const code = aperture ? "aperture" : "mask";
  const list = document[key];
  const body = list.length ? `<ul class="region-list">${list.map(item => {
    const range = aperture ? `${roles[item.role]} · ${fmt(item.x, 1)}, ${fmt(item.y, 1)}`
      : `${fmt(item.left, 0)},${fmt(item.top, 0)} → ${fmt(item.right, 0)},${fmt(item.bottom, 0)} · ${item.frame ? "单帧" : "全组"}`;
    return `<li><span><b>${E(item.label)}</b><small>${E(range)}</small></span>
      ${action("编辑", "edit-region", {style: "tiny ghost", attrs: `data-id="${item.id}" data-kind="${key}"`})}</li>`;
  }).join("")}</ul>` : box(aperture ? "可在图面点选圆心，或输入参考帧坐标。" : "暂无人工掩膜。");
  return panel(label, body, {controls: action(aperture ? "加孔径" : "加坏区", "add-region", {
    style: "tiny", attrs: `data-kind="${code}"`, disabled: !document.reference,
  })});
}

export function alignment(view) {
  const doc = view.doc;
  const labels = doc.frames.map(item => [item.id, item.filename]);
  const chosen = doc.frames.find(frame => frame.id === doc.reference);
  const data = Object.fromEntries((view.registration?.rows || []).map(row => [row.frame, row]));
  const rows = doc.frames.map(frame => {
    const off = frame.id === doc.reference ? {source: "reference", dx: 0, dy: 0} : doc.alignments[frame.id];
    const decision = data[frame.id];
    return {cells: [
      E(frame.filename), fmt(off?.dx), fmt(off?.dy), fmt(off?.score ?? decision?.score, 3),
      status(stateLabel(off?.source ?? decision?.state ?? "unaligned"), Boolean(off)),
      action("改位移", "offset", {style: "tiny ghost", attrs: `data-id="${frame.id}"`, disabled: frame.id === doc.reference}),
    ]};
  });
  const pair = view.pair || {a: chosen?.id, b: doc.frames.find(frame => frame.id !== chosen?.id)?.id, normalize: "median"};
  return `<div class="workspace-grid"><div class="column">
    ${panel("参考帧位移", rows.length ? table(["曝光", "ΔX px", "ΔY px", "相似度", "状态", ""], rows) : box("请先导入科学曝光。"), {
      bare: true, hint: chosen ? `固定基准 · ${chosen.filename}` : "",
      controls: action("自动对准", "register", {style: "primary", disabled: doc.frames.length < 2}),
    })}</div><div class="column">
    ${panel("对准后差分", `<form id="comparison-form" class="forms">${pick("基准影像 A", "a", labels, pair.a)}
      ${pick("比对影像 B", "b", labels, pair.b)}${pick("差分零位", "normalize", [["median", "对齐两图背景"], ["none", "保留背景差"]], pair.normalize, true)}
      <div class="actions wide">${action("查看差分", "compare", {disabled: labels.length < 2})}
        ${action(view.blink ? "停止闪烁" : "闪烁浏览", "blink", {style: "ghost", disabled: labels.length < 2})}</div></form>
      <div id="difference-output">${view.pairImage ? `<figure class="wide-preview"><img src="${view.pairImage}" alt="两张已对准曝光的差分">
       <figcaption>橙色为 A 较亮，蓝色为 B 较亮。紫色区域缺少共同覆盖。</figcaption></figure>` : ""}</div>`,
      {hint: "已对准图像，固定画幅。", controls: action("设置基准", "reference", {style: "tiny ghost", disabled: !labels.length})})}
  </div></div>`;
}

export function sourceChoices(doc) {
  return Object.fromEntries(Object.entries(roles).map(([role]) => [
    role,
    doc.apertures.filter(item => item.role === role).map(item => [item.id, item.label]),
  ]));
}

function curveRows(view, curve) {
  return curve.rows.map(row => ({
    cls: view.frame === row.frame ? "active" : "",
    cells: [
      action(row.filename, "open-frame", {style: "ghost tiny", attrs: `data-id="${row.frame}"`}),
      time(row.midpoint),
      ...["target", "check"].map(role => row[role]?.state === "measured"
        ? `<span class="mono">${fmt(row[role].magnitude, 4)} ± ${fmt(row[role].error, 4)}</span>`
        : E(stateLabel(row[role]?.state))),
      status(stateLabel(row.target.state), row.target.state === "measured"),
    ],
  }));
}

export function photometry(view) {
  const doc = view.doc;
  const options = sourceChoices(doc);
  const curve = view.curve;
  const saved = view.curveOptions || {band: currentBand(doc)};
  const controls = `<div class="actions">${action("保存结果", "save-result", {style: "tiny", attrs: 'data-kind="photometry"', disabled: !curve})}
    ${action("下载曲线 CSV", "result-csv", {style: "tiny ghost", attrs: 'data-kind="photometry"', disabled: !curve})}</div>`;
  const selection = `<form id="curve-form" class="forms">${pick("滤镜", "band", filters(doc), saved.band)}
    ${pick("目标星", "target", options.target, saved.target)}${pick("比较星", "comparison", options.comparison, saved.comparison)}
    ${pick("检查星", "check", [["", "不使用"], ...options.check], saved.check)}
    <div class="wide actions"><button class="button primary" type="submit" ${options.target.length && options.comparison.length ? "" : "disabled"}>计算光度曲线</button>
    ${action("孔径设置", "photometry-settings", {style: "ghost"})}</div></form>`;
  const chart = curve ? `<div class="panel-body"><div class="legend-inline"><span><i class="target"></i>目标 − 比较</span>
    <span><i class="check"></i>检查 − 比较</span><span>UTC · mag</span></div>
    <canvas class="chart" id="curve-chart" aria-label="差分光度曲线"></canvas></div>
    ${table(["曝光", "中点 UTC", "目标差分", "检查差分", "状态"], curveRows(view, curve))}` : box("选择孔径与滤镜，计算当前差分光度。");
  return `<div class="workspace-grid"><div class="column">
    ${panel("差分光度曲线", chart, {hint: curve ? `${curve.summary.measured} / ${curve.summary.frames} 张曝光可量测` : "", controls, bare: true})}</div>
    <div class="column">${panel("本次光度测量", selection, {hint: "在原生曝光中落下参考坐标的孔径。"})}
    ${regionPanel(view, "apertures")}${curve ? measurements(view, curve) : ""}</div></div>`;
}

function measurements(view, curve) {
  const row = curve.rows.find(row => row.frame === view.frame) || curve.rows[0];
  if (!row) return "";
  return panel("单帧孔径读数", table(["源点", "净 ADU/s", "覆盖", "背景 ADU"], row.measurements.map(measurement => ({
    cells: [E(measurement.label), fmt(measurement.flux_per_second, 1), pct(measurement.coverage), fmt(measurement.background?.level, 1)],
  }))), {hint: row.filename, bare: true});
}

export function quality(view) {
  const doc = view.doc;
  const opts = view.qualityOptions || {band: "all", stream: "calibrated", fwhm_limit: 5, elongation_limit: 1.7};
  const rows = (view.quality?.rows || []).map(row => ({cells: [
    action(row.filename, "open-frame", {style: "tiny ghost", attrs: `data-id="${row.frame}"`}),
    String(row.stars_used), fmt(row.median_fwhm), fmt(row.median_elongation), fmt(row.sky?.median, 1),
    fmt(row.sky_spread, 1), pct(row.masked_fraction), row.flags.length ? row.flags.map(flag => badge(stateLabel(flag), "warn")).join(" ") : status("正常", true),
  ]}));
  return `<div class="column">${panel("质检设置", `<form id="quality-form" class="forms three">
    ${pick("滤镜", "band", filters(doc, true), opts.band)}${pick("测量图层", "stream", Object.entries(sources).slice(0, 2), opts.stream)}
    ${input("星点 FWHM 提示阈值 px", "fwhm_limit", opts.fwhm_limit, {type: "number", min: 1, max: 25})}
    ${input("拖尾轴比提示阈值", "elongation_limit", opts.elongation_limit, {type: "number", min: 1.05, max: 9})}
    <div class="actions"><button type="submit" class="button primary" ${doc.frames.length ? "" : "disabled"}>运行曝光质检</button></div></form>`)}
    ${panel("曝光质量记录", rows.length ? table(["曝光", "星点数", "FWHM px", "拖尾轴比", "天空 ADU", "背景跨度 ADU", "坏区", "提示"], rows) : box("运行质检后逐帧显示星点和背景情况。"),
    {bare: true, controls: `<div class="actions">${action("保存结果", "save-result", {style: "tiny", attrs: 'data-kind="quality"', disabled: !rows.length})}
    ${action("下载质检 CSV", "result-csv", {style: "tiny ghost", attrs: 'data-kind="quality"', disabled: !rows.length})}</div>`})}</div>`;
}

export function stacks(view) {
  const doc = view.doc;
  const opts = view.stackOptions || {band: currentBand(doc), mode: "mean", min_support: Math.min(2, doc.frames.length) || 1, subtract_background: false};
  const stack = view.stack;
  const modes = [["mean", "均值"], ["median", "中位"], ["exposure", "按曝光时间加权"]];
  return `<div class="column">${panel("对准叠加", `<form id="stack-form" class="forms three">${pick("滤镜", "band", filters(doc), opts.band)}
    ${pick("像素光度合并方式", "mode", modes, opts.mode)}
    ${input("最低覆盖帧数", "min_support", opts.min_support, {type: "number", step: 1, min: 1, max: 48})}
    ${check("扣除每帧天空中位水平", "subtract_background", opts.subtract_background)}
    <div class="actions"><button type="submit" class="button primary" ${doc.frames.length ? "" : "disabled"}>生成叠加</button></div></form>
    ${stack ? `<div class="two-previews"><figure><img src="${stack.image}" alt="合并后的科学曝光">
      <figcaption>叠加图 · ADU/s</figcaption></figure><figure><img src="${stack.coverage}" alt="每像素的覆盖帧数">
      <figcaption>覆盖图 · 0 到 ${stack.report.used.length} 帧</figcaption></figure></div>` : ""}`, {
    hint: "在固定参考画幅内合并已对准曝光。", controls: stack ? `<div class="actions">${action("科学图 FITS", "stack-export", {style: "tiny", attrs: 'data-format="fits"'})}
    ${action("覆盖图 FITS", "stack-export", {style: "tiny ghost", attrs: 'data-format="coverage_fits"'})}</div>` : "",
  })}${stack ? panel("叠加依据", `<p class="help">合格画幅 ${pct(stack.report.qualified_fraction)} · 已用 ${stack.report.used.length} 张 · 未用 ${stack.report.skipped.length} 张</p>
    ${table(["科学曝光", "覆盖", "ΔX / ΔY px"], stack.report.used.map(row => ({cells: [E(row.filename), pct(row.valid_fraction), row.offset.map(x => fmt(x)).join(" / ")]})))}
    ${stack.report.skipped.length ? `<p class="help">${stack.report.skipped.map(row => E(row.filename) + " " + stateLabel(row.state)).join(" · ")}</p>` : ""}`) : ""}</div>`;
}

export function archives(view) {
  const archived = view.archive;
  const items = view.archives;
  const body = items === null ? box("正在读取归档…") : !items.length ? box("暂无已保存的光度或曝光质检。") :
    `<ul class="archive-list">${items.map(item => `<li><span><b>${E(item.label)}</b>
    <small>${item.kind === "photometry" ? "孔径光度" : item.kind === "quality" ? "曝光质检" : "对准"} · r${item.revision} · ${shortDate(item.created)} ${time(item.created)} UTC</small></span>
    <div class="actions">${action("查看结果", "archive", {style: "tiny", attrs: `data-id="${item.id}"`})}
    <a class="button tiny ghost" href="${apiPath(view.doc.id, `/snapshots/${item.id}?format=csv`)}">导出 ↓</a></div></li>`).join("")}</ul>`;
  const result = archived?.result;
  const data = result ? panel(archived.label, archived.kind === "photometry"
    ? `<canvas class="chart" id="archive-chart" aria-label="已保存光度曲线"></canvas>${table(["曝光", "中点 UTC", "目标差分", "检查差分", "状态"], curveRows(view, result))}`
    : table(["曝光", "星点数", "FWHM px", "拖尾轴比"], result.rows.map(row => ({cells: [E(row.filename), E(row.stars_used), fmt(row.median_fwhm), fmt(row.median_elongation)]}))),
    {hint: `按保存时的结果查看 · 观测版本 r${archived.revision}`, controls: action("移除归档", "delete-archive", {style: "tiny danger", attrs: `data-id="${archived.id}"`})}) : "";
  return `<div class="column">${panel("已保存结果", body, {controls: action("刷新结果", "archives", {style: "tiny ghost"})})}${data}</div>`;
}
