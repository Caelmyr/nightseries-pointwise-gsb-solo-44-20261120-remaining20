import {
  $, E, action, apiPath, assetKinds, check, download, field, fmt, input, panel, pick, promptForm, queries,
  request, roles, shortDate, showMessage, sources, stateLabel,
} from "./ui.js";
import { definitions, fieldTools } from "./tools-view.js";
import { curveChart, drawRegions, histogramChart, profileChart } from "./charts.js";
import {
  acquisition, alignment, archives, inspection, noSeries, photometry, quality, selectedFrame, stacks, tabs, title,
} from "./views.js";

const root = $("#workspace");
const nav = $("#series-nav");
const state = {list: [], view: null, epoch: 0};
const view = () => state.view;
const routes = {inspect: inspection, acquisition, alignment, photometry, quality, stack: stacks, field: fieldTools, archives};
let blinkTimer;
let resizeDebounce;
let renderEpoch = 0;

function withActive(pinned, handler) {
  if (view() === pinned) handler();
}

function mutableOptions(data = {}) {
  return {...data, version: view().doc.revision};
}

function currentPath(suffix = "") {
  return apiPath(view().doc.id, suffix);
}

function queryCurrent() {
  return "?" + queries({version: view().doc.revision});
}

function currentStream(frame) {
  return frame.kind === "science" ? view().stream : "raw";
}

function releaseUrl(url) {
  if (url?.startsWith("blob:")) URL.revokeObjectURL(url);
}

function resetResults(session) {
  if (session.stack) {
    releaseUrl(session.stack.image);
    releaseUrl(session.stack.coverage);
  }
  releaseUrl(session.pairImage);
  for (const url of Object.values(session.toolResult?.images || {})) releaseUrl(url);
  Object.assign(session, {
    curve: null, quality: null, calibration: null, stack: null, registration: null,
    pairImage: null, toolResult: null, stars: null, pixel: null, chart: null, profile: null, points: [],
  });
}

function stopBlink() {
  if (view()) view().blink = false;
  clearInterval(blinkTimer);
  blinkTimer = null;
}

function renderNav() {
  nav.innerHTML = state.list.length ? state.list.map(item =>
    `<button class="series-button ${view()?.doc.id === item.id ? "selected" : ""}" type="button" data-action="series" data-id="${item.id}">
      <b>${E(item.name)}</b><small>${item.frames} 张 · ${item.start ? shortDate(item.start) : item.synthetic ? "合成" : "待导入"}</small></button>`
  ).join("") : '<span class="empty-nav">暂无已保存观测</span>';
}

async function listSeries() {
  const results = await request("/api/series");
  state.list = results.items;
  renderNav();
}

function selectFrame(id, openView = false) {
  if (!view()) return;
  const session = view();
  session.frame = id;
  session.pixel = null;
  session.stars = null;
  session.points = [];
  session.profile = null;
  session.chart = null;
  if (openView) session.tab = "inspect";
  const frame = selectedFrame(session);
  if (frame?.kind !== "science" && session.tool !== "pixel") session.tool = "pixel";
  render();
}

async function selectSeries(id, supplied) {
  const epoch = ++state.epoch;
  const doc = supplied || await request(apiPath(id));
  if (epoch !== state.epoch) return;
  if (view()) resetResults(view());
  stopBlink();
  state.view = {
    doc, tab: "inspect", frame: doc.reference || doc.frames[0]?.id || doc.calibration_frames[0]?.id,
    stream: "calibrated", flags: false, tool: "pixel", points: [], archives: null, pair: null,
  };
  localStorage.setItem("nightseries-observation", id);
  renderNav();
  render();
}

async function updateDocument(doc, { chooseLast = false } = {}) {
  const session = view();
  if (!session || session.doc.id !== doc.id) return;
  resetResults(session);
  stopBlink();
  session.doc = doc;
  const visible = [...doc.frames, ...doc.calibration_frames];
  if (chooseLast || !visible.some(frame => frame.id === session.frame)) {
    session.frame = (chooseLast ? doc.frames.at(-1)?.id : doc.reference) || visible[0]?.id;
  }
  await listSeries();
  if (view() === session) render();
}

function render() {
  const session = view();
  ++renderEpoch;
  if (!session) {
    root.innerHTML = noSeries();
    return;
  }
  root.innerHTML = title(session) + routes[session.tab](session);
  const image = $("#frame-image");
  if (image) {
    image.addEventListener("load", repaint);
    image.addEventListener("error", () => showMessage("读取影像失败，可以重开观测或查看原始层。", true), {once: true});
  }
  requestAnimationFrame(repaint);
}

function repaint() {
  const session = view();
  if (!session) return;
  const overlay = $("#overlay");
  const image = $("#frame-image");
  const frame = selectedFrame(session);
  if (overlay && frame && image?.naturalWidth) {
    drawRegions(overlay, session.doc, frame, currentStream(frame), session.stars, session.points);
  }
  const curve = $("#curve-chart");
  if (curve && session.curve) curveChart(curve, session.curve, session.frame);
  const archived = $("#archive-chart");
  if (archived && session.archive) curveChart(archived, session.archive.result, session.frame);
  const histogram = $("#histogram-chart");
  if (histogram) histogramChart(histogram, session.histogram);
  const profile = $("#profile-chart");
  if (profile) profileChart(profile, session.profile);
}

function valueEditor(title, content, handler, submit) {
  promptForm(title, content, async (data, form) => {
    await handler(data, form);
  }, submit);
}

function updateSettings(patch) {
  return request(currentPath(), {method: "PATCH", data: mutableOptions(patch)}).then(doc => updateDocument(doc));
}

function fieldEditor(config) {
  return config.map(([label, name, value, opts]) => input(label, name, value, opts)).join("");
}

function newSeries() {
  valueEditor("新建观测", input("名称", "name", "", {required: true, wide: true, placeholder: "例如 N-042 · 9 月测光"}), async data => {
    const doc = await request("/api/series", {method: "POST", data});
    await listSeries();
    await selectSeries(doc.id, doc);
    showMessage("观测已创建，接下来可导入曝光。");
  }, "创建");
}

function observationSettings() {
  const {doc} = view();
  valueEditor("观测设置", input("观测名称", "name", doc.name, {required: true, wide: true}) +
    field("值班备注", `<textarea name="notes" maxlength="350">${E(doc.notes)}</textarea>`, true) +
    action("删除整个观测", "delete-series", {style: "tiny danger"}), updateSettings);
}

function upload(initial) {
  const content = [
    pick("影像用途", "kind", Object.entries(assetKinds), initial),
    field("原图文件", '<input type="file" name="files" accept=".fit,.fits,.fts,.png,.tif,.tiff" multiple required>', true),
    input("曝光开始 UTC（图像头已含时可留空）", "started", "", {placeholder: "2026-09-12T20:38:00Z", wide: true}),
    input("曝光秒数", "exposure", "", {type: "number", min: 0, max: 86400}),
    input("滤镜标签", "band", "", {placeholder: "FITS 可留空"}),
    input("导入备注", "notes", "", {wide: true}),
    '<p class="help wide">FITS 可从每张主头读取开始时间、曝光秒数与滤镜。文本框非空时应用于本次选中的全部文件；无 FITS 时间的灰度图可逐张导入。</p>',
  ].join("");
  valueEditor("导入曝光", content, async (_, form) => {
    const payload = new FormData(form);
    payload.set("version", view().doc.revision);
    const path = currentPath("/exposures");
    const doc = await request(path, {method: "POST", body: payload});
    await updateDocument(doc);
    showMessage("原曝光已保存。");
  }, "导入");
}

function editExposure(frameId) {
  const {doc} = view();
  const frame = [...doc.frames, ...doc.calibration_frames].find(item => item.id === frameId);
  if (!frame) return;
  const fields = [
    ...(frame.kind === "science" ? [input("开始时间 UTC", "started", frame.started, {wide: true, required: true})] : []),
    ...(frame.kind !== "bias" ? [input("曝光秒数", "exposure", frame.exposure, {type: "number", min: .001, max: 86400, required: true})] : []),
    input("滤镜标签", "band", frame.band), input("备注", "notes", frame.notes, {wide: true}),
    check("参与计算", "enabled", frame.enabled),
    action("删除曝光", "delete-exposure", {style: "tiny danger", attrs: `data-id="${frame.id}"`}),
  ].join("");
  valueEditor(frame.filename, fields, async (data, form) => {
    data.enabled = form.elements.enabled.checked;
    const result = await request(currentPath(`/exposures/${frameId}`), {method: "PATCH", data: mutableOptions(data)});
    await updateDocument(result);
  });
}

function photoSettings() {
  const p = view().doc.photometry;
  const count = {type: "number", required: true};
  const fields = [
    ["孔径半径 px", "radius", p.radius, {...count, min: 1.5, max: 28}],
    ["背景环内半径 px", "sky_inner", p.sky_inner, {...count, min: 3, max: 46}],
    ["背景环外半径 px", "sky_outer", p.sky_outer, {...count, min: 4, max: 60}],
    ["最低孔径覆盖 fraction", "min_fraction", p.min_fraction, {...count, min: .5, max: 1}],
    ["增益 e⁻/ADU", "gain", p.gain, {...count, min: .02, max: 1000}],
    ["探测器饱和值 ADU", "saturation", p.saturation, {...count, min: 1}],
  ];
  valueEditor("孔径与背景", fieldEditor(fields), data => updateSettings({photometry: data}));
}

function calibrationSettings() {
  const settings = view().doc.calibration;
  const fields = [
    check("应用偏置", "bias", settings.bias),
    check("应用暗场", "dark", settings.dark),
    check("应用平场", "flat", settings.flat),
    input("平场最低有效响应", "flat_floor", settings.flat_floor, {type: "number", min: .05, max: .9, required: true}),
  ].join("");
  valueEditor("校准开关与有效像素", fields, (data, form) => {
    for (const key of ["bias", "dark", "flat"]) data[key] = form.elements[key].checked;
    return updateSettings({calibration: data});
  });
}

function stretchSettings() {
  const s = view().doc.stretch;
  const fields = [
    pick("显示拉伸", "mode", [["asinh", "反双曲正弦"], ["linear", "线性"], ["log", "对数"]], s.mode, true),
    input("黑点百分位", "low", s.low, {type: "number", min: 0, max: 99.9, required: true}),
    input("白点百分位", "high", s.high, {type: "number", min: .1, max: 100, required: true}),
    check("反相", "invert", s.invert),
  ].join("");
  valueEditor("影像拉伸", fields, (data, form) => {
    data.invert = form.elements.invert.checked;
    return updateSettings({stretch: data});
  });
}

async function inspectHeader() {
  const session = view();
  const frame = selectedFrame(session);
  const stats = await request(currentPath(`/exposures/${frame.id}/summary`));
  withActive(session, () => {
    let lines = Object.entries(stats.header).map(([key, value]) => `<div><span>${E(key)}</span><b>${E(value)}</b></div>`).join("");
    lines += `<div><span>画幅</span><b>${stats.width} × ${stats.height}</b></div><div><span>原图中位</span><b>${fmt(stats.raw_statistics.median, 2)}</b></div>
      <div><span>无效像素</span><b>${stats.raw_statistics.invalid_pixels}</b></div>`;
    valueEditor("原始曝光头与取值域", `<div class="wide meta-list">${lines}</div>`, async () => {}, "关闭");
  });
}

function regionEditor(kind, original = null, position = {}) {
  const doc = view().doc;
  if (!doc.dimensions) return;
  const isAperture = kind === "apertures";
  const data = original || position;
  const dims = doc.dimensions;
  const int = {type: "number", min: 0, required: true};
  let fields = input(isAperture ? "源点标签" : "坏区名称", "label", data.label, {wide: true, required: true});
  if (isAperture) {
    fields += pick("孔径角色", "role", Object.entries(roles), data.role || "target", true);
    fields += fieldEditor([
      ["参考 X px", "x", data.x, {...int, max: dims.width - 1}],
      ["参考 Y px", "y", data.y, {...int, max: dims.height - 1}],
    ]);
  } else {
    fields += fieldEditor([
      ["左 X", "left", data.left, {...int, max: dims.width - 1}],
      ["上 Y", "top", data.top, {...int, max: dims.height - 1}],
      ["右 X（开区间）", "right", data.right, {...int, max: dims.width}],
      ["下 Y（开区间）", "bottom", data.bottom, {...int, max: dims.height}],
    ]);
    fields += pick("应用范围", "frame", [["", "整个探测器"], ...doc.frames.map(frame => [frame.id, frame.filename])], data.frame || "", true);
  }
  if (original) {
    fields += action("删除区域", "delete-region", {style: "tiny danger", attrs: `data-id="${original.id}" data-kind="${kind}"`});
  }
  valueEditor(isAperture ? "参考坐标孔径" : "原生坐标坏区", fields, async payload => {
    const suffix = `/regions/${kind}${original ? "/" + original.id : ""}`;
    const next = await request(currentPath(suffix), {method: original ? "PATCH" : "POST", data: mutableOptions(payload)});
    await updateDocument(next);
  });
}

function chooseReference() {
  const {doc} = view();
  const options = doc.frames.map(frame => [frame.id, frame.filename]);
  const note = doc.apertures.length ? "现有孔径绑定基准坐标，需先移除孔径方可更换。" : "切换后需重新测定各曝光位移。";
  valueEditor("固定参考帧", pick("参考曝光", "frame", options, doc.reference, true) +
    `<p class="help wide">${note}</p>`, async data => {
    const next = await request(currentPath("/reference"), {method: "PUT", data: mutableOptions(data)});
    await updateDocument(next);
  });
}

function offsetEditor(id) {
  const doc = view().doc;
  const off = doc.alignments[id] || {};
  const frame = doc.frames.find(row => row.id === id);
  if (!frame) return;
  const num = {type: "number", required: true, min: -512, max: 512};
  const fields = [
    input("ΔX px", "dx", off.dx ?? 0, num),
    input("ΔY px", "dy", off.dy ?? 0, num),
    '<p class="help wide">这里的位移从参考坐标指向本帧的原生像素坐标。</p>',
    ...(doc.alignments[id] ? [action("移除位移", "delete-offset", {style: "tiny danger", attrs: `data-id="${id}"`})] : []),
  ];
  valueEditor(frame.filename + " 位移", fields.join(""), async data => {
    const next = await request(currentPath(`/align/${id}`), {method: "PUT", data: mutableOptions(data)});
    await updateDocument(next);
  });
}

function register() {
  valueEditor("自动对准", input("最大搜索位移 px", "max_shift", 30, {type: "number", min: 2, max: 100}) +
    check("保留现有偏移", "only_missing", true), async (data, form) => {
    const session = view();
    data.only_missing = form.elements.only_missing.checked;
    const result = await request(currentPath("/register"), {method: "POST", data: mutableOptions(data)});
    await updateDocument(result.document);
    withActive(session, () => {
      session.registration = result.result;
      const measured = result.result.rows.filter(row => row.state === "measured").length;
      const uncertain = result.result.rows.filter(row => ["unreliable", "unavailable"].includes(row.state)).length;
      showMessage(`更新 ${measured} 个偏移 · ${uncertain} 张需人工检查`);
      render();
    });
  }, "计算并应用可靠位移");
}

function pairFromForm() {
  const data = Object.fromEntries(new FormData($("#comparison-form")));
  if (!data.a || !data.b || data.a === data.b) throw new Error("请选择两张不同曝光。");
  return data;
}

async function compare() {
  const session = view();
  const pair = pairFromForm();
  stopBlink();
  const path = currentPath(`/difference/${pair.a}/${pair.b}?` + queries({normalize: pair.normalize, version: session.doc.revision}));
  const blob = await request(path, {type: "blob"});
  withActive(session, () => {
    releaseUrl(session.pairImage);
    session.pairImage = URL.createObjectURL(blob);
    session.pair = pair;
    render();
  });
}

function blink() {
  const session = view();
  if (session.blink) {
    stopBlink();
    render();
    return;
  }
  const pair = pairFromForm();
  if ([pair.a, pair.b].some(id => id !== session.doc.reference && !session.doc.alignments[id])) {
    throw new Error("两张曝光都需要现有对准位移。");
  }
  const url = id => currentPath(`/exposures/${id}/image.png?` + queries({stream: "aligned", version: session.doc.revision}));
  session.pair = pair;
  session.blink = true;
  render();
  const host = $("#difference-output");
  host.innerHTML = '<figure class="wide-preview"><img alt="闪烁比较的已对准星场"><figcaption></figcaption></figure>';
  let first = true;
  const toggle = () => {
    const id = first ? pair.a : pair.b;
    $("img", host).src = url(id);
    $("figcaption", host).textContent = "正在查看 " + session.doc.frames.find(row => row.id === id).filename;
    first = !first;
  };
  toggle();
  blinkTimer = setInterval(toggle, 570);
}

async function frameApi(method, query = {}) {
  const session = view();
  const frame = selectedFrame(session);
  if (!frame) return null;
  const params = queries({stream: currentStream(frame), version: session.doc.revision, ...query});
  return request(currentPath(`/exposures/${frame.id}/${method}?` + params), {type: query.format === "csv" ? "blob" : "json"});
}

function overlayPoint(event, canvas, dims) {
  const area = canvas.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(dims.width - 1, (event.clientX - area.left) / area.width * dims.width - .5)),
    y: Math.max(0, Math.min(dims.height - 1, (event.clientY - area.top) / area.height * dims.height - .5)),
  };
}

async function overlayClick(event, canvas) {
  const session = view();
  const {doc, frame, tool} = session;
  const point = overlayPoint(event, canvas, doc.dimensions);
  if (tool === "pixel") {
    const sample = await frameApi("inspect", point);
    withActive(session, () => {
      if (session.frame !== frame) return;
      session.pixel = sample;
      render();
    });
    return;
  }
  if (tool === "aperture") {
    if (session.stream !== "aligned" && frame !== doc.reference) {
      const offset = doc.alignments[frame];
      if (!offset) throw new Error("请先完成此曝光的对准，或在参考帧添加孔径。");
      point.x -= offset.dx;
      point.y -= offset.dy;
    }
    regionEditor("apertures", null, {x: fmt(point.x, 2), y: fmt(point.y, 2)});
    return;
  }
  if (session.points.length >= 2) session.points = [];
  session.points.push(point);
  repaint();
  if (session.points.length < 2) return;
  const [a, b] = session.points;
  if (tool === "mask") {
    regionEditor("masks", null, {
      left: Math.floor(Math.min(a.x, b.x)), top: Math.floor(Math.min(a.y, b.y)),
      right: Math.floor(Math.max(a.x, b.x)) + 1, bottom: Math.floor(Math.max(a.y, b.y)) + 1,
      frame,
    });
    return;
  }
  const coordinates = {x0: a.x, y0: a.y, x1: b.x, y1: b.y};
  const result = await frameApi("profile", coordinates);
  withActive(session, () => {
    if (session.frame !== frame) return;
    session.profile = result;
    session.profileQuery = coordinates;
    session.chart = "profile";
    render();
  });
}

async function analysis(kind, data) {
  const session = view();
  const options = {...data};
  if (kind === "quality" && options.band === "all") delete options.band;
  const result = await request(currentPath(`/analysis/${kind}`), {method: "POST", data: mutableOptions(options)});
  withActive(session, () => {
    if (kind === "photometry") {
      session.curveOptions = options;
      session.curve = result;
    } else {
      session.qualityOptions = {...options, band: data.band};
      session.quality = result;
    }
    render();
  });
}

function calculationOptions(kind) {
  const options = {...(kind === "photometry" ? view().curveOptions : view().qualityOptions)};
  if (options.band === "all") delete options.band;
  return options;
}

async function resultCsv(kind) {
  const formats = {photometry: "light-curve.csv", quality: "exposure-quality.csv"};
  const body = await request(currentPath(`/analysis/${kind}`), {
    method: "POST", type: "blob", data: mutableOptions({...calculationOptions(kind), format: "csv"}),
  });
  download(body, formats[kind]);
}

function saveResult(kind) {
  valueEditor("保存当前结果", input("结果题名", "label", "", {required: true, wide: true}), async ({label}) => {
    await request(currentPath("/snapshots"), {
      method: "POST", data: mutableOptions({kind, options: calculationOptions(kind), label}),
    });
    view().archives = null;
    showMessage("已保存。可在结果归档查看。");
  });
}

async function stack(data) {
  const session = view();
  const path = currentPath("/stack");
  const options = mutableOptions(data);
  const report = await request(path, {method: "POST", data: options});
  const [image, coverage] = await Promise.all(
    ["image", "coverage"].map(format => request(path, {method: "POST", data: {...options, format}, type: "blob"}))
  );
  withActive(session, () => {
    if (session.stack) {
      releaseUrl(session.stack.image);
      releaseUrl(session.stack.coverage);
    }
    session.stack = {report, image: URL.createObjectURL(image), coverage: URL.createObjectURL(coverage)};
    session.stackOptions = data;
    render();
  });
}

async function exportStack(format) {
  const names = {fits: "aligned-stack.fits", coverage_fits: "stack-support.fits", status_fits: "stack-support-status.fits"};
  const bytes = await request(currentPath("/stack"), {
    method: "POST", data: mutableOptions({...view().stackOptions, format}), type: "blob",
  });
  download(bytes, names[format] || "stack.fits");
}

async function loadArchives() {
  const session = view();
  const items = await request(currentPath("/snapshots"));
  withActive(session, () => {
    session.archives = items.items;
    if (session.tab === "archives") render();
  });
}

function toolInputs(target, kind) {
  const values = Object.fromEntries(new FormData(target));
  for (const checkbox of target.querySelectorAll("input[type=checkbox]")) values[checkbox.name] = checkbox.checked;
  if (values.band === "all") delete values.band;
  if (values.threshold === "") delete values.threshold;
  if (kind === "surface") {
    for (const key of ["target", "sky"]) {
      values[key] = {};
      for (const boundary of ["left", "top", "right", "bottom"]) {
        const field = key + "_" + boundary;
        values[key][boundary] = values[field];
        delete values[field];
      }
    }
  }
  return values;
}

function toolRequest(session, key, options, format = null, type = "json") {
  const definition = definitions[key];
  const params = {...options, version: session.doc.revision};
  if (format) params.format = format;
  const path = typeof definition.route === "function" ? definition.route(session.doc, options) : definition.route;
  if (definition.method === "GET") {
    return request(apiPath(session.doc.id, path + "?" + queries(params)), {type});
  }
  return request(apiPath(session.doc.id, path), {method: "POST", data: params, type});
}

async function runTool(key, options) {
  const session = view();
  let data = await toolRequest(session, key, options, ["contact", "cutout"].includes(key) ? "json" : null);
  if (key === "detector") data = data.report;
  if (key === "sky" && data.field) {
    const path = apiPath(session.doc.id, "/sky/pixel?" + queries({
      x: options.x,
      y: options.y,
      version: session.doc.revision,
    }));
    data.point = await request(path);
  }
  const images = {};
  const formats = definitions[key].images ?? [];
  if (key !== "sky_audit" || data.worst) {
    for (const format of formats) {
      images[format] = await toolRequest(session, key, options, format, "blob");
    }
  }
  withActive(session, () => {
    for (const url of Object.values(session.toolResult?.images || {})) releaseUrl(url);
    session.fieldOptions ??= {};
    session.fieldOptions[key] = options;
    session.toolResult = {key, data, images: Object.fromEntries(
      Object.entries(images).map(([format, blob]) => [format, URL.createObjectURL(blob)]))};
    if (session.tab === "field") render();
  });
}

function csvTextarea(label, example) {
  return field(label, `<textarea name="csv" required spellcheck="false" placeholder="${E(example)}"></textarea>`, true);
}

function editLog() {
  const schema = "filename,started_utc,exposure_seconds,filter,notes,enabled";
  const example = schema + "\nfield-V-01.fits,2026-09-12T20:38:00Z,42,V,,true";
  valueEditor("校对曝光日志", csvTextarea("粘贴 UTF-8 CSV", example) +
    `<p class="help wide">按唯一文件名绑定已导入的原图。空值保留原记录，时间要带 UTC Z 或时区偏移。</p>`, async data => {
    const result = await request(currentPath("/timeline/log"), {method: "POST", data: mutableOptions(data)});
    const draft = result.result;
    if (draft.issues.length) throw new Error(draft.issues.map(row => `第 ${row.line} 行 · ${row.filename} · ${row.reason}`).join("\n"));
    const altered = draft.updates.filter(row => Object.keys(row.after).length);
    if (!altered.length) {
      showMessage("日志和本机现有记录一致。");
      return;
    }
    setTimeout(() => {
      const summary = altered.map(row => `<li>${E(row.filename)} · ${E(Object.keys(row.after).join("、"))}</li>`).join("");
      valueEditor("应用核对后的曝光日志", `<ul class="wide help">${summary}</ul>`, async () => {
        const next = await request(currentPath("/timeline/log"), {
          method: "POST", data: {...data, apply: true, version: draft.revision},
        });
        await updateDocument(next.document);
        showMessage(`已更新 ${altered.length} 张曝光元数据`);
      }, "写入曝光记录");
    }, 0);
  }, "预览变动");
}

function editCatalog() {
  valueEditor("导入本地星表", csvTextarea("列名 label,ra,dec,magnitude", "label,ra,dec,magnitude\nAtlas,0.021,23.4,12.3") +
    '<p class="help wide">赤经和赤纬使用 ICRS 十进制度。保存后可投影到当前参考图幅中。</p>', async data => {
    const next = await request(currentPath("/sky/catalog"), {method: "PUT", data: mutableOptions(data)});
    await updateDocument(next);
    showMessage("本地星表已存入观测。");
  }, "保存星表");
}

function editPlate() {
  const fields = [
    csvTextarea("定标星 CSV，参考坐标从零开始", "label,x,y,ra,dec,weight"),
    pick("拟合方法", "mode", [["all", "全定标星"], ["robust", "按角距剔除异常"]], "all"),
    input("星点残差容差（角秒）", "tolerance", 6, {type: "number", min: .1, max: 120}),
  ].join("");
  valueEditor("定标星拟合", fields, async data => {
    const result = await request(currentPath("/sky/plate"), {method: "POST", data: mutableOptions(data)});
    const measure = result.result;
    const revision = result.document.revision;
    const values = `<p class="help wide">拟合 RMS ${fmt(measure.rms_arcsec, 3)}″，${measure.anchors.filter(row => row.included).length} 星参与。</p>`;
    setTimeout(() => valueEditor("应用图幅定标", values, async () => {
      const next = await request(currentPath("/sky/plate"), {method: "POST", data: {...data, apply: true, version: revision}});
      await updateDocument(next.document);
      showMessage("已写入当前参考帧的天球定标。");
    }), 0);
  }, "拟合预览");
}

function confirmDelete(message, handler) {
  valueEditor("确认移除", `<p class="help wide">${E(message)}</p>`, handler, "移除");
}

const actions = {
  reload: listSeries,
  new: newSeries,
  series: ({id}) => selectSeries(id),
  demo: async ({scenario}) => {
    showMessage("正在渲染合成曝光…");
    const doc = await request(`/api/demo/${scenario}`, {method: "POST"});
    await listSeries();
    await selectSeries(doc.id, doc);
  },
  tab: async ({tab}) => {
    if (!(tab in tabs)) return;
    stopBlink();
    view().tab = tab;
    render();
    if (tab === "archives" && view().archives === null) await loadArchives();
  },
  frame: ({id}) => selectFrame(id),
  "open-frame": ({id}) => selectFrame(id, true),
  upload: ({kind}) => upload(kind || "science"),
  "observation-settings": observationSettings,
  "edit-exposure": ({id}) => editExposure(id),
  header: inspectHeader,
  flags: () => {
    view().flags = !view().flags;
    render();
  },
  stretch: stretchSettings,
  tool: ({tool}) => {
    view().tool = tool;
    view().points = [];
    render();
  },
  histogram: async () => {
    const session = view();
    const data = await frameApi("histogram");
    withActive(session, () => {
      session.histogram = data;
      session.chart = "histogram";
      render();
    });
  },
  stars: async () => {
    const session = view();
    if (session.stars) {
      session.stars = null;
      render();
      return;
    }
    const result = await frameApi("stars");
    withActive(session, () => {
      session.stars = result.stars;
      showMessage(`找到 ${result.stars_used} 个可量测星点`);
      render();
    });
  },
  "profile-csv": async () => download(await frameApi("profile", {...view().profileQuery, format: "csv"}), "pixel-profile.csv"),
  "edit-calibration": calibrationSettings,
  calibration: async () => {
    const session = view();
    const result = await request(currentPath("/calibration"));
    withActive(session, () => {
      session.calibration = result;
      render();
    });
  },
  "photometry-settings": photoSettings,
  "edit-region": ({id, kind}) => regionEditor(kind, view().doc[kind].find(item => item.id === id)),
  "add-region": ({kind}) => regionEditor(kind === "aperture" ? "apertures" : "masks"),
  "delete-region": ({id, kind}) => confirmDelete("该区域将从当前观测中移除。", async () => {
    const result = await request(currentPath(`/regions/${kind}/${id}`) + queryCurrent(), {method: "DELETE"});
    await updateDocument(result);
  }),
  "delete-exposure": ({id}) => confirmDelete("原始曝光和关联的单帧坏区将从本机观测移除。", async () => {
    const result = await request(currentPath(`/exposures/${id}`) + queryCurrent(), {method: "DELETE"});
    await updateDocument(result);
  }),
  "delete-series": () => confirmDelete("将移除本机这组观测的曝光、区域和归档结果。", async () => {
    await request(currentPath() + queryCurrent(), {method: "DELETE"});
    resetResults(view());
    state.view = null;
    localStorage.removeItem("nightseries-observation");
    await listSeries();
    render();
  }),
  reference: chooseReference,
  offset: ({id}) => offsetEditor(id),
  "delete-offset": async ({id}) => {
    const result = await request(currentPath(`/align/${id}`) + queryCurrent(), {method: "DELETE"});
    $("#editor").close();
    await updateDocument(result);
  },
  register,
  compare,
  blink,
  "result-csv": ({kind}) => resultCsv(kind),
  "save-result": ({kind}) => saveResult(kind),
  archives: loadArchives,
  archive: async ({id}) => {
    const session = view();
    const entry = await request(currentPath(`/snapshots/${id}`));
    withActive(session, () => {
      session.archive = entry;
      render();
    });
  },
  "delete-archive": ({id}) => confirmDelete("将删除这份已保存的结果。", async () => {
    await request(currentPath(`/snapshots/${id}`), {method: "DELETE"});
    view().archive = null;
    await loadArchives();
  }),
  "stack-export": ({format}) => exportStack(format),
  "log-edit": editLog,
  "sky-catalog": editCatalog,
  "sky-plate": editPlate,
  "sky-plate-clear": () => confirmDelete("移除本组手工定标，保留参考原图及其图像头。", async () => {
    const next = await request(currentPath("/sky/plate") + queryCurrent(), {method: "DELETE"});
    await updateDocument(next);
  }),
  "install-badmask": async () => {
    const session = view();
    const selected = [...document.querySelectorAll("[data-detector-region]:checked")].map(item => item.dataset.detectorRegion);
    const data = {...session.fieldOptions.detector, apply: true, selected};
    const result = await toolRequest(session, "detector", data);
    await updateDocument(result.document);
    showMessage("选择的坏区已写入探测器掩膜。");
  },
  "tool-export": async ({format, name}) => {
    const session = view();
    const key = session.toolResult.key;
    const blob = await toolRequest(session, key, session.fieldOptions[key], format, "blob");
    download(blob, name);
  },
};

async function operate(task, button) {
  if (button) button.disabled = true;
  try {
    await task();
  } catch (error) {
    showMessage(error.message, true);
  } finally {
    if (button?.isConnected) button.disabled = false;
  }
}

document.addEventListener("click", event => {
  const trigger = event.target.closest("[data-action]");
  if (trigger && actions[trigger.dataset.action]) {
    operate(() => actions[trigger.dataset.action](trigger.dataset), trigger);
    return;
  }
  if (event.target.id === "overlay") operate(() => overlayClick(event, event.target));
});

document.addEventListener("pointermove", event => {
  if (event.target.id !== "overlay" || !view()?.doc.dimensions) return;
  const point = overlayPoint(event, event.target, view().doc.dimensions);
  $("#cursor-readout").textContent = `X ${fmt(point.x, 1)} / Y ${fmt(point.y, 1)}`;
});

document.addEventListener("change", event => {
  const node = event.target;
  if (node.id === "stream") {
    const session = view();
    session.stream = node.value;
    if (node.value === "aligned" && session.tool === "mask") session.tool = "pixel";
    session.chart = null;
    session.pixel = null;
    session.stars = null;
    session.points = [];
    render();
  }
  if (node.id === "field-tool" && node.value in definitions) {
    view().fieldTool = node.value;
    render();
  }
  if (node.id === "bundle-in" && node.files.length) {
    const body = new FormData();
    body.set("bundle", node.files[0]);
    operate(async () => {
      const result = await request("/api/series/import", {method: "POST", body});
      await listSeries();
      await selectSeries(result.id, result);
      showMessage("观测包已导入到独立观测。");
    });
    node.value = "";
  }
});

document.addEventListener("submit", event => {
  const {target} = event;
  if (!["curve-form", "quality-form", "stack-form", "field-form"].includes(target.id)) return;
  event.preventDefault();
  const data = Object.fromEntries(new FormData(target));
  operate(() => {
    if (target.id === "field-form") {
      const key = view().fieldTool || "timeline";
      return runTool(key, toolInputs(target, key));
    }
    if (target.id === "stack-form") {
      data.subtract_background = target.elements.subtract_background.checked;
      return stack(data);
    }
    return analysis(target.id === "curve-form" ? "photometry" : "quality", data);
  }, $("button[type=submit]", target));
});

window.addEventListener("resize", () => {
  const epoch = renderEpoch;
  clearTimeout(resizeDebounce);
  resizeDebounce = setTimeout(() => { if (epoch === renderEpoch) repaint(); }, 100);
});

operate(async () => {
  render();
  await listSeries();
  const last = localStorage.getItem("nightseries-observation");
  if (state.list.some(item => item.id === last)) await selectSeries(last);
});
