export const $ = (value, root = document) => root.querySelector(value);
export const E = value => String(value ?? "").replace(/[&<>"']/g, part =>
  ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[part]);

export const roles = {target: "目标星", comparison: "比较星", check: "检查星"};
export const assetKinds = {science: "科学曝光", bias: "偏置", dark: "暗场", flat: "平场"};
export const sources = {raw: "原始像素", calibrated: "校准后", aligned: "对准后"};
const reasons = {
  measured: "已测量", retained: "保留原值", unreliable: "峰值存疑", unavailable: "不可用",
  unaligned: "待对准", insufficient_aperture: "孔径覆盖不足", insufficient_sky: "背景环不足",
  saturated: "饱和", nonpositive: "净光度不为正", few_stars: "少星", broad_stars: "星点过宽",
  elongated: "拖尾", masked: "掩膜面积大", no_support: "有效像素不足", manual: "手工", automatic: "自动", reference: "基准",
  local_skew: "分区方向不一致", orientation_unknown: "共同方向证据不足",
  consistent: "与共同方向一致", deviating: "本区偏离",
  masked_zone: "掩膜遮挡", round_stars: "星点近圆形", scatter: "方向散乱",
  round_frames: "全图星点近圆，无共同拖尾方向", masked_frames: "全图星点被掩膜遮挡",
  no_common_direction: "无共同方向", one_zone_only: "仅一区可定方向", zones_disagree: "各区方向不一致",
  angle_shift: "方向偏离", elongation_change: "拖尾程度不同", defined: "已定",
};
export const stateLabel = state => reasons[state] || state?.replace(/^target_/, "目标星：").replace(/^comparison_/, "比较星：").replace(/([：])(.+)$/, (_, a, b) => a + (reasons[b] || b)) || "—";
export const fmt = (value, digits = 2) => value == null || !Number.isFinite(Number(value)) ? "—" : Number(value).toFixed(digits);
export const pct = value => value == null ? "—" : fmt(value * 100, 1) + "%";
export const time = value => value ? E(value.slice(11, 19)) : "—";
export const shortDate = value => value ? E(value.slice(0, 10)) : "—";

export function choose(items, current) {
  return items.map(([value, label]) => `<option value="${E(value)}" ${value === current ? "selected" : ""}>${E(label)}</option>`).join("");
}

export function action(label, id, options = {}) {
  const { style = "", attrs = "", disabled = false } = options;
  return `<button type="button" class="button ${style}" data-action="${id}" ${disabled ? "disabled" : ""} ${attrs}>${E(label)}</button>`;
}

export function panel(title, content, {hint = "", controls = "", cls = "", bare = false} = {}) {
  return `<section class="panel ${cls}"><div class="panel-heading"><div><h2>${E(title)}</h2>
    ${hint ? `<p>${E(hint)}</p>` : ""}</div>${controls}</div>${bare ? content : `<div class="panel-body">${content}</div>`}</section>`;
}

export function field(label, control, wide = false) {
  return `<label class="field ${wide ? "wide" : ""}">${E(label)}${control}</label>`;
}

export function input(label, name, value = "", opts = {}) {
  const { type = "text", step = "any", min = "", max = "", placeholder = "", required = false, wide = false } = opts;
  return field(label, `<input name="${name}" type="${type}" value="${E(value)}" ${required ? "required" : ""} placeholder="${E(placeholder)}"
    ${type === "number" ? `step="${step}" min="${min}" max="${max}"` : ""}>`, wide);
}

export const pick = (label, name, values, current, wide = false) => field(label, `<select name="${name}">${choose(values, current)}</select>`, wide);

export const check = (label, name, value) => `<label class="check"><input type="checkbox" name="${name}" ${value ? "checked" : ""}>${E(label)}</label>`;

export function table(head, rows) {
  return `<div class="table-wrap"><table><thead><tr>${head.map(name => `<th scope="col">${E(name)}</th>`).join("")}</tr></thead><tbody>
    ${rows.map(({ cells, cls = "" }) => `<tr class="${cls}">${cells.map(cell => `<td>${cell}</td>`).join("")}</tr>`).join("")}
    </tbody></table></div>`;
}

export const box = value => `<div class="empty-box">${E(value)}</div>`;
export const badge = (value, cls = "") => `<span class="pill ${cls}">${E(value)}</span>`;
export const status = (value, good) => `<span class="status-text ${good ? "" : "warn"}">${E(value)}</span>`;
export const queries = values => new URLSearchParams(Object.entries(values).filter(([, value]) => value !== null && value !== undefined));

export function apiPath(series, suffix = "") {
  return `/api/series/${series}${suffix}`;
}

export async function request(url, options = {}) {
  const { data, method = "GET", type = "json", body, signal } = options;
  const response = await fetch(url, {
    method, signal,
    headers: data !== undefined ? {"Content-Type": "application/json"} : {},
    body: data !== undefined ? JSON.stringify(data) : body,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    const fail = new Error(error.error || `请求失败（${response.status}）`);
    fail.status = response.status;
    throw fail;
  }
  if (response.status === 204) return null;
  return type === "blob" ? await response.blob() : await response.json();
}

export function promptForm(title, content, handle, submit = "保存") {
  const dialog = $("#editor");
  if (dialog.open) dialog.close();
  dialog.innerHTML = `<div class="panel-heading"><h2 id="dialog-title">${E(title)}</h2>
    <button class="icon" type="button" aria-label="关闭">×</button></div>
    <form class="panel-body"><div class="forms">${content}</div><p class="dialog-error" role="alert"></p>
      <div class="actions"><button type="button" class="button ghost" data-cancel>取消</button>
      <button type="submit" class="button primary">${E(submit)}</button></div></form>`;
  $(".icon", dialog).onclick = () => dialog.close();
  $("[data-cancel]", dialog).onclick = () => dialog.close();
  const form = $("form", dialog);
  form.onsubmit = async event => {
    event.preventDefault();
    const go = $("button[type=submit]", form);
    const messages = $(".dialog-error", form);
    go.disabled = true;
    messages.textContent = "";
    try {
      await handle(Object.fromEntries(new FormData(form)), form);
      dialog.close();
    } catch (err) {
      messages.textContent = err.message;
    } finally {
      go.disabled = false;
    }
  };
  dialog.showModal();
}

let dismiss;
export function showMessage(text, error = false) {
  const node = $("#message");
  node.textContent = text;
  node.dataset.state = error ? "error" : "normal";
  clearTimeout(dismiss);
  dismiss = setTimeout(() => node.textContent = "", error ? 8200 : 4200);
}

export function download(blob, filename) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 3000);
}
