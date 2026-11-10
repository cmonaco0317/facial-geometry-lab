(() => {
  "use strict";
  const form = document.getElementById("profile-review-form");
  const image = document.getElementById("profile-source");
  const canvas = document.getElementById("profile-canvas");
  const status = document.getElementById("profile-status");
  if (!form || !image || !canvas || !status) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const fields = [...form.querySelectorAll("[data-anchor]")];
  const anchors = [...form.querySelectorAll('[name="active_anchor"]')];
  const missing = [...form.querySelectorAll("[data-missing]")];
  const width = Number(canvas.dataset.width), height = Number(canvas.dataset.height);
  const zoom = document.getElementById("profile-zoom");
  const level = document.getElementById("profile-level");
  const metric = document.getElementById("profile-metric");
  const field = (k, a) => fields.find(f => f.dataset.anchor === k && f.dataset.coordinate === a);
  const absent = k => missing.find(f => f.dataset.missing === k);
  const read = () => Object.fromEntries(anchors.map(a => {
    const x = field(a.value, "x").valueAsNumber, y = field(a.value, "y").valueAsNumber;
    return [a.value, absent(a.value).checked || !Number.isFinite(x) || !Number.isFinite(y) ? null : [x / 100, y / 100]];
  }));
  const raw = () => Object.fromEntries(anchors.map(a => [a.value, {x:field(a.value, "x").value, y:field(a.value, "y").value, missing:absent(a.value).checked}]));
  const writeRaw = values => anchors.forEach(a => { const v = values[a.value];field(a.value,"x").value=v.x;field(a.value,"y").value=v.y;absent(a.value).checked=v.missing; });
  const empty = () => Object.fromEntries(anchors.map(a => [a.value,{x:"",y:"",missing:true}]));
  const editable = () => JSON.stringify({points:raw(),facing:form.elements.facing.value,level:level.value});
  const initial = editable();
  let undo = [], dirty = form.dataset.draft === "true", transform = null, dragging = null, numericStart = null;
  const changed = () => {
    form.elements.confirmed.checked = false;
    form.elements.lateral_confirmed.checked = false;
    form.elements.acknowledged.checked = false;
    dirty = true;
    status.textContent = "Unsaved profile edits. Preview the readings, then review and confirm before saving.";
  };
  const pushUndo = snapshot => { undo.push(snapshot);if (undo.length > 60) undo.shift(); };
  const remember = () => pushUndo(raw());
  const pair = (points, a, b) => {
    if (!points[a] || !points[b]) return;
    ctx.beginPath();ctx.moveTo(points[a][0] * width, points[a][1] * height);
    ctx.lineTo(points[b][0] * width, points[b][1] * height);ctx.stroke();
  };
  const draw = () => {
    if (!image.complete || !image.naturalWidth) return;
    const angle = Math.max(-45, Math.min(45, Number(level.value) || 0)) * Math.PI / 180;
    const c = Math.cos(angle), s = Math.sin(angle);
    const bw = width * Math.abs(c) + height * Math.abs(s), bh = width * Math.abs(s) + height * Math.abs(c);
    const pixelRatio = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(Math.min(1000, bw) * pixelRatio);
    canvas.height = Math.round(canvas.width * bh / bw);
    canvas.style.width = `${100 * Number(zoom.value)}%`;
    canvas.style.maxWidth = "none";
    const scale = canvas.width / bw;
    const tx = canvas.width / 2, ty = canvas.height / 2;
    transform = {c, s, scale, tx, ty};
    ctx.setTransform(1, 0, 0, 1, 0, 0);ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(scale * c, scale * s, -scale * s, scale * c, tx, ty);
    ctx.translate(-width / 2, -height / 2);
    ctx.drawImage(image, 0, 0, width, height);
    const points = read();
    const groups = {
      convexity: [["nasion", "subnasale"], ["subnasale", "pogonion"]],
      total_convexity: [["nasion", "tip"], ["tip", "pogonion"]],
      upper_eline: [["tip", "pogonion"], ["nasion", "menton"]],
      lower_eline: [["tip", "pogonion"], ["nasion", "menton"]],
      nasal_projection_ratio: [["nasion", "subnasale"], ["nasion", "menton"]],
      nasal_length_ratio: [["nasion", "tip"], ["nasion", "menton"]],
      lower_face_ratio: [["subnasale", "menton"], ["nasion", "menton"]],
      lip_offset_difference: [["tip", "pogonion"], ["nasion", "menton"]]
    };
    ctx.strokeStyle = "#ffe099";ctx.lineWidth = Math.max(width, height) * .003;
    const active = metric.value;
    const lines = active === "all" ? Object.values(groups).flat() : groups[active] || [];
    lines.forEach(([a, b]) => pair(points, a, b));
    const perpendicular = (key, a, b) => {
      if (![key, a, b].every(k => points[k])) return;
      const q = [points[key][0] * width, points[key][1] * height];
      const u = [points[a][0] * width, points[a][1] * height], v = [points[b][0] * width, points[b][1] * height];
      const dx = v[0] - u[0], dy = v[1] - u[1], den = dx * dx + dy * dy;
      if (den < 1e-8) return;
      const t = ((q[0] - u[0]) * dx + (q[1] - u[1]) * dy) / den;
      ctx.save();ctx.setLineDash([width * .01, width * .006]);ctx.beginPath();
      ctx.moveTo(q[0], q[1]);ctx.lineTo(u[0] + t * dx, u[1] + t * dy);ctx.stroke();ctx.restore();
    };
    if (["all", "nasal_projection_ratio"].includes(active)) perpendicular("tip", "nasion", "subnasale");
    if (["all", "upper_eline", "lip_offset_difference"].includes(active)) perpendicular("upper_lip", "tip", "pogonion");
    if (["all", "lower_eline", "lip_offset_difference"].includes(active)) perpendicular("lower_lip", "tip", "pogonion");
    const arc = (a, b, c) => {
      if (![a, b, c].every(k => points[k])) return;
      const u = [points[a][0] * width, points[a][1] * height], v = [points[b][0] * width, points[b][1] * height], q = [points[c][0] * width, points[c][1] * height];
      const radius = Math.min(Math.hypot(u[0] - v[0], u[1] - v[1]), Math.hypot(q[0] - v[0], q[1] - v[1])) * .18;
      if (radius < 1e-8) return;
      const start = Math.atan2(u[1] - v[1], u[0] - v[0]), end = Math.atan2(q[1] - v[1], q[0] - v[0]);
      const delta = ((end - start + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
      ctx.beginPath();ctx.arc(v[0], v[1], radius, start, start + delta, delta < 0);ctx.stroke();
    };
    if (["all", "convexity"].includes(active)) arc("nasion", "subnasale", "pogonion");
    if (["all", "total_convexity"].includes(active)) arc("nasion", "tip", "pogonion");
    const radius = Math.max(width, height) * .009;
    anchors.forEach((a, i) => {
      const p = points[a.value];if (!p) return;
      const x = p[0] * width, y = p[1] * height;
      ctx.beginPath();ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fillStyle = a.checked ? "#ffe099" : "#47e5c2";ctx.fill();
      ctx.strokeStyle = "#122820";ctx.stroke();
      ctx.font = `bold ${Math.max(width, height) * .023}px sans-serif`;ctx.fillStyle = "white";
      ctx.fillText(String(i + 1), x + radius * 1.5, y - radius * 1.5);
    });
    // Screen-horizontal guide; no assumption about anatomical horizontal is stored.
    ctx.setTransform(1, 0, 0, 1, 0, 0);ctx.strokeStyle = "#ffffff88";ctx.lineWidth = pixelRatio;
    ctx.setLineDash([6 * pixelRatio, 6 * pixelRatio]);ctx.beginPath();ctx.moveTo(0, canvas.height / 2);ctx.lineTo(canvas.width, canvas.height / 2);ctx.stroke();ctx.setLineDash([]);
  };
  const locate = event => {
    const r = canvas.getBoundingClientRect(), t = transform;
    const x = ((event.clientX - r.left) * canvas.width / r.width - t.tx) / t.scale;
    const y = ((event.clientY - r.top) * canvas.height / r.height - t.ty) / t.scale;
    const p = [(t.c * x + t.s * y + width / 2) / width, (-t.s * x + t.c * y + height / 2) / height];
    return p.every(v => v >= 0 && v <= 1) ? p : null;
  };
  const place = (name, p) => { if (!p) return;absent(name).checked = false;field(name, "x").value = (p[0] * 100).toFixed(6);field(name, "y").value = (p[1] * 100).toFixed(6);changed();draw(); };
  canvas.addEventListener("pointerdown", event => {
    if (!transform) return;
    const p = locate(event);if (!p) return;
    event.preventDefault();canvas.focus();remember();
    let active = anchors.find(a => a.checked);
    const points = read();
    const candidates = anchors.map(a => ({anchor:a,distance:points[a.value] ? Math.hypot((points[a.value][0]-p[0])*width,(points[a.value][1]-p[1])*height) : Infinity})).filter(v=>v.distance<Math.max(width,height)*.025);
    candidates.sort((a,b)=>a.distance-b.distance || Number(b.anchor.checked)-Number(a.anchor.checked));
    const near = candidates[0]?.anchor;
    if (near) { active = near;near.checked = true; }
    dragging = active.value;canvas.setPointerCapture(event.pointerId);place(dragging, p);
  });
  canvas.addEventListener("pointermove", event => { if (dragging) place(dragging, locate(event)); });
  const finish = () => { dragging = null; };
  canvas.addEventListener("pointerup", finish);canvas.addEventListener("pointercancel", finish);
  canvas.addEventListener("keydown", event => {
    const delta = {ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]}[event.key];
    if (!delta) return;event.preventDefault();const active = anchors.find(a => a.checked).value;
    const p = read()[active] || [.5,.5], step = event.shiftKey ? 10 : 1;remember();
    place(active, [Math.max(0, Math.min(1, p[0] + delta[0] * step / width)), Math.max(0, Math.min(1, p[1] + delta[1] * step / height))]);
  });
  fields.forEach(f => {
    f.addEventListener("focus", () => { numericStart = raw(); });
    f.addEventListener("input", () => { if (numericStart) { pushUndo(numericStart);numericStart = null; } absent(f.dataset.anchor).checked = false;changed();draw(); });
  });
  missing.forEach(f => f.addEventListener("change", () => { const before = raw();before[f.dataset.missing].missing = !f.checked;pushUndo(before);changed();draw(); }));
  anchors.forEach(a => a.addEventListener("change", draw));
  level.addEventListener("input", () => { changed();draw(); });
  form.elements.facing.addEventListener("change", changed);
  zoom.addEventListener("change", draw);metric.addEventListener("change", draw);
  document.getElementById("profile-undo").addEventListener("click", () => { if (!undo.length) return;writeRaw(undo.pop());changed();draw(); });
  document.getElementById("profile-reset").addEventListener("click", () => { remember();writeRaw(empty());changed();draw(); });
  form.addEventListener("submit", () => { dirty = false; });
  window.addEventListener("beforeunload", event => { if (dirty) { event.preventDefault();event.returnValue = ""; } });
  canvas.hidden = true;
  const initialize = () => { image.hidden = true;canvas.hidden = false;draw(); };
  if (image.complete && image.naturalWidth) initialize();else image.addEventListener("load", initialize, {once:true});
  window.addEventListener("pageshow", () => { setTimeout(() => { dirty = form.dataset.draft === "true" || editable() !== initial;form.elements.confirmed.checked = false;form.elements.lateral_confirmed.checked = false;form.elements.acknowledged.checked = false;draw(); }, 0); });
})();
