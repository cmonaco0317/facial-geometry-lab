/* Origin-scoped launch capability; no archive key or cross-port cookie. */
(() => {
  "use strict";
  // Bind synchronously: a failed unlock or disabled JS must never put passwords in GET URLs.
  document.querySelectorAll("[data-history-import], [data-history-save], [data-preview-form]").forEach((form) => {
    form.addEventListener("submit", (event) => event.preventDefault());
  });
  const storageKey = "facial-history-launch-v1";
  let auth = null;
  const statuses = () => document.querySelectorAll("[data-history-status]");
  const notice = (message) => statuses().forEach((node) => { node.textContent = message; });
  const el = (tag, text, parent) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = String(text);
    if (parent) parent.append(node);
    return node;
  };
  const button = (parent, label, action) => {
    const node = el("button", label, parent); node.type = "button";
    node.addEventListener("click", () => run(action)); return node;
  };
  const run = async (action) => {
    try { await action(); } catch (error) { notice(error.message || "The local operation failed."); }
  };
  const runOnce = (form, action, message) => {
    if (form.dataset.pending === "yes") return;
    form.dataset.pending = "yes";
    const submit = form.querySelector("button");
    if (submit) submit.disabled = true;
    notice(message);
    run(async () => { try { await action(); } finally { form.dataset.pending = "no"; if (submit) submit.disabled = false; } });
  };
  const api = async (data, binary = false) => {
    if (!auth) throw new Error("History is locked in this tab. Reopen Facial Analysis from the app to unlock it.");
    const response = await fetch(data.operation === "import" ? "/history/import" : "/history/api", {
      method: "POST", credentials: "same-origin", cache: "no-store",
      headers: { "Content-Type": "application/json", "X-Archive-Capability": auth.capability, "X-Archive-CSRF": auth.csrf },
      body: JSON.stringify(data)
    });
    if (!response.ok) throw new Error(response.status === 429 ? "Archive attempts are limited. Wait 15 minutes." :
      "Operation failed. Check the selected report, required fields, password, Keychain access and storage limits. Existing numerical snapshots remain unchanged.");
    return binary ? response.blob() : response.json();
  };
  const download = (blob, name) => {
    const url = URL.createObjectURL(blob), link = el("a");
    link.href = url; link.download = name; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const table = (parent, headings, rows) => {
    const wrap = el("div", undefined, parent); wrap.className = "table-wrap";
    const node = el("table", undefined, wrap), head = el("tr", undefined, el("thead", undefined, node));
    headings.forEach((h) => el("th", h, head)); const body = el("tbody", undefined, node);
    rows.forEach((row) => { const tr = el("tr", undefined, body); row.forEach((v) => el("td", v === null ? "—" : v, tr)); });
  };
  const embeddedGeometry = record => Boolean(record.snapshot.observations?.some(r=>r.sources.some(s=>s.points?.length)));
  const frozenContext = (parent,record) => {
    if(!record.context?.saved_report)return;
    const payload=JSON.parse(record.context.saved_report.report_json),details=el("details",undefined,parent);
    el("summary","Frozen appearance, consultation, questions and illustrations",details);
    el("p","These are the saved report’s own crops and labels. Crop framing and pose can differ between reports; their display is not alignment, calibrated scale or physical progress.",details);
    for(const family of payload.appearance?.families || []) {
      el("h4",family.title,details);
      if(!family.findings.length)el("p",family.missing_reason,details);
      for(const row of family.findings) {
        const figure=el("figure",undefined,details),img=el("img",undefined,figure);
        img.src="data:image/jpeg;base64,"+row.jpeg_b64;img.alt=`Saved photo ${row.photo_number}: ${family.title} crop`;img.loading="lazy";img.className="appearance-image";
        el("figcaption",`Photo ${row.photo_number} · ${row.role} · ${row.visibility} · ${row.method}`,figure);
        el("p",row.note,details);el("p",row.uncertainty,details);
        el("pre",JSON.stringify({method:row.geometry_method,readings:row.measurements,pixel_descriptors:row.pixel_descriptors},null,2),details);
      }
    }
    const review=payload.consultation?.reviewer_record;
    if(review) {el("h4","Reviewer attribution at the saved revision",details);el("pre",JSON.stringify(review,null,2),details);}
    const questions=payload.report_questions;
    el("p",`${questions?.answers?.length || 0} saved answers. Their earlier evidence stays bound to saved revisions; open the full report to inspect its annotations and links.`,details);
    const experiment=payload.photo_experiment?.entry;
    if(experiment)el("p",`Saved ${experiment.kind} illustration · source photo ${experiment.photo_number}. Illustrative only; measured observations and scores remain separate.`,details);
    button(details,"Open this frozen report as HTML",async()=>download(await api({operation:"report",id:record.id,format:"html"},true),"saved-report.html"));
  };
  const reportView = (parent, record) => {
    const section = el("section", undefined, parent), s = record.snapshot;
    el("h3", record.name, section);
    el("p", `${s.profile} · ${s.score === null ? "No composite" : s.score + " / 100"} · ${s.formula}`, section);
    el("p", `Snapshot ${s.created_at}. Capture ${s.captured_at || "not recorded"}. ${s.protocol}; camera/setup ${s.camera || "not recorded"}.`, section);
    if (record.context) {
      const c = record.context.capture;
      el("p", `Report format ${record.context.report_version} · archive context ${record.context.version} · instrument ${s.instrument}.`, section);
      el("p", `User-recorded conditions: pose ${c.pose}; expression ${c.expression}; lighting ${c.lighting}; distance/setup ${c.distance || "not recorded"}. These labels are unverified.`, section);
      el("p", record.context.saved_report ? "Full frozen report retained, including crops, saved question contexts and illustrations." : "Numerical history only. Appearance crops, reviewer/Q&A context and illustrations were not retained.", section);
      if(record.context.workflow)el("p","Editable correction and reviewer journals are retained, including earlier or removed notes. Imported records are unverified; reviewer identity and qualifications are not authenticated.",section);
      if(record.context.sources?.length) {
        table(section,["Source slot","Construction-associated views","Pose estimate: yaw / pitch / roll","Expression/view label and basis","Original frame / source hash"],record.context.sources.map(p=>[
          p.photo_number,p.views.map(v=>v.replaceAll("_"," ")).join(", "),`${p.pose.yaw ?? "unknown"} / ${p.pose.pitch ?? "unknown"} / ${p.pose.roll ?? "unknown"} degrees (uncalibrated)`,`${p.role} · ${p.role_basis.replaceAll("_"," ")}`,`${p.width} × ${p.height} · ${p.source_sha256}`
        ]));
      }
    } else el("p", "Legacy numerical report: detailed capture and full-report context unavailable.", section);
    if (record.imported_at) el("p", `Imported ${record.imported_at}. Imported observations are not authenticated measurements.`, section);
    el("p", "Geometry summary only. No validated attractiveness rating or certainty implied by decimal precision.", section);
    table(section, ["Component", "Value", "Weight", "Contribution"], s.components.map((c) => [c.id, c.value, (100*c.effective_weight).toFixed(1) + "%", (c.value*c.effective_weight).toFixed(2)]));
    table(section, ["Measurement", "Reading", "Unit", "Method", "Side / photo"], s.readings.map((r) => [r.label, r.value.toFixed(3), r.unit, r.method, `${r.side} ${r.photo || ""}`]));
    if (s.observations && s.observations.length) {
      el("h4", "Frozen measurement evidence", section);
      table(section, ["Reading", "Value", "Status", "Method", "Source / revision", "Eligibility or missing reason"], s.observations.map((r) => [r.label, r.value === null ? "Unavailable" : `${r.value.toFixed(r.display_decimals)} ${r.unit}`, r.status, r.method, r.sources.map((p) => `Photo ${p.photo_number}, review ${p.review_revision}`).join("; ") || "No compatible source", r.withheld_reason || r.score_exclusion_reason]));
      el("p", embeddedGeometry(record) ? "This imported or legacy snapshot contains point geometry. Removing evidence below clears those points while preserving numeric readings and updating its checksum." : "This numerical snapshot omits point coordinates. An opted-in full report can separately retain all its source points, crops and earlier contexts. Selected original photos and annotations are stored separately.", section);
    }
    s.ideas.forEach((idea) => el("p", `${idea.title}: ${idea.action}`, section));
    frozenContext(section,record);
    const metadata = el("details", undefined, section); el("summary", "Frozen comparison metadata", metadata);
    el("pre", JSON.stringify({formula_contract:s.formula_contract,reference:s.reference,cohort:s.cohort,fallback:s.fallback,instrument:s.instrument,routes:s.routes,harmony_ids:s.harmony_ids,preferences:s.preferences}, null, 2), metadata);
    if (record.evidence.length) {
      const photos = el("details", undefined, section); el("summary", `Show ${record.evidence.length} saved source photos`, photos);
      el("p", "Uncropped originals, scaled uniformly to this container. No alignment or warping is applied.", photos);
      record.evidence.forEach((item) => { const image = el("img", undefined, photos); image.src = "data:image/jpeg;base64," + item.jpeg_b64; image.alt = `Saved source photo ${item.ordinal}`; image.className = "history-photo"; });
    }
  };
  const show = async (id) => {
    const record = await api({operation:"read",id}), detail = document.querySelector("[data-history-detail]");
    detail.replaceChildren(); reportView(detail, record);
    if (record.context?.saved_report) {
      button(detail, "Restore frozen report as HTML", async () => download(await api({operation:"report",id,format:"html"},true),"saved-report.html"));
      button(detail, "Download exact full-report JSON", async () => download(await api({operation:"report",id,format:"json"},true),"saved-report.json"));
      el("p", "The frozen report retains its original readings, appearance, reviewer status, saved questions and simulated context. It is separate from a new editable analysis. Imported reports remain unverified.", detail);
    }
    const label = el("label", "Report label", detail), input = el("input", undefined, label); input.value = record.name; input.maxLength = 120;
    button(detail, "Rename report", async () => { await api({operation:"rename",id,name:input.value}); await refresh(); notice("Report label saved. Numerical snapshot unchanged."); });
    const passwordLabel = el("label", "Password for portable backup (12+ characters)", detail), password = el("input", undefined, passwordLabel);
    password.type = "password"; password.minLength = 12; password.maxLength = 1024; password.autocomplete = "off";
    button(detail, "Download encrypted backup", async () => {
      const value = password.value; password.value = "";
      if (value.length < 12) throw new Error("Choose a backup password of at least 12 characters.");
      download(await api({operation:"backup",id,password:value}, true), "facial-history.fab"); notice("Encrypted backup downloaded. Keep its password separately.");
    });
    button(detail, "Download readable numerical JSON", async () => {
      const {input_digest, ...readable} = record.snapshot;
      download(new Blob([JSON.stringify(readable,null,2)],{type:"application/json"}),"facial-history-readable.json");
    });
    if (record.evidence.length) {
      button(detail, "Restore evidence into a new temporary session", async () => { const result = await api({operation:"restore",id}); location.assign(result.location); });
    }
    if (record.evidence.length || record.context?.saved_report || embeddedGeometry(record)) {
      button(detail, "Remove photos, point evidence and full-report context", async () => {
        if (!confirm("Remove saved photos, point evidence and full-report context including earlier crops, questions and illustrations? Numeric readings stay. Point-bearing legacy snapshots receive an updated checksum after their coordinates are removed.")) return;
        await api({operation:"remove_evidence",id}); await refresh(); await show(id); notice("Photos, point evidence and full-report context removed. Numerical snapshot retained.");
      });
    }
    detail.scrollIntoView({block:"start"});
  };
  const refresh = async () => {
    const list = document.querySelector("[data-history-list]"); if (!list) return;
    const data = await api({operation:"list"}); list.replaceChildren();
    const before = document.querySelector("[data-compare-before]"), after = document.querySelector("[data-compare-after]");
    for(const select of [before,after]) {select.replaceChildren();const option=el("option","Choose a report",select);option.value="";}
    let bytes = 0;
    data.records.forEach((r) => {
      const row = el("div", undefined, list); row.className = "history-row"; row.dataset.archiveId = r.id;
      if (!r.status) {
        el("p", `${r.name} · ${r.profile} · capture ${r.captured_at || "not recorded"} · saved ${r.created_at} · ${r.score === null ? "No composite" : r.score} · ${r.evidence_count} source photos${r.context_saved ? " · full report retained" : " · numerical only"}`, row);
        for(const select of [before,after]) {const option=el("option",`${r.name} · capture ${r.captured_at || "not recorded"} · saved ${r.created_at}`,select);option.value=r.id;}
        button(row,"Open",()=>show(r.id)); bytes += r.bytes;
      } else el("p", "Locked, missing-key or damaged archive. Files were preserved.", row);
      button(row,"Delete this archive",async()=>{
        if (!confirm("Delete this saved archive and its evidence from this app? Separate backups and exports remain.")) return;
        await api({operation:"delete",id:r.id}); await refresh(); document.querySelector("[data-history-detail]").replaceChildren(); notice("Archive deleted from this app.");
      });
    });
    notice(`${data.records.length} saved reports · ${(bytes/1000000).toFixed(2)} MB readable archives. ${data.records.length ? "" : "Nothing saved yet."}`);
  };
  const init = async () => {
    const match = /^#history=([A-Za-z0-9_-]{40,96})$/.exec(location.hash);
    if (match) {
      history.replaceState(null,"",location.pathname + location.search);
      const response = await fetch("/history/auth",{method:"POST",credentials:"same-origin",headers:{"Content-Type":"application/json"},body:JSON.stringify({capability:match[1]})});
      if (!response.ok) throw new Error("History unlock failed. Reopen the app after the attempt limit clears.");
      auth = await response.json(); sessionStorage.setItem(storageKey,JSON.stringify(auth));
    } else {
      try { auth = JSON.parse(sessionStorage.getItem(storageKey)); } catch { auth = null; }
    }
    if (!auth) { notice("History is locked in this tab. Open Facial Analysis from the app to enable it. Temporary analysis remains available."); return; }
    notice("Private history available. Nothing is saved automatically."); await refresh();
    const panel = document.querySelector("[data-history-session]");
    const session = panel ? {session:panel.dataset.historySession,session_csrf:panel.dataset.historySessionCsrf} : null;
    if (panel) {
      const result = await api({operation:"photos",...session});
      if (result.restored_archive) {
        el("p", "This is a new analysis from saved evidence. Points require fresh review; the original report remains in history.", panel);
        const link = el("a", "Open the original saved report", panel);
        link.href = "/history?report=" + encodeURIComponent(result.restored_archive.id);
        el("p",result.restored_archive.workflow_available ? "Saved appearance corrections and reviewer revision history were restored. Only selected byte-matched source photos are active. Earlier questions and illustrations keep their saved contexts; old reviewer attribution does not establish review of this new analysis." : "This archive has no editable correction/reviewer journal. Its retained full report, if any, remains available separately.",panel);
      }
      result.photos.forEach((photo)=>{
        const label = el("label",undefined,panel.querySelector("[data-history-photos]")), input = el("input",undefined,label);
        input.type="checkbox";input.value=photo.id;input.dataset.archivePhoto="yes";
        el("span",`Keep source photo ${photo.ordinal} and any saved review points (${photo.width} × ${photo.height})`,label);
        const option=el("option",`Photo ${photo.ordinal}`,panel.querySelector("[data-preview-photos]"));option.value=photo.id;
      });
      panel.querySelector("[data-history-save]").addEventListener("submit",event=>{
        event.preventDefault();const target=event.currentTarget;runOnce(target,async()=>{
          const form=new FormData(target),data=Object.fromEntries(form);
          data.capture={};
          for(const key of ["pose","expression","lighting","distance"]) {data.capture[key]=data["capture_"+key];delete data["capture_"+key];}
          data.keep_context=form.has("keep_context");
          if (data.captured_at) data.captured_at=new Date(data.captured_at).toISOString();
          const photos=Array.from(panel.querySelectorAll("[data-archive-photo]:checked"),node=>node.value);
          await api({operation:"save",...session,...data,photos});notice(`Saved an encrypted numerical snapshot${photos.length ? " with selected source photos" : " without original source photos"}${data.keep_context ? " and a full frozen report containing crops and saved contexts" : "; full report context omitted"}. Ending this session will keep it.`);
        }, "Saving encrypted snapshot…");
      });
      let previewURL=null;
      panel.querySelector("[data-preview-form]").addEventListener("submit",event=>{
        event.preventDefault();run(async()=>{
          const form=new FormData(event.currentTarget),data=Object.fromEntries(form);data.original=Boolean(data.original);
          for (const key of ["x","y","scale"]) data[key]=Number(data[key]);
          const blob=await api({operation:"preview",...session,...data},true),output=panel.querySelector("[data-preview-output]");
          if(previewURL)URL.revokeObjectURL(previewURL);previewURL=URL.createObjectURL(blob);output.replaceChildren();
          const image=el("img",undefined,output);image.src=previewURL;image.alt="Local illustrative overlay, not a measurement or prediction";image.className="preview-image";
          button(output,"Download labeled illustration",async()=>download(blob,"illustrative-preview.svg"));notice("Illustration rendered locally. Original source photo unchanged.");
        });
      });
    }
    document.querySelectorAll("[data-history-controls]").forEach((fieldset)=>{fieldset.disabled=false;});
    document.querySelector("[data-history-refresh]")?.addEventListener("click",()=>run(refresh));
    document.querySelector("[data-history-compare]")?.addEventListener("click",()=>run(async()=>{
      const ids=[document.querySelector("[data-compare-before]").value,document.querySelector("[data-compare-after]").value];
      if(ids.some(id=>!id)||ids[0]===ids[1])throw new Error("Choose distinct before and after reports. Capture dates are shown in each choice.");
      const result=await api({operation:"compare",id:ids[0],other_id:ids[1]}),detail=document.querySelector("[data-history-detail]"),c=result.comparison;
      detail.replaceChildren();el("h2",`Comparison: ${c.status.replaceAll("_"," ")}`,detail);
      button(detail,"Download comparison and retained evidence JSON",async()=>download(new Blob([JSON.stringify(result,null,2)],{type:"application/json"}),"facial-history-comparison.json"));
      el("p","The comparison download includes retained source photos, crops, point geometry and earlier saved notes when present. Share only what you intend.",detail);
      el("p",c.score_delta===null ? "Overall delta withheld because evidence or methods differ." : `Geometry-score difference: ${c.score_delta}. Difference observed; significance not established.`,detail);
      if(c.differences.length)el("p","Changed: "+c.differences.join(", "),detail);
      if(c.observed_range)el("p",`Observed range across these two input sets: ${c.observed_range[0]}–${c.observed_range[1]}. This is not a confidence interval.`,detail);
      c.warnings.forEach(w=>el("p",w,detail));
      el("p", "Arithmetic differences subtract stored values from the same construction and method, including when capture conditions differ. They do not establish matched captures. Empirical repeat-capture limits are unavailable; no change is interpreted as physical progress, including small or zero changes. Source slots do not establish same-person identity.", detail);
      const value = n => n === null || n === undefined ? "Unavailable" : String(n);
      if(c.observations?.length) {
        table(detail,["Construction","Before","After","Arithmetic difference","Before / after method and status","Why progress cannot be interpreted"],c.observations.map(r=>[
          r.label,`${value(r.before)} ${r.before_unit || ""}`,`${value(r.after)} ${r.after_unit || ""}`,value(r.delta),
          `${r.before_method || "Absent"} (${r.before_status || "absent"}) → ${r.after_method || "Absent"} (${r.after_status || "absent"})`,r.reasons.join(" ")
        ]));
        c.observations.forEach(r=>{
          const d=el("details",undefined,detail);el("summary",r.label+" · source and provenance",d);
          el("p",r.before_provenance+" → "+r.after_provenance,d);
          el("p",[r.before_missing,r.after_missing].filter(Boolean).join(" "),d);
          el("pre",JSON.stringify({before:r.before_sources,after:r.after_sources,before_capture:r.before_capture,after_capture:r.after_capture},null,2),d);
        });
      } else el("p","Legacy reports contain no frozen observation inventory. Detailed source/method comparison is unavailable.",detail);
      if(c.readings.length)table(detail,["Legacy measurement","Before","After","Arithmetic difference","Unit","Method / side"],c.readings.map(r=>[r.label,value(r.before),value(r.after),value(r.delta),r.unit,`${r.method} ${r.side}`]));
      const sides=el("div",undefined,detail);sides.className="history-comparison";reportView(sides,result.before);reportView(sides,result.after);
      detail.scrollIntoView({block:"start"});
    }));
    document.querySelector("[data-history-import]")?.addEventListener("submit",event=>{
      event.preventDefault();run(async()=>{
        const form=event.currentTarget,file=form.elements.backup.files[0],password=form.elements.password.value;form.elements.password.value="";
        if(!file || file.size>6000016)throw new Error("Choose a supported backup of at most 6 MB.");
        const bytes=new Uint8Array(await file.arrayBuffer());let binary="";
        for(let offset=0;offset<bytes.length;offset+=32768)binary+=String.fromCharCode(...bytes.subarray(offset,offset+32768));
        await api({operation:"import",backup:btoa(binary),password});await refresh();notice("Backup imported as a separate encrypted archive. Observations remain unverified.");
      });
    });
    const selected = new URLSearchParams(location.search).get("report");
    if(selected && document.querySelector("[data-history-detail]"))await show(selected);
  };
  run(init);
})();
