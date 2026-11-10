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
  const reportView = (parent, record) => {
    const section = el("section", undefined, parent), s = record.snapshot;
    el("h3", record.name, section);
    el("p", `${s.profile} · ${s.score === null ? "No composite" : s.score + " / 100"} · ${s.formula}`, section);
    el("p", `Snapshot ${s.created_at}. Capture ${s.captured_at || "not recorded"}. ${s.protocol}; camera/setup ${s.camera || "not recorded"}.`, section);
    if (record.imported_at) el("p", `Imported ${record.imported_at}. Imported observations are not authenticated measurements.`, section);
    el("p", "Geometry summary only. No validated attractiveness rating or certainty implied by decimal precision.", section);
    table(section, ["Component", "Value", "Weight", "Contribution"], s.components.map((c) => [c.id, c.value, (100*c.effective_weight).toFixed(1) + "%", (c.value*c.effective_weight).toFixed(2)]));
    table(section, ["Measurement", "Reading", "Unit", "Method", "Side / photo"], s.readings.map((r) => [r.label, r.value.toFixed(3), r.unit, r.method, `${r.side} ${r.photo || ""}`]));
    s.ideas.forEach((idea) => el("p", `${idea.title}: ${idea.action}`, section));
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
      button(detail, "Remove saved photos and point evidence", async () => {
        if (!confirm("Remove saved photos and point evidence from this archive? The numerical report stays.")) return;
        await api({operation:"remove_evidence",id}); await refresh(); await show(id); notice("Saved evidence removed. Numerical report retained.");
      });
    }
    detail.scrollIntoView({block:"start"});
  };
  const refresh = async () => {
    const list = document.querySelector("[data-history-list]"); if (!list) return;
    const data = await api({operation:"list"}); list.replaceChildren();
    let bytes = 0;
    data.records.forEach((r) => {
      const row = el("div", undefined, list); row.className = "history-row";
      if (!r.status) {
        const label = el("label", undefined, row), input = el("input", undefined, label); input.type = "checkbox"; input.value = r.id; input.dataset.compareReport = "yes";
        el("span", `${r.name} · ${r.profile} · ${r.score === null ? "No composite" : r.score} · ${r.evidence_count} source photos`, label);
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
      result.photos.forEach((photo)=>{
        const label = el("label",undefined,panel.querySelector("[data-history-photos]")), input = el("input",undefined,label);
        input.type="checkbox";input.value=photo.id;input.dataset.archivePhoto="yes";
        el("span",`Keep source photo ${photo.ordinal} and any saved review points (${photo.width} × ${photo.height})`,label);
        const option=el("option",`Photo ${photo.ordinal}`,panel.querySelector("[data-preview-photos]"));option.value=photo.id;
      });
      panel.querySelector("[data-history-save]").addEventListener("submit",event=>{
        event.preventDefault();const target=event.currentTarget;runOnce(target,async()=>{
          const form=new FormData(target),data=Object.fromEntries(form);
          if (data.captured_at) data.captured_at=new Date(data.captured_at).toISOString();
          const photos=Array.from(panel.querySelectorAll("[data-archive-photo]:checked"),node=>node.value);
          await api({operation:"save",...session,...data,photos});notice(`Saved an encrypted numerical snapshot${photos.length ? " with selected evidence" : " without photos or point geometry"}. Ending this session will keep it.`);
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
      const ids=Array.from(document.querySelectorAll("[data-compare-report]:checked"),node=>node.value);
      if(ids.length!==2)throw new Error("Select exactly two reports to compare.");
      const result=await api({operation:"compare",id:ids[0],other_id:ids[1]}),detail=document.querySelector("[data-history-detail]"),c=result.comparison;
      detail.replaceChildren();el("h2",`Comparison: ${c.status.replaceAll("_"," ")}`,detail);
      el("p",c.score_delta===null ? "Overall delta withheld because evidence or methods differ." : `Geometry-score difference: ${c.score_delta}. Difference observed; significance not established.`,detail);
      if(c.differences.length)el("p","Changed: "+c.differences.join(", "),detail);
      if(c.observed_range)el("p",`Observed range across these two input sets: ${c.observed_range[0]}–${c.observed_range[1]}. This is not a confidence interval.`,detail);
      c.warnings.forEach(w=>el("p",w,detail));
      table(detail,["Measurement","Before","After","Delta","Unit","Method / side"],c.readings.map(r=>[r.label,r.before.toFixed(3),r.after.toFixed(3),r.delta.toFixed(3),r.unit,`${r.method} ${r.side}`]));
      const sides=el("div",undefined,detail);sides.className="history-comparison";reportView(sides,result.before);reportView(sides,result.after);
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
  };
  run(init);
})();
