(() => {
  "use strict";
  const form=document.getElementById("frontal-review-form");
  if (!form) return;
  const image=document.getElementById("frontal-source"), canvas=document.getElementById("frontal-canvas"), status=document.getElementById("frontal-status");
  const ctx=canvas.getContext("2d");
  if (!ctx) return;
  const inputs=[...form.querySelectorAll("[data-anchor]")], anchors=[...form.querySelectorAll('[name="active_anchor"]')];
  const visibility=[...form.querySelectorAll("[data-missing]")], regions=[...form.querySelectorAll(".anchor-region")];
  const field=(key,axis)=>inputs.find(f=>f.dataset.anchor===key&&f.dataset.coordinate===axis);
  const missing=key=>visibility.find(f=>f.dataset.missing===key);
  const region=document.getElementById("frontal-region");
  const forehead=document.getElementById("forehead-visibility");
  const blocked=()=>new Set(forehead.value==="forehead_hidden"||(forehead.value==="unassessed"&&forehead.dataset.automaticStatus&&forehead.dataset.automaticStatus!=="forehead_clear")?["hairline","mesh_top","forehead_a","forehead_b"]:forehead.value==="visible"?[]:["hairline"]);
  const syncForehead=()=>{
    const hidden=blocked();
    anchors.forEach(a=>{
      const unavailable=hidden.has(a.value);
      ["x","y"].forEach(axis=>{field(a.value,axis).disabled=unavailable;});
      a.disabled=unavailable;
      missing(a.value).disabled=false;
      if(unavailable)missing(a.value).checked=true;
    });
  };
  let mirror=false, original=false, undo=[], dragging=false;
  const snapshot=()=>({values:inputs.map(f=>f.value),hidden:visibility.map(f=>f.checked),forehead:forehead.value});
  const remember=()=>{undo.push(snapshot());if(undo.length>60)undo.shift();};
  const edited=()=>{syncForehead();forehead.dataset.before=forehead.value;form.elements.confirmed.checked=false;form.elements.acknowledged.checked=false;original=false;rememberView();document.getElementById("frontal-original").setAttribute("aria-pressed","false");status.textContent="Draft changed. Calculate draft readings, then review and save.";draw();};
  const draw=()=>{
    if(!image.naturalWidth)return;
    ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,canvas.width,canvas.height);
    if(mirror){ctx.translate(canvas.width,0);ctx.scale(-1,1);}
    ctx.drawImage(image,0,0,canvas.width,canvas.height);ctx.setTransform(1,0,0,1,0,0);
    anchors.forEach((a,i)=>{
      if(blocked().has(a.value))return;
      const row=a.closest("fieldset");
      if(region.value!=="all"&&row.dataset.region!==region.value)return;
      const fx=field(a.value,"x"),fy=field(a.value,"y");
      const x=original?(fx.dataset.original===""?NaN:Number(fx.dataset.original)):fx.valueAsNumber;
      const y=original?(fy.dataset.original===""?NaN:Number(fy.dataset.original)):fy.valueAsNumber;
      if((!original&&missing(a.value).checked)||!Number.isFinite(x)||!Number.isFinite(y))return;
      const px=(mirror?100-x:x)*canvas.width/100,py=y*canvas.height/100;
      ctx.beginPath();ctx.arc(px,py,a.checked?6:4,0,Math.PI*2);ctx.fillStyle=a.checked?"#ffe099":"#47e5c2";ctx.fill();ctx.strokeStyle="#122820";ctx.lineWidth=2;ctx.stroke();
      ctx.font="bold 14px sans-serif";ctx.strokeText(String(i+1),px+8,py-7);ctx.fillStyle="#fff";ctx.fillText(String(i+1),px+8,py-7);
    });
  };
  const rememberView=()=>{
    const state=(history.state&&typeof history.state==="object")?history.state:{};
    history.replaceState({...state,frontalView:{mirror,original}},"");
  };
  const syncView=()=>{
    syncForehead();
    forehead.dataset.before=forehead.value;
    regions.forEach(r=>{r.hidden=region.value!=="all"&&r.dataset.region!==region.value;});
    const zoom=Number(document.getElementById("frontal-zoom").value);
    canvas.style.width=`${Number.isFinite(zoom)&&zoom>=1&&zoom<=3?zoom*100:100}%`;canvas.style.maxWidth="none";
    const view=history.state?.frontalView;
    mirror=view?.mirror===true;original=view?.original===true;
    document.getElementById("frontal-mirror").setAttribute("aria-pressed",String(mirror));
    document.getElementById("frontal-original").setAttribute("aria-pressed",String(original));
    form.elements.confirmed.checked=false;form.elements.acknowledged.checked=false;
    draw();
  };
  // Browsers may restore native form values after script initialization.
  // Reconcile the drawing and controls on the next task, for both bfcache and reload Back.
  window.addEventListener("pageshow",()=>{setTimeout(syncView,0);});
  const initialize=()=>{canvas.width=Math.min(1200,image.naturalWidth);canvas.height=Math.round(canvas.width*image.naturalHeight/image.naturalWidth);image.hidden=true;canvas.hidden=false;syncView();};
  canvas.hidden=true;
  if(image.complete&&image.naturalWidth)initialize();else image.addEventListener("load",initialize,{once:true});
  const place=event=>{
    const active=anchors.find(a=>a.checked);if(!active||original||blocked().has(active.value))return;
    const rect=canvas.getBoundingClientRect();let x=Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width));
    if(mirror)x=1-x;
    field(active.value,"x").value=(x*100).toFixed(8);field(active.value,"y").value=(Math.max(0,Math.min(1,(event.clientY-rect.top)/rect.height))*100).toFixed(8);
    missing(active.value).checked=false;edited();
  };
  canvas.addEventListener("pointerdown",e=>{if(original)return;remember();dragging=true;canvas.setPointerCapture(e.pointerId);place(e);});
  canvas.addEventListener("pointermove",e=>{if(dragging)place(e);});
  canvas.addEventListener("pointerup",()=>{dragging=false;});canvas.addEventListener("pointercancel",()=>{dragging=false;});
  canvas.addEventListener("keydown",e=>{
    if(!["ArrowLeft","ArrowRight","ArrowUp","ArrowDown"].includes(e.key)||original)return;
    e.preventDefault();const active=anchors.find(a=>a.checked);if(!active||missing(active.value).checked)return;
    remember();const axis=e.key==="ArrowLeft"||e.key==="ArrowRight"?"x":"y",f=field(active.value,axis);
    let step=(e.key==="ArrowLeft"||e.key==="ArrowUp"?-1:1)*(e.shiftKey?1:.1);if(axis==="x"&&mirror)step=-step;
    f.value=Math.max(0,Math.min(100,f.valueAsNumber+step)).toFixed(8);edited();
  });
  inputs.forEach(f=>{f.addEventListener("focus",()=>{f.dataset.before=f.value;});f.addEventListener("input",()=>{remember();const last=undo[undo.length-1];last.values[inputs.indexOf(f)]=f.dataset.before??f.value;f.dataset.before=f.value;edited();});});
  visibility.forEach(f=>f.addEventListener("change",()=>{const before=snapshot();before.hidden[visibility.indexOf(f)]=!f.checked;undo.push(before);edited();}));
  forehead.addEventListener("change",()=>{const before=snapshot();before.forehead=forehead.dataset.before??"unassessed";undo.push(before);forehead.dataset.before=forehead.value;edited();});
  forehead.addEventListener("focus",()=>{forehead.dataset.before=forehead.value;});
  anchors.forEach(a=>a.addEventListener("change",draw));
  region.addEventListener("change",()=>{regions.forEach(r=>{r.hidden=region.value!=="all"&&r.dataset.region!==region.value;});const next=anchors.find(a=>region.value==="all"||a.closest("fieldset").dataset.region===region.value);if(next)next.checked=true;draw();});
  document.getElementById("frontal-zoom").addEventListener("change",e=>{canvas.style.width=`${Number(e.target.value)*100}%`;canvas.style.maxWidth="none";});
  document.getElementById("frontal-mirror").addEventListener("click",e=>{mirror=!mirror;e.currentTarget.setAttribute("aria-pressed",String(mirror));rememberView();draw();});
  document.getElementById("frontal-original").addEventListener("click",e=>{original=!original;e.currentTarget.setAttribute("aria-pressed",String(original));rememberView();status.textContent=original?"Showing immutable automatic points. Toggle back to edit.":"Showing current draft points.";draw();});
  const reset=(keys,resetVisibility=false)=>{remember();if(resetVisibility)forehead.value="unassessed";keys.forEach(key=>{["x","y"].forEach(axis=>{field(key,axis).value=field(key,axis).dataset.original;});missing(key).checked=field(key,"x").value==="";});edited();};
  form.querySelectorAll("[data-reset]").forEach(b=>b.addEventListener("click",()=>reset([b.dataset.reset])));
  document.getElementById("frontal-reset").addEventListener("click",()=>reset(anchors.map(a=>a.value),true));
  document.getElementById("frontal-undo").addEventListener("click",()=>{const old=undo.pop();if(!old)return;inputs.forEach((f,i)=>{f.value=old.values[i];});visibility.forEach((f,i)=>{f.checked=old.hidden[i];});forehead.value=old.forehead;edited();});
})();
