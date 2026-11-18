(() => {
  "use strict";
  const image = document.getElementById("experiment-original");
  const x = document.getElementById("geometry-target-x");
  const y = document.getElementById("geometry-target-y");
  const status = document.getElementById("geometry-target-status");
  if (!image || !x || !y || !status) return;
  image.addEventListener("click", (event) => {
    const rect = image.getBoundingClientRect();
    if (!(rect.width > 0 && rect.height > 0)) return;
    x.value = String(Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)));
    y.value = String(Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)));
    status.textContent = "Target updated. Save the illustration to compare it with the original.";
  });
})();
