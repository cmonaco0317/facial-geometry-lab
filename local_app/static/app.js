(() => {
  "use strict";

  const updateCount = (input) => {
    const form = input.closest("form");
    const output = form ? form.querySelector("[data-file-count]") : null;
    if (!output) return;

    const count = input.files.length;
    if (count === 0) {
      output.textContent = input.id === "more-photos"
        ? "No additional photos selected."
        : "No photos selected yet.";
    } else if (count === 1) {
      output.textContent = "1 photo selected.";
    } else {
      output.textContent = `${count} photos selected.`;
    }
  };

  document.querySelectorAll("[data-photo-input]").forEach((input) => {
    input.addEventListener("change", () => updateCount(input));
  });
})();
