/* Minimal Data Ops progressive enhancement */
(function () {
  // Preserve scroll after POSTs is handled by browser; no-op hook for future drawers.
  document.querySelectorAll("[data-do-confirm]").forEach(function (el) {
    el.addEventListener("click", function (e) {
      if (!window.confirm(el.getAttribute("data-do-confirm"))) e.preventDefault();
    });
  });
})();
