/* Minimal Data Ops progressive enhancement */

(function () {

  document.querySelectorAll("[data-do-confirm]").forEach(function (el) {

    el.addEventListener("click", function (e) {

      if (!window.confirm(el.getAttribute("data-do-confirm"))) e.preventDefault();

    });

  });



  function wireTableFilters(table, opts) {

    if (!table) return;

    var dataRows = table.querySelectorAll(opts.dataRowSelector);

    var impactRows = opts.impactRowSelector

      ? table.querySelectorAll(opts.impactRowSelector)

      : [];

    var search = opts.searchInput;

    var attention = opts.attentionCheckbox;



    function rowMatches(row) {

      var q = (search && search.value || "").trim().toLowerCase();

      var onlyAttention = attention && attention.checked;

      var text = (row.textContent || "").trim().toLowerCase();

      var matchQ = !q || text.indexOf(q) !== -1;

      var matchA = !onlyAttention || row.getAttribute("data-needs-attention") === "1";

      return matchQ && matchA;

    }



    function apply() {

      var visible = {};

      dataRows.forEach(function (row) {

        var show = rowMatches(row);

        row.hidden = !show;

        var id = row.getAttribute("data-dataset-id") || row.getAttribute("data-source-id");

        if (id) visible[id] = show;

      });

      impactRows.forEach(function (row) {

        var id = row.getAttribute("data-dataset-id");

        row.hidden = !visible[id];

      });

    }



    if (search) search.addEventListener("input", apply);

    if (attention) attention.addEventListener("change", apply);

    apply();

  }



  wireTableFilters(document.querySelector("[data-do-release-table]"), {

    dataRowSelector: "tr[data-do-release-row]",

    impactRowSelector: "tr[data-do-release-impact]",

    searchInput: document.querySelector("[data-do-release-search]"),

    attentionCheckbox: document.querySelector("[data-do-needs-attention-only]"),

  });



  wireTableFilters(document.querySelector("[data-do-diag-table]"), {

    dataRowSelector: "tr[data-do-diag-row]",

    searchInput: document.querySelector("[data-do-diag-search]"),

    attentionCheckbox: document.querySelector("[data-do-diag-needs-attention-only]"),

  });



  function wireBusyForms(root) {

    var scope = root || document;

    scope.querySelectorAll("form[data-do-busy-form]").forEach(function (form) {

      if (form.getAttribute("data-do-busy-wired") === "1") return;

      form.setAttribute("data-do-busy-wired", "1");

      form.addEventListener("submit", function (e) {

        if (form.getAttribute("data-do-busy-active") === "1") {

          e.preventDefault();

          return;

        }

        var btn = form.querySelector("[data-do-busy-submit]");

        if (!btn) return;

        form.setAttribute("data-do-busy-active", "1");

        form.classList.add("is-busy");

        form.setAttribute("aria-busy", "true");

        btn.disabled = true;

        btn.setAttribute("aria-disabled", "true");

        var label = form.getAttribute("data-do-busy-label") || "Working…";

        var detail = form.getAttribute("data-do-busy-detail") || label;

        btn.textContent = label;

        var status = form.querySelector("[data-do-busy-status]");

        if (status) {

          status.hidden = false;

          status.textContent = detail;

        }

      });

    });

  }



  wireBusyForms(document);



  var modal = document.getElementById("do-source-modal");

  var modalBody = modal && modal.querySelector("[data-do-modal-body]");

  var modalPanel = modal && modal.querySelector(".do-modal-panel");

  var lastFocus = null;



  function closeModal() {

    if (!modal) return;

    modal.hidden = true;

    document.body.classList.remove("do-modal-open");

    if (modalBody) modalBody.innerHTML = '<p class="do-muted do-small">Loading…</p>';

    if (lastFocus && lastFocus.focus) lastFocus.focus();

  }



  function openModal(url) {

    if (!modal || !modalBody) {

      window.location.href = url.replace("/panel", "");

      return;

    }

    lastFocus = document.activeElement;

    modal.hidden = false;

    document.body.classList.add("do-modal-open");

    modalBody.innerHTML = '<p class="do-muted do-small">Loading…</p>';

    fetch(url, { credentials: "same-origin", headers: { Accept: "text/html" } })

      .then(function (resp) {

        if (!resp.ok) throw new Error("load failed");

        return resp.text();

      })

      .then(function (html) {

        modalBody.innerHTML = html;

        wireBusyForms(modalBody);

        var closeBtn = modalBody.querySelector("[data-do-modal-close]");

        if (closeBtn) closeBtn.focus();

      })

      .catch(function () {

        modalBody.innerHTML = '<p class="do-flash-error">Could not load dataset detail.</p>';

      });

    if (modalPanel) modalPanel.focus();

  }



  if (modal) {

    modal.addEventListener("click", function (e) {

      if (e.target.closest("[data-do-modal-close]")) closeModal();

    });

    document.addEventListener("keydown", function (e) {

      if (modal.hidden) return;

      if (e.key === "Escape") closeModal();

    });

  }



  document.querySelectorAll("[data-do-source-panel]").forEach(function (btn) {

    btn.addEventListener("click", function () {

      openModal(btn.getAttribute("data-do-source-panel"));

    });

  });

})();

