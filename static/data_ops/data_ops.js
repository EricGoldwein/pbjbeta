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

        var submitted = e.submitter || null;

        var btn = (submitted && submitted.hasAttribute && submitted.hasAttribute("data-do-busy-submit"))

          ? submitted

          : form.querySelector("[data-do-busy-submit]");

        if (!btn) return;

        if (submitted && submitted.name) {

          var keep = form.querySelector("input[data-do-submitter-keep]");

          if (!keep) {

            keep = document.createElement("input");

            keep.type = "hidden";

            keep.setAttribute("data-do-submitter-keep", "1");

            form.appendChild(keep);

          }

          keep.name = submitted.name;

          keep.value = submitted.value;

        }

        form.setAttribute("data-do-busy-active", "1");

        form.classList.add("is-busy");

        form.setAttribute("aria-busy", "true");

        btn.disabled = true;

        btn.setAttribute("aria-disabled", "true");

        var label = (btn.getAttribute && btn.getAttribute("data-do-busy-label")) || form.getAttribute("data-do-busy-label") || "Working…";

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



  function wireDashboardRunForm(form) {

    if (!form) return;

    var wrap = form.querySelector("[data-do-password-wrap]");

    var password = form.querySelector("#dashboard_password");
    var publishPassword = form.querySelector("[data-do-publish-password]");
    var publishWrap = form.querySelector("[data-do-publish-password-wrap]");
    var PW_STORE = "doDashFacilityPassword";

    function readStoredPassword() {
      try {
        var raw = window.sessionStorage.getItem(PW_STORE);
        if (!raw) return "";
        var saved = JSON.parse(raw);
        var ccnEl = form.querySelector('input[name="ccn"]');
        var ccn = (ccnEl && ccnEl.value) || "";
        if (!saved || saved.ccn !== ccn) return "";
        return saved.password || "";
      } catch (err) { return ""; }
    }

    function storePassword(value) {
      try {
        var ccnEl = form.querySelector('input[name="ccn"]');
        var ccn = (ccnEl && ccnEl.value) || "";
        if (!ccn || !value) return;
        window.sessionStorage.setItem(PW_STORE, JSON.stringify({ ccn: ccn, password: value }));
      } catch (err) {}
    }

    function syncPasswordFields(fromPublish) {
      if (!password || !publishPassword) return;
      if (fromPublish) {
        if (publishPassword.value && !password.value) password.value = publishPassword.value;
      } else {
        if (password.value && !publishPassword.value) publishPassword.value = password.value;
      }
    }

    function fillFromSession() {
      var stored = readStoredPassword();
      if (!stored) return;
      if (password && !password.value) password.value = stored;
      if (publishPassword && !publishPassword.value) publishPassword.value = stored;
    }

    function syncAccess() {

      var open = form.querySelector('input[name="access_mode"]:checked');

      var isOpen = open && open.value === "open";

      if (wrap) wrap.hidden = !!isOpen;

      if (password) {

        password.disabled = !!isOpen;

        password.required = !isOpen;

      }
      if (publishWrap) publishWrap.hidden = !!isOpen;
      if (publishPassword) {
        publishPassword.disabled = !!isOpen;
      }

    }

    form.querySelectorAll("[data-do-access-mode]").forEach(function (el) {

      el.addEventListener("change", syncAccess);

    });
    if (password) {
      password.addEventListener("input", function () {
        syncPasswordFields(false);
        if (password.value) storePassword(password.value);
      });
    }
    if (publishPassword) {
      publishPassword.addEventListener("input", function () {
        syncPasswordFields(true);
        if (publishPassword.value) storePassword(publishPassword.value);
      });
    }

    form.addEventListener("click", function (e) {

      var btn = e.target.closest("[name=intent]");

      if (!btn || !btn.value) return;

      var confirmInput = form.querySelector("#confirm_ccn");

      var confirmBox = form.querySelector('input[name="confirm_publish"]');

      var needsPublish = btn.value.indexOf("deploy_") === 0;

      if (confirmInput) confirmInput.required = needsPublish;

      if (confirmBox) confirmBox.required = needsPublish;
      syncPasswordFields(needsPublish);

    });

    fillFromSession();
    syncAccess();

    form.addEventListener("submit", function (e) {

      if (!window.fetch) return;

      e.preventDefault();

      e.stopImmediatePropagation();

      var submitted = e.submitter;

      var intent = (submitted && submitted.value) || "";
      var isDeploy = intent.indexOf("deploy_") === 0;
      syncPasswordFields(isDeploy);
      var effectivePw = "";
      if (isDeploy && publishPassword && publishPassword.value) effectivePw = publishPassword.value;
      else if (password && password.value) effectivePw = password.value;
      else if (publishPassword && publishPassword.value) effectivePw = publishPassword.value;
      if (effectivePw) storePassword(effectivePw);

      var body = new FormData(form);

      if (submitted && submitted.name) body.set(submitted.name, submitted.value);
      if (effectivePw) body.set("dashboard_password", effectivePw);

      var pwField = form.querySelector("#dashboard_password");
      if (pwField) pwField.classList.remove("is-invalid");
      if (publishPassword) publishPassword.classList.remove("is-invalid");

      var panel = document.getElementById("do-builder-progress");

      if (panel) panel.hidden = false;

      progressUserHidden = false;

      form.classList.add("is-busy");

      form.setAttribute("aria-busy", "true");

      var prior = document.querySelector("[data-do-builder-result]");
      if (prior) prior.hidden = true;
      var dismiss = document.querySelector("[data-do-progress-dismiss]");
      if (dismiss) dismiss.hidden = false;

      fetch(form.action, {

        method: "POST",

        body: body,

        credentials: "same-origin",

        headers: { "X-Requested-With": "XMLHttpRequest", "Accept": "application/json" },

      }).then(function (res) {

        return res.json().then(function (data) {

          if (!res.ok || !data.job_id) {

            var blocked = data.detail || (data.errors || []).join("; ") || "Could not start";

            paintProgress({ percent: 0, label: "Blocked", detail: blocked, state: "error", steps: [] });

            showBuilderResult({ state: "error", detail: blocked, intent: body.get("intent"), ccn: body.get("ccn") });

            clearBuilderRunBusy();

            if (/password/i.test(blocked)) {
              var intentVal = (body.get("intent") || "");
              var isDep = String(intentVal).indexOf("deploy_") === 0;
              var pwPub = form.querySelector("[data-do-publish-password]");
              var pw = form.querySelector("#dashboard_password");
              var target = (isDep && pwPub) ? pwPub : (pw || pwPub);
              if (target) {
                target.classList.add("is-invalid");
                target.focus();
                try { target.scrollIntoView({ block: "center", behavior: "smooth" }); } catch (err) {}
              }
            }

            return;

          }

          paintProgress(data);

          pollJob(data.job_id);

        });

      }).catch(function () {

        paintProgress({ percent: 0, label: "Blocked", detail: "Could not start the build", state: "error", steps: [] });
        clearBuilderRunBusy();

      });

    }, true);

  }

  function clearBuilderRunBusy() {
    var form = document.querySelector("[data-do-dash-run]");
    if (!form) return;
    form.classList.remove("is-busy");
    form.classList.remove("is-progressing");
    form.removeAttribute("aria-busy");
    form.removeAttribute("data-do-busy-active");
  }

  var progressUserHidden = false;

  function paintProgress(data) {

    var panel = document.getElementById("do-builder-progress");

    if (!panel) return;

    if (progressUserHidden && data.state === "running") return;

    panel.hidden = false;

    var pct = document.querySelector("[data-do-progress-pct]");

    var fill = document.querySelector("[data-do-progress-fill]");

    var label = document.querySelector("[data-do-progress-label]");

    var detail = document.querySelector("[data-do-progress-detail]");

    var list = document.querySelector("[data-do-progress-steps]");
    var dismiss = document.querySelector("[data-do-progress-dismiss]");

    var n = typeof data.percent === "number" ? data.percent : 0;

    if (pct) pct.textContent = String(n);

    if (fill) fill.style.width = n + "%";

    if (label) label.textContent = data.label || "";

    if (detail) {
      var failedNote = "";
      (data.steps || []).forEach(function (step) {
        if (step.state === "failed" && step.note && step.note !== "Failed") failedNote = step.note;
      });
      detail.textContent = failedNote || data.detail || "";
    }
    if (panel) panel.classList.toggle("is-failed", data.state === "error" || data.ok === false);
    if (dismiss) dismiss.hidden = data.state === "running";

    if (data.state === "error" || data.ok === false) {
      var allWaiting = !(data.steps || []).some(function (step) {
        return step.state && step.state !== "pending";
      });
      if (allWaiting) data.steps = [];
    }

    if (list) {
      list.innerHTML = (data.steps || []).map(function (step, i) {
        var st = step.state || "pending";
        var note = step.note || (st === "done" ? "Done" : st === "active" ? "Now" : st === "failed" ? "Failed" : "Waiting");
        return '<li class="is-' + st + '">'
          + '<span class="do-step-num">' + (i + 1) + '</span>'
          + '<span class="do-step-copy"><span class="do-step-title">' + (step.label || "") + '</span>'
          + '<span class="do-step-note">' + note + '</span></span></li>';
      }).join("");
    }

  }

  function pollJob(jobId) {
    var misses = 0;

    function tick() {

      fetch("/actions/dashboard/job/" + encodeURIComponent(jobId), {

        credentials: "same-origin",

        headers: { "Accept": "application/json" },

      }).then(function (res) { return res.json().then(function (data) { return { res: res, data: data }; }); }).then(function (pack) {
        var data = pack.data || {};
        if (!pack.res.ok) {
          paintProgress({ percent: 0, label: "Blocked", detail: data.detail || data.error || "Lost the build job.", state: "error", steps: [] });
          showBuilderResult({ state: "error", detail: data.detail || data.error || "Lost the build job.", intent: data.intent });
          clearBuilderRunBusy();
          return;
        }

        misses = 0;
        paintProgress(data);

        var runForm = document.querySelector("[data-do-dash-run]");
        if (runForm) runForm.classList.toggle("is-progressing", data.state === "running");

        if (data.state === "running") {

          window.setTimeout(tick, 1000);

          return;

        }

        clearBuilderRunBusy();

        showBuilderResult(data);
        if (data.state === "ok" && (data.intent || "") === "build") {
          window.setTimeout(function () { window.location.reload(); }, 800);
        }

      }).catch(function () {
        misses += 1;
        if (misses >= 5) {
          paintProgress({ percent: 0, label: "Blocked", detail: "Lost contact with the build job.", state: "error", steps: [] });
          showBuilderResult({ state: "error", detail: "Lost contact with the build job." });
          clearBuilderRunBusy();
          return;
        }
        window.setTimeout(tick, 2000);

      });

    }

    tick();

  }

  function safeVercelUrl(url) {
    var s = String(url || "").trim();
    if (!s) return "";
    if (s.indexOf("https://") !== 0) return "";
    if (s.indexOf(".vercel.app") < 0) return "";
    if (/[\s"'<>]/.test(s)) return "";
    return s;
  }

  function publishEnvLabel(intent) {
    if (intent === "deploy_staging") return "Staging";
    if (intent === "deploy_production") return "Production";
    return "Publish";
  }

  function showBuilderResult(data) {
    var el = document.querySelector("[data-do-builder-result]");
    if (!el) return;
    var intent = data.intent || "";
    var state = data.state || "";
    var text = "";
    var kind = "is-ok";
    var publicUrl = safeVercelUrl(data.public_url);
    var deploymentUrl = safeVercelUrl(data.deployment_url);
    if (state === "ok" && intent === "preview") {
      kind = "is-warn";
      text = "Preview finished. That was a dry run — no files were written. Click Build locally to package the dashboard.";
    } else if (state === "ok" && intent.indexOf("deploy_") === 0) {
      text = publishEnvLabel(intent) + " publish finished.";
    } else if (state === "ok") {
      text = "Local build finished. Open it above, or publish below. This computer only until you publish.";
    } else {
      kind = "is-bad";
      var fallback = intent.indexOf("deploy_") === 0 ? "Publish failed." : "Build failed.";
      text = data.detail && data.detail !== "Build failed." && data.detail !== "Publish failed."
        ? data.detail
        : (data.label ? data.label + " — " + (data.detail || fallback) : fallback);
    }
    el.className = "do-builder-result " + kind;
    el.textContent = "";
    var title = document.createElement("div");
    title.className = "do-builder-result-title";
    title.textContent = text;
    el.appendChild(title);

    var showPublishLinks = intent.indexOf("deploy_") === 0 && (publicUrl || deploymentUrl);
    if (showPublishLinks) {
      var links = document.createElement("div");
      links.className = "do-builder-result-links";
      if (state === "ok" && publicUrl) {
        var primary = document.createElement("a");
        primary.className = "do-btn do-btn-primary do-builder-result-cta";
        primary.href = publicUrl;
        primary.target = "_blank";
        primary.rel = "noopener";
        primary.textContent = "Open " + publishEnvLabel(intent) + " dashboard";
        links.appendChild(primary);
        var aliasRow = document.createElement("p");
        aliasRow.className = "do-builder-result-url do-mono do-small";
        aliasRow.appendChild(document.createTextNode(publishEnvLabel(intent) + " URL: "));
        var aliasA = document.createElement("a");
        aliasA.href = publicUrl;
        aliasA.target = "_blank";
        aliasA.rel = "noopener";
        aliasA.textContent = publicUrl;
        aliasRow.appendChild(aliasA);
        links.appendChild(aliasRow);
      } else if (deploymentUrl) {
        var soft = document.createElement("p");
        soft.className = "do-builder-result-url do-mono do-small";
        soft.appendChild(document.createTextNode("Deployment URL (check before sharing): "));
        var softA = document.createElement("a");
        softA.href = deploymentUrl;
        softA.target = "_blank";
        softA.rel = "noopener";
        softA.textContent = deploymentUrl;
        soft.appendChild(softA);
        links.appendChild(soft);
      }
      if (state === "ok" && deploymentUrl && deploymentUrl !== publicUrl) {
        var depRow = document.createElement("p");
        depRow.className = "do-builder-result-url do-mono do-small do-muted";
        depRow.appendChild(document.createTextNode("Latest deployment: "));
        var depA = document.createElement("a");
        depA.href = deploymentUrl;
        depA.target = "_blank";
        depA.rel = "noopener";
        depA.textContent = deploymentUrl;
        depRow.appendChild(depA);
        links.appendChild(depRow);
      }
      el.appendChild(links);
    }

    el.hidden = false;
    if (kind === "is-bad" || (intent && intent.indexOf("deploy_") === 0)) {
      try { el.scrollIntoView({ block: "nearest", behavior: "smooth" }); } catch (err) {}
    }
    try {
      window.sessionStorage.setItem("doDashLastResult", JSON.stringify({
        ccn: data.ccn,
        intent: intent,
        state: state,
        detail: data.detail,
        local_view_url: data.local_view_url,
        public_url: publicUrl || data.public_url || "",
        deployment_url: deploymentUrl || data.deployment_url || "",
        t: Date.now()
      }));
    } catch (err) {}
  }

  function restoreBuilderResult() {
    var el = document.querySelector("[data-do-builder-result]");
    var ccnInput = document.getElementById("ccn");
    if (!el || !ccnInput) return;
    try {
      var raw = window.sessionStorage.getItem("doDashLastResult");
      if (!raw) return;
      var saved = JSON.parse(raw);
      var ccn = (ccnInput.value || "").trim();
      if (!saved || saved.ccn !== ccn) return;
      if (Date.now() - (saved.t || 0) > 6 * 60 * 60 * 1000) return;
      if (saved.state === "error") return;
      showBuilderResult(saved);
      var panel = document.getElementById("do-builder-progress");
      if (panel) panel.hidden = true;
    } catch (err) {}
  }

  function wireProgressDismiss() {
    var btn = document.querySelector("[data-do-progress-dismiss]");
    if (!btn) return;
    btn.addEventListener("click", function () {
      progressUserHidden = true;
      var panel = document.getElementById("do-builder-progress");
      if (panel) panel.hidden = true;
      btn.hidden = true;
    });
  }

  wireDashboardRunForm(document.querySelector("[data-do-dash-run]"));
  restoreBuilderResult();
  wireOpenLocal();
  wireProgressDismiss();

  function wireOpenLocal() {
    var links = document.querySelectorAll("[data-do-open-local]");
    if (!links.length) return;
    links.forEach(function (el) {
      el.addEventListener("click", function (e) {
        var href = el.getAttribute("href") || "";
        if (href.indexOf("http") !== 0) return;
        e.preventDefault();
        window.open(href, "_blank", "noopener");
        var ccn = el.getAttribute("data-ccn") || "";
        var wrap = el.closest(".do-builder-open");
        var status = wrap ? wrap.querySelector("[data-do-open-local-status]") : null;
        if (status) {
          status.hidden = false;
          status.textContent = "Starting the local server if it is not already running…";
        }
        fetch("/actions/dashboard/open-local", {
          method: "POST",
          credentials: "same-origin",
          headers: {
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "X-Requested-With": "XMLHttpRequest"
          },
          body: "ccn=" + encodeURIComponent(ccn)
        }).then(function (res) { return res.json(); }).then(function (data) {
          if (status) status.textContent = (data && data.detail) || "Opened in a new tab.";
        }).catch(function () {
          if (status) status.textContent = "Opened the local URL. If it failed to load, click Open again in a few seconds.";
        });
      });
    });
  }

})();
