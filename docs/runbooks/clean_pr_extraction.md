# Clean PR extraction runbook

**Purpose:** Extract small, reviewable PRs from dirty WIP without merging the WIP branch wholesale or polluting `origin/main`.

**Audience:** PBJapp hygiene sidecar and anyone slicing dashboard or guardrail work from a long-lived feature branch.

---

## 1. Why we use clean worktrees

| Principle | Rule |
|-----------|------|
| Dirty branch is source only | `feature/v2-ui-pr1-style-guide-clean` (and similar) holds WIP. **Do not merge it wholesale.** |
| New PRs start clean | Every extraction PR branches from **`origin/main`**, not from the dirty feature branch. |
| Copy, don't drag | Copy **only scoped, approved files** from the dirty worktree into the clean PR worktree. |
| One PR = one purpose | Style guide, guardrails, a partial template slice — each gets its own branch and PR. |

This keeps reviewable diffs, avoids accidental CMS/generated data in git, and lets dashboard PR-B slices proceed in parallel with hygiene chores.

---

## 2. Naming convention

### Worktree directories (sibling to main repo)

| Role | Path example |
|------|----------------|
| Main dirty WIP | `C:\Users\egold\PycharmProjects\PBJapp` |
| Clean PR worktree | `PBJapp-prA1`, `PBJapp-prA1b`, `PBJapp-prCmin`, `PBJapp-prRunbook`, … |

Use a short suffix that matches the PR purpose (`prA1` = first hygiene slice, `prB1min` = minimal dashboard partial, etc.).

### Git branch names

Describe **purpose**, not the dirty feature branch:

- `chore/data-path-audit-guardrails`
- `chore/packaging-cms-slice-guardrails`
- `chore/minimal-deployment-guardrails`
- `docs/worktree-extraction-runbook`
- `feature/v2-shell-partials` (when the slice is intentionally product-scoped)

---

## 3. Safe extraction workflow

### Prerequisites

- Dirty WIP is intact in the main `PBJapp` worktree; you are **reading** from it, not restructuring it.
- You know the exact file list for this PR before creating the worktree.

### Steps

1. **Fetch `origin`**
   ```powershell
   git -C C:\Users\egold\PycharmProjects\PBJapp fetch origin
   ```

2. **Create clean worktree + branch from `origin/main`**
   ```powershell
   git -C C:\Users\egold\PycharmProjects\PBJapp worktree add -b <branch-name> `
     C:\Users\egold\PycharmProjects\PBJapp-pr<suffix> origin/main
   ```

3. **Copy only approved files** from dirty WIP into the clean worktree  
   Use explicit paths — `Copy-Item` per file or a small list. Do not copy whole trees unless the PR scope is explicitly a tree.

4. **Stage only approved files**
   ```powershell
   cd C:\Users\egold\PycharmProjects\PBJapp-pr<suffix>
   git add docs/runbooks/some_file.md   # example: one path at a time
   ```

5. **Run minimal checks** for the PR type (examples):
   - Docs only: confirm staged file count and paths.
   - V2 template/JS: `python scripts/check_v2_inline_js.py`, `python scripts/check_v2_evidence_layout.py`, etc.
   - Guardrails: relevant `scripts/check_*.py` or `python -m pytest tests/... -q`

6. **Verify staged file list**
   ```powershell
   git diff --cached --name-only
   git diff --cached --stat
   ```

7. **Commit and push**
   ```powershell
   git commit -m "Short imperative subject"
   git push -u origin HEAD
   ```

8. **Open PR** against `main`.

9. **Verify GitHub “Files changed”** on the PR page before merge — must match the intended scope exactly.

---

## 4. Hard rules

| Never | Why |
|-------|-----|
| `git add -A` | Stages hundreds of untracked CMS CSVs, deploy bundles, and local artifacts. |
| `git clean` | Irreversible deletion of untracked work. |
| Stash hundreds of untracked files casually | Easy to lose WIP or mix scopes; use explicit copy + stage instead. |
| Merge `origin/main` into dirty WIP without a plan | Creates merge noise and confuses extraction boundaries. |
| Include generated CSVs / local CMS data unless explicitly intended | Blobs review, breaks clones, violates `.gitignore` intent. |
| One PR with unrelated purposes | Reviewers cannot safely merge; rollback becomes painful. |

**Also:** Do not modify the dirty feature branch as part of an unrelated hygiene or docs PR.

---

## 5. Quiet-output checklist

Prefer **counts and scoped path checks** over giant diffs or full PowerShell streams.

| Check | Command (quiet) |
|-------|-------------------|
| Staged file count | `(git diff --cached --name-only).Count` |
| Staged paths only | `git diff --cached --name-only` |
| Unstaged noise (should be empty for docs-only PR) | `git status -sb` |
| Short stat | `git diff --cached --stat` |

In Cursor or chat, report **short checklists** (branch, staged count, paths, checks run, safe Y/N) — not full terminal dumps.

---

## 6. PR checklist template

Copy and fill before every commit:

```txt
Branch:
Files staged count:
Files staged:
Checks run:
Any files outside scope staged? YES/NO
Safe to commit? YES/NO
```

Example (docs-only):

```txt
Branch: docs/worktree-extraction-runbook
Files staged count: 1
Files staged: docs/runbooks/clean_pr_extraction.md
Checks run: git diff --cached --name-only; path scope review
Any files outside scope staged? NO
Safe to commit? YES
```

---

## 7. Post-merge note

- **Do not** delete branches or remove worktrees automatically after merge.
- **First** confirm the PR is merged on GitHub and the branch is no longer needed for follow-up slices.
- **Stale worktree cleanup** should start as a **separate read-only inventory** (branch name, last commit, merged Y/N) before any `git worktree remove`.

Retiring `PBJapp-prA1` and similar is a deliberate hygiene task — not part of every extraction PR.

---

## Warning: `provider_info_extracted/`

On **`origin/main`**, `provider_info_extracted/` is **untracked** (gitignored). An older dirty branch may still **track** those CSVs in its local or remote history — that is **branch-local legacy state**.

**Do not** “fix” `provider_info_extracted/` index or untrack files casually inside unrelated PRs. Treat it as a dedicated, scoped chore PR after explicit audit (`docs/repo_cleanup_plan.md` §B.6).

---

## Related docs

- `docs/repo_cleanup_plan.md` — repo layers and phase-2 classification
- `docs/testing_backlog.md` — pre-deploy checks and pytest inventory
- `docs/runbooks/v2_facility_deploy_runbook.md` — facility deploy (product path; separate from this extraction pattern)
