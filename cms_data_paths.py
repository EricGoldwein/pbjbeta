"""

Canonical CMS national data paths (nurse, non-nurse, EIN).



Superdynamic v2 facility bundles are built from **per-CCN slices** under

``deployments/pbj320-<CCN>/``. National source files live only in the paths below.



Path roots resolve via ``data_path_resolver`` (env → ``data_paths.local.json`` →

repo-relative default). An explicit ``root`` argument keeps legacy test/offline

behavior: ``<root>/<RelativeName>`` with no env/json lookup.



EIN uses two national shapes (CMS delivery, not our choice):

  - ``EIN/monolithic/`` — multi-quarter PUF zip (bulk history)

  - ``EIN/quarters/`` — single-quarter zips for releases newer than the monolithic snapshot



Extraction code merges both; you ingest new quarters into ``quarters/`` only.

"""



from __future__ import annotations



from pathlib import Path



from data_path_resolver import DATA_ROOT_KEYS, resolve_data_path, repo_root as _resolver_repo_root



_REPO_ROOT = Path(__file__).resolve().parent





def repo_root() -> Path:

    return _resolver_repo_root()





def optional_repo_root(candidate: str | Path | None) -> Path | None:
    """
    Map a caller-supplied repo path to ``_data_dir(..., root=...)``.

    Returns ``None`` when ``candidate`` is omitted or matches the canonical
    resolved repo root (env/json aware). Returns an explicit ``Path`` only for
    non-canonical roots (tests, temp dirs).
    """
    if candidate is None:
        return None
    explicit = Path(candidate).resolve()
    if explicit == repo_root():
        return None
    return explicit


def _data_dir(key: str, root: Path | None = None) -> Path:

    if root is not None:

        return (root / DATA_ROOT_KEYS[key]).resolve()

    return resolve_data_path(key).path





def standardized_nurse_dir(root: Path | None = None) -> Path:

    return _data_dir("standardized_pbj", root)





def standardized_nonnurse_dir(root: Path | None = None) -> Path:

    return _data_dir("standardized_nonnurse", root)





def nonnurse_raw_dir(root: Path | None = None) -> Path:

    return _data_dir("nonnursecsv", root)





def nurse_raw_dir(root: Path | None = None) -> Path:

    return _data_dir("pbjcsv", root)





def ein_root(root: Path | None = None) -> Path:

    return _data_dir("ein", root)





def ein_monolithic_dir(root: Path | None = None) -> Path:

    return ein_root(root) / "monolithic"





def ein_quarters_dir(root: Path | None = None) -> Path:

    return ein_root(root) / "quarters"





def ein_staging_dir(root: Path | None = None) -> Path:

    return ein_root(root) / "staging"





def ein_extracted_dir(root: Path | None = None) -> Path:

    return ein_root(root) / "extracted"





def ein_quarters_manifest(root: Path | None = None) -> Path:

    return ein_quarters_dir(root) / "manifest.json"





def deployments_dir(root: Path | None = None) -> Path:

    return _data_dir("deployments", root)





def facility_deploy_dir(ccn: str, root: Path | None = None) -> Path:

    return deployments_dir(root) / f"pbj320-{str(ccn).strip().zfill(6)}"





def provider_info_dir(root: Path | None = None) -> Path:

    """CMS monthly provider-info archives (yearly outer zips + extracted NH_*.csv)."""

    return _data_dir("provider_info", root)





def provider_info_normalized_dir(root: Path | None = None) -> Path:

    return _data_dir("provider_info_normalized", root)





def provider_info_extracted_dir(root: Path | None = None) -> Path:

    return _data_dir("provider_info_extracted", root)





def indexed_dir(root: Path | None = None) -> Path:

    return _data_dir("indexed", root)





def metrics_backups_dir(root: Path | None = None) -> Path:

    return _data_dir("metrics_backups", root)





def ownership_dir(root: Path | None = None) -> Path:

    """Monthly NH_Ownership_* facility contact files (CMS provider-info zip; not SNF_All_Owners)."""

    return (root or repo_root()) / "ownership"





def citations_dir(root: Path | None = None) -> Path:

    """NH health deficiency citations (separate from provider-info monthly zips)."""

    return (root or repo_root()) / "Citations"





def provider_release_manifest_dir(release_key: str, root: Path | None = None) -> Path:

    """Tracked manifest folder for one CMS provider release (e.g. ``2026-06``)."""

    return provider_info_dir(root) / "_manifests" / release_key


