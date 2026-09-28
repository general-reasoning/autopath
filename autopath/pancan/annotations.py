"""CPTAC annotation index for enriching MDS tile shards.

Discovers and indexes clinical, pathological, and genomic annotations
from the CPTAC data hierarchy.  All paths are resolved relative to a
*source root* (the :class:`PancanTileClip` source directory) using
:mod:`fsspec` for filesystem-agnostic access.

Annotation Structure
--------------------
Each tile's ``annotations`` dict has the following shape::

    {
        'case_id':       str,   # e.g. 'C3N-01179'
        'cohort':        str,   # e.g. 'CCRCC_QUPATH_new'
        'gdc_clinical':  dict | None,
        'gdc_pathology': dict | None,
        'gdc_exposure':  dict | None,
        'gdc_follow_up': dict | None,
        'qupath':        dict | None,
        'mutations':     dict | None,   # gene_name → True (only mutated genes)
    }
"""

import csv
import functools
import io
import os
import re
from typing import Optional

import fsspec

from dbx import Logger

logger = Logger()


# ── Sentinel handling ───────────────────────────────────────────────

_GDC_SENTINEL = "'--"


def _clean_gdc_value(v: str) -> Optional[str]:
    """Convert GDC sentinel ``'--`` to ``None``, pass through otherwise."""
    if v == _GDC_SENTINEL or v.strip() == '':
        return None
    return v


# ── Case-ID extraction ─────────────────────────────────────────────

_CASE_ID_RE = re.compile(r'(C\d[A-Z]-\d{5}|\d{2}[A-Z]{2}\d{3})')


def extract_case_id(bag_name: str) -> Optional[str]:
    """Extract the CPTAC case ID (e.g. `C3N-01179` or `02OV035`) from a bag name."""
    m = _CASE_ID_RE.match(bag_name)
    return m.group(1) if m else None


# ── Relative path constants ────────────────────────────────────────
# All paths are relative to the clip source root.

GDC_CLINICAL_DIR = 'clinical data - downloaded from GDC'

# Cohort directory → QuPath annotation CSV basename.
QUPATH_CSV_MAP = {
    'BRCA_QUPATH_new':  'CPTAC_BRCA_anns_slides.csv',
    'CCRCC_QUPATH_new': 'CPTAC_CCRCC_anns_slides.csv',
    'COAD_QUPATH_new':  'CPTAC_COAD_anns_slides.csv',
    'GBM_QUPATH_new':   'CPTAC_GBM_anns_slides.csv',
    'HNSCC':            'CPTAC_HNSCC_anns_slides.csv',
    'LSCC_QUPATH_new':  'CPTAC_LSCC_anns_slides.csv',
    'LUAD_QUPATH':      'CPTAC_LUAD_anns_slides.csv',
    'OV_QUPATH_new':    'CPTAC_OV_anns_slides.csv',
    'PDA':              'CPTAC_PDA_anns_slides.csv',
    'UCEC':             'CPTAC_UCEC_anns_slides.csv',
}

# Cohort directory → Zhenyu mutation CSV basename.
MUTS_CSV_MAP = {
    'BRCA_QUPATH_new':  'CPTAC_BRCA_anns_slides_zhenyu_muts_full.csv',
    'CCRCC_QUPATH_new': 'CPTAC_CCRCC_anns_slides_zhenyu_muts_full.csv',
    'COAD_QUPATH_new':  'CPTAC_COAD_anns_slides_zhenyu_muts_full.csv',
    'HNSCC':            'CPTAC_HNSC_anns_slides_zhenyu_muts_full.csv',
    'LSCC_QUPATH_new':  'CPTAC_LSCC_anns_slides_zhenyu_muts_full.csv',
    'LUAD_QUPATH':      'CPTAC_LUAD_anns_slides_zhenyu_muts_full.csv',
    'OV_QUPATH_new':    'CPTAC_OV_anns_slides_zhenyu_muts_full.csv',
    'PDA':              'CPTAC_PDA_anns_slides_zhenyu_muts_full.csv',
    'UCEC':             'CPTAC_UCEC_anns_slides_zhenyu_muts_full.csv',
}

# Columns that are *not* gene-mutation indicators in the mutation CSVs.
# Everything else is treated as a boolean gene flag.
_MUTS_META_COLUMNS = {
    'slide', 'submitter_id', 'tumor_code', 'short_title',
    'histological_type', 'tumor_stage_pathological',
    'method_of_pathologic_diagnosis', '...3',
}


# ── TSV / CSV parsing helpers ──────────────────────────────────────

def _read_tsv(fs, path):
    """Read a GDC TSV via *fs*, returning a list of row dicts."""
    with fs.open(path, 'r', newline='') as f:
        text = f.read()
    reader = csv.DictReader(io.StringIO(text), delimiter='\t')
    return list(reader)


def _read_csv(fs, path):
    """Read a CSV via *fs*, returning a list of row dicts."""
    with fs.open(path, 'r', newline='', encoding='utf-8-sig') as f:
        text = f.read()
    reader = csv.DictReader(io.StringIO(text))
    return list(reader)


# ── GDC TSV indexing ───────────────────────────────────────────────

def _index_gdc_tsv(fs, path):
    """Parse a GDC TSV and return ``{case_submitter_id: {col: value}}``

    Only the first row per case is kept.  Sentinel values (``'--``)
    are replaced with ``None``.
    """
    index = {}
    if not fs.exists(path):
        logger.info(f"GDC TSV not found, skipping: {path}")
        return index
    rows = _read_tsv(fs, path)
    for row in rows:
        cid = row.get('case_submitter_id')
        if cid is None or cid in index:
            continue
        cleaned = {}
        for k, v in row.items():
            if k in ('case_id', 'case_submitter_id', 'project_id'):
                continue  # identifiers, not annotations
            val = _clean_gdc_value(v)
            if val is not None:
                cleaned[k] = val
        index[cid] = cleaned
    return index


# ── QuPath CSV indexing ────────────────────────────────────────────

def _index_qupath_csv(fs, path):
    """Parse a QuPath annotation CSV and return ``{submitter_id: dict}``.

    Skips the ``slide`` column (contains embedded JSON-like data).
    Only the first row per submitter is kept.
    """
    index = {}
    if not fs.exists(path):
        logger.info(f"QuPath CSV not found, skipping: {path}")
        return index
    rows = _read_csv(fs, path)
    for row in rows:
        sid = row.get('submitter_id')
        if sid is None or sid in index:
            continue
        cleaned = {}
        for k, v in row.items():
            if k in ('slide', 'submitter_id'):
                continue
            if v is not None and v.strip() not in ('', 'NA', 'N/A'):
                cleaned[k] = v.strip()
        index[sid] = cleaned
    return index


# ── Mutation CSV indexing ──────────────────────────────────────────

def _index_muts_csv(fs, path):
    """Parse a Zhenyu mutation CSV and return ``{submitter_id: {gene: True}}``.

    Only genes with value ``TRUE`` are included in the per-case dict.
    """
    index = {}
    if not fs.exists(path):
        logger.info(f"Mutation CSV not found, skipping: {path}")
        return index
    rows = _read_csv(fs, path)
    for row in rows:
        sid = row.get('submitter_id')
        if sid is None or sid in index:
            continue
        muts = {}
        for k, v in row.items():
            if k in _MUTS_META_COLUMNS:
                continue
            if v is not None and v.strip().upper() == 'TRUE':
                muts[k] = True
        index[sid] = muts
    return index


# ── AnnotationIndex ────────────────────────────────────────────────

class AnnotationIndex:
    """Lazily-loaded, cached index of all CPTAC annotations.

    Parameters
    ----------
    source_root : str
        The clip source directory (``$PANCAN_CPTAC_ROOT``).
    fs : fsspec.AbstractFileSystem
        Filesystem for resolving paths.
    """

    def __init__(self, source_root: str, fs: fsspec.AbstractFileSystem):
        self.source_root = source_root
        self.fs = fs
        self._gdc_clinical = None
        self._gdc_pathology = None
        self._gdc_exposure = None
        self._gdc_follow_up = None
        self._qupath_cache = {}     # cohort → dict
        self._muts_cache = {}       # cohort → dict

    # ── Lazy loaders ────────────────────────────────────────────

    def _load_gdc(self):
        """Load all GDC TSV files (clinical, pathology, exposure, follow_up)."""
        if self._gdc_clinical is not None:
            return  # already loaded
        gdc_dir = os.path.join(self.source_root, GDC_CLINICAL_DIR)
        # Discover project subdirectories (e.g. clinical.project-cptac-3.*)
        self._gdc_clinical = {}
        self._gdc_pathology = {}
        self._gdc_exposure = {}
        self._gdc_follow_up = {}
        if not self.fs.exists(gdc_dir):
            logger.info(f"GDC clinical directory not found: {gdc_dir}")
            return
        for subdir in self.fs.ls(gdc_dir, detail=False):
            if not self.fs.isdir(subdir):
                continue
            for tsv_name, target in [
                ('clinical.tsv',        self._gdc_clinical),
                ('pathology_detail.tsv', self._gdc_pathology),
                ('exposure.tsv',        self._gdc_exposure),
                ('follow_up.tsv',       self._gdc_follow_up),
            ]:
                tsv_path = os.path.join(subdir, tsv_name)
                if self.fs.exists(tsv_path):
                    idx = _index_gdc_tsv(self.fs, tsv_path)
                    # Merge (earlier projects don't overwrite later ones)
                    for k, v in idx.items():
                        target.setdefault(k, v)
        n = len(self._gdc_clinical)
        logger.info(f"Loaded GDC annotations: {n} cases")

    def _load_qupath(self, cohort: str):
        """Load the QuPath annotation CSV for *cohort*."""
        if cohort in self._qupath_cache:
            return
        csv_name = QUPATH_CSV_MAP.get(cohort)
        if csv_name is None:
            self._qupath_cache[cohort] = {}
            return
        csv_path = os.path.join(self.source_root, cohort, csv_name)
        self._qupath_cache[cohort] = _index_qupath_csv(self.fs, csv_path)

    def _load_muts(self, cohort: str):
        """Load the Zhenyu mutation CSV for *cohort*."""
        if cohort in self._muts_cache:
            return
        csv_name = MUTS_CSV_MAP.get(cohort)
        if csv_name is None:
            self._muts_cache[cohort] = {}
            return
        csv_path = os.path.join(self.source_root, cohort, csv_name)
        self._muts_cache[cohort] = _index_muts_csv(self.fs, csv_path)

    # ── Public API ──────────────────────────────────────────────

    def lookup(self, case_id: str, cohort: str) -> dict:
        """Return the annotation dict for *case_id* in *cohort*.

        Returns a nested dict suitable for storage as an MDS ``json``
        column.  Missing tiers are ``None``.
        """
        self._load_gdc()
        self._load_qupath(cohort)
        self._load_muts(cohort)

        return {
            'case_id':       case_id,
            'cohort':        cohort,
            'gdc_clinical':  self._gdc_clinical.get(case_id),
            'gdc_pathology': self._gdc_pathology.get(case_id),
            'gdc_exposure':  self._gdc_exposure.get(case_id),
            'gdc_follow_up': self._gdc_follow_up.get(case_id),
            'qupath':        self._qupath_cache.get(cohort, {}).get(case_id),
            'mutations':     self._muts_cache.get(cohort, {}).get(case_id),
        }


# ── Worker-local cached accessor ───────────────────────────────────

@functools.lru_cache(maxsize=1)
def _cached_index(source_root: str) -> AnnotationIndex:
    """Return a worker-local :class:`AnnotationIndex`.

    Cached by *source_root* so that parallel workers sharing
    the same clip source each build exactly one index.
    """
    fs, _ = fsspec.core.url_to_fs(source_root)
    return AnnotationIndex(source_root, fs)


def get_annotations(source_root: str, case_id: str, cohort: str) -> dict:
    """Convenience: look up annotations for a single case.

    Uses a worker-local cached :class:`AnnotationIndex`.

    Parameters
    ----------
    source_root : str
        The clip source directory (``$PANCAN_CPTAC_ROOT``).
    case_id : str
        The CPTAC case ID (e.g. ``'C3N-01179'``).
    cohort : str
        The cohort directory name (e.g. ``'CCRCC_QUPATH_new'``).
    """
    return _cached_index(source_root).lookup(case_id, cohort)
