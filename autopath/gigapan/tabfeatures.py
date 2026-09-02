"""tabfeatures — DeepFeatureBag, DeepFeatureClip, and Bipolar variants using dbx.datafeatures."""

from dataclasses import dataclass, field, fields
import json
import os

import dbx
from dbx.datablocks import Datablock, DatajournalEntry
from dbx.datafeatures import (
    BipolarDatafeatureTab,
    BipolarDatafeatureTable,
    Datacollator,
    DatafeatureTab,
    DatafeatureTable,
)
from dbx.datamodels import DatamodelEvaluatorFactory
from dbx.datapoints import DIRTOPIC, SLICETOPIC, DatapointTab, DatapointTable


class TileCollator(Datacollator):
    """Default collator for tile datasets, extracting signal ('tiles', 'tile')."""

    VERSION = 1

    @dataclass
    class VAR(Datablock.VAR):
        signals: list = field(default_factory=lambda: [('tiles', 'tile')])
        labels: list = field(default_factory=list)
        length: int | None = None


class TileAnnotationCollator(Datacollator):
    """Default collator for tile datasets, extracting signal ('tiles', 'tile') and 
        ('annotations', {annotation}), defaulting ot ('annotations', 'cohort')."""

    VERSION = 1

    @dataclass
    class VAR(Datablock.VAR):
        signals: list = field(default_factory=lambda: [('tiles', 'tile')])
        labels: list = field(default_factory=lambda: [('annotations', 'cohort')])
        length: int | None = None


def tile_collator() -> TileCollator:
    """Constructor for the default tile collator."""
    return TileCollator()


def tile_annotation_collator(label: str) -> TileAnnotationCollator:
    """Constructor for the default tile-annotation collator."""
    return TileAnnotationCollator()


class DeepFeatureBag(DatafeatureTab):
    """Deep feature bag storing multi-layer activations."""

    VERSION = 1
    TOPICS = {'features': SLICETOPIC}
    LEGACY_SIGNATURE = True

    @dataclass
    class VAR(Datablock.VAR):
        datapoint_tab: DatapointTab
        evaluator_factory: DatamodelEvaluatorFactory
        collator: Datacollator | None = None
        feature_namemap: dict[str, str] | None = None
        shard_size_limit_bytes: int = 1 << 26
        shard_size: int = 1024

    @staticmethod
    def UNSAFE_redirector(bag, idx=None, stack=None, *, journal=None):
        """Redirect callable that maps a bag to its built paths via journal lookup.

        Matches journal build events against `bag.var.datapoint_tab.tag`.
        Returns `{'paths': paths}` if valid, or `None` if invalid.
        """
        try:
            datapoint_tab = getattr(bag.var, 'datapoint_tab', None)
            dp_tag = getattr(datapoint_tab, 'tag', None)
            if not dp_tag:
                return None

            journals_to_check = []
            if journal is not None and len(journal) > 0:
                journals_to_check.append(journal)
            elif hasattr(bag, 'journal'):
                try:
                    bj = bag.journal()
                    if bj is not None and len(bj) > 0:
                        journals_to_check.append(bj)
                except Exception:
                    pass

            if not journals_to_check:
                return None

            matched_entry = None

            def is_valid_paths(paths):
                if not paths or not isinstance(paths, dict):
                    return False
                for p in paths.values():
                    if not p or not bag.fs.exists(p):
                        return False
                    idx_p = os.path.join(p, 'index.json')
                    if bag.fs.exists(idx_p):
                        try:
                            content = bag.fs.cat(idx_p)
                            if not json.loads(content).get('shards'):
                                return False
                        except Exception:
                            return False
                return True

            for j in journals_to_check:
                df = j[j['event'] == 'build:end'] if ('event' in j.columns and 'build:end' in j['event'].values) else j

                if idx is not None and 0 <= idx < len(df):
                    cand = df.iloc[idx]
                    cand_sig = str(cand.get('signature', cand.get('type', '')))
                    cand_tag = str(cand.get('tag', ''))
                    if dp_tag in cand_sig or dp_tag in cand_tag:
                        entry = DatajournalEntry(cand)
                        paths = entry.paths
                        if is_valid_paths(paths):
                            matched_entry = cand
                            break

                for _, row in df.iloc[::-1].iterrows():
                    row_sig = str(row.get('signature', row.get('type', '')))
                    row_tag = str(row.get('tag', ''))
                    if dp_tag in row_sig or dp_tag in row_tag:
                        entry = DatajournalEntry(row)
                        paths = entry.paths
                        if is_valid_paths(paths):
                            matched_entry = row
                            break
                if matched_entry is not None:
                    break

            if matched_entry is not None:
                entry = DatajournalEntry(matched_entry)
                paths = entry.paths
                if is_valid_paths(paths):
                    return {'paths': paths}
            return None

        except Exception as e:
            if hasattr(bag, 'log'):
                bag.log.detailed(f"UNSAFE_redirector failed: {e}")
            return None

    def unsafe_redirector(self, idx=None, stack=None, *, journal=None):
        return self.UNSAFE_redirector(self, idx=idx, stack=stack, journal=journal)

    def __validate__(self, **kwargs):
        feature_paths = self.paths()
        point_paths = self.var.datapoint_tab.paths()
        for feature_key, feature_path in feature_paths.items():
            if self.tag not in feature_path:
                self.log.warning(f"Tag {self.tag} not in feature path for topic {feature_key}: {feature_path}")
                return False
        for point_key, point_path in point_paths.items():
                if self.tag not in point_path:
                    self.log.warning(f"Tag {self.tag} not in point path for topic {point_key}: {point_path}")
                    return False
        return True
        

class DeepFeatureClip(DatafeatureTable):
    """Deep feature clip storing multi-layer activations across bags."""

    VERSION = 1
    TAB = DeepFeatureBag
    TOPICS = {'tabs': DIRTOPIC, 'tab_paths': DIRTOPIC, 'done': 'done'}
    LEGACY_SIGNATURE = True

    @dataclass
    class VAR(Datablock.VAR):
        datapoint_table: DatapointTable
        evaluator_factory: DatamodelEvaluatorFactory
        collator: Datacollator | None = None
        feature_namemap: dict | None = None
        shard_size_limit_bytes: int = 1 << 26
        shard_size: int = 1024


class BipolarDeepFeatureBag(BipolarDatafeatureTab):
    """Bipolar-encoded deep feature bag."""

    VERSION = 1
    TOPICS = {'bipolar_features': SLICETOPIC, 'tab_bipolar_features': SLICETOPIC}
    LEGACY_SIGNATURE = True

    @dataclass
    class VAR(Datablock.VAR):
        featuretab: DatafeatureTab
        layer: str = 'final'
        threshold: float = 0.5
        ternarize: bool = False


class BipolarDeepFeatureClip(BipolarDatafeatureTable):
    """Bipolar-encoded deep feature clip across bags."""

    VERSION = 1
    TAB = BipolarDeepFeatureBag
    TOPICS = {'tabs': DIRTOPIC, 'tab_paths': DIRTOPIC, 'done': 'done'}

    @dataclass
    class VAR(Datablock.VAR):
        featuretable: DatafeatureTable
        layer: str = 'final'
        threshold: float = 0.5
        ternarize: bool = False
