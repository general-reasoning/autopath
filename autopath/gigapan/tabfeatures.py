"""tabfeatures — DeepFeatureBag, DeepFeatureClip, and Bipolar variants using dbx.datafeatures."""

from dataclasses import dataclass, field, fields
import dbx
from dbx.datablocks import Datablock
from dbx.datapoints import DatapointTab, DatapointTable, DIRTOPIC, SLICETOPIC
from dbx.datafeatures import (
    Datacollator,
    DatafeatureTab,
    DatafeatureTable,
    BipolarDatafeatureTab,
    BipolarDatafeatureTable,
)


from dbx.datamodels import DatamodelEvaluatorFactory


class TileCollator(Datacollator):
    """Default collator for tile datasets, extracting signal ('tiles', 'tile')."""

    VERSION = 1

    @dataclass
    class VAR(Datablock.VAR):
        signals: list = field(default_factory=lambda: [('tiles', 'tile')])
        labels: list = field(default_factory=list)
        length: int | None = None


def tile_collator() -> TileCollator:
    """Constructor for the default tile collator."""
    return TileCollator()


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
    def UNSAFE_redirect_callable(bag, idx=None, *, journal=None):
        """Redirect callable that maps a bag to its built paths via journal lookup.

        Searches `journal` for a build event matching `bag` (by `idx` or by matching `source`/`tag`),
        and returns `{'paths': paths}` if valid, or `None` if invalid.
        """
        try:
            import os
            from dbx.datablocks import DatajournalEntry

            if journal is None and hasattr(bag, 'journal'):
                try:
                    journal = bag.journal()
                except Exception:
                    journal = None

            if journal is None or len(journal) == 0:
                return None

            df = journal[journal['event'] == 'build:end'] if ('event' in journal.columns and 'build:end' in journal['event'].values) else journal

            datapoint_tab = getattr(bag.var, 'datapoint_tab', None)
            search_keys = []
            if datapoint_tab is not None:
                dp_var = getattr(datapoint_tab, 'var', getattr(datapoint_tab, 'cfg', getattr(datapoint_tab, 'config', None)))
                if dp_var is not None and hasattr(dp_var, 'source') and dp_var.source:
                    search_keys.append(os.path.basename(dp_var.source))
                    search_keys.append(dp_var.source)
                if hasattr(datapoint_tab, 'tag') and datapoint_tab.tag:
                    search_keys.append(datapoint_tab.tag)
            if hasattr(bag, 'tag') and bag.tag:
                search_keys.append(bag.tag)

            matched_entry = None
            if idx is not None and 0 <= idx < len(df):
                cand = df.iloc[idx]
                cand_sig = str(cand.get('signature', cand.get('type', '')))
                cand_tag = str(cand.get('tag', ''))
                if any(k and (k in cand_sig or k in cand_tag) for k in search_keys):
                    matched_entry = cand

            if matched_entry is None:
                for _, row in df.iloc[::-1].iterrows():
                    row_sig = str(row.get('signature', row.get('type', '')))
                    row_tag = str(row.get('tag', ''))
                    if any(k and (k in row_sig or k in row_tag) for k in search_keys):
                        matched_entry = row
                        break

            if matched_entry is not None:
                entry = DatajournalEntry(matched_entry)
                paths = entry.paths
                if paths:
                    first_path = next(iter(paths.values()))
                    if bag.valid_path(first_path):
                        return {'paths': paths}
            return None
        except Exception as e:
            if hasattr(bag, 'log'):
                bag.log.detailed(f"UNSAFE_redirect_callable failed: {e}")
            return None

    def unsafe_redirect_callable(self, idx=None, journal=None):
        return self.UNSAFE_redirect_callable(self, idx=idx, journal=journal)





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
