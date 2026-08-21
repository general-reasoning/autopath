"""tabfeatures — DeepFeatureBag, DeepFeatureClip, and Bipolar variants using dbx.datafeatures."""

from dataclasses import dataclass, field
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

    @dataclass
    class VAR(Datablock.VAR):
        datapoint_tab: DatapointTab | None = None
        tilebag: DatapointTab | None = None
        evaluator_factory: DatamodelEvaluatorFactory | None = None
        collator: Datacollator | None = None
        feature_namemap: dict[str, str] | None = None
        shard_size_limit_bytes: int = 1 << 26
        shard_size: int = 1024

        def __post_init__(self):
            tab = self.datapoint_tab if self.datapoint_tab is not None else self.tilebag
            object.__setattr__(self, 'datapoint_tab', tab)
            object.__setattr__(self, 'tilebag', tab)


class DeepFeatureClip(DatafeatureTable):
    """Deep feature clip storing multi-layer activations across bags."""

    VERSION = 1
    TAB = DeepFeatureBag
    TOPICS = {'tabs': DIRTOPIC, 'tab_paths': DIRTOPIC, 'done': 'done'}

    @dataclass
    class VAR(Datablock.VAR):
        datapoint_table: DatapointTable | None = None
        tilebagclip: DatapointTable | None = None
        evaluator_factory: DatamodelEvaluatorFactory | None = None
        collator: Datacollator | None = None
        feature_namemap: dict | None = None
        shard_size_limit_bytes: int = 1 << 26
        shard_size: int = 1024

        def __post_init__(self):
            table = self.datapoint_table if self.datapoint_table is not None else self.tilebagclip
            object.__setattr__(self, 'datapoint_table', table)
            object.__setattr__(self, 'tilebagclip', table)


class BipolarDeepFeatureBag(BipolarDatafeatureTab):
    """Bipolar-encoded deep feature bag."""

    VERSION = 1
    TOPICS = {'bipolar_features': SLICETOPIC, 'tab_bipolar_features': SLICETOPIC}

    @dataclass
    class VAR(Datablock.VAR):
        featuretab: DatafeatureTab | None = None
        bag: DatafeatureTab | None = None
        layer: str = 'final'
        threshold: float = 0.5
        ternarize: bool = False

        def __post_init__(self):
            tab = self.featuretab if self.featuretab is not None else self.bag
            object.__setattr__(self, 'featuretab', tab)
            object.__setattr__(self, 'bag', tab)


class BipolarDeepFeatureClip(BipolarDatafeatureTable):
    """Bipolar-encoded deep feature clip across bags."""

    VERSION = 1
    TAB = BipolarDeepFeatureBag
    TOPICS = {'tabs': DIRTOPIC, 'tab_paths': DIRTOPIC, 'done': 'done'}

    @dataclass
    class VAR(Datablock.VAR):
        featuretable: DatafeatureTable | None = None
        clip: DatafeatureTable | None = None
        stats_probe: object | None = None
        layer: str = 'final'
        threshold: float = 0.5
        bag_aggregation_threshold: float = 0.5
        ternarize: bool = False
        ternarize_tiles: bool = False

        def __post_init__(self):
            tbl = self.featuretable if self.featuretable is not None else self.clip
            object.__setattr__(self, 'featuretable', tbl)
            object.__setattr__(self, 'clip', tbl)


