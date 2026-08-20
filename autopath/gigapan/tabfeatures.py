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
        tilebag: DatapointTab = None
        datapoint_tab: DatapointTab = None
        evaluator_factory: object = None
        collator: object = None
        feature_namemap: dict[str, str] | None = None
        shard_size_limit_bytes: int = 1 << 26
        shard_size: int = 1024

    def __post_init__(self):
        super().__post_init__()
        tab = self.var.tilebag if self.var.tilebag is not None else self.var.datapoint_tab
        object.__setattr__(self.var, 'datapoint_tab', tab)
        if self.var.collator is None:
            object.__setattr__(self.var, 'collator', tile_collator())


class DeepFeatureClip(DatafeatureTable):
    """Deep feature clip storing multi-layer activations across bags."""

    VERSION = 1
    TAB = DeepFeatureBag
    TOPICS = {'tabs': DIRTOPIC, 'built_tabs': DIRTOPIC, 'done': 'done'}

    @dataclass
    class VAR(Datablock.VAR):
        tilebagclip: DatapointTable = None
        datapoint_table: DatapointTable = None
        evaluator_factory: object = None
        collator: object = None
        feature_namemap: dict | None = None
        shard_size_limit_bytes: int = 1 << 26
        shard_size: int = 1024

    def __post_init__(self):
        super().__post_init__()
        table = self.var.tilebagclip if self.var.tilebagclip is not None else self.var.datapoint_table
        object.__setattr__(self.var, 'datapoint_table', table)
        if self.var.collator is None:
            object.__setattr__(self.var, 'collator', tile_collator())
class BipolarDeepFeatureBag(BipolarDatafeatureTab):
    """Bipolar-encoded deep feature bag."""

    VERSION = 1
    TOPICS = {'bipolar_features': SLICETOPIC, 'tab_bipolar_features': SLICETOPIC}

    @dataclass
    class VAR(Datablock.VAR):
        featuretab: DatafeatureTab = None
        layer: str = 'final'
        threshold: float = 0.5
        ternarize: bool = False


class BipolarDeepFeatureClip(BipolarDatafeatureTable):
    """Bipolar-encoded deep feature clip across bags."""

    VERSION = 1
    TAB = BipolarDeepFeatureBag
    TOPICS = {'tabs': DIRTOPIC, 'built_tabs': DIRTOPIC, 'done': 'done'}

    @dataclass
    class VAR(Datablock.VAR):
        clip: DatafeatureTable = None
        featuretable: DatafeatureTable = None
        stats_probe: object = None
        layer: str = 'final'
        bag_aggregation_threshold: float = 0.5
        threshold: float = 0.5
        ternarize_tiles: bool = False
        ternarize: bool = False

    def __post_init__(self):
        super().__post_init__()
        ft = self.var.clip if self.var.clip is not None else self.var.featuretable
        object.__setattr__(self.var, 'featuretable', ft)
        agg_thresh = self.var.bag_aggregation_threshold if self.var.bag_aggregation_threshold is not None else self.var.threshold
        object.__setattr__(self.var, 'threshold', agg_thresh)
        tern = self.var.ternarize_tiles if self.var.ternarize_tiles is not None else self.var.ternarize
        object.__setattr__(self.var, 'ternarize', tern)

