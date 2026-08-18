"""tabfeatures — DeepFeatureBag, DeepFeatureClip, and Bipolar variants using dbx.datafeatures."""

from dataclasses import dataclass
import dbx
from dbx.datapoints import DatapointTab, DatapointTable
from dbx.datafeatures import (
    DatafeatureTab,
    DatafeatureTable,
    BipolarDatafeatureTab,
    BipolarDatafeatureTable,
)


class DeepFeatureBag(DatafeatureTab):
    """Deep feature bag storing multi-layer activations."""

    VERSION = 1

    @dataclass
    class VAR(DatafeatureTab.VAR):
        tilebag: DatapointTab = None
        datapoint_tab: DatapointTab = None

    def __post_init__(self):
        super().__post_init__()
        tab = self.var.tilebag if self.var.tilebag is not None else self.var.datapoint_tab
        if tab is not None:
            object.__setattr__(self.var, 'datapoint_tab', tab)


class DeepFeatureClip(DatafeatureTable):
    """Deep feature clip storing multi-layer activations across bags."""

    VERSION = 1
    TAB = DeepFeatureBag

    @dataclass
    class VAR(DatafeatureTable.VAR):
        tilebagclip: DatapointTable = None
        datapoint_table: DatapointTable = None

    def __post_init__(self):
        super().__post_init__()
        table = self.var.tilebagclip if self.var.tilebagclip is not None else self.var.datapoint_table
        if table is not None:
            object.__setattr__(self.var, 'datapoint_table', table)


class BipolarDeepFeatureBag(BipolarDatafeatureTab):
    """Bipolar-encoded deep feature bag."""

    VERSION = 1

    @dataclass
    class VAR(BipolarDatafeatureTab.VAR):
        pass


class BipolarDeepFeatureClip(BipolarDatafeatureTable):
    """Bipolar-encoded deep feature clip across bags."""

    VERSION = 1
    TAB = BipolarDeepFeatureBag

    @dataclass
    class VAR(BipolarDatafeatureTable.VAR):
        clip: DatafeatureTable = None
        featuretable: DatafeatureTable = None

    def __post_init__(self):
        super().__post_init__()
        ft = self.var.clip if self.var.clip is not None else self.var.featuretable
        if ft is not None:
            object.__setattr__(self.var, 'featuretable', ft)
