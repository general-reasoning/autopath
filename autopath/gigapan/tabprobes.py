"""tabprobes — DeepFeatureAffineLogisticProbe and DeepFeatureStatsProbe using dbx.probes."""

from dataclasses import dataclass
import dbx
from dbx.datablocks import Datablock
from dbx.probes import (
    FeatureAffineLogisticProbe,
    FeatureStatsProbe,
)


class DeepFeatureAffineLogisticProbe(FeatureAffineLogisticProbe):
    """Logistic regression probe for deep feature layers."""

    VERSION = 1

    @dataclass
    class VAR(FeatureAffineLogisticProbe.VAR):
        pass


class DeepFeatureStatsProbe(FeatureStatsProbe):
    """Statistics probe for deep feature layers."""

    VERSION = 2

    SPECIALIZATIONS = [
        Datablock.Specialization(
            spec={},
            topics={'count': 'count.npz'},
            version=1,
            all_recorded=True,
            note='Pre-instance topics build',
        )
    ]

    @dataclass
    class VAR(FeatureStatsProbe.VAR):
        pass


class BipolarDeepFeatureAffineLogisticProbe(FeatureAffineLogisticProbe):
    """Logistic regression probe for bipolar-encoded deep feature layers."""

    VERSION = 1

    @dataclass
    class VAR(FeatureAffineLogisticProbe.VAR):
        pass


class BipolarDeepFeatureStatsProbe(FeatureStatsProbe):
    """Statistics probe for bipolar-encoded deep feature layers."""

    VERSION = 2

    @dataclass
    class VAR(FeatureStatsProbe.VAR):
        pass
