"""tabprobes — DeepFeatureAffineLogisticProbe and DeepFeatureStatsProbe using dbx.dataprobes."""

from dataclasses import dataclass
import dbx
from dbx.dataprobes import (
    DatafeatureAffineLogisticProbe,
    DatafeatureStatsProbe,
)


class DeepFeatureAffineLogisticProbe(DatafeatureAffineLogisticProbe):
    """Logistic regression probe for deep feature layers."""

    VERSION = 1

    @dataclass
    class VAR(DatafeatureAffineLogisticProbe.VAR):
        pass


class DeepFeatureStatsProbe(DatafeatureStatsProbe):
    """Statistics probe for deep feature layers."""

    VERSION = 1

    @dataclass
    class VAR(DatafeatureStatsProbe.VAR):
        pass


class BipolarDeepFeatureAffineLogisticProbe(DatafeatureAffineLogisticProbe):
    """Logistic regression probe for bipolar-encoded deep feature layers."""

    VERSION = 1

    @dataclass
    class VAR(DatafeatureAffineLogisticProbe.VAR):
        pass


class BipolarDeepFeatureStatsProbe(DatafeatureStatsProbe):
    """Statistics probe for bipolar-encoded deep feature layers."""

    VERSION = 1

    @dataclass
    class VAR(DatafeatureStatsProbe.VAR):
        pass
