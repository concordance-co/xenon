"""Discovery hooks for Assistant Axis layer and capping calibration.

Known-model settings are encoded in ``specs.py``. This module is reserved for a
future reusable procedure that discovers those settings for new models instead
of requiring each project to hand-roll that calibration.
"""

from __future__ import annotations

from typing import Any


def discover_assistant_axis_layer_and_capping(*args: Any, **kwargs: Any) -> None:
    """TODO: discover the best Assistant Axis layer and activation-cap setting.

    The Yora repository includes known-model configs and capping config loading,
    but not a reusable end-to-end layer/cap discovery routine. The expected
    implementation should evaluate candidate layers and cap percentiles against
    held-out persona/role-play prompts following the Assistant Axis paper. The
    placeholder is intentionally explicit so unknown-model users get a clear
    failure rather than a silently arbitrary layer choice.
    """

    del args, kwargs
    raise NotImplementedError(
        "Assistant Axis layer and capping discovery is not implemented yet. "
        "Use known model configs when available or pass an explicit layer."
    )
