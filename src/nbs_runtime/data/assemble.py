"""Turn raw extracted layers into the Layers the first pass uses. Pure numpy, offline, testable.

Unit conversions happen here, once (DATASET_INPUTS.md: "convert each native value once"):
  SOC  stored dg/kg   x 0.1 -> g/kg
  pH   stored pH x 10 x 0.1 -> pH units
Depths are then combined by thickness (0-5, 5-15, 15-30 cm); a missing depth leaves the cell unknown.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ..firstpass import Layers
from ..rasters import (
    PH_X10_TO_PH,
    SOC_DG_PER_KG_TO_G_PER_KG,
    TOPSOIL_LAYERS,
    host_fractions_from_shares,
    share_sum_warnings,
    soil_value_warnings,
    thickness_weighted,
)


def assemble_layers(
    slope_deg: np.ndarray,
    soc_raw: Mapping[str, np.ndarray],
    ph_raw: Mapping[str, np.ndarray],
    shares: Mapping[int, np.ndarray] | None,
    groups: Mapping[str, Sequence[int]],
    notes: Sequence[str] = (),
) -> Layers:
    """With shares=None the host gate is NOT applied: every cell counts as host (an exploratory overview)."""
    soc_by = {d: np.asarray(a, dtype=float) * SOC_DG_PER_KG_TO_G_PER_KG for d, a in soc_raw.items()}
    ph_by = {d: np.asarray(a, dtype=float) * PH_X10_TO_PH for d, a in ph_raw.items()}
    soc = thickness_weighted([soc_by[d] for d in TOPSOIL_LAYERS])
    ph = thickness_weighted([ph_by[d] for d in TOPSOIL_LAYERS])
    slope = np.asarray(slope_deg, dtype=float)
    if shares is None:
        fe, fu = np.ones_like(slope), np.zeros_like(slope)
        extra = ["host gate NOT applied: every cell is treated as host, so this is not the policy's gate"]
    else:
        fe, fu = host_fractions_from_shares(groups, shares)
        extra = share_sum_warnings(shares)
    return Layers(
        slope_deg=slope, soc_g_kg=soc, ph=ph,
        eligible_fraction=fe, unknown_fraction=fu,
        soc_by_depth=soc_by, ph_by_depth=ph_by, synthetic=False,
        host_gate_applied=shares is not None,
        notes=[*notes, *soil_value_warnings(soc, ph), *extra],
    )
