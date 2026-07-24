from __future__ import annotations

import datetime
import sys
from pathlib import Path
from typing import Any, Dict

# Keep sibling imports working even when this file is loaded via a root shim.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Public scheme names.
SCHEME_AUS_BASE = "aus_base"
SCHEME_SI_DOBAVA = "si_dobava"
SCHEME_SI_SAMOOSKRBA = "si_samooskrba"

SUPPORTED_SCHEMES = (
    SCHEME_AUS_BASE,
    SCHEME_SI_DOBAVA,
    SCHEME_SI_SAMOOSKRBA,
)

# Multi-user modes are intentionally unsupported in this dispatcher for now.
SKIPPED_MULTI_USER_SCHEMES = (
    "si_skupnost",
    "si_obracun_skupnosti",
    "si_obracun_souporabe",
)


def Aus_Base(
    smp_market_price_mwh: float,
    total_consumed_kwh: float,
    utc_date: datetime.datetime,
    interval_minutes: int = 30,
) -> dict:
    """Legacy Australian interval pricing function kept as default behavior."""

    # -------------------------------------------------------------------------
    # 1. CONSTANTS & TAXES
    # -------------------------------------------------------------------------
    GST_RATE = 0.10
    DAYS_IN_MONTH = 30

    # -------------------------------------------------------------------------
    # 2. CONSTANT COSTS (Fixed Monthly/Daily Fees)
    # -------------------------------------------------------------------------
    monthly_subscription_ex_gst = 20.00
    daily_supply_ex_gst = 1.09

    intervals_per_day = (24 * 60) / float(interval_minutes)
    intervals_per_month = intervals_per_day * DAYS_IN_MONTH

    constant_cost_ex_gst = (daily_supply_ex_gst / intervals_per_day) + (monthly_subscription_ex_gst / intervals_per_month)
    constant_cost_inc_gst = constant_cost_ex_gst * (1 + GST_RATE)

    # -------------------------------------------------------------------------
    # 3. VARIABLE COSTS (Price of Electricity)
    # -------------------------------------------------------------------------
    # Existing pipeline uses this conversion; kept for backward compatibility.
    spot_price_kwh = smp_market_price_mwh / 0.615

    MLF = 0.995
    DLF = 1.045
    adjusted_spot_kwh = spot_price_kwh * MLF * DLF

    nem_time = utc_date + datetime.timedelta(hours=10)
    hour = nem_time.hour

    if 15 <= hour < 21:
        network_rate_kwh = 0.2360
    elif 10 <= hour < 15:
        network_rate_kwh = 0.0270
    else:
        network_rate_kwh = 0.0720

    env_market_rate_kwh = 0.0250

    if total_consumed_kwh >= 0:
        total_rate_kwh_ex_gst = adjusted_spot_kwh + network_rate_kwh + env_market_rate_kwh
        variable_cost_ex_gst = total_consumed_kwh * total_rate_kwh_ex_gst
        variable_cost_inc_gst = variable_cost_ex_gst * (1 + GST_RATE)
    else:
        variable_cost_ex_gst = total_consumed_kwh * adjusted_spot_kwh
        variable_cost_inc_gst = variable_cost_ex_gst

    return {
        "constant_price_aud": round(constant_cost_inc_gst, 10),
        "variable_price_aud": round(variable_cost_inc_gst, 10),
    }


def _normalize_aus_result(raw: Dict[str, Any], scheme: str) -> Dict[str, Any]:
    constant_price = float(raw.get("constant_price_aud", 0.0))
    variable_price = float(raw.get("variable_price_aud", 0.0))
    return {
        "scheme": scheme,
        "currency": "AUD",
        "constant_price_aud": round(constant_price, 10),
        "variable_price_aud": round(variable_price, 10),
    }


def _normalize_si_result(
    raw: Dict[str, Any],
    scheme: str,
    *,
    apply_ddv: bool,
) -> Dict[str, Any]:
    # Interval SI outputs are taxable item totals without monthly fixed charges.
    from si_tarife import DDV

    postavke = raw.get("obdavcljive_postavke", {})
    taxable = float(sum(float(v) for v in postavke.values()))
    dobropis = float(raw.get("dobropis_odkup", 0.0))

    if apply_ddv:
        interval_total = (taxable * (1.0 + float(DDV))) - dobropis
    else:
        interval_total = taxable - dobropis

    return {
        "scheme": scheme,
        "currency": "EUR",
        # Compatibility keys are kept for RL/MILP integration stability.
        "constant_price_aud": 0.0,
        "variable_price_aud": round(interval_total, 10),
        "taxable_interval_eur": round(taxable, 10),
        "dobropis_odkup_eur": round(dobropis, 10),
        "ddv_included": bool(apply_ddv),
    }


def _resolve_si_dobava(
    smp_market_price_mwh: float,
    total_consumed_kwh: float,
    utc_date: datetime.datetime,
    interval_minutes: int,
    **kwargs: Any,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    from si_obracun import dobava
    from si_paketi import PAKETI

    paket_id = str(kwargs.get("paket_id", "GENI_REDNI"))
    if paket_id not in PAKETI:
        raise ValueError(f"Unknown SI package '{paket_id}'.")

    raw = dobava(
        market_price_mwh=float(smp_market_price_mwh),
        total_consumed_kwh=float(total_consumed_kwh),
        utc_date=utc_date,
        interval_minutes=int(interval_minutes),
        paket=PAKETI[paket_id],
        pravila=kwargs.get("pravila"),
        meritve_15min=bool(kwargs.get("meritve_15min", True)),
    )
    normalized = _normalize_si_result(
        raw,
        SCHEME_SI_DOBAVA,
        apply_ddv=bool(kwargs.get("apply_ddv", True)),
    )
    normalized["paket_id"] = paket_id
    return normalized, raw


def _resolve_si_samooskrba(
    smp_market_price_mwh: float,
    total_consumed_kwh: float,
    utc_date: datetime.datetime,
    interval_minutes: int,
    **kwargs: Any,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    from si_obracun import samooskrba
    from si_paketi import PAKETI

    paket_id = str(kwargs.get("paket_id", "GENI_SAMO_REDNI"))
    if paket_id not in PAKETI:
        raise ValueError(f"Unknown SI package '{paket_id}'.")

    total_produced_kwh = kwargs.get("total_produced_kwh")
    consumed = float(total_consumed_kwh)
    if total_produced_kwh is None:
        # If only net flow is known (common in RL), infer a simple split.
        if consumed < 0.0:
            produced = abs(consumed)
            consumed = 0.0
        else:
            produced = 0.0
    else:
        produced = float(total_produced_kwh)

    raw = samooskrba(
        market_price_mwh=float(smp_market_price_mwh),
        total_consumed_kwh=consumed,
        utc_date=utc_date,
        interval_minutes=int(interval_minutes),
        total_produced_kwh=produced,
        paket=PAKETI[paket_id],
        pravila=kwargs.get("pravila"),
        meritve_15min=bool(kwargs.get("meritve_15min", True)),
    )
    normalized = _normalize_si_result(
        raw,
        SCHEME_SI_SAMOOSKRBA,
        apply_ddv=bool(kwargs.get("apply_ddv", True)),
    )
    normalized["paket_id"] = paket_id
    normalized["total_produced_kwh"] = round(produced, 10)
    return normalized, raw


def _resolve_single_scheme(
    scheme: str,
    smp_market_price_mwh: float,
    total_consumed_kwh: float,
    utc_date: datetime.datetime,
    interval_minutes: int,
    **kwargs: Any,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if scheme in SKIPPED_MULTI_USER_SCHEMES:
        raise ValueError(
            f"Scheme '{scheme}' is skipped for now because it requires multi-user/community context."
        )

    if scheme == SCHEME_AUS_BASE:
        raw = Aus_Base(
            smp_market_price_mwh=float(smp_market_price_mwh),
            total_consumed_kwh=float(total_consumed_kwh),
            utc_date=utc_date,
            interval_minutes=int(interval_minutes),
        )
        return _normalize_aus_result(raw, SCHEME_AUS_BASE), raw

    if scheme == SCHEME_SI_DOBAVA:
        return _resolve_si_dobava(
            smp_market_price_mwh,
            total_consumed_kwh,
            utc_date,
            interval_minutes,
            **kwargs,
        )

    if scheme == SCHEME_SI_SAMOOSKRBA:
        return _resolve_si_samooskrba(
            smp_market_price_mwh,
            total_consumed_kwh,
            utc_date,
            interval_minutes,
            **kwargs,
        )

    raise ValueError(
        f"Unknown scheme '{scheme}'. Supported schemes: {', '.join(SUPPORTED_SCHEMES)}."
    )


def calculate_interval_price(
    smp_market_price_mwh: float,
    total_consumed_kwh: float,
    utc_date: datetime.datetime,
    interval_minutes: int = 30,
    *,
    scheme: str = SCHEME_AUS_BASE,
    compare_all: bool = False,
    include_raw: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Unified interval pricing dispatcher for RL/MILP.

    Compatibility contract:
    - Always returns keys `constant_price_aud` and `variable_price_aud`.
    - Default behavior is equivalent to `Aus_Base`.

    Extended features:
    - `scheme`: choose one supported single-user pricing scheme.
    - `compare_all`: include normalized (and optionally raw) outputs for all supported schemes.
    - Always includes explicit currency metadata.
    """

    normalized, raw = _resolve_single_scheme(
        scheme,
        smp_market_price_mwh,
        total_consumed_kwh,
        utc_date,
        interval_minutes,
        **kwargs,
    )

    result: Dict[str, Any] = dict(normalized)
    result["comparison_enabled"] = bool(compare_all)

    if include_raw:
        result["raw_result"] = raw

    if compare_all:
        comparisons: Dict[str, Any] = {}
        for candidate in SUPPORTED_SCHEMES:
            candidate_normalized, candidate_raw = _resolve_single_scheme(
                candidate,
                smp_market_price_mwh,
                total_consumed_kwh,
                utc_date,
                interval_minutes,
                **kwargs,
            )
            if include_raw:
                comparisons[candidate] = {
                    "normalized": candidate_normalized,
                    "raw": candidate_raw,
                }
            else:
                comparisons[candidate] = candidate_normalized
        result["comparisons"] = comparisons

    return result


def list_pricing_schemes(include_skipped: bool = False) -> tuple[str, ...]:
    """Return supported schemes; optionally include known skipped schemes."""
    if include_skipped:
        return SUPPORTED_SCHEMES + SKIPPED_MULTI_USER_SCHEMES
    return SUPPORTED_SCHEMES