from __future__ import annotations

import datetime
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Keep sibling imports working even when this file is loaded via a root shim.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# -----------------------------------------------------------------------------
# Public scheme names and semantics
# -----------------------------------------------------------------------------
# SCHEME_AUS_BASE:
#   Legacy AU benchmark pricing used in existing RL pipelines.
#
# SCHEME_SI_DOBAVA:
#   Slovenian supply-only billing logic (no self-supply production context).
#   Can be paired with TARIFNI/AKTIVNI/DINAMICNI supply modes.
#
# SCHEME_SI_SAMOOSKRBA:
#   Slovenian self-supply billing logic with interval netting between
#   consumption and production. Supports buyback styles including
#   NI/FIKSNI/AKTIVNI/DINAMICNI/NET_METERING via selected package.
#
# SI pricing mode vocabulary:
# - TipCene.TARIFNI: fixed tariff plan (ET or VT/MT)
# - TipCene.AKTIVNI: 4-tariff active plan (solar/off-peak/standard/peak)
# - TipCene.DINAMICNI: SIPX-linked dynamic plan
#
# SI buyback mode vocabulary:
# - TipOdkupa.NI: no buyback
# - TipOdkupa.FIKSNI: fixed buyback rate
# - TipOdkupa.AKTIVNI: 4-tariff buyback
# - TipOdkupa.DINAMICNI: SIPX minus spread
# - TipOdkupa.NET_METERING: annual net-metering, no interval credit
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


def _parse_tip_cene(value: Optional[Any]) -> Optional[Any]:
    if value is None:
        return None
    from si_paketi import TipCene

    if isinstance(value, TipCene):
        return value

    txt = str(value).strip().lower()
    for candidate in TipCene:
        if candidate.value == txt:
            return candidate
    raise ValueError(
        "Unknown pricing_mode. Use one of: "
        + ", ".join(t.value for t in TipCene)
    )


def _parse_tip_odkupa(value: Optional[Any]) -> Optional[Any]:
    if value is None:
        return None
    from si_paketi import TipOdkupa

    if isinstance(value, TipOdkupa):
        return value

    txt = str(value).strip().lower()
    for candidate in TipOdkupa:
        if candidate.value == txt:
            return candidate
    raise ValueError(
        "Unknown buyback_mode. Use one of: "
        + ", ".join(t.value for t in TipOdkupa)
    )


def _resolve_meritve_15min(
    interval_minutes: int,
    meritve_15min: Optional[bool],
    warnings: List[str],
) -> bool:
    if meritve_15min is None:
        if int(interval_minutes) != 15:
            warnings.append(
                "Non-15-minute interval detected; auto-setting meritve_15min=False."
            )
            return False
        return True

    if bool(meritve_15min) and int(interval_minutes) != 15:
        warnings.append(
            "meritve_15min=True with non-15-minute interval may be inconsistent with metering granularity."
        )
    return bool(meritve_15min)


def _resolve_pravila(
    utc_date: datetime.datetime,
    pravila: Any,
    pricing_reference_year: Optional[int],
    warnings: List[str],
) -> Any:
    if pravila is not None:
        if pricing_reference_year is not None:
            warnings.append(
                "Both pravila and pricing_reference_year were provided; using explicit pravila."
            )
        return pravila

    if pricing_reference_year is None:
        return None

    from si_obracun import Pravila

    ref_year = int(pricing_reference_year)
    data_year = int(utc_date.year)
    if data_year != ref_year:
        warnings.append(
            f"Using pricing rules from {ref_year} for data timestamp year {data_year}."
        )

    if ref_year == 2027:
        return Pravila.od_2027()
    return Pravila.ob_datumu(datetime.date(ref_year, 1, 1))


def _infer_consumed_produced(
    total_consumed_kwh: float,
    total_produced_kwh: Optional[float],
    warnings: List[str],
) -> Tuple[float, float]:
    consumed = float(total_consumed_kwh)
    if total_produced_kwh is not None:
        return consumed, float(total_produced_kwh)

    if consumed < 0.0:
        warnings.append(
            "Negative net flow without explicit production; inferring production=abs(net) and consumption=0."
        )
        return 0.0, abs(consumed)
    return consumed, 0.0


def _select_si_package(
    scheme: str,
    smp_market_price_mwh: float,
    total_consumed_kwh: float,
    utc_date: datetime.datetime,
    interval_minutes: int,
    *,
    paket_id: Optional[str],
    pricing_mode: Optional[Any],
    buyback_mode: Optional[Any],
    provider: Optional[str],
    pravila: Any,
    meritve_15min: bool,
    apply_ddv: bool,
    total_produced_kwh: Optional[float],
    warnings: List[str],
) -> str:
    from si_obracun import dobava, samooskrba
    from si_paketi import PAKETI, Shema, TipOdkupa

    if paket_id is not None:
        chosen = str(paket_id)
        if chosen not in PAKETI:
            raise ValueError(f"Unknown SI package '{chosen}'.")
        paket = PAKETI[chosen]
        if scheme == SCHEME_SI_DOBAVA and paket.zahteva_pv:
            warnings.append(
                f"Selected package '{chosen}' is self-supply oriented while scheme is '{SCHEME_SI_DOBAVA}'."
            )
        if scheme == SCHEME_SI_SAMOOSKRBA and not paket.dovoljuje_pv:
            warnings.append(
                f"Selected package '{chosen}' does not allow PV/self-supply while scheme is '{SCHEME_SI_SAMOOSKRBA}'."
            )
        return chosen

    resolved_tip_cene = _parse_tip_cene(pricing_mode)
    resolved_tip_odkupa = _parse_tip_odkupa(buyback_mode)
    provider_key = str(provider).strip().lower() if provider is not None else None

    candidates = []
    for pid, paket in PAKETI.items():
        if scheme == SCHEME_SI_DOBAVA and paket.zahteva_pv:
            continue
        if scheme == SCHEME_SI_SAMOOSKRBA and not paket.dovoljuje_pv:
            continue

        if resolved_tip_cene is not None and paket.tip_cene is not resolved_tip_cene:
            continue
        if resolved_tip_odkupa is not None and paket.tip_odkupa is not resolved_tip_odkupa:
            continue
        if provider_key is not None and paket.dobavitelj.strip().lower() != provider_key:
            continue

        if scheme == SCHEME_SI_SAMOOSKRBA and resolved_tip_odkupa is TipOdkupa.NET_METERING:
            if Shema.NET_METERING not in paket.dovoljene_sheme:
                continue

        candidates.append((pid, paket))

    if not candidates:
        raise ValueError(
            "No SI package matches selected filters for the requested scheme. "
            "Provide explicit paket_id or relax pricing_mode/buyback_mode/provider filters."
        )

    scored: List[Tuple[float, str]] = []
    for pid, paket in candidates:
        if scheme == SCHEME_SI_DOBAVA:
            raw = dobava(
                market_price_mwh=float(smp_market_price_mwh),
                total_consumed_kwh=float(total_consumed_kwh),
                utc_date=utc_date,
                interval_minutes=int(interval_minutes),
                paket=paket,
                pravila=pravila,
                meritve_15min=meritve_15min,
            )
        else:
            consumed, produced = _infer_consumed_produced(
                total_consumed_kwh,
                total_produced_kwh,
                [],
            )
            raw = samooskrba(
                market_price_mwh=float(smp_market_price_mwh),
                total_consumed_kwh=consumed,
                utc_date=utc_date,
                interval_minutes=int(interval_minutes),
                total_produced_kwh=produced,
                paket=paket,
                pravila=pravila,
                meritve_15min=meritve_15min,
            )
        normalized = _normalize_si_result(raw, scheme, apply_ddv=apply_ddv)
        scored.append((float(normalized["variable_price_aud"]), pid))

    scored.sort(key=lambda item: (item[0], item[1]))
    chosen_pid = scored[0][1]
    if len(scored) > 1:
        warnings.append(
            f"Multiple packages matched filters; auto-selected cheapest interval candidate '{chosen_pid}'."
        )
    return chosen_pid


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
    *,
    paket_id: Optional[str],
    pricing_mode: Optional[Any],
    buyback_mode: Optional[Any],
    provider: Optional[str],
    pravila: Any,
    meritve_15min: Optional[bool],
    apply_ddv: bool,
    warnings: List[str],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    from si_obracun import dobava
    from si_paketi import PAKETI

    meritve = _resolve_meritve_15min(interval_minutes, meritve_15min, warnings)

    if float(total_consumed_kwh) < 0.0:
        warnings.append(
            "Supply-only scheme received negative net consumption; data likely includes export/production."
        )

    resolved_paket_id = _select_si_package(
        SCHEME_SI_DOBAVA,
        smp_market_price_mwh,
        total_consumed_kwh,
        utc_date,
        interval_minutes,
        paket_id=paket_id,
        pricing_mode=pricing_mode,
        buyback_mode=buyback_mode,
        provider=provider,
        pravila=pravila,
        meritve_15min=meritve,
        apply_ddv=apply_ddv,
        total_produced_kwh=None,
        warnings=warnings,
    )

    raw = dobava(
        market_price_mwh=float(smp_market_price_mwh),
        total_consumed_kwh=float(total_consumed_kwh),
        utc_date=utc_date,
        interval_minutes=int(interval_minutes),
        paket=PAKETI[resolved_paket_id],
        pravila=pravila,
        meritve_15min=meritve,
    )
    normalized = _normalize_si_result(
        raw,
        SCHEME_SI_DOBAVA,
        apply_ddv=apply_ddv,
    )
    normalized["paket_id"] = resolved_paket_id
    normalized["warnings"] = list(warnings)
    normalized["selection_mode"] = "paket_id" if paket_id is not None else "filtered_auto"
    normalized["meritve_15min_effective"] = meritve
    return normalized, raw


def _resolve_si_samooskrba(
    smp_market_price_mwh: float,
    total_consumed_kwh: float,
    utc_date: datetime.datetime,
    interval_minutes: int,
    *,
    paket_id: Optional[str],
    pricing_mode: Optional[Any],
    buyback_mode: Optional[Any],
    provider: Optional[str],
    pravila: Any,
    meritve_15min: Optional[bool],
    apply_ddv: bool,
    total_produced_kwh: Optional[float],
    warnings: List[str],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    from si_obracun import samooskrba
    from si_paketi import PAKETI, TipOdkupa

    meritve = _resolve_meritve_15min(interval_minutes, meritve_15min, warnings)

    consumed, produced = _infer_consumed_produced(
        total_consumed_kwh,
        total_produced_kwh,
        warnings,
    )

    resolved_paket_id = _select_si_package(
        SCHEME_SI_SAMOOSKRBA,
        smp_market_price_mwh,
        total_consumed_kwh,
        utc_date,
        interval_minutes,
        paket_id=paket_id,
        pricing_mode=pricing_mode,
        buyback_mode=buyback_mode,
        provider=provider,
        pravila=pravila,
        meritve_15min=meritve,
        apply_ddv=apply_ddv,
        total_produced_kwh=produced,
        warnings=warnings,
    )

    paket = PAKETI[resolved_paket_id]
    if paket.tip_odkupa is TipOdkupa.NET_METERING:
        warnings.append(
            "NET_METERING package selected: interval export credit is zero; balancing happens on annual settlement."
        )

    if produced <= 0.0 and consumed > 0.0:
        warnings.append(
            "Self-supply scheme selected but no production was provided/inferred for this interval."
        )

    raw = samooskrba(
        market_price_mwh=float(smp_market_price_mwh),
        total_consumed_kwh=consumed,
        utc_date=utc_date,
        interval_minutes=int(interval_minutes),
        total_produced_kwh=produced,
        paket=paket,
        pravila=pravila,
        meritve_15min=meritve,
    )
    normalized = _normalize_si_result(
        raw,
        SCHEME_SI_SAMOOSKRBA,
        apply_ddv=apply_ddv,
    )
    normalized["paket_id"] = resolved_paket_id
    normalized["total_produced_kwh"] = round(produced, 10)
    normalized["warnings"] = list(warnings)
    normalized["selection_mode"] = "paket_id" if paket_id is not None else "filtered_auto"
    normalized["meritve_15min_effective"] = meritve
    return normalized, raw


def _resolve_single_scheme(
    scheme: str,
    smp_market_price_mwh: float,
    total_consumed_kwh: float,
    utc_date: datetime.datetime,
    interval_minutes: int,
    *,
    paket_id: Optional[str],
    pricing_mode: Optional[Any],
    buyback_mode: Optional[Any],
    provider: Optional[str],
    pravila: Any,
    meritve_15min: Optional[bool],
    apply_ddv: bool,
    total_produced_kwh: Optional[float],
    warnings: List[str],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if scheme in SKIPPED_MULTI_USER_SCHEMES:
        raise ValueError(
            f"Scheme '{scheme}' is skipped for now because it requires multi-user/community context."
        )

    if scheme == SCHEME_AUS_BASE:
        if any(v is not None for v in (pricing_mode, buyback_mode, paket_id, provider)):
            warnings.append(
                "SI package selection filters are ignored for aus_base scheme."
            )
        raw = Aus_Base(
            smp_market_price_mwh=float(smp_market_price_mwh),
            total_consumed_kwh=float(total_consumed_kwh),
            utc_date=utc_date,
            interval_minutes=int(interval_minutes),
        )
        normalized = _normalize_aus_result(raw, SCHEME_AUS_BASE)
        normalized["warnings"] = list(warnings)
        return normalized, raw

    if scheme == SCHEME_SI_DOBAVA:
        return _resolve_si_dobava(
            smp_market_price_mwh,
            total_consumed_kwh,
            utc_date,
            interval_minutes,
            paket_id=paket_id,
            pricing_mode=pricing_mode,
            buyback_mode=buyback_mode,
            provider=provider,
            pravila=pravila,
            meritve_15min=meritve_15min,
            apply_ddv=apply_ddv,
            warnings=warnings,
        )

    if scheme == SCHEME_SI_SAMOOSKRBA:
        return _resolve_si_samooskrba(
            smp_market_price_mwh,
            total_consumed_kwh,
            utc_date,
            interval_minutes,
            paket_id=paket_id,
            pricing_mode=pricing_mode,
            buyback_mode=buyback_mode,
            provider=provider,
            pravila=pravila,
            meritve_15min=meritve_15min,
            apply_ddv=apply_ddv,
            total_produced_kwh=total_produced_kwh,
            warnings=warnings,
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
    paket_id: Optional[str] = None,
    pricing_mode: Optional[Any] = None,
    buyback_mode: Optional[Any] = None,
    provider: Optional[str] = None,
    pravila: Any = None,
    pricing_reference_year: Optional[int] = None,
    meritve_15min: Optional[bool] = None,
    apply_ddv: bool = True,
    total_produced_kwh: Optional[float] = None,
    compare_all: bool = False,
    include_raw: bool = False,
) -> Dict[str, Any]:
    """Unified interval pricing dispatcher for RL/MILP.

    Key features:
    - Select an explicit SI package with `paket_id`, or
    - Select by pricing model via `pricing_mode` (TipCene) and
      `buyback_mode` (TipOdkupa), optionally constrained by `provider`.
    - Non-15-minute intervals are supported. If `meritve_15min` is omitted and
      `interval_minutes != 15`, the dispatcher auto-sets `meritve_15min=False`.
        - `pricing_reference_year` can force SI pricing rules from a chosen year
            (e.g., 2026 or 2027) even when input data timestamps are older.
    - Returns `warnings` when selected configuration is inconsistent with
      provided data context.
    """

    warnings: List[str] = []
    resolved_pravila = _resolve_pravila(
        utc_date,
        pravila,
        pricing_reference_year,
        warnings,
    )

    normalized, raw = _resolve_single_scheme(
        scheme,
        smp_market_price_mwh,
        total_consumed_kwh,
        utc_date,
        interval_minutes,
        paket_id=paket_id,
        pricing_mode=pricing_mode,
        buyback_mode=buyback_mode,
        provider=provider,
        pravila=resolved_pravila,
        meritve_15min=meritve_15min,
        apply_ddv=bool(apply_ddv),
        total_produced_kwh=total_produced_kwh,
        warnings=warnings,
    )

    result: Dict[str, Any] = dict(normalized)
    result["comparison_enabled"] = bool(compare_all)
    result.setdefault("warnings", list(warnings))

    if include_raw:
        result["raw_result"] = raw

    if compare_all:
        comparisons: Dict[str, Any] = {}
        for candidate in SUPPORTED_SCHEMES:
            candidate_warnings: List[str] = []
            candidate_normalized, candidate_raw = _resolve_single_scheme(
                candidate,
                smp_market_price_mwh,
                total_consumed_kwh,
                utc_date,
                interval_minutes,
                paket_id=paket_id,
                pricing_mode=pricing_mode,
                buyback_mode=buyback_mode,
                provider=provider,
                pravila=resolved_pravila,
                meritve_15min=meritve_15min,
                apply_ddv=bool(apply_ddv),
                total_produced_kwh=total_produced_kwh,
                warnings=candidate_warnings,
            )
            if candidate_warnings and "warnings" not in candidate_normalized:
                candidate_normalized["warnings"] = candidate_warnings
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