from __future__ import annotations

import importlib.util
from pathlib import Path

_IMPL_FILE = Path(__file__).resolve().parent / "New pricing functions" / "Pricing_Functions.py"
_SPEC = importlib.util.spec_from_file_location("_household_pricing_impl", _IMPL_FILE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load pricing implementation from {_IMPL_FILE}")

_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

Aus_Base = _MODULE.Aus_Base
calculate_interval_price = _MODULE.calculate_interval_price
list_pricing_schemes = _MODULE.list_pricing_schemes
SUPPORTED_SCHEMES = _MODULE.SUPPORTED_SCHEMES
SKIPPED_MULTI_USER_SCHEMES = _MODULE.SKIPPED_MULTI_USER_SCHEMES
SCHEME_AUS_BASE = _MODULE.SCHEME_AUS_BASE
SCHEME_SI_DOBAVA = _MODULE.SCHEME_SI_DOBAVA
SCHEME_SI_SAMOOSKRBA = _MODULE.SCHEME_SI_SAMOOSKRBA
