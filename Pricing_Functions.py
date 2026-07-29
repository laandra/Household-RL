"""Root-level import shim.

`Environment.py` / `Basic_Functions.py` import pricing functions from the
repo root (`from Pricing_Functions import calculate_interval_price`), but the
actual implementation lives in `New pricing functions/Pricing_Functions.py`
(a directory name with a space, which can't be imported as a normal Python
package). This shim loads that file directly via `importlib` and re-exports
its public surface.
"""
import importlib.util
from pathlib import Path

_IMPL_FILE = Path(__file__).resolve().parent / "New pricing functions" / "Pricing_Functions.py"
_SPEC = importlib.util.spec_from_file_location("_household_pricing_impl", _IMPL_FILE)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

Aus_Base = _MODULE.Aus_Base
calculate_interval_price = _MODULE.calculate_interval_price
list_pricing_schemes = _MODULE.list_pricing_schemes
resolve_block_for_datetime = _MODULE.resolve_block_for_datetime
resolve_reset_window_id = _MODULE.resolve_reset_window_id
compute_prorated_fixed_charge_eur = _MODULE.compute_prorated_fixed_charge_eur

PRIVZETO_REFERENCNO_LETO = _MODULE.PRIVZETO_REFERENCNO_LETO

SUPPORTED_SCHEMES = _MODULE.SUPPORTED_SCHEMES
SKIPPED_MULTI_USER_SCHEMES = _MODULE.SKIPPED_MULTI_USER_SCHEMES
SCHEME_AUS_BASE = _MODULE.SCHEME_AUS_BASE
SCHEME_SI_DOBAVA = _MODULE.SCHEME_SI_DOBAVA
SCHEME_SI_SAMOOSKRBA = _MODULE.SCHEME_SI_SAMOOSKRBA
