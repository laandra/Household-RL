# Auto-load repository diagnostics in any Python process started from this workspace.
try:
    import auto_diagnostics

    auto_diagnostics.enable_auto_diagnostics()
except Exception:
    # Keep startup resilient; diagnostics must never block simulation code.
    pass
