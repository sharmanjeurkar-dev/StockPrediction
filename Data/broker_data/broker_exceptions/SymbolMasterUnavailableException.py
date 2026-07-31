class SymbolMasterUnavailableException(Exception):
    """Raised when the symbol master can't be fetched and no cache exists to fall back on."""

    pass
