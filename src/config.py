import os

class PathConfig:
    def __init__(self, results_base: str, splits_base: str, processed_base: str):
        self.results_base = results_base
        self.splits = splits_base
        self.processed = processed_base
        self.saved_models = os.path.join(results_base, "saved_models")
        self.tables = os.path.join(results_base, "tables")
        self.figures = os.path.join(results_base, "figures")

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __contains__(self, key: str):
        return hasattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

def get_paths(quick: bool = False) -> PathConfig:
    results_base = os.getenv("RESULTS_BASE", "results_quick" if quick else "results")
    splits_base = os.getenv("SPLITS_BASE", "data/splits_quick" if quick else "data/splits")
    is_quick = quick or ("quick" in results_base.lower()) or ("quick" in splits_base.lower()) or (os.getenv("QUICK", "").lower() in ["1", "true", "yes"])
    processed_base = os.getenv("PROCESSED_BASE", "data/processed_quick" if is_quick else "data/processed")
    return PathConfig(results_base, splits_base, processed_base)
