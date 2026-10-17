"""Fed News Forecasting: does Federal Reserve text help forecast the federal funds rate?

Package layout (docs/REVIEW_MAP.md explains every module and the run order):

    config.py, http.py   paths, fixed design settings, cached HTTP
    collect/             raw sources -> dated tables (ALFRED vintages, markets, FOMC calendar, Fed documents, labels)
    features/            stance scores and the monthly / per-meeting feature panels
    models/              ridge, ordered logit, Taylor rule, the linear walk-forward, LSTM/GRU/TFT building blocks
    evaluation/          metrics, registered tests, Holm, report helpers, figures
    studies/             one folder per study: runners, study-specific features and evaluation

Until 2026-10-17 every module sat directly in this package (``fed_news.walkforward``, ``fed_news.tune``...).
Those names are recorded in the pre-registrations, results files, the audit and git history, so they still work:
``import fed_news.tune`` and ``python -m fed_news.tune forecast`` load the module at its new location. The old
name and the new name refer to the same module object. The mapping is MOVED below.
"""
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys

# old module name -> subpackage it now lives in
_LOCATION = {
    "collect": ["alfred", "corpus", "fomc_calendar", "fraser", "indices", "labels", "market", "meetings",
                "regional_corpus"],
    "features": ["panel", "stance", "text_features", "haiku_index"],
    "models": ["walkforward", "neural", "tft"],  # fed_news.models itself is now the package (re-exports linear.py)
    "evaluation": ["stats_tests", "evaluate", "clark_west", "plots", "robustness"],
    "studies.plan1_linear": ["run_models"],
    "studies.plan2_neural_original": ["neural_walk", "run_neural", "evaluate_h2a", "frozen", "diag_tft"],
    "studies.plan5_event_study": ["events", "event_study", "naive_stance", "ablation", "market_bench",
                                  "evaluate_text"],
    "studies.study2_regional": ["stance2", "study2", "study2_frozen", "evaluate_study2"],
    "studies.study3_decision_text": ["decision_embed", "decision_text", "study3", "evaluate_study3"],
    "studies.study4_haiku_logit": ["haiku_compare", "study4_sensitivity"],
    "studies.study4_networks_v2": ["tune", "verify_v2", "study4_pooled"],
    "studies.study5_horizon": ["study5"],
    "studies.poc": ["stmt_change", "poc_stmt", "chronogpt", "poc_scorer", "poc_finetune", "poc_llm_labels",
                    "poc_haiku_leak", "poc_haiku_memory"],
}
MOVED = {f"fed_news.{m}": f"fed_news.{pkg}.{m}" for pkg, mods in _LOCATION.items() for m in mods}


class _AliasLoader(importlib.machinery.SourceFileLoader):
    """Import: hand back the module already loaded under its new name. ``python -m``: runpy calls get_code(),
    which compiles the module's file, so the old command runs the moved code unchanged."""

    def __init__(self, old: str, new: str, path: str):
        super().__init__(old, path)
        self._new = new

    def create_module(self, spec):
        module = importlib.import_module(self._new)
        self._spec = module.__spec__
        return module

    def exec_module(self, module):
        module.__spec__ = self._spec  # keep the module's own spec; the import system set the alias spec on it


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        new = MOVED.get(fullname)
        if new is None:
            return None
        origin = importlib.util.find_spec(new).origin
        return importlib.util.spec_from_file_location(fullname, origin, loader=_AliasLoader(fullname, new, origin))


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.append(_AliasFinder())
