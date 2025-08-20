# emile_mini/__init__.py
from importlib.metadata import version as _pkg_version, PackageNotFoundError

# Package version (matches pyproject name)
try:
    __version__ = _pkg_version("emile-mini")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Always-present core pieces
from .config import QSEConfig
from .agent import EmileAgent
from .embodied_qse_emile import EmbodiedQSEAgent, EmbodiedEnvironment, SensoriMotorBody

# ----- Optional imports (guarded so import emile_mini never crashes) -----
# Social
try:
    from .social_qse_agent_v2 import SocialQSEAgent, SocialEnvironment, run_social_experiment
    _HAS_SOCIAL = True
except Exception:
    _HAS_SOCIAL = False

# Dynamics runner (coupling)
try:
    from .qse_agent_dynamics_runner import QSEAgentDynamicsRunner
    _HAS_RUNNER = True
except Exception:
    QSEAgentDynamicsRunner = None
    _HAS_RUNNER = False

# Core metrics collector
try:
    from .qse_core_metric_runner_c import QSEMetricsCollector, run_qse_metrics_collection
    _HAS_CORE = True
except Exception:
    QSEMetricsCollector = None
    run_qse_metrics_collection = None
    _HAS_CORE = False

# Analyzer
try:
    from .analyze_qse_dynamics import QSEDynamicsAnalyzer
    _HAS_ANALYZER = True
except Exception:
    QSEDynamicsAnalyzer = None
    _HAS_ANALYZER = False

# Complete navigation system
try:
    from .complete_navigation_system_d import (
        ProactiveEmbodiedQSEAgent,
        ClearPathEnvironment,
        test_complete_navigation,
    )
    _HAS_NAV = True
except Exception:
    ProactiveEmbodiedQSEAgent = None
    ClearPathEnvironment = None
    test_complete_navigation = None
    _HAS_NAV = False

# (Optional) Cognitive battery helpers
try:
    from .cognitive_battery import run_protocol_A, run_protocol_C1, run_protocol_C2_nav
    _HAS_BATTERY = True
except Exception:
    run_protocol_A = None
    run_protocol_C1 = None
    run_protocol_C2_nav = None
    _HAS_BATTERY = False

# ----- Public API -----
__all__ = [
    "__version__",
    "QSEConfig",
    "EmileAgent",
    "EmbodiedQSEAgent",
    "EmbodiedEnvironment",
    "SensoriMotorBody",
]

if _HAS_SOCIAL:
    __all__ += ["SocialQSEAgent", "SocialEnvironment", "run_social_experiment"]

if _HAS_RUNNER:
    __all__ += ["QSEAgentDynamicsRunner"]

if _HAS_CORE:
    __all__ += ["QSEMetricsCollector", "run_qse_metrics_collection"]

if _HAS_ANALYZER:
    __all__ += ["QSEDynamicsAnalyzer"]

if _HAS_NAV:
    __all__ += ["ProactiveEmbodiedQSEAgent", "ClearPathEnvironment", "test_complete_navigation"]

# Optionally export the battery helpers if present
if _HAS_BATTERY:
    __all__ += [
        "run_protocol_A",
        "run_protocol_C1",
        "run_protocol_C2_nav",
    ]
