# emile_mini/__init__.py
from importlib.metadata import version as _pkg_version, PackageNotFoundError

# Dynamic package version (matches the name in pyproject.toml)
try:
    __version__ = _pkg_version("emile-mini")
except PackageNotFoundError:
    __version__ = "0.0.0"  # fallback when running from a local tree

from .config import QSEConfig
from .embodied_qse_emile import EmbodiedQSEAgent, EmbodiedEnvironment, SensoriMotorBody
from .social_qse_agent_v2 import SocialQSEAgent, SocialEnvironment, run_social_experiment
from .agent import EmileAgent

__all__ = [
    "__version__",
    "QSEConfig",
    "EmbodiedQSEAgent",
    "EmbodiedEnvironment",
    "EmileAgent",
    "SensoriMotorBody",
    "SocialQSEAgent",
    "SocialEnvironment",
    "run_social_experiment",
]
