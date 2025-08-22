# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Fixed

## [0.5.0] - 2025-08-22

### Added

- **Enhanced Navigation System**: ProactiveEmbodiedQSEAgent and ClearPathEnvironment with robust heading and cue following capabilities
- **Cognitive Battery Runner**: Comprehensive Protocol A, C1, and C2 implementations with PPO comparison hooks
- **Navigation Reporting**: Quadrant and density sweep analysis with CSV/JSON export capabilities
- **Context Module Improvements**: Hysteresis and dwell-time mechanisms to prevent thrashing in dynamic environments
- **Memory Module Enhancements**: Structured episodic storage with JSON-safe sanitization, comprehensive stats and search APIs
- **Embodied QSE Environment**: Enhanced energy balance mechanics, improved forage systems, and richer object interactions
- **QSE Metrics Collector**: Advanced emergent time analysis, rupture episode detection, quantum entropy measurement with summary and gates
- **CLI Subcommands**: Complete command suite including social, maze, nav-demo, nav-ppo-train, nav-compare, nav-report, and battery commands
- **Multimodal Input Support**: Optional multimodal inputs (text/image/audio) with deterministic fusion (disabled by default; enable via MULTIMODAL_ENABLED)
- **Dynamic Attention Policies**: ModalityAttentionPolicy and dynamic attention control with `EmileAgent.set_attention_mode()`

### Changed

- Improved stability and robustness across all cognitive modules
- Enhanced logging and debugging capabilities throughout the system
- Extended agent APIs to support multimodal input processing while maintaining backward compatibility
- Optimized performance for large-scale navigation and learning tasks

### Technical Notes

- **Backward Compatible**: All changes maintain existing API compatibility
- **Deterministic Behavior**: Enhanced reproducibility across experiments
- **Modular Architecture**: Clear separation of concerns with optional component loading
- **Comprehensive Testing**: Enhanced validation and testing frameworks

## [0.3.0] - Previous Release

### Added

- Bidirectional QSE validation
- Autopoiesis measurement framework
- Comprehensive RL evaluation suite
- Core émile-mini functionality with embodied and social learning capabilities
