# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2025-01-28

### Added

- **Multimodal Input Support**: Added optional multimodal inputs (text/image/audio) with deterministic fusion (disabled by default; enable via MULTIMODAL_ENABLED).
- **Dynamic Attention Policies**: Added ModalityAttentionPolicy and `EmileAgent.set_attention_mode("listening"/"reading"/"looking", steps=N)` for dynamic attention control.
- **Enhanced Symbolic Reasoning**: SymbolicReasoner can now accept `modality_features` and add a bounded curvature term into Σ when enabled.
- **Embodied Multimodal Path**: Fuses vision + proprioception into Σ when enabled for embodied agents.
- **New Configuration Flags**: 
  - `MULTIMODAL_ENABLED` (default: false) - Controls multimodal feature availability
  - `MODALITY_INFLUENCE_SCALE` - Controls the influence scale of multimodal features

### Changed

- Extended agent APIs to support multimodal input processing while maintaining backward compatibility
- Enhanced embodied agents with automatic multimodal feature extraction when enabled

### Technical Notes

- **Backward Compatible**: Default behavior unchanged when `MULTIMODAL_ENABLED=False`
- **Deterministic Fusion**: Uses early fusion with fixed projection for consistent results
- **Bounded Influence**: Multimodal features have controlled impact on core QSE dynamics
- **Lightweight Adapters**: Dependency-free encoders for text, image, and audio processing

## [0.3.0] - Previous Release

### Added

- Bidirectional QSE validation
- Autopoiesis measurement framework
- Comprehensive RL evaluation suite
- Core émile-mini functionality with embodied and social learning capabilities
