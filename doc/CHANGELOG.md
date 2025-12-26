# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Project restructuring to follow Research Project Management template
- New pipeline CLI (`src/pipeline.py`) with subcommand pattern
- Stage-based module organization (s00-s05)
- Configuration module (`src/config.py`)
- Quarto manuscript system (`manuscript_quarto/`)
- Comprehensive documentation (`doc/`)
- This changelog

### Changed
- Renamed directories:
  - `data/raw/` → `data_raw/`
  - `data/processed/` → `data_work/`
  - `docs/` → `doc/`
  - `outputs/figures/` → `figures/`
- Reorganized source code into stages pattern
- Unified imports through config module

### Removed
- Empty directories (`spatial/`, `visualization/`, `scripts/`)

---

## [0.1.0] - Initial Development

### Added
- Core SEM analysis modules:
  - `capacity_sem.data.loader` - QPR data loading
  - `capacity_sem.data.external_data` - External covariates
  - `capacity_sem.features.timeliness` - Duration metrics
  - `capacity_sem.features.experience_indicators` - Experience measures
  - `capacity_sem.features.program_stratification` - Activity classification
  - `capacity_sem.models.sem_specifications` - 24 model specifications
  - `capacity_sem.models.sem_fitting` - Model estimation
  - `capacity_sem.models.sem_diagnostics` - Fit evaluation
- Embedded population, severity, and employment datasets

---

## Format

Based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

Categories:
- **Added** for new features
- **Changed** for changes in existing functionality
- **Deprecated** for soon-to-be removed features
- **Removed** for now removed features
- **Fixed** for any bug fixes
- **Security** for vulnerabilities
