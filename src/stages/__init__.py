"""
Pipeline stages for Capacity-SEM analysis.

Each stage module handles a specific phase of the analysis pipeline:
- s00_ingest: Load QPR and external data
- s00b_standardize: Standardize QPR data with fixed denominators (NEW)
- s01_link: Construct analysis panel
- s01b_features: Build features from standardized data (NEW)
- s01c_program_types: Aggregate program type features (NEW)
- s02_features: Compute indicators and features (legacy)
- s03_estimation: Fit SEM models
- s03c_kaifa_external_replication: Provisional external Kaifa SEM reconstruction
- s03d_kaifa_recovered_analysis: Recovered Kaifa notebook analysis port
- s04_robustness: Robustness checks
- s05_figures: Generate figures
- s07_capacity_summary: Capacity summary reporting
"""

from . import s00_ingest
from . import s00b_standardize
from . import s01_link
from . import s01b_features
from . import s01c_program_types
from . import s02_features
from . import s03_estimation
from . import s03c_kaifa_external_replication
from . import s03d_kaifa_recovered_analysis
from . import s04_robustness
from . import s05_figures
from . import s07_capacity_summary

__all__ = [
    's00_ingest',
    's00b_standardize',
    's01_link',
    's01b_features',
    's01c_program_types',
    's02_features',
    's03_estimation',
    's03c_kaifa_external_replication',
    's03d_kaifa_recovered_analysis',
    's04_robustness',
    's05_figures',
    's07_capacity_summary',
]
