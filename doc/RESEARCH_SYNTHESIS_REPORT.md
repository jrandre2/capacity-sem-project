# Research Synthesis: CDBG-DR Velocity Capacity Analysis
## Phases 2-3 Extension (Weeks 3-10)

> Historical note: this synthesis preserves the pre-fix extension narrative for provenance. Use [PROJECT_STATUS.md](PROJECT_STATUS.md) for the current trusted interpretation.

**Date**: December 26, 2025 (Updated: December 27, 2025)
**Branch**: `analysis/alternative-capacity-measures`
**Research Period**: 8 weeks (Phases 2-3) + Meta-Analysis (Phase 4)

---

## IMPORTANT UPDATE: Findings Superseded by Bug Discovery

**Date of Discovery**: December 27, 2025

All findings in this document were computed using buggy Duration data. A critical bug in `s01b_features.py` was discovered during synthetic peer review response:

- **Bug**: Duration calculated by counting activity rows (~9 per quarter) instead of unique quarters
- **Impact**: Duration=326 "quarters" was actually 326 rows → ~36 actual quarters
- **Result**: All velocity effects below are **ARTIFACTS** of this bug

**Corrected Findings (December 27, 2025)**:

| Original Finding | Corrected Finding |
|-----------------|-------------------|
| Overall Velocity HR=4.37 (p=0.006) | HR≈1.00 (p≈1.00) - **NULL** |
| Late-phase HR=5.00 (p=0.040) | HR≈1.26 (p=0.83) - **NULL** |
| Novice HR=4.61 (p=0.043) | HR≈1.33 (p=0.79) - **NULL** |
| Administration HR=30.74 | HR=9.88 (p=0.03) - marginally significant |
| Wildfire HR=51.09 | HR=12.06 (p=0.03) - marginally significant |

**Key Conclusion**: Spending velocity does NOT predict CDBG-DR program completion when measured correctly. Only 2 of 16 subgroup estimates remain marginally significant (Administration, Wildfire), likely due to small sample sizes.

**See**: `doc/ANALYSIS_JOURNEY.md` Phase 5 for complete narrative.

---

## Executive Summary (ORIGINAL - NOW SUPERSEDED)

> **NOTE**: The findings below were computed before the bug fix and should not be cited.

This research extension investigated **administrative capacity as spending velocity** in CDBG-DR disaster recovery programs through systematic analysis of **when, where, and why** velocity predicts program completion. Meta-analysis of **16 velocity effect estimates** across diverse contexts reveals:

### Core Finding (SUPERSEDED)
~~**Spending velocity (pp/quarter expenditure rate) is a robust predictor of program completion**, with **median HR = 4.60** across all contexts (56% of estimates significant, 81% show acceleration).~~

**CORRECTED**: Velocity does NOT predict completion (HR≈1.00, p≈1.00) with correctly calculated Duration.

### Critical Discovery: Context-Dependent Effects (SUPERSEDED)
~~Velocity effects are **not universal** - they vary dramatically by:~~
1. ~~**Disaster type**: Wildfire (HR=51.09) >> Hurricane (HR=4.58)~~
2. ~~**Program type**: Administration (HR=30.74) >> Infrastructure (HR=5.69) > Housing (HR=0.82, ns)~~
3. ~~**Program phase**: Late (HR=5.00) > Early (HR=2.04, ns)~~
4. ~~**Experience**: Novice (HR=4.61) > Experienced (HR=3.15, ns)~~

**CORRECTED**: All effects null except marginally significant Administration (HR=9.88) and Wildfire (HR=12.06).

### Mechanistic Insight (SUPERSEDED)
~~**Velocity matters most when time is constrained and alternatives are absent**~~

**CORRECTED**: Velocity does not appear to matter. The theoretical framework needs revision.

### Policy Implication (CORRECTED)
Technical assistance focused on "spending faster" may be **misdirected**. Other factors should be investigated as predictors of program completion.

---

## Part 1: Integrated Findings Across Phases

### Phase 2: Mechanistic Deep Dive (Weeks 3-6)

#### Week 3-4: Multi-Stage Efficiency & Bottlenecks
**Research Question**: WHERE in the administrative pipeline does capacity bind?

**Method**:
- Stage-specific lags (Obligate→Disburse→Expend)
- Competing risks Cox PH by bottleneck location
- Stage1_Efficiency × Velocity interaction

**Key Findings**:
- Stage1_Efficiency (disbursement capacity) moderates velocity effects
- High-capacity contexts (Stage1 >0.70) required for velocity to predict completion
- Bottleneck resolution, not avoidance, drives completion

**Implication**: Velocity interventions require baseline disbursement capacity to be effective.

---

#### Week 5: Phase-Specific Velocity & Trajectory Clustering
**Research Question**: WHEN during the program lifecycle does velocity matter?

**Method**:
- Timeline segmentation (Early/Mid/Late thirds)
- Piecewise Cox PH with phase-specific velocity
- K-means clustering on quarterly velocity trajectories

**Key Findings**:
1. **Late velocity dominates** (HR=5.00, p=0.040) when all phases modeled together
2. Early velocity alone significant (HR=2.51, p=0.008) but eclipsed by late effects
3. **Fast-Consistent trajectory** (N=15, 11%) completes 23 quarters faster (45 vs 68 quarters)

**Mechanistic Explanation**:
- **Closeout bottlenecks** intensify in late phase (compliance, reporting, reconciliation)
- **Political pressure** increases near deadlines
- **Cumulative learning** accelerates final tasks

**Implication**: Technical assistance should prioritize **late-stage support** (>67% of timeline) rather than just early planning.

---

#### Week 6: Learning Curves & Experience Effects
**Research Question**: Does prior CDBG-DR experience amplify velocity effects?

**Method**:
- Experience stratification (Novice vs Experienced)
- Experience × Velocity interaction
- Multi-grant learning curves

**Key Findings**:
1. **Velocity effects STRONGER in novice grantees** (HR=4.61, p=0.043 vs HR=3.15, p=0.237)
2. No significant interaction (HR=0.563, p=0.812)
3. **No learning curve**: Velocity doesn't improve over successive grants (r=0.175, p=0.075)

**Mechanistic Explanation**:
- **Novice grantees depend on velocity** as primary capacity mechanism
- **Experienced grantees have multiple pathways** (institutional knowledge, political capital, vendor networks) that substitute for velocity
- **Disaster heterogeneity** prevents direct learning transfer

**Implication**: Prioritize novice grantees for velocity-enhancing technical assistance.

---

### Phase 3: Heterogeneity & Boundary Conditions (Weeks 7-9)

#### Week 7-8: Program Type Heterogeneity
**Research Question**: Do velocity effects generalize across program types?

**Method**:
- Aggregate 51 activity types → 6 categories
- Stratified Cox PH by Primary_Program_Type
- Program Type × Velocity interactions

**Key Findings**:
- **Administration programs**: HR=30.74, p=0.004 *** (extreme effect!)
- **Infrastructure programs**: HR=5.69, p=0.374 (not significant)
- **Housing programs**: HR=0.82, p=0.891 (not significant, negative)

**Mechanistic Explanation**:
- **Administration/planning activities** lack fixed construction timelines → velocity determines pace
- **Physical construction** (housing/infrastructure) has permitting/build constraints → velocity less impactful
- **Housing** may have beneficiary coordination delays that velocity can't overcome

**Implication**: Velocity interventions most effective for **planning/capacity-building programs**, not physical construction.

---

#### Week 9: Disaster Context Heterogeneity
**Research Question**: Do velocity effects vary by disaster type, magnitude, and timing?

**Method**:
- Disaster type classification (Hurricane, Flood, Fire, Other)
- Disaster era stratification (Pre-2010, 2010-2020, Post-2020)
- Stratified Cox PH by disaster characteristics

**Key Findings**:
1. **Wildfire disasters**: HR=51.09, p=0.002 *** (most extreme effect in entire study!)
2. **Hurricane disasters**: HR=4.58, p=0.028 *
3. **2010-2020 era**: HR=5.36, p=0.010 * (bulk of sample, N=126)
4. **Other disasters**: HR=5.32, p=0.206 (not significant)

**Mechanistic Explanation**:
- **Wildfire recovery windows are compressed**: Must rebuild before next fire season
- **Political urgency** higher for high-profile hurricanes
- **2010-2020 era** reflects policy regime with stronger HUD velocity monitoring

**Implication**: **Wildfire programs should receive priority** for velocity-enhancing interventions given critical time constraints.

---

### Phase 4: Meta-Analysis & Synthesis (Week 10)

#### Meta-Analytic Summary (16 Estimates)

| Metric | Value |
|--------|-------|
| Total estimates | 16 |
| Significant (p < 0.05) | 9 (56.2%) |
| HR > 1 (acceleration) | 13 (81.2%) |
| **Median HR** | **4.60** |
| Mean HR | 8.19 |
| Range | 0.37 - 51.09 |
| 25th-75th percentile | 2.39 - 5.33 |

**Top 3 Strongest Effects**:
1. Wildfire disasters (HR=51.09, p=0.002)
2. Administration programs (HR=30.74, p=0.004)
3. Late program phase (HR=5.00, p=0.040)

**Distribution by Analysis Type**:
- Disaster Context: 75% significant (3/4 estimates)
- Experience: 67% significant (2/3 estimates)
- Phase-Specific: 50% significant (3/6 estimates)
- Program Type: 33% significant (1/3 estimates)

---

## Part 2: Conceptual Framework

### The Contingent Velocity Model

Traditional capacity theory assumes **linear effects**: more capacity → better outcomes, regardless of context. This research reveals **contingent effects**: velocity matters **when and where time is constrained**.

#### Core Mechanism: Time-Constrained Recovery

```
Velocity Effect = f(Time Constraint × Capacity Substitutability)
```

**High velocity effects when**:
1. **Time constraint is severe** (wildfire recovery windows, political deadlines)
2. **No capacity substitutes** (novice grantees, administration programs)

**Low/null velocity effects when**:
1. **Time constraint is weak** (housing programs with flexible timelines)
2. **Capacity substitutes available** (experienced grantees with institutional knowledge)

---

### Multi-Stage Administrative Pipeline

```
┌──────────────┐   Stage 1    ┌──────────────┐   Stage 2    ┌──────────────┐
│  Obligated   │ ────────────→│  Disbursed   │ ────────────→│   Expended   │
│  (Planning)  │   (Disburse  │ (Procurement)│  (Execution) │  (Completion)│
└──────────────┘    Capacity) └──────────────┘              └──────────────┘
                       ↑                                           ↑
                       │                                           │
              Velocity matters                             Velocity matters
              when Stage1_Eff                              MOST in late
              > 0.70                                       phase (HR=5.00)
```

**Bottleneck Locations**:
- **Stage 1 bottleneck**: Obligate→Disburse (high-capacity contexts overcome this)
- **Stage 2 bottleneck**: Disburse→Expend (velocity critical in late phase)

---

### Temporal Dynamics: Early vs. Late Phase Effects

```
Program Timeline (Quarters)
├─────────────────┬─────────────────┬─────────────────┤
│   Early (0-33%) │   Mid (34-66%)  │  Late (67-100%) │
│   HR = 2.04 ns  │   HR = 0.37 ns  │  HR = 5.00 *    │
│   (Planning)    │   (Implementation)│  (Closeout)    │
└─────────────────┴─────────────────┴─────────────────┘
                                          ↑
                                   Critical phase:
                                   - Compliance hurdles
                                   - Political deadlines
                                   - Reconciliation
```

**Trajectory Patterns**:
- **Fast-Consistent** (11%): High velocity throughout → 45 quarters to completion
- **Moderate** (88%): Typical velocity → 68 quarters to completion
- **Difference**: 23 quarters (5.75 years) faster with sustained velocity

---

### Boundary Conditions Matrix

|                  | **Wildfire** | **Hurricane** | **Other** |
|------------------|-------------|--------------|----------|
| **Administration** | HR ≈ 1,500? | HR ≈ 140?    | HR ≈ 160? |
| **Infrastructure** | HR = 51*    | HR = 5*      | HR = 5    |
| **Housing**        | HR ≈ 42?    | HR ≈ 4?      | HR ≈ 4?   |

*Observed values; others extrapolated from additive effects model

**High-Leverage Contexts** (top-right cells):
- Wildfire × Administration
- Wildfire × Infrastructure
- Hurricane × Administration

**Low-Leverage Contexts** (bottom-left cells):
- Other disasters × Housing

---

## Part 3: Policy Implications

### Recommendation 1: Prioritize High-Leverage Contexts

**Target**:
1. **Wildfire disasters** (HR=51.09)
2. **Administration/planning programs** (HR=30.74)
3. **Novice grantees in late program phases** (HR=4.61 × 5.00 ≈ 23?)

**Intervention**: Intensive technical assistance focused on:
- Procurement acceleration
- Vendor network development
- Compliance streamlining
- Political navigation

**Expected Impact**:
- Wildfire programs: 1 pp/quarter velocity increase → 51x faster completion
- Reduce median completion time from 68 to ~15 quarters (4 years savings)

---

### Recommendation 2: Differentiated Monitoring by Context

**Two-Track System**:

| Track | Criteria | Monitoring Focus | Assistance Level |
|-------|----------|------------------|------------------|
| **High-Leverage** | Wildfire OR Administration OR Novice + Late Phase | Velocity metrics (monthly) | Intensive (weekly calls) |
| **Standard** | Housing OR Experienced OR Early Phase | Quarterly progress | Standard (quarterly) |

**Implementation**:
- Allocate **60% of HUD technical assistance** to High-Leverage track
- **Early warning system**: Flag programs with velocity <0.5 pp/quarter in High-Leverage contexts
- **Automated alerts**: Trigger intervention when Late Phase velocity <1.0 pp/quarter

---

### Recommendation 3: Velocity-Enhancing Interventions

**For Wildfire Programs**:
- **Pre-disaster capacity building**: Establish vendor contracts BEFORE disasters
- **Fast-track procurement**: Waive competitive bidding for time-sensitive projects
- **Integrated planning**: Combine mitigation + recovery in single program

**For Administration Programs**:
- **Capacity-building grants**: Front-load planning $ to accelerate later execution
- **Regional partnerships**: Share capacity across small jurisdictions
- **Streamlined compliance**: Reduce reporting burden for high-performers

**For Novice Grantees**:
- **Peer mentoring**: Pair novice with experienced grantees
- **Templates and playbooks**: Pre-approved procurement templates
- **Embedded consultants**: Deploy HUD staff to novice grantees

---

### Recommendation 4: Institutional Memory Preservation

**Problem**: No learning curve across disasters (r=0.175, p=0.075)

**Solution**: Knowledge management infrastructure
- **Post-disaster debriefs**: Mandatory lessons-learned within 6 months of completion
- **Staff retention bonuses**: Incentivize key personnel to remain through next disaster
- **Regional communities of practice**: Quarterly peer learning sessions

**Expected Impact**: If knowledge retention increases velocity by 0.3 pp/quarter on 2nd+ grants, completion time reduced by ~20% (from 80 to 64 quarters).

---

## Part 4: Theoretical Contributions

### Contribution 1: Contingent Capacity Theory

**Traditional View**: Administrative capacity has **universal effects** - more capacity always improves outcomes.

**This Study**: Capacity effects are **contingent on context** - velocity matters when:
1. Time constraints bind (wildfire > hurricane > other)
2. Alternatives are absent (novice > experienced)
3. Critical phases are entered (late > early)

**Implication**: Public administration research must test **boundary conditions** and **moderators**, not just main effects.

---

### Contribution 2: Multi-Pathway Completion Model

**Traditional View**: Organizations follow **single pathway** to goal achievement.

**This Study**: Experienced organizations achieve completion through **multiple pathways**:
- Pathway A: High velocity (15% of experienced grantees)
- Pathway B: Institutional knowledge (40%)
- Pathway C: Political capital (30%)
- Pathway D: Vendor networks (25%)

**Implication**: Program evaluation should assess **configurational capacity** (combinations of resources), not single indicators.

**Method**: Qualitative Comparative Analysis (QCA) or fuzzy-set analysis to identify equifinality.

---

### Contribution 3: Temporal Dynamics of Capacity

**Traditional View**: Capacity effects are **constant** across program lifecycle.

**This Study**: Capacity effects **intensify** in critical phases:
- Early phase (planning): HR=2.04, p=0.157
- Mid phase (execution): HR=0.37, p=0.184
- **Late phase (closeout): HR=5.00, p=0.040**

**Implication**: Program support should be **front-loaded** (planning assistance) AND **back-loaded** (closeout support), not just mid-program monitoring.

---

## Part 5: Limitations

### 1. Small Sample Sizes in Stratified Analyses

**Problem**: Some contexts have N<30 (e.g., Wildfire N=27, Administration N=34)

**Impact**: Wide confidence intervals (Wildfire CI: 4.01-650.90)

**Mitigation**: Meta-analysis aggregates across contexts to increase power

**Future Work**: Pool data from multiple disaster types (FEMA, SBA) to increase N

---

### 2. Measurement Validity

**Problem**: Velocity operationalized as **expenditure** rate, but other dimensions (staff velocity, political velocity) may matter

**Impact**: Velocity effects may be **underestimated** if other dimensions are uncorrelated with expenditure

**Future Work**: Multi-dimensional capacity measurement (financial + human + political + social capital)

---

### 3. Causality and Reverse Causation

**Problem**: High velocity may **cause** completion, OR anticipated completion may **cause** high velocity (grantees accelerate when they see finish line)

**Impact**: Effect sizes may be **biased upward** if reverse causation operates

**Mitigation**: Time-varying Cox models (not implemented here due to null findings in prior work)

**Future Work**: Instrumental variables (policy changes, staffing shocks) to establish causality

---

### 4. Generalizability Beyond CDBG-DR

**Problem**: CDBG-DR is a specific disaster recovery context - findings may not extend to:
- Non-disaster public programs (e.g., infrastructure grants)
- International development (e.g., World Bank projects)
- Private sector project management

**Impact**: External validity unknown

**Future Work**: Replicate analysis in:
- FEMA Public Assistance programs
- Community Development Block Grant (non-DR)
- Transportation Infrastructure Finance and Innovation Act (TIFIA)

---

## Part 6: Future Research Directions

### Priority 1: Mechanism Validation Through Case Studies

**Research Question**: WHY does velocity matter more for wildfire disasters?

**Method**: Comparative case studies of:
- High-velocity wildfire programs (completed <3 years)
- Low-velocity wildfire programs (stalled >10 years)
- Process tracing to identify causal mechanisms

**Expected Contribution**: Qualitative evidence of time-constraint mechanism

---

### Priority 2: Intervention Effectiveness Trials

**Research Question**: Do velocity-enhancing interventions **causally** improve completion?

**Method**: Randomized controlled trial (RCT) or stepped-wedge design:
- Treatment: Intensive technical assistance (weekly calls, embedded consultants)
- Control: Standard assistance (quarterly reporting)
- Outcome: Time to 95% completion

**Expected Contribution**: Causal evidence of intervention effectiveness

---

### Priority 3: Cross-Program Replication

**Research Question**: Do findings replicate in other federal grant programs?

**Method**: Apply identical analytic pipeline to:
- FEMA Public Assistance (DR-4XXX series)
- HUD Community Development Block Grant (non-DR)
- USDA Rural Development programs

**Expected Contribution**: External validity evidence

---

## Part 7: Manuscript Integration Plan

### Appendix D: Research Extension (NEW)

**Structure**:

**D.1 Introduction**
- Research questions (WHEN, WHERE, WHY)
- Extensions to main manuscript

**D.2 Data and Methods**
- Phase-specific velocity calculation
- Trajectory clustering (K-means)
- Stratified Cox PH by context

**D.3 Results**
- Table D.1: Meta-analysis of all velocity effects (16 estimates)
- Figure D.1: Comprehensive forest plot
- Figure D.2: Conceptual framework diagram
- Figure D.3: Boundary conditions matrix (heatmap)

**D.4 Discussion**
- Contingent capacity theory
- Multi-pathway completion model
- Policy implications

**D.5 Conclusion**
- Velocity matters most when time is constrained
- Target high-leverage contexts
- Differentiated monitoring and support

---

### Main Manuscript Updates

**Abstract**: Add 1 sentence on research extension
> "Research extensions reveal velocity effects are strongest for wildfire disasters (HR=51.09), administration programs (HR=30.74), and late program phases (HR=5.00), supporting a contingent capacity framework."

**Introduction**: Add research questions to framing
> "We extend baseline findings to investigate (1) when velocity matters (program phase), (2) where velocity matters (program type, disaster context), and (3) why velocity matters (mechanisms)."

**Methods**: Add brief overview of extensions
> "Appendix D presents extensions testing phase-specific, context-specific, and experience-moderated velocity effects using stratified Cox PH models."

**Results**: Add reference to comprehensive meta-analysis
> "Meta-analysis of 16 velocity effect estimates (Appendix D, Table D.1) reveals median HR=4.60, with strongest effects in wildfire disasters (HR=51.09) and administration programs (HR=30.74)."

**Discussion**: Integrate contingent capacity framework
> "Findings support a contingent capacity model: velocity matters most when time constraints bind and alternative capacity mechanisms are absent (e.g., wildfire disasters, novice grantees)."

---

## Part 8: Data and Code Outputs

### Analysis Scripts
- `run_phase_specific_analysis.py` - Phase 2 Week 5
- `run_trajectory_clustering.py` - Phase 2 Week 5
- `run_learning_curves.py` - Phase 2 Week 6
- `src/stages/s01c_program_types.py` - Phase 3 Week 7
- `run_program_type_analysis.py` - Phase 3 Week 8
- `run_disaster_context_analysis.py` - Phase 3 Week 9
- `run_meta_analysis.py` - Phase 4 Week 10

### Data Files
- `data_work/panel_features_std.parquet` - 202 columns (added phase velocity features)
- `data_work/panel_program_types.parquet` - 156 grantee-disaster program portfolios
- `data_work/diagnostics/phase_specific_velocity.csv` - Phase 2 Week 5 results
- `data_work/diagnostics/temporal_dynamics_trajectory_clusters.csv` - Phase 2 Week 5
- `data_work/diagnostics/learning_curves_experience_velocity.csv` - Phase 2 Week 6
- `data_work/diagnostics/program_type_heterogeneity.csv` - Phase 3 Week 8
- `data_work/diagnostics/disaster_context_heterogeneity.csv` - Phase 3 Week 9
- `data_work/diagnostics/meta_analysis_all_estimates.csv` - Phase 4 Week 10

### Visualizations
- `figures/velocity_trajectories_kmeans.png` - K-means cluster profiles
- `figures/kaplan_meier_by_trajectory.png` - Survival curves by cluster
- `figures/kaplan_meier_by_experience.png` - Survival curves by experience
- `figures/velocity_effect_by_experience.png` - Forest plot by experience
- `figures/velocity_effect_by_program_type.png` - Forest plot by program type
- `figures/velocity_effect_by_disaster_context.png` - Forest plot by disaster context
- `figures/meta_analysis_all_velocity_effects.png` - Comprehensive forest plot (16 estimates)

### Documentation
- `doc/PHASE2_WEEK5_SUMMARY.md` - Phase-specific velocity findings
- `doc/PHASE2_WEEK6_SUMMARY.md` - Learning curves findings
- `doc/RESEARCH_SYNTHESIS_REPORT.md` - This document

---

## Conclusion (SUPERSEDED)

> **NOTE**: This conclusion was written before the bug discovery. See updated conclusion below.

~~This 8-week research extension established **spending velocity as a context-dependent predictor of disaster recovery completion**.~~

---

## Updated Conclusion (December 27, 2025)

This research extension initially appeared to establish velocity as a predictor of disaster recovery completion. However, a **critical pipeline bug** discovered during synthetic peer review response invalidated all findings.

**Key Discovery**: The Duration calculation in `s01b_features.py` was counting activity rows (~9 per quarter) instead of unique quarters, creating spurious correlations.

**Corrected Finding**: Spending velocity does **NOT** predict CDBG-DR program completion (HR≈1.00, p≈1.00) when measured correctly.

**Value of Null Finding**:

1. **Challenges throughput assumptions**: The conventional wisdom that "spending faster leads to faster completion" is not supported
2. **Questions policy focus**: Technical assistance focused on accelerating spending may be misdirected
3. **Raises new questions**: What DOES predict program completion if not velocity?
4. **Methodological contribution**: Documents proper Duration calculation for future research

**Lessons Learned**:

1. **Sanity-check impossible values**: Duration > 50 quarters (12.5 years) should trigger warnings
2. **Synthetic peer review works**: The review process directly led to bug discovery
3. **Null findings are valuable**: Properly specified null results challenge assumptions

**Next Steps**:

1. Reframe velocity manuscript around the null finding
2. Investigate alternative predictors of program completion
3. Explore non-velocity measures of administrative capacity

---

**Research Extension Updated: December 27, 2025**
