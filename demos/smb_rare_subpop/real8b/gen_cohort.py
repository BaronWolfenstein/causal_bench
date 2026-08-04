"""Hierarchical care-pathway oncology cohort generator for the SMB encoder ablation.

Emits MEDS-format events (subject_id, time, code, table) + per-patient labels.

Design intent
-------------
* Structure is multi-level BY CONSTRUCTION: cancer_type -> receptor subtype ->
  stage -> regimen -> cycle -> per-cycle events. That makes the rare subgroup a
  distinct *branch* of the pathway tree (defensibly separable in embedding space)
  and gives a known-depth positive control for a later STRUCT-gate instrument demo.
* `make_cohort(..., structured=True)` is the positive control (real temporal +
  compositional structure). `structured=False` shuffles each patient's event
  stream in time while preserving the exact multiset of codes -> a flat negative
  control with matched marginals (STRUCT gate should NOT separate it from its own
  time-shuffle).
* All codes are in-domain MSK oncology (ICD-10, CPT, receptor/biomarker labs,
  antineoplastic agents) so the frozen encoder is not fed OOD tokens.

Rare branch = metastatic triple-negative breast cancer (TNBC): distinct at every
level (receptors, stage IV + mets sites, salvage regimens, deeper myelosuppression,
more AEs). Common branch = early-stage ER+/HER2- breast cancer on AC-T + endocrine.

Causal layer
------------
Treatment A = pembrolizumab (immunotherapy) receipt. Propensity is confounded by
indication (stage/subtype/PD-L1). Positivity violation is engineered into the RARE
region: metastatic PD-L1+ patients are almost all treated -> few untreated rare
controls -> ATE not identified there without augmentation. Outcome Y = 12-month
progression with a known, heterogeneous treatment effect (stronger in PD-L1+), so
the true effect is computable for the demo's payoff panel.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

DAY = np.timedelta64(1, "D")
_BASE = np.datetime64("2022-01-03")

# ----------------------------------------------------------------------------- #
# code vocabularies (in-domain oncology)
# ----------------------------------------------------------------------------- #
DX_BREAST = "DIAGNOSIS//C50.911"          # malignant neoplasm, breast, unspecified
METS_SITES = ["C79.51", "C78.7", "C79.31", "C78.00"]  # bone, liver, brain, lung
AE_CODES = ["D70.1", "D64.9", "G62.0", "R11.2", "R50.9"]  # neutropenia, anemia, neuropathy, nausea, fever

# antineoplastic + supportive agents
COMMON_CHEMO = ["doxorubicin", "cyclophosphamide", "paclitaxel"]
COMMON_ENDOCRINE = ["anastrozole", "tamoxifen", "letrozole"]
RARE_CHEMO = ["carboplatin", "gemcitabine", "sacituzumab_govitecan", "eribulin"]
RARE_TARGETED = ["olaparib"]              # if gBRCA
IMMUNO = "pembrolizumab"                  # == treatment A
SUPPORTIVE = ["filgrastim", "ondansetron", "dexamethasone"]

CPT_INFUSION = "96413"
CPT_SURGERY = ["19301", "19303"]          # lumpectomy, mastectomy
CPT_IMAGING = ["77067", "78815", "71260"] # mammography, PET/CT, chest CT
CPT_BIOPSY = "19083"

LAB_NAMES = ["WBC", "ANC", "HGB", "PLT", "CREATININE", "ALT"]

# signal-free, in-vocabulary distractor events shared by all patients. Injected at
# rate `ambiguity` to make emission NON-INJECTIVE (same phenotype -> variable codes),
# the Parley/PCFG-ambiguity realism knob real EHR has and clean synthetic lacks.
DISTRACTORS = ["LAB//GLUCOSE//{v}", "PROCEDURE//CPT//99213", "DIAGNOSIS//Z51.11",
               "MEDICATION//acetaminophen", "MEDICATION//ondansetron",
               "LAB//SODIUM//{v}", "PROCEDURE//CPT//36415", "DIAGNOSIS//R53.83"]


def _lab_value(rng, name, tox, noise_mult=1.0):
    """Draw a lab value; `tox` in [0,1] deepens stress. noise_mult>1 widens the
    distribution so fast/slow trajectories overlap (emission ambiguity)."""
    nm = noise_mult
    if name == "WBC":
        return round(float(np.clip(rng.normal(6.5 - 3.0 * tox, 1.5 * nm), 0.5, 14)), 1)
    if name == "ANC":
        return round(float(np.clip(rng.normal(3.8 - 2.6 * tox, 1.1 * nm), 0.1, 9)), 1)
    if name == "HGB":
        return round(float(np.clip(rng.normal(12.8 - 3.2 * tox, 1.0 * nm), 6.5, 16)), 1)
    if name == "PLT":
        return int(np.clip(rng.normal(240 - 120 * tox, 45 * nm), 20, 450))
    if name == "CREATININE":
        return round(float(np.clip(rng.normal(0.9 + 0.4 * tox, 0.2 * nm), 0.4, 3.5)), 2)
    if name == "ALT":
        return int(np.clip(rng.normal(28 + 45 * tox, 12 * nm), 8, 300))
    return 0


def _emit(events, t, code, table):
    events.append((t, code, table))


def make_patient(rng, is_rare, ambiguity=0.0, force_A=None, u_conf=1.0):
    """Return (events, label) for one patient. events = list[(time, code, table)].

    ambiguity in [0,1] makes the task HARDER: value noise widens (fast/slow overlap),
    the fast_prog effect weakens, trajectories lengthen, and signal-free distractor
    codes are injected (non-injective emission). This stresses whether the encoder
    regimes actually differ under real-EHR-like difficulty vs a clean toy.
    """
    events = []
    t = _BASE + int(rng.integers(0, 300)) * DAY
    noise_mult = 1.0 + 2.5 * ambiguity

    # fast_prog: a DYNAMICS-defined phenotype (orthogonal to the rare vocabulary
    # axis). It manifests ONLY through trajectory shape -- steeper lab decline,
    # earlier/more frequent restaging, more AEs, fewer completed cycles -- with NO
    # unique code token. This is the hard, non-saturating discriminator: a model
    # that captures trajectory dynamics should separate it few-shot better than one
    # that leans on vocabulary. Signal lives in lab-VALUE trends + event cadence.
    fast_prog = int(rng.random() < 0.30)
    ramp = (0.045 + 0.075 * (1 - 0.45 * ambiguity)) if fast_prog else 0.045  # weaker signal when ambiguous
    restage_every = 2 if fast_prog else 3
    # latent frailty U: an UNMEASURED confounder -- affects treatment AND outcome but
    # is emitted as NO code, so the baseline embedding cannot capture it. Drives the
    # QBA panel (residual confounding augmentation can't touch).
    frailty = int(rng.random() < 0.35)

    if is_rare:
        subtype = "TNBC"
        stage = 4
        pdl1 = int(rng.random() < 0.45)             # CPS>=10 positive
        brca = int(rng.random() < 0.20)
        n_cycles = int(rng.integers(4, 10))
        chemo = list(rng.choice(RARE_CHEMO, size=2, replace=False))
        base_tox = 0.55
    else:
        subtype = "ERpos"
        stage = int(rng.integers(1, 4))             # I-III
        pdl1 = int(rng.random() < 0.10)
        brca = int(rng.random() < 0.05)
        n_cycles = int(rng.integers(4, 8))
        chemo = list(COMMON_CHEMO)
        base_tox = 0.20
    if fast_prog:
        n_cycles = max(3, n_cycles - 2)             # early progression -> fewer cycles
    n_cycles += int(rng.integers(0, round(6 * ambiguity) + 1))  # longer trajectories when hard

    # ---- diagnosis / staging / receptors (level 1-2) ----
    _emit(events, t, DX_BREAST, "diagnosis")
    _emit(events, t, f"DIAGNOSIS//STAGE//{stage}", "diagnosis")
    if is_rare:
        for site in rng.choice(METS_SITES, size=int(rng.integers(1, 3)), replace=False):
            _emit(events, t + int(rng.integers(0, 10)) * DAY, f"DIAGNOSIS//{site}", "diagnosis")
        _emit(events, t, "LAB//ER//negative", "measurement")
        _emit(events, t, "LAB//PR//negative", "measurement")
        _emit(events, t, "LAB//HER2//negative", "measurement")
        _emit(events, t, f"LAB//PD-L1//{'positive' if pdl1 else 'negative'}", "measurement")
        if brca:
            _emit(events, t, "LAB//BRCA//mutated", "measurement")
    else:
        _emit(events, t, "LAB//ER//positive", "measurement")
        _emit(events, t, f"LAB//PR//{'positive' if rng.random() < 0.8 else 'negative'}", "measurement")
        _emit(events, t, "LAB//HER2//negative", "measurement")

    # ---- surgery / biopsy (level 3) ----
    if is_rare:
        _emit(events, t + 7 * DAY, f"PROCEDURE//CPT//{CPT_BIOPSY}", "procedure")
    else:
        _emit(events, t + int(rng.integers(10, 30)) * DAY,
              f"PROCEDURE//CPT//{rng.choice(CPT_SURGERY)}", "procedure")

    # ---- treatment assignment A (immunotherapy) ----
    # MODERATE propensity with real overlap (v4 fix: the old deterministic 0.95/0.02
    # made e->1 given the perfectly-encoded confounders, so augmentation could never
    # restore overlap). Confounded by indication (stage/pdl1/rare) AND latent frailty U.
    logit_a = -1.0 + 0.35 * stage + 0.8 * pdl1 + 1.0 * int(is_rare) - 1.3 * u_conf * frailty
    p_a = 1.0 / (1.0 + np.exp(-logit_a))
    # force_A overrides propensity (for balanced-treatment augmentation cohorts);
    # the MEDS stay consistent because the pembrolizumab event is emitted from A below.
    A = int(force_A) if force_A is not None else int(rng.random() < p_a)

    # ---- cycles (level 4) -> per-cycle events (level 5) ----
    ct = t + 30 * DAY
    tx_start = ct                                   # treatment-decision boundary (pre-therapy)
    for c in range(n_cycles):
        tox = float(np.clip(base_tox + ramp * c + rng.normal(0, 0.05), 0, 1))
        _emit(events, ct, f"PROCEDURE//CPT//{CPT_INFUSION}", "procedure")
        for drug in chemo:
            _emit(events, ct, f"MEDICATION//{drug}", "medication")
        if A:
            _emit(events, ct, f"MEDICATION//{IMMUNO}", "medication")
        if is_rare and brca:
            _emit(events, ct + DAY, f"MEDICATION//{RARE_TARGETED[0]}", "medication")
        if not is_rare and c == n_cycles - 1:
            _emit(events, ct + 14 * DAY, f"MEDICATION//{rng.choice(COMMON_ENDOCRINE)}", "medication")
        # labs mid-cycle (noise widens with ambiguity)
        for name in LAB_NAMES:
            _emit(events, ct + 10 * DAY, f"LAB//{name}//{_lab_value(rng, name, tox, noise_mult)}", "measurement")
        # signal-free distractor events (non-injective emission)
        for _ in range(int(rng.poisson(3 * ambiguity))):
            tmpl = rng.choice(DISTRACTORS)
            code = tmpl.format(v=round(float(rng.normal(100, 20)), 0)) if "{v}" in tmpl else tmpl
            tbl = ("measurement" if code.startswith("LAB") else "procedure" if code.startswith("PROCEDURE")
                   else "medication" if code.startswith("MEDICATION") else "diagnosis")
            _emit(events, ct + int(rng.integers(0, 18)) * DAY, code, tbl)
        # supportive care / AEs scale with toxicity
        if rng.random() < 0.3 + 0.5 * tox:
            _emit(events, ct + 3 * DAY, f"MEDICATION//{rng.choice(SUPPORTIVE)}", "medication")
        if rng.random() < 0.15 + 0.55 * tox:
            _emit(events, ct + 6 * DAY, f"DIAGNOSIS//{rng.choice(AE_CODES)}", "diagnosis")
        # restaging imaging cadence (faster for fast progressors)
        if c % restage_every == restage_every - 1:
            _emit(events, ct + 18 * DAY, f"PROCEDURE//CPT//{rng.choice(CPT_IMAGING)}", "procedure")
        ct = ct + int(rng.integers(19, 24)) * DAY

    # ---- outcome Y: 12-month progression (known DGP) ----
    # baseline logit rises with stage + TNBC; immunotherapy lowers it, more so if PD-L1+
    lp = -1.4 + 0.55 * stage + (1.3 if is_rare else 0.0) + 0.9 * u_conf * frailty  # U confounds outcome (off when u_conf=0)
    tau = -(0.6 + 1.1 * pdl1)                       # heterogeneous treatment effect (log-odds)
    lp_treated = lp + tau
    p_prog_factual = 1.0 / (1.0 + np.exp(-(lp + tau * A)))
    Y = int(rng.random() < p_prog_factual)

    # continuous outcome (tumor-burden change) -- LINEAR in confounders + HETEROGENEOUS
    # treatment (much stronger in rare) + U, so linear estimators are well-specified (v6).
    # U contributes only when u_conf=1. tau_cont IS the individual causal effect.
    tau_cont = -1.0 - 2.0 * int(is_rare)
    y_cont = (2.0 + 0.8 * stage + 1.5 * int(is_rare) + 0.5 * pdl1
              + tau_cont * A + 1.2 * u_conf * frailty + rng.normal(0, 1.0))

    label = dict(
        is_rare=int(is_rare), subtype=subtype, stage=stage, pdl1=pdl1, brca=brca,
        fast_prog=fast_prog, frailty=frailty, p_a=round(float(p_a), 4), A=A, Y=Y,
        y_cont=round(float(y_cont), 4), tau_cont=float(tau_cont), n_cycles=n_cycles,
        p_prog_factual=round(float(p_prog_factual), 4),
        p_prog_untreated=round(float(1 / (1 + np.exp(-lp))), 4),
        p_prog_treated=round(float(1 / (1 + np.exp(-lp_treated))), 4),
        end_time=(ct + 30 * DAY),
        baseline_time=tx_start,                     # pre-treatment anchor for a VALID confounder embedding
    )
    return events, label


def make_cohort(n, rare_frac=0.15, seed=0, structured=True, ambiguity=0.0, u_conf=1.0):
    """Build a cohort of `n` patients.

    structured=True  -> real temporal order (positive control).
    structured=False -> per-patient event *times* shuffled while preserving the
                        exact multiset of codes (flat negative control, matched
                        marginals) for the STRUCT-gate instrument demo.
    Returns (meds_df, labels_df).
    """
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for i in range(n):
        sid = f"P{i:05d}"
        is_rare = rng.random() < rare_frac
        events, label = make_patient(rng, is_rare, ambiguity=ambiguity, u_conf=u_conf)
        times = [e[0] for e in events]
        if not structured:
            perm = rng.permutation(len(times))
            times = [times[k] for k in perm]        # scramble time<->event binding
        for (t0, code, table), t in zip(events, times):
            rows.append((sid, t, code, table))
        label["subject_id"] = sid
        labels.append(label)
    meds = pd.DataFrame(rows, columns=["subject_id", "time", "code", "table"])
    meds["time"] = pd.to_datetime(meds["time"])
    meds = meds.sort_values(["subject_id", "time"]).reset_index(drop=True)
    lab = pd.DataFrame(labels)
    lab["end_time"] = pd.to_datetime(lab["end_time"])
    lab["baseline_time"] = pd.to_datetime(lab["baseline_time"])
    return meds, lab


def make_augmentation_cohort(n, seed=100, ambiguity=0.0, u_conf=1.0):
    """Rare-only patients with BALANCED forced treatment (A~0.5) and TRUE outcomes
    from the known DGP. Fills the rare region with overlap-restoring controls+treated
    whose Y is structurally faithful (not a fitted model's guess) -- the honest,
    toy-faithful augmentation. In production the FE (trust-validated) generates these
    rare points; here we use the structural DGP so the causal panel has ground truth.
    """
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for i in range(n):
        sid = f"AUG{i:05d}"
        events, label = make_patient(rng, True, ambiguity=ambiguity, force_A=int(i % 2 == 0), u_conf=u_conf)
        for (t0, code, table) in events:
            rows.append((sid, t0, code, table))
        label["subject_id"] = sid
        labels.append(label)
    meds = pd.DataFrame(rows, columns=["subject_id", "time", "code", "table"])
    meds["time"] = pd.to_datetime(meds["time"])
    meds = meds.sort_values(["subject_id", "time"]).reset_index(drop=True)
    lab = pd.DataFrame(labels); lab["end_time"] = pd.to_datetime(lab["end_time"])
    lab["baseline_time"] = pd.to_datetime(lab["baseline_time"])
    return meds, lab


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    meds, lab = make_cohort(n, seed=0)
    print("meds rows:", len(meds), "| patients:", lab.subject_id.nunique())
    print("rare frac:", round(lab.is_rare.mean(), 3),
          "| A rate rare/common:",
          round(lab[lab.is_rare == 1].A.mean(), 3), "/", round(lab[lab.is_rare == 0].A.mean(), 3))
    print("Y rate rare/common:",
          round(lab[lab.is_rare == 1].Y.mean(), 3), "/", round(lab[lab.is_rare == 0].Y.mean(), 3))
    print("unique codes:", meds.code.nunique(), "| events/patient:", round(len(meds) / n, 1))
    # naive true ATE on progression (risk difference, full population, known DGP)
    ate = (lab.p_prog_treated - lab.p_prog_untreated).mean()
    print("true ATE (risk diff, progression):", round(float(ate), 4))
