# Sleep Scoring ML Project


# Build and MVP 
### Steps to MVP
1. locate a small dataset to work with
2. build a basic preprocessing pipeline
3. build a basic model (flexibility is nice here)
4. build a training loop
5. build tests
6. build analysis code


### Develop new architectures
Experiment 


### Optimize for performance

### Optimize for speed

Keep envorinment up to date and clean


Sleep-Stage Classifier (Rodent EEG) — Project Spec

1) Goal & scope
	•	Objective: Train a neural network to classify Wake / NREM (SWS) / REM from rodent EEG (+optional EMG/accel) in near-real time.
	•	Primary outputs: per-epoch labels (e.g., 10-s epochs common in rodents) and a continuous hypnogram.
	•	Secondary outputs: uncertainty per epoch, quality flags (artifact, missing signal).
	•	Success criteria (Phase 1):
	•	Macro F1 ≥ 0.82, per-class F1(REM) ≥ 0.75, Cohen’s κ ≥ 0.80 on held-out animals.
	•	Latency ≤ 10 ms/epoch on a single GPU for batch inference; ≤ 50 ms on CPU.

2) Data & labeling
	•	Signals: 1 EEG channel (500 Hz target), EMG (1 channel, 500 Hz optional)
	•	Epoching: 50s centered around 10s non-overlapping windows (adjustable via config).
	•	Labels: expert scored Wake / NREM / REM; allow unknown for ambiguous.
	•	Splits: animal-wise split (no subject leakage). Suggested: 70/15/15 (train/val/test) with ≥ 5 animals in test.
	•	Metadata: animal ID, strain, age, recording start UTC, lights-on/off times.

3) Preprocessing
	•	Artifact handling: flatline, saturations, motion bursts → flag epochs; either drop or feed artifact flag channel.
	•	Synchronization: align EEG/EMG/accel timestamps; enforce constant epoch boundaries.

4) Feature strategy

Two tracks (you can run both; they’re complementary):

A. End-to-end time-series models
	•	Model sees raw (or lightly filtered) signals + optional engineered side-channels (artifact flag, light/dark).
	•	Temporal context via sequence models.

B. Hybrid features
	•	Add per-epoch features: log-power bands (δ 0.5–4, θ 6–9, σ 10–15, β 15–30, γ 30–80), θ/δ, EMG RMS, spectral entropy, line-noise ratio.
	•	Concatenate to learned embeddings; helps with small-data regimes.

5) Modeling options (recommended stack)

Baseline (simple & strong):
	•	1D CNN + BiLSTM over each epoch with ±3 epochs of context (i.e., 7-epoch window).
	•	Final per-epoch head with class-weighted cross-entropy (or focal loss).

Sequence-first (best temporal consistency):
	•	CNN front-end → Transformer/Conformer/BiGRU across long windows (5–15 min) to model transitions.
	•	Add HMM/CRF smoothing on logits for continuity (penalize impossible jumps like NREM→Wake→REM in 1 step).

Modern alternative:
	•	State-Space Model (SSM) / S4D / RWKV-TS backbone for long-range temporal patterns at low compute.

Optional self-supervised pretraining (if ≥50 hrs unlabeled):
	•	BYOL-A / wav2vec-style masked reconstruction on EEG; then fine-tune head for stages. Big gains on REM.

6) Losses & class imbalance
	•	Start with weighted CE (weights ∝ 1/√freq).
	•	Compare focal loss (γ=2) for REM sensitivity.
	•	If you treat stages as ordered (Wake ↔ NREM ↔ REM), optionally try ordinal loss head and report κ.

7) Evaluation
	•	Metrics: macro F1, per-class F1, precision/recall, Cohen’s κ, AUROC per class, confusion matrix.
	•	Structure-aware metrics: median bout length error, % impossible transitions, hypnogram edit distance.
	•	Unit of CV: leave-one-animal-out CV (LOAO) plus fixed hold-out site/day.
	•	Calibration: reliability curves + ECE per class.
	•	Error analysis dashboards: time-aligned hypnogram, spectrogram, EMG envelope, predicted probs.

8) Post-processing (temporal smoothing)
	•	Viterbi decoding with transition prior (estimated from training).
	•	3-state transition matrix; add small self-loop prior to reduce spurious flips.
	•	Option: median filter (3–5 epochs) as ablation.

9) Data augmentation
	•	Time-shift within epoch (±100 ms), small amplitude jitter, Gaussian noise (SNR ≥ 20 dB), random notch width, SpecAugment on log-mels, EMG dropout (simulate electrode loss). Keep physiologically plausible.

10) Experiments (Phase-1 grid)
	•	Context: {no context, ±3 epochs, 10-min window}
	•	Backbone: {CNN+BiLSTM, CNN+Transformer-4L, SSM-tiny}
	•	Loss: {weighted CE, focal}
	•	EMG: {none, as channel, EMG-only ablation}
	•	Smoothing: {none, HMM, median-5}
Track all with MLflow; compare by macro F1 & κ on animal-held-out test.

11) MLOps & reproducibility
	•	Data versioning: DVC (raw → interim → processed), checksums per recording.
	•	Tracking: MLflow for params/metrics/artifacts; save confusion matrices and hypnograms.
	•	Configs: Hydra or YAML; no hard-coded paths.
	•	Testing: unit tests for preprocessing (filter freq response, epoch alignment), leakage tests (ensure no animal overlap).
	•	Containers: Dockerfile with CUDA tag; deterministic seeds; save environment (conda env export).
	•	CI: run lint, unit tests, 1 tiny smoke-train.

12) Deliverables
	•	Trained checkpoints (top-3 by val κ), ONNX export, reproducible scripts.
	•	Inference CLI & REST microservice (FastAPI): .edf/.mat → CSV labels + JSON probs.
	•	Research report: dataset summary, protocol, metrics, ablations, common failure modes.

13) Risks & mitigations
	•	Label noise: consensus scoring or SNCL (soft-label training); robust loss.
	•	Class imbalance (REM rare): oversample REM epochs, focal loss, hard-example mining.
	•	Site variability: per-recording normalization, light/dark covariate, domain-adversarial head if multi-site.
	•	Artifacts: explicit artifact detector channel; drop vs. learn.

⸻

14) Reference implementation skeleton

Repo layout

Notes:

Subsampling a 24 interval to balance the classes leads to biased sampling. 
REM is the minority class and represents a small fraction of a 24 hour period. 
Subsampling the majority class to match REM counts produces a sampling from across the 24 hour period. Very few awake and slow wave sleep bouts are immediately adjacent to each other
Retaining all of the REM bouts means that there is a higher proportion of REM bouts that are adjecent to other REM bouts

Counts of adjacent bouts 
R    168
S     15
A      8

## Testing TODO List

### Unit Tests Needed

The following functions currently lack unit tests and should be prioritized for test coverage:

#### utils/sleep_utils.py
- [ ] `verify_edf()` - EDF file validation
- [ ] `extract_sleep_stages_from_edf()` - Sleep stage extraction from EDF
- [ ] `save_sleep_stages_datatable()` - Save sleep stages to file
- [ ] `subsample_datatable()` - File-based subsampling
- [ ] `extract_activity_signal()` - Activity signal extraction
- [ ] `extract_sleep_context()` - Sleep context extraction

#### utils/config_utils.py
- [x] `load_config_from_hydra_dump()` - Load Hydra dumped config
- [x] `load_config_with_compose_api()` - Load config with Compose API
- [x] `unpack_config()` - Unpack nested config

#### utils/hardware_utils.py
- [x] `resolve_device()` - Device resolution logic

#### utils/io_utils.py
- [x] `list_files_with_extension()` - File listing utility
- [x] `ensure_dir()` - Directory creation utility
- [x] `save_to_subject_hdf5()` - HDF5 save operations
- [x] `load_from_subject_hdf5()` - HDF5 load operations
- [x] `aggregate_csvs()` - CSV aggregation

#### utils/labels.py
- [x] `normalize_stage()` - Sleep stage normalization
- [x] `labels_from_annotations()` - Label creation from annotations
- [x] `labels_from_csv()` - Label creation from CSV

#### utils/dataset_utils.py
- [x] `split_train_test()` - Train/test splitting

#### dataio/edf_reader.py
- [ ] `_pick_channels()` - Channel selection logic
- [ ] `load_edf()` - EDF loading functionality

#### cv.py (Cross-validation functions)
- [ ] `kfold_split()` - K-fold cross-validation
- [ ] `stratified_kfold_split()` - Stratified K-fold
- [ ] `time_series_split()` - Time series splitting
- [ ] `group_kfold_split()` - Group K-fold
- [ ] `stratified_group_kfold_split()` - Stratified group K-fold
- [ ] `cross_validate_model()` - Model cross-validation
- [ ] `evaluate_cv_results()` - CV results evaluation
- [ ] `nested_cross_validation()` - Nested cross-validation
- [ ] `_get_scoring_func()` - Scoring function helper
- [ ] `train_test_split_temporal()` - Temporal train/test split
- [ ] `rolling_window_validation()` - Rolling window validation

#### models/cnn_bilstm.py
- [ ] `_make_conv_stack()` - Convolution stack builder
- [ ] `build_model()` - Model builder function

#### models/implicit_cnn.py
- [ ] `build_model()` - Model builder function

### Integration Tests Needed

#### Complex workflows that need integration testing:
- [ ] End-to-end EDF processing pipeline
- [ ] Model training workflow
- [ ] Cross-validation pipeline
- [ ] Data loading and preprocessing
- [ ] Config loading and validation

### Current Test Coverage Status
- **Total functions identified**: 39
- **Functions with unit tests**: 6 (sleep_utils.py functions only)
- **Coverage gap**: 33 functions need unit tests
- **Priority**: Start with utility functions as they're used throughout the codebase
