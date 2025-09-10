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
	•	Primary outputs: per-epoch labels (e.g., 4-s epochs common in rodents) and a continuous hypnogram.
	•	Secondary outputs: uncertainty per epoch, quality flags (artifact, missing signal).
	•	Success criteria (Phase 1):
	•	Macro F1 ≥ 0.82, per-class F1(REM) ≥ 0.75, Cohen’s κ ≥ 0.80 on held-out animals.
	•	Latency ≤ 10 ms/epoch on a single GPU for batch inference; ≤ 50 ms on CPU.

2) Data & labeling
	•	Signals: 1–2 EEG channels (512 Hz target), EMG (1 channel, 512–1k Hz optional), optional accelerometer.
	•	Epoching: 4-s non-overlapping windows (adjustable via config).
	•	Labels: expert scored Wake / NREM / REM; allow unknown for ambiguous.
	•	Splits: animal-wise split (no subject leakage). Suggested: 70/15/15 (train/val/test) with ≥ 5 animals in test.
	•	Metadata: animal ID, strain, age, recording start UTC, lights-on/off times.

3) Preprocessing
	•	Filtering: EEG band-pass 0.5–100 Hz; notch at mains (50/60 Hz). EMG 10–300 Hz.
	•	Re-referencing: common average or mastoid equivalent if available.
	•	Resampling: to 512 Hz; z-score per-recording (fit on train only).
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
