# Future Development

## Purpose

This document outlines practical next-stage development directions for the **Endometrial Infection Classification App** in its current research-to-deployment form. It is intended to help future contributors, collaborators, and reviewers extend the system in ways that preserve research rigor, interpretability, and translational value.

The current project already provides:

- binary classification of endometrial scan images into `infected` and `uninfected`
- a TensorFlow inference pipeline exposed through FastAPI and Gradio
- downloadable public test images for demonstration and app validation
- explanatory overlays and prediction metadata for inference transparency
- an EDA Lab that summarizes dataset curation, split strategy, optimization traces, and held-out evaluation
- leakage-control safeguards based on exact duplicate removal and grouped perceptual-similarity splitting

The next stage of development should focus on making the solution:

- clinically richer
- methodologically stronger
- more interpretable
- more reusable in real institutional settings

## Current project scope

At the time of writing, the project supports:

- training from the original infected and uninfected archives
- exact duplicate removal and near-duplicate-aware grouped splitting
- a productionized browser interface for image upload and prediction
- confidence and class-probability reporting
- lightweight attention-style explainability artifacts for each inference run
- internal held-out evaluation with documented safeguards
- deployment on a zero-cost infrastructure stack

This is a strong foundation, but it should still be understood as a **research-support classifier**, not a final clinical system.

## Future enrichment directions

### 1. Study-level and patient-level partitioning

**Possible direction**

Move beyond image-level and similarity-group-level splitting by introducing patient, scan-session, study, or acquisition-level partitioning where metadata are available.

**Why it matters**

This is one of the most important next steps for scientific rigor. If multiple related images come from the same patient or acquisition session, random or image-level splits can still overestimate generalization even after near-duplicate controls.

**Implementation considerations**

- capture or reconstruct study identifiers during data ingestion
- store patient or session metadata in the audit manifests
- enforce split assignment at the highest trustworthy grouping level
- report clearly which grouping level was used for every experiment

### 2. External validation on independent data

**Possible direction**

Evaluate the model on data collected from a different source, device pipeline, institution, or time period.

**Why it matters**

Internal held-out performance is useful, but external validation is the strongest test of whether the model generalizes beyond the current archive.

**Implementation considerations**

- curate a separate external dataset with the same label structure
- report distribution shifts such as image resolution, noise, and acquisition differences
- compare internal versus external metrics directly
- document any drop in calibration, recall, or specificity

### 3. Region-of-interest localization and segmentation support

**Possible direction**

Introduce an explicit region-of-interest pipeline so the model focuses on the relevant endometrial structures rather than the entire frame.

**Why it matters**

A classifier can achieve strong performance by exploiting background or acquisition artifacts. ROI localization or segmentation can reduce shortcut learning and make the system more anatomically grounded.

**Implementation considerations**

- define whether the future objective is bounding-box localization, weak localization, or pixel-level segmentation
- compare full-frame classification against ROI-based classification
- evaluate whether ROI guidance improves robustness and interpretability
- preserve the current model as a baseline for controlled comparison

### 4. Stronger explainability and explanation validation

**Possible direction**

Upgrade the current saliency-based explanation workflow to include more robust interpretability methods and structured review by domain experts.

**Why it matters**

Current heatmaps are helpful as interpretability aids, but they are not sufficient evidence that the model is reasoning from clinically meaningful regions.

**Implementation considerations**

- compare saliency, Grad-CAM-style methods, occlusion sensitivity, and perturbation testing
- measure explanation stability across similar scans
- involve expert review where possible to assess whether highlighted regions are clinically plausible
- separate user-facing explanation wording from technical explanation outputs

### 5. Confidence calibration and uncertainty estimation

**Possible direction**

Improve the meaning of model confidence so that displayed scores correspond more closely to real predictive reliability.

**Why it matters**

Raw classifier probabilities can be overconfident. In a medical setting, misinterpreted confidence can be more harmful than low confidence.

**Implementation considerations**

- evaluate calibration using reliability curves and expected calibration error
- test post-hoc methods such as temperature scaling
- distinguish class probability from decision certainty in the UI
- introduce abstention or review thresholds for uncertain predictions

### 6. Multi-class or severity-aware modeling

**Possible direction**

Expand the label space beyond binary classification if future data support richer diagnostic groupings.

**Why it matters**

Binary classification is a useful first step, but real clinical and research workflows may benefit from distinguishing severity levels, subtypes, or adjacent pathological categories.

**Implementation considerations**

- only expand label space when annotation quality is strong enough
- preserve a well-tested binary baseline for comparison
- evaluate class imbalance carefully before moving to finer labels
- redesign the UI and reporting logic for multi-class probability interpretation

### 7. Multimodal clinical context integration

**Possible direction**

Combine image data with supporting metadata such as age, symptoms, laboratory findings, acquisition mode, or clinician notes where ethically and legally appropriate.

**Why it matters**

Many medical decisions are not made from imaging alone. A multimodal system may produce more reliable and context-aware outputs than an image-only model.

**Implementation considerations**

- define a clear schema for structured metadata
- build separate ingestion and validation rules for non-image inputs
- compare image-only and multimodal baselines
- ensure sensitive clinical variables are handled carefully and transparently

### 8. Robustness testing and distribution-shift analysis

**Possible direction**

Stress-test the model against noisy, compressed, low-contrast, differently sized, or partially degraded images.

**Why it matters**

A model that performs well on clean internal data may fail under realistic usage conditions. Robustness testing helps reveal fragility early.

**Implementation considerations**

- create controlled perturbation test suites
- benchmark sensitivity to blur, brightness, cropping, and compression
- identify classes of images where performance degrades fastest
- surface robustness findings in the EDA or documentation

### 9. Fairness, bias, and subgroup analysis

**Possible direction**

Assess whether the system performs differently across demographic or acquisition-related subgroups if such metadata become available.

**Why it matters**

Medical AI systems can appear strong overall while performing unevenly for certain populations or devices.

**Implementation considerations**

- define subgroups in a statistically responsible way
- avoid subgroup claims when sample sizes are too small
- report confidence intervals, not only point estimates
- document any imbalance or underrepresentation that limits conclusions

### 10. Stronger benchmarking and experiment tracking

**Possible direction**

Formalize the research workflow so multiple architectures, training configurations, and evaluation protocols can be compared consistently.

**Why it matters**

As the project grows, it becomes increasingly important to separate one-off experiments from traceable, reproducible evidence.

**Implementation considerations**

- add structured experiment manifests
- compare architectures such as MobileNetV2, EfficientNet, ConvNeXt, and compact ViT variants
- track data version, split strategy, augmentation regime, and calibration setting for every run
- store benchmark summaries in a stable, reviewable format

### 11. Clinician-in-the-loop review workflow

**Possible direction**

Evolve the app from a pure classifier interface into a review-support tool that can capture expert feedback.

**Why it matters**

Research systems become more valuable when they support iterative review, disagreement analysis, and annotation refinement rather than only single-shot predictions.

**Implementation considerations**

- add fields for expert agreement or disagreement
- capture review comments without overwriting raw model outputs
- support exporting reviewed cases for future study
- preserve traceability between the original prediction and the reviewed assessment

### 12. Better public deployment hardening

**Possible direction**

Prepare the app for more sustained public use and reproducible releases.

**Why it matters**

The current zero-cost stack is a strong MVP, but heavier usage, larger artifacts, and stricter reproducibility requirements will eventually need stronger release discipline.

**Implementation considerations**

- pin model and artifact versions more explicitly
- add release notes for changes in metrics or evaluation protocol
- profile startup time and memory use
- define what should happen when the model file, audit manifests, or explanation pipeline are unavailable

### 13. Privacy, governance, and ethical-use guardrails

**Possible direction**

Strengthen documentation and implementation around privacy, appropriate use, and research governance.

**Why it matters**

Medical-image projects require more than technical accuracy. Responsible deployment depends on clear use boundaries and data handling practices.

**Implementation considerations**

- document what kind of data should and should not be uploaded
- state clearly that the system is not a standalone diagnosis tool
- define retention behavior for uploaded images in hosted environments
- add governance notes for institutions that want to adapt the project

## Recommended development principles

- Keep the training, evaluation, inference, and UI layers modular.
- Preserve the current audit-friendly design and expand it where possible.
- Prefer research transparency over impressive-looking metrics.
- Treat interpretability as an aid, not proof of clinical reasoning.
- Preserve exact documentation of any change that affects data curation, model behavior, or evaluation claims.
- Do not present internal held-out results as external validation.
- Prefer additions that increase trustworthiness and usability, not just architectural complexity.

## Contributor expectations

Future contributors should:

- document the motivation for any substantial data, model, or UI change
- preserve backward compatibility where practical, or explain clearly when it is intentionally broken
- extend tests when behavior changes
- update audit artifacts, README notes, and evaluation wording when methodology changes
- avoid silently changing the evidence standard used to report model quality

## Acknowledgment requirement

Any future work, derivative implementation, enrichment effort, publication, presentation, or adaptation based materially on this project should acknowledge:

- **Okon Prince**
- **Dr. Obi Cajetan**

Where appropriate, that acknowledgment should appear in the repository, documentation, publication, presentation, or deployed product notes.

## Suggested immediate next steps

The most valuable short-term priorities are:

1. Introduce study-level or patient-level partitioning if metadata can be recovered.
2. Run external validation on an independent image set.
3. Add calibration analysis and uncertainty-aware decision thresholds.
4. Expand the current explanation workflow with stronger interpretability methods.
5. Benchmark at least one stronger architecture against the current MobileNetV2 baseline.
6. Add a clinician-review capture workflow for research feedback loops.

## Closing note

The project is already more than a notebook demonstration. It is a meaningful bridge between medical imaging research and a real deployed interface. Future development should preserve that strength by improving not only model performance, but also evidence quality, interpretability, reproducibility, and practical usefulness.
