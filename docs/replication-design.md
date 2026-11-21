# EEG Imagined-Movement Decoder: Design

Date: 2026-11-09

## Goal

Build and train our own model that decodes imagined movements from real EEG
recordings, for a TKS replicate project. It reproduces the motor-imagery result
from the EEGNet paper (Lawhern et al., 2018), then answers one follow-up question
of our own.

**Success means:**

1. EEGNet beats random guessing (25%) by a clear margin on every evaluation.
2. Our results show the same pattern as the paper: EEGNet is about level with
   FBCSP when trained on each person, and slightly ahead when tested on new people.
3. Our accuracies are compared with the values in the paper's Figures 3 and 5.
   The paper's text gives no accuracy numbers for this dataset, so the values are
   read off those figures, and the report says so.
4. Charts, a short demo and a write-up are ready to present.

## Data

BCI Competition IV 2a, downloaded through MOABB as `BNCI2014_001`.

- 9 people, 22 EEG channels and 4 classes: left hand, right hand, feet, tongue
- 2 recording sessions per person, 576 trials per person in total
- Session 1 is the competition training set and session 2 is the test set

## Preprocessing (from the paper)

1. Band-pass filter to 4-40 Hz.
2. Resample from 250 Hz to 128 Hz.
3. Cut each trial to 0.5-2.5 s after the cue, giving 256 samples.

The paper follows the pipeline in Schirrmeister et al. and does not list every
setting. Anything we have to choose ourselves goes under "Differences from the
paper" in the report.

## Models

**CSP + LDA:** a quick check that the data pipeline works. It is not part of the
comparison with the paper.

**FBCSP (the paper's baseline):** 9 filter bands across 4-40 Hz, 2 CSP filter pairs
per band, and elastic-net logistic regression (alpha = 0.95, with lambda chosen on
validation data).

**EEGNet-8,2 in PyTorch:**

- F1 = 8, D = 2, temporal kernel length 32 samples
- Max-norm limit of 1 on spatial filters and 0.25 on the output layer
- Dropout 0.5 for within-subject and 0.25 for cross-subject
- Adam with default settings, cross-entropy loss, no class weights
- Up to 500 epochs, keeping the weights with the lowest validation loss
- Batch size 32 (our choice, since the paper doesn't state one)

## Evaluation

**Within-subject (replication):** four-fold blockwise cross-validation for each
person, with 2 blocks to train, 1 to validate and 1 to test. The paper doesn't define
a block for this dataset, so we split each person's 576 trials into 4 equal
consecutive blocks of 144, and the report lists this as our interpretation.

**Cross-subject (replication):** for each test person, randomly pick 5 of the others
to train on and use the remaining 3 for validation, using their session 1 data, then
test on the test person's session 2. Repeat 10 times per person for 90 runs, and
report the mean and standard error.

**Calibration experiment (our question):** how much of a new person's data does it
take to close the gap between the cross-subject and within-subject results?

1. Train on the other 8 people.
2. Fine-tune on 0, 12, 24, 48 or 96 trials from the new person's session 1, with the
   same number of trials per class.
3. Test on the new person's session 2.
4. Repeat for all 9 people, with 5 random trial selections per person at each size.

This goes beyond the procedure we are replicating, but it is a project extension and
not a claim of new science. The write-up has to acknowledge that related work exists.

## Keeping results honest

- The test data is never used for training, fine-tuning, early stopping or choosing
  settings.
- Anything fitted to data, such as CSP filters, is fitted on training trials only.
- Random seeds are fixed, and each run's configuration and results are saved.
- Results are reported as they come out, including where they differ from the paper.

## Code layout

```
requirements.txt
src/data.py        load and preprocess the dataset
src/splits.py      within-subject, cross-subject and calibration splits
src/eegnet.py      EEGNet model
src/baselines.py   CSP + LDA and FBCSP
src/train.py       training loop with early stopping
src/evaluate.py    run an experiment and save results as CSV
src/plots.py       turn saved results into charts
tests/             tests for the items listed under "Testing"
results/           saved results and charts
```

Runs on Python 3.12 in a virtual environment, using PyTorch, MNE, MOABB and
scikit-learn. Training uses the Mac's GPU (MPS) and can fall back to the CPU. The
target machine is an Apple M5 with 16 GB of RAM.

## Testing

- EEGNet-8,2 has exactly 1,716 parameters on this data, matching the paper's Table 3.
- Splits never put the same trial in both training and test, and a cross-subject test
  person never appears in training or validation.
- Calibration selections contain the requested number of trials, balanced by class.
- Preprocessed trials have the expected shape: 22 channels by 256 samples.
- A small EEGNet run can overfit a handful of trials, which shows the training loop
  learns.

## Build order

Each stage leaves something usable, so the project can stop after any of them.

1. Environment and data: the dataset loads, and preprocessing passes its tests.
2. Baselines: CSP + LDA, then FBCSP.
3. Replication: EEGNet within-subject and cross-subject, compared with the paper. This
   is the minimum complete project.
4. Calibration experiment.
5. Presentation: charts, a demo of predictions on unseen trials, and the write-up.

## Out of scope

- Collecting our own EEG or running a live headset demo
- Comparing other architectures
- Other datasets or decoding tasks
