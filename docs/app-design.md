# AI Brainwave Decoder app: design

Date: 2026-11-21

## Why

This is the TKS build project. It goes on GitHub and TKS Life, so it has to make
sense to someone who opens a link without knowing the project. The app lets anyone
pick a dataset, a person and a model, train it, and see how accurate it gets and how
fast. A Code view shows the command and the real Python running.

## Goals

- Real training of EEGNet, FBCSP and CSP + LDA from a browser page.
- Three datasets: BCI IV 2a (4 movements, 22 channels), BCI IV 2b (left vs right
  hand, 3 channels), PhysioNet motor imagery (left vs right hand, 64 channels, first
  10 people).
- A time limit. The chart shows accuracy over elapsed time, the best accuracy within
  the limit and when it was first reached.
- A Code view with the run command, a live log and the source, highlighting the
  line that matches the current step.
- A hosted replay page (GitHub Pages) that plays back recorded real runs, clearly
  labelled as recordings.
- Code without comments, plain and consistent.

## Non-goals

- Live EEG headsets or real-time decoding.
- Training in the browser.
- Accounts, saving runs to a server, or running many trainings at once.

## Anti-goals

- Anything that presents a recording or simulation as live training.
- Showing test accuracy in a way that implies it was used to pick the model.

## Approaches considered

1. Local Python app only: real training, but a TKS Life viewer can't try it.
2. Everything in the browser: works from a link, but means rewriting the models in
   JavaScript, probably losing FBCSP, and shipping large data files.
3. Local app plus a hosted replay page. **Chosen:** it gives real training plus a
   link that works, for little extra work.

## How it works

- `app.py` starts a small server using only the Python standard library and opens
  the page.
- The server's API:
  - `GET /api/options` lists datasets and people.
  - `GET /api/source/<file>` returns source from a fixed list of files.
  - `POST /api/runs` starts one training run in a background thread. Only one runs
    at a time.
  - `GET /api/runs/<id>/events` streams the run's events as server-sent events.
  - `POST /api/runs/<id>/stop` stops a run.
- `src/live.py` runs one training job and emits events:
  - `log` lines with an optional code marker;
  - `epoch` with losses and test accuracy;
  - `result` and `done`.
- Splits follow the replication's within-subject fold 0. Fold 0 tests on block 0,
  validates on block 1 and trains on blocks 2 and 3, with channel z-scoring fitted
  on the training trials.
- EEGNet keeps the weights with the lowest validation loss, as in the replication.
- Test accuracy is computed every epoch for display only. The page says the kept
  model is chosen by validation loss.
- `train_model` gains an `on_epoch` callback and a `time_limit`. `FBCSP.fit` gains
  an optional step callback so its stages can be shown live.
- The datasets are loaded through MOABB, with a 4–40 Hz band-pass, 0.5–2.5 s after
  the cue, resampled to 128 Hz (256 samples). Each is cached under
  `data/processed/`. The first use downloads it, and the log says so.
- `web/` holds `index.html`, `style.css` and `app.js`. If the API answers, the page
  runs live. If not, as on GitHub Pages, it loads `replay/runs.json` and plays the
  recorded events at real speed, with a speed control and a "Recorded run" label.
- `python -m src.record` records the replay runs: person 1 for every dataset and
  model, with a 2-minute limit.
- `.github/workflows/pages.yml` publishes `web/` to GitHub Pages.

## Builder's calls

- MIT licence.
- Folder layout and file names.
- Recording person 1 only for the replay page.
- PhysioNet is limited to 10 people.
- The README credits the AI assistant used to build it.
