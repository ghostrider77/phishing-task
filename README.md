## Setup

The dependencies and the project itself is `uv`-managed. The `Makefile` contains couple of useful command shortcuts.

Create the environment by running the `make sync` command.

## Baseline classification model

- Run one of the notebooks: `make notebook` starts Jupyter in the browser. Alternatively, one can run those from VSCode.
  - `01-data-and-model-exploration` contains my work including my notes and remarks
  - `02-model-training` is the training code copied from the first notebook without the experimental and throwaway things, sor one can repeat training quickly.
  - `03-prediction` is the inference notebook where one can test how the `predict` function works
  - `04-run-model` is the notebook that runs inference on `test.csv` and creates `predictions.csv`
- `predict.py`, the requested file for inference, it can be called from the notebook above.


The experimentation and model training notebook expect the presence of the `data` folder with the training files.
The trained model is saved to the `model` folder. Each run of the notebooks above will override the previous model.
