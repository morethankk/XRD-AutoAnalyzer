# PyTorch Refactor Overview

This document summarises the migration of the XRD-AutoAnalyzer neural
network pipeline from TensorFlow/Keras to PyTorch.

## Architecture

- **Model structure** – The convolutional architecture closely mirrors the
  original design. Separate branches are retained for XRD and PDF analysis
  with identical convolution and pooling configurations. Dense layers use
  the same number of neurons (3,100 and 1,200).
- **Dropout** – A custom `CustomDropout` layer applies dropout regardless of
  the module's training state to facilitate Monte Carlo dropout during
  inference.

## Training Pipeline

- Training uses standard PyTorch loops with `torch.utils.data.DataLoader`
  for batching. The loss function switches to `CrossEntropyLoss` which
  operates on raw logits, providing numerical stability.
- Optimisation continues to use the Adam optimiser. Dataset splitting and
  shuffling logic is preserved from the previous implementation.

## Performance Considerations

- The flatten dimension for the fully-connected head is computed
  dynamically using a dummy forward pass, ensuring the model adapts to the
  convolutional stack without manual calculation.
- Evaluation utilities compute accuracy using vectorised operations to
  minimise Python overhead.

## Testing Strategy

- Syntax of all modified modules is validated via `python -m py_compile`.
- Due to the absence of a PyTorch installation in the execution
  environment, full unit tests could not be executed. Users should run the
  existing example scripts after installing PyTorch to validate end-to-end
  training and inference.

