# Limitations and deviations

- The exact certificate proves conservation of the displayed invariants; it
  does not independently formalize the completeness classification over all
  `C1` functions.
- The high-dimensional audit uses independent random inputs and squared loss,
  not CIFAR-10 images or a trained ViT. Scale is used only to rule out a
  small-matrix implementation artifact; the universal evidence is algebraic.
- The formal model-scale run is routed to HF `cpu-upgrade` because its runtime
  is uncertain. No GPU is used.

