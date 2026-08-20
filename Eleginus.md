# Eleginus Changelog

## 2026-08-20

- Replaced the independent Policy and Value pair with a single incremental Value network. Eleginus now relies on principal-variation search rather than a learned Policy for move selection.
- Reduced the king-conditioned Value accumulator to 160 channels and replaced the king-conditioned relation table with a shared 768-class piece-square relation table. The complete float32 checkpoint remains below 20 MiB.
- Retained the two-perspective king-conditioned feature accumulator, material-dependent Value buckets and incrementally maintained relation state.
- Changed move ordering to use transposition-table moves, tactical scores, killer moves and history scores across iterative-deepening passes.
- Changed the supervised HDF5 schema to contain only encoded states and Value targets. Existing Eleginus datasets and checkpoints must be regenerated for this architecture.

## 2026-08-20 - ISCA Value Field

- Replaced the ordered-piece-pair relation state with an incremental square-control field. Canonical pseudo-attacks produce factorized source, destination and geometry messages for each controlled square.
- Added square-local control encoding, mean pooling and bucket-conditioned SoftMax attention. Native inference updates changed control edges and square representations incrementally, while a material-bucket change recomputes the 64-square pooling state.
- Added learned material and bishop-pair residuals alongside the existing PSQT residual. The sparse accumulator, control summary and residual terms jointly determine the Value estimate.
- Derived the control graph from the existing sparse state encoding, so Value-only Eleginus HDF5 datasets remain usable. Checkpoints from the preceding network layout require retraining because their trainable tensor shapes differ.
