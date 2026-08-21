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

## 2026-08-20 - ISCA Initialization

- Assigned explicit initialization scales to every ISCA embedding and projection. Edge embeddings use a larger scale than occupancy, count and square context embeddings, while the local encoder begins inside the active region of SCReLU and square-attention logits begin near a uniform distribution.

## 2026-08-20 - Incremental PVS State

- Replaced per-child copies of the board and Value accumulator with reversible move application on a worker-local PVS stack. The ISCA accumulator stores one attack bitboard for each canonical piece and updates only moved pieces and sliders whose rays cross a changed square.
- Indexed quiet-move history by side to move, applied bounded gravity updates and assigned negative feedback to quiet moves that failed before a later quiet beta cutoff. Late-move reduction now measures the position of a move among quiet candidates rather than among all candidates.
- Extended incremental-inference tests to compare every persistent ISCA component with full refreshes across deterministic random legal trajectories and to verify exact restoration after undo.

## 2026-08-20 - Transposition Clusters

- Packed each transposition entry into 16 bytes and grouped four entries into one 64-byte cluster. Probing examines the cluster associated with the complete position key, while replacement favors current-generation, deeper and exact records. Thread-local table shards continue to share the configured Hash memory budget.

## 2026-08-21 - Shared Material Residual

- Shared the learned material and bishop-pair residual across all Value buckets. Positions from every material phase now update the same six coefficients, while the nonlinear Value pathway and attention query remain conditioned on the current material bucket.

## 2026-08-21 - Calibrated Multi-Query Control

- Changed control-edge aggregation to preserve signed source, destination and geometry embeddings until the square-local projection applies SCReLU.
- Replaced the single square-pooling query with four independent queries for each material bucket. Their outputs form one 16-component control summary while retaining incremental square updates.
- Added a fixed material baseline fitted from a uniformly distributed sample of the existing Value targets before supervised optimization. Training gives moderately greater weight to positions whose targets lie farther from equality, while the nonlinear pathways learn the remaining positional variation.
- Added periodic exact reconstruction of incrementally maintained Value state to bound floating-point drift during long PVS apply-and-undo sequences.
