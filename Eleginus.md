# Eleginus

Eleginus is a sparse, incrementally evaluated Value architecture paired with deterministic
best-first minimax (BFM) search.

## 1. Notation

- $\mathcal X$ is the set of complete chess states maintained by the rules engine. A state includes
  piece placement, side to move, castling rights, en-passant state, move counters and the history
  required to adjudicate repetition.
- $\mathcal A(x)$ is the legal-action set in state $x\in\mathcal X$, and $T(x,a)$ is the
  deterministic successor after $a\in\mathcal A(x)$.
- $c_x$ is the side to move in $x$, and $\bar c_x$ is the opposing color.
- $g(x,c)\in\{0,\frac12,1\}$ is the exact score for color $c$ in a terminal state $x$.
- $\phi_E(x)$ is the sparse Eleginus encoding of $x$.
- $\mathcal I_E=\{0,\ldots,4671\}$ is the fixed action-index set, and $i_E(a)$ is the index of a
  legal action.
- $\theta_V$ denotes the trainable parameters of the Value network.
- $R_t\in[0,1]$ is the supervision target for the state $x_t$ recorded before ply $t$.
- $V(x)\in[0,1]$ is the Value network prediction computed from $\phi_E(x)$.

Neural targets and terminal scores deliberately share the same range and perspective, so search can
replace a leaf prediction directly with an exact rule outcome. The selected supervision source then
determines the statistical meaning of Value: result supervision estimates expected game score, while
annotation supervision estimates a transformed numerical annotation.

## 2. State and Action Encoding

The rules engine operates on $x$, and the network observes $\phi_E(x)$. This division keeps the
neural representation compact while the complete rules state preserves legality and terminal
adjudication. Consequently, states that share an encoding may remain different search nodes with
different legal histories.

### 2.1 State Encoding

Squares are numbered by

$$
q(r,f)=8r+f,
\qquad r,f\in\{0,\ldots,7\},
$$

where $q(0,0)$ is `a1` and $q(7,7)$ is `h8`. For a color perspective
$c\in\{\mathrm W,\mathrm B\}$, the orientation map is

$$
\omega_c(q(r,f))=
\begin{cases}
q(r,f),&c=\mathrm W,\\
q(7-r,f),&c=\mathrm B.
\end{cases}
$$

Under this map, Black's perspective uses a vertical reflection that preserves files. Eleginus then
combines the oriented square with perspective-relative ownership. Piece types are ordered as pawn,
knight, bishop, rook, queen and king, and the perspective-relative category of piece $p$ is

$$
\rho_c(p)=
\mathrm{type}(p)+6\mathbf 1[\mathrm{color}(p)\ne c],
$$

so $\rho_c(p)\in\{0,\ldots,11\}$. Let $\kappa_c(x)$ be the oriented square of color $c$'s king.
A piece $p$ on square $q$ activates the king-conditioned identifier

$$
\iota_c(x,p,q)=768\kappa_c(x)+64\rho_c(p)+\omega_c(q),
$$

which lies in $\{0,\ldots,49151\}$. Every piece, including both kings, contributes one such
identifier to each color perspective. Because the identifier includes $\kappa_c(x)$, the same
piece-square relation acquires a different representation whenever the perspective king moves to
another square.

Eleginus supplements piece placement with castling and en-passant context. Relative to perspective
$c$, castling rights form the mask

$$
m_c(x)=
\mathbf 1[\text{own kingside}]
+2\mathbf 1[\text{own queenside}]
+4\mathbf 1[\text{opposing kingside}]
+8\mathbf 1[\text{opposing queenside}],
$$

which activates $49152+m_c(x)$. The en-passant context is

$$
e(x)=
\begin{cases}
0,&\text{if no en-passant square exists},\\
1+f_{\mathrm{ep}},&\text{if the en-passant square is on file }f_{\mathrm{ep}},
\end{cases}
$$

and activates $49168+e(x)$. The resulting vocabulary is

| Identifier range | Meaning | Cardinality |
| --- | --- | ---: |
| $0\ldots49151$ | king-conditioned piece placement | 49152 |
| $49152\ldots49167$ | perspective-relative castling rights | 16 |
| $49168\ldots49176$ | en-passant context | 9 |
| $49177$ | zero-valued padding | 1 |

Each perspective contains at most 32 piece identifiers, one castling identifier and one en-passant
identifier. A fixed-width representation therefore provides 34 slots per perspective and assigns
the padding identifier to every unused slot. The in-memory encoder stores White and Black features
separately together with the side to move. Before dense evaluation, the network places the
side-to-move perspective first and the opposing perspective second.

Piece placement, castling rights, en-passant context and side to move determine the neural
projection, while move counters and repetition history remain in the complete rules state. Complete
states that share the projected fields consequently receive the same neural prediction while
retaining separate search nodes and path histories. As search extends a path, the rules engine uses
that complete history to recognize a repetition or move-count terminal and supplies the
corresponding exact draw score.

### 2.2 Action Encoding

Preprocessing stores the played move beside every Value record, although the current supervised
objective does not consume that move. The fixed codec gives this stored field and the checkpoint
descriptor a stable action space containing 4096 source-destination indices and 576 underpromotion
indices:

$$
|\mathcal I_E|=64\times64+64\times9=4672.
$$

Let $f(a)$ be the source square and $t(a)$ the encoded destination. The source-destination block
represents ordinary moves and queen promotions through

$$
i_E(a)=64f(a)+t(a).
$$

Within the same block, castling uses the king's final square as $t(a)$ when the internal move
representation refers to the rook. The underpromotion block then distinguishes knight, bishop and
rook promotions by direction and promoted piece. If
$\Delta_f(a)\in\{-1,0,1\}$ is the destination-file displacement and

$$
r(a)=
\begin{cases}
0,&\text{knight},\\
1,&\text{bishop},\\
2,&\text{rook},
\end{cases}
$$

then

$$
i_E(a)=4096+9f(a)+3\bigl(\Delta_f(a)+1\bigr)+r(a).
$$

Decoding searches $\mathcal A(x)$ for the legal move with the requested index. Because the codec is
interpreted within this legal-action set, the complete state supplies its castling, en-passant and
promotion semantics.

## 3. Network

### 3.1 Sparse Accumulator

Let $\mathcal F_c(x)$ be the active identifiers from perspective $c$. Eleginus represents these
identifiers with a learned embedding table

$$
E\in\mathbb R^{49178\times256},
$$

whose padding row is fixed at zero. The raw perspective accumulator is

$$
u_c(x)=\sum_{j\in\mathcal F_c(x)}E_j.
$$

Before dense evaluation, each component is clipped to $[0,1]$. The two clipped accumulators are
then ordered from the side-to-move perspective:

$$
h_c(x)=\mathrm{clip}_{[0,1]}\bigl(u_c(x)\bigr),
$$

$$
h(x)=h_{c_x}(x)\mathbin\Vert h_{\bar c_x}(x)
\in\mathbb R^{512}.
$$

This ordering gives one network a consistent moving-player viewpoint and preserves the asymmetry
between the two color-conditioned feature sets.

### 3.2 Value Network

The dense evaluator consists of two rectified affine layers followed by a scalar sigmoid:

$$
y(x)=\mathrm{ReLU}\left(W_1h(x)+b_1\right)
\in\mathbb R^{64},
$$

$$
r(x)=\mathrm{ReLU}\left(W_2y(x)+b_2\right)
\in\mathbb R^{32},
$$

$$
V(x)=\sigma\left(w_3^\top r(x)+b_3\right)
\in[0,1].
$$

The network contains 12,624,513 scalar parameters:

| Component | Shape | Parameters |
| --- | --- | ---: |
| sparse embedding | $49178\times256$ | 12,589,568 |
| first affine layer | $512\rightarrow64$ | 32,832 |
| second affine layer | $64\rightarrow32$ | 2,080 |
| scalar output | $32\rightarrow1$ | 33 |

The embedding contains 99.72% of all parameters, which concentrates representational capacity in
sparse feature lookup. In float32, the parameter tensors occupy about 48.16 MiB. Once the relevant
rows have been accumulated, the position-dependent dense tail requires 34,848
multiply-accumulate operations: $512\times64+64\times32+32$.

The same sparse-dense division also organizes how features interact. Within each perspective, a
commutative sum first aggregates the active identifiers and the componentwise clipping then bounds
the resulting accumulator. The king-conditioned identifiers carry board geometry into this sum,
after which the $512\rightarrow64\rightarrow32$ dense tail combines simultaneously active features.
Its two bottlenecks thereby express global interactions through a compact shared state.

### 3.3 Initialization

The padding row is initialized to zero, and every active embedding row uses an independent
$\mathcal N(0,1)$ initialization. The first two affine layers use fan-in initialization,

$$
W_{ij},b_i\sim
U\left(-\frac1{\sqrt{n_{\mathrm{in}}}},
\frac1{\sqrt{n_{\mathrm{in}}}}\right).
$$

The final weight and bias are initialized to zero. This final layer makes the initialized network
satisfy $V(x)=1/2$ across the complete state space.

### 3.4 Incremental Evaluation

Search realizes the Efficiently Updatable Neural Network (NNUE) principle by carrying the unclipped
accumulators along each tree edge. If $x'=T(x,a)$, then for either color perspective

$$
u_c(x')=u_c(x)
-\sum_{j\in\mathcal F_c(x)\setminus\mathcal F_c(x')}E_j
+\sum_{j\in\mathcal F_c(x')\setminus\mathcal F_c(x)}E_j.
$$

A normal move usually changes the moving piece, a captured piece and the affected rule contexts. A
king move changes the king-conditioned identifier of every piece in that king's perspective, so
more rows are replaced, while the same set-difference equation remains exact. Clipping follows the
update immediately before the dense layers.

For embedding width 256, a full refresh adds at most $2\times34\times256=17408$ scalar components.
Because an incremental update performs work proportional to the symmetric difference of the old
and new feature sets, an ordinary move usually requires a small row update, while a king move can
replace almost every piece row in one perspective and approach refresh cost. The transition type
therefore determines the sparse-input saving; the dense tail performs the same 34,848
multiply-accumulates after every accumulator update.

### 3.5 Training and Search Evaluation

Training evaluates the network in float32 batches with automatic differentiation, while search uses
a scalar float32 realization of the same embedding and affine equations. The scalar path bounds the
input to the final sigmoid so that its exponential remains numerically stable:

$$
\sigma\left(
\mathrm{clip}_{[-30,30]}
\left(w_3^\top r(x)+b_3\right)
\right).
$$

Because both numerical paths use the same learned embedding and affine parameters, this bounded
sigmoid completes the search-time realization of the trained Value function.

## 4. Preprocessing

### 4.1 Value Targets

For every mainline move $a_t$ recorded in Portable Game Notation (PGN) and played from state $x_t$,
preprocessing forms the record

$$
d_t=\bigl(\phi_E(x_t),i_E(a_t),R_t\bigr).
$$

A PGN `FEN` header supplies $x_0$ when present; otherwise, $x_0$ is the standard initial state.

Each preprocessing run selects exactly one target source for the entire dataset. Result mode derives
every target from the completed game result, while annotation mode derives targets from comments
and neutral defaults. For a game ending at terminal state $x_T$, result supervision uses

$$
R_t=g(x_T,c_{x_t})
\in\left\{0,\frac12,1\right\}.
$$

The target is complemented when the side to move changes from one color to the other. Result mode
retains games whose PGN result is `1-0`, `0-1` or `1/2-1/2`.

In annotation mode, a comment attached to move $a_{t-1}$ describes the resulting state $x_t$. The
first standalone decimal, with an optional sign, is interpreted as a pawn evaluation $s_t$ from
White's perspective. It is converted to the side-to-move perspective by

$$
\widehat s_t=
\begin{cases}
s_t,&c_{x_t}=\mathrm W,\\
-s_t,&c_{x_t}=\mathrm B,
\end{cases}
$$

and then mapped to the Value range:

$$
R_t=\frac{1+\tanh(\widehat s_t/3)}2.
$$

The initial state has no preceding move comment and therefore receives the neutral target $1/2$.
For $t>0$, the target for $x_t$ comes from the comment attached to $a_{t-1}$ when that comment
contains a numerical evaluation; otherwise, $x_t$ also receives $1/2$. A numerical comment attached
to the final move has no later pre-move state to label. Annotation mode admits a game when at least
one of its move comments contains a numerical evaluation, so unannotated gaps in an admitted game
contribute neutral targets and draw the fitted conditional mean toward $1/2$.

### 4.2 HDF5 Schema

The Hierarchical Data Format version 5 (HDF5) output contains three row-aligned datasets. `states`
has type `uint16` and shape $[N,2,34]$;
its first perspective is the side to move and its second perspective is the opponent. `moves` has
type `uint16` and shape $[N]$, while `values` has type `float32` and shape $[N]$. The move index
preserves the recorded action under $i_E$, although the Value-only supervised objective uses only
`states` and `values`.

The root attributes identify the schema as follows:

- `arch_type=eleginus`
- `state_encoding=eleginus_sparse_features_v1`
- `move_encoding=sd_64x64_underpromo9`
- `target_schema=side_to_move_expectation_01_v1`
- `value_perspective=side_to_move`
- `source=pgn_comments` with `has_cmt=1`, or `source=pgn_result` with `has_cmt=0`

Comment-derived data additionally records `comment_eval_perspective=white` and the transformation
shown in Section 4.1. The writer requires every move index to belong to $\mathcal I_E$ and every
Value target to be finite and contained in $[0,1]$. The reader verifies the five schema identities,
the $[N,2,34]$ state shape and equal row counts across all three datasets before training begins.

## 5. Supervised Training

For a minibatch $\mathcal B$, supervised learning minimizes mean squared prediction error:

$$
L_V^{(\mathcal B)}=
\frac1{|\mathcal B|}
\sum_{x_t\in\mathcal B}
\left(V(x_t)-R_t\right)^2.
$$

For a fixed data distribution, the population minimizer of this objective is

$$
V^*(\phi)=\mathbb E\left[R_t\mid\phi_E(x_t)=\phi\right].
$$

This equation gives the precise interpretation of Value. With result targets, it is the conditional
expected game score in the training distribution. With annotation targets, it is the conditional
mean of the transformed evaluator score. If several complete states collapse to the same encoding,
including states with different repetition or move-count histories, their targets contribute to the
same conditional mean.

Automatic differentiation computes the gradient of $L_V^{(\mathcal B)}$ with respect to
$\theta_V$. Before AdamW applies an update, the learner clips the global Euclidean norm of that
gradient to at most one. Every record contributes equal weight, so the active embedding rows and
dense parameters receive gradients from the same minibatch objective. The reference settings are:

| Quantity | Value |
| --- | ---: |
| epochs | 1 |
| minibatch size | 512 |
| learning rate | $10^{-3}$ |
| weight decay | $10^{-5}$ |
| gradient-norm limit | 1 |
| random seed | 2026 |

For minibatch size $B$, the learner reads contiguous blocks of $\max(4096,16B)$ records. Each epoch
first shuffles the block order and then independently shuffles the records within every loaded
block, giving randomness at both storage scales.

Each training invocation constructs AdamW with fresh first- and second-moment estimates. The
reference seed controls new-model initialization, chunk-order shuffling and row-order shuffling.
Supplying an existing checkpoint initializes $\theta_V$ from its saved parameters while retaining
the newly constructed optimizer state.

CPU and CUDA evaluate the same float32 objective. Their backend-specific reduction orders determine
the exact sequence of floating-point operations and may therefore produce different final parameter
bits.

Training writes the final parameters after all selected epochs or optimizer steps have completed.
The value `0` for `--max-steps` leaves the number of optimizer steps unrestricted, so `--epochs`
then determines when training ends.

## 6. Search

### 6.1 Search State and Static Evaluation

Eleginus search constructs an explicit tree of complete rules states. Each node couples its chess
state with the incremental Value accumulator, its parent, depth and subtree size, thereby retaining
the information needed for both evaluation and tree traversal. Its static and backed evaluations
then summarize the neural or terminal evidence used by minimax. The static side-to-move value is

$$
v_0(x)=
\begin{cases}
g(x,c_x),&x\text{ is terminal},\\
V(x),&x\text{ is ongoing}.
\end{cases}
$$

Before any children have been generated, the backed value is initialized as
$\overline v(x)=v_0(x)$. This initialization uses Value for an ongoing frontier leaf and the exact
game result for a terminal node; consequently, a terminal root returns its exact score immediately.

With result supervision, neural and terminal values have the same expected-score interpretation.
With annotation supervision, search mixes transformed annotation targets at ongoing leaves with
exact game scores at terminal leaves. Their common range makes minimax arithmetic well-defined,
while their relative calibration follows the annotation distribution and target transformation.

The tree preserves one node per path. Two paths reaching equivalent board positions therefore
remain separate nodes and retain their own repetition and move-count histories.

### 6.2 Best-First Selection

Frontier selection uses closeness to an even evaluation. Define the evenness score

$$
E(x)=1-2\left|v_0(x)-\frac12\right|,
$$

which is one at $v_0(x)=1/2$ and zero at either endpoint. An ongoing frontier node below the
depth limit receives priority

$$
F(x)=-\lambda_d d(x)+\lambda_eE(x),
$$

where $d(x)$ is its depth in plies, $\lambda_d$ is the depth coefficient and $\lambda_e$ is the
evenness coefficient. The depth term favors shorter unresolved lines, while the evenness term favors
leaves whose static evaluation is least decisive. In particular, the formula assigns the maximum
evenness score to $1/2$ and progressively smaller scores toward either endpoint. The resulting
priority therefore allocates additional expansion to near-even leaves where modest value changes
may alter move ordering.

The priority is fixed when a child enters the frontier because it depends on that child's immutable
static value and depth. Later minimax backups update ancestor backed values while each queued child
retains its original priority. One global maximum-priority queue contains all eligible frontier
nodes, and each search step expands a node with maximal $F(x)$. The queue defines no secondary key
for nodes with equal priorities.

### 6.3 Expansion and Minimax Backup

A positive search budget first expands the root and thereby creates the initial frontier. Each later
expansion selects the frontier node with maximal priority and performs the same sequence:

1. The rules engine enumerates every legal action from the selected state.
2. Each successor receives an incrementally updated Value accumulator.
3. Terminal successors receive exact scores, while ongoing successors receive neural values.
4. Eligible ongoing successors below the depth limit enter the frontier.

Because every expansion generates and evaluates the complete legal child set, terminal outcomes and
depth-limited leaves immediately contribute static values. Ongoing children strictly shallower than
the maximum depth additionally enter the frontier and become candidates for later expansion.

After the children have been evaluated, adversarial backup proceeds to the root:

$$
\overline v(x)=
\begin{cases}
v_0(x),&x\text{ is a leaf},\\[4pt]
\displaystyle\max_{a\in\mathcal A(x)}
\left(1-\overline v(T(x,a))\right),&x\text{ is expanded}.
\end{cases}
$$

The complement converts a child's side-to-move score to its parent's perspective. Repeating this
operation through the ancestors yields a finite minimax tree whose unresolved leaves are supplied
by $V$. Replacing a static leaf estimate with the minimax result of its generated children can move
an ancestor value in either direction. Backed root values may therefore rise or fall as the
expansion budget grows.

### 6.4 Root Decision

For a root action $a$, the final backed action value is

$$
Q_B(x,a)=1-\overline v(T(x,a)).
$$

The selected move satisfies

$$
a^*\in\underset{a\in\mathcal A(x)}{\mathrm{arg\,max}}\;Q_B(x,a),
$$

and the reported position value is $\overline v(x)$. Root decision uses these backed action values
directly. If several actions have exactly the same backed value, the first one in the rules engine's
legal-move enumeration is selected.

The initialization in Section 3.3 makes every ongoing prediction exactly $1/2$. At initialization,
all ongoing children at a common depth consequently have identical static values and priorities.
Exact terminal outcomes discovered inside the finite tree distinguish their corresponding moves;
the queue ordering and legal-move order resolve the remaining ties. Supervised training then
introduces position-dependent Value differences.

### 6.5 Search Limits and Algorithmic Properties

One expansion denotes one parent whose complete legal child set has been generated. The evaluated
node count consequently increases by the number of generated children and is generally much larger
than the expansion count. Search stops upon reaching the expansion budget or exhausting the eligible
frontier. The reference settings are:

| Quantity | Value |
| --- | ---: |
| expansion budget | 32 |
| maximum depth | 64 plies |
| depth coefficient $\lambda_d$ | 0.12 |
| evenness coefficient $\lambda_e$ | 0.20 |

Under these limits, the root expansion evaluates every legal successor through the scalar Value
path. Subsequent expansions preserve path-specific nodes, generate complete legal child sets and use
the global queue to choose the next frontier node. Minimax backup connects these local expansions to
the root decision, completing a finite best-first minimax procedure.

## 7. Checkpoint and Embedded Runtime

An Eleginus training checkpoint is a `.pth` archive with two top-level entries. `model` contains the
Value network parameters, while `arch` contains the fixed architecture descriptor:

| Field | Value |
| --- | ---: |
| `type_id` | 3 |
| `feature_count` | 49178 |
| `feature_slots` | 34 |
| `accumulator` | 256 |
| `hidden` | 64 |
| `bottleneck` | 32 |
| `action_size` | 4672 |

Loading a checkpoint validates every descriptor field before restoring the parameters. Saving first
writes a temporary archive and then replaces the destination, so the destination changes only after
the complete archive has been written.

Standalone inference uses a different representation. The embedding tool reads a `.pth` checkpoint
on the CPU, extracts its float32 parameter arrays and appends them to a copy of a weightless
executable template. The appended payload records a format marker, the runtime dimensions and the
length of every parameter array. The resulting executable locates and validates this payload at
startup before constructing `CpuValue`.

The embedded evaluator uses the scalar equations in Section 3.5 and has no LibTorch runtime
dependency. Its native payload requires a little-endian target with IEEE 754 binary32 floats. The
embedding operation writes through a temporary file and requires distinct template and output paths.

## 8. UCI Interface

The embedded Universal Chess Interface (UCI) executable identifies the engine as `Eleginus` and the
author as `Gadidae`, then loads its Value parameters from its own file. It exposes one engine option,
`BFMExpansions`, with default value 32 and advertised range 1 through 1,000,000. Each `go` command
performs one synchronous search with the current expansion budget and the fixed depth and priority
coefficients from Section 6.5. The expansion budget is determined by `BFMExpansions`; UCI time,
depth and node fields do not change it.

For a completed search with root value $\overline v(x)$, the value written to the UCI centipawn
field is

$$
\mathrm{cp}(x)=
\mathrm{round}\left(2000\left(\overline v(x)-\frac12\right)\right).
$$

This score uses the root side-to-move perspective. The `depth` field reports the number of expanded
parents, the `nodes` field reports the number of generated and evaluated children and the one-move
principal variation contains the selected root move. The subsequent `bestmove` field reports the
same move.

## 9. Verification

The Eleginus test executable checks action-codec round trips for ordinary moves, castling and
promotions. It also compares full and incremental accumulators, compares LibTorch and scalar CPU
Value outputs, verifies checkpoint and embedded-runtime round trips, checks HDF5 perspective order
and validates the expansion budget and terminal score used by search. These checks run through CTest
as `eleginustests` when the training-side Eleginus targets are enabled.
