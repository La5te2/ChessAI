# Melano

Melano is a geometry-aware Policy-Value transformer for chess. It represents each position as one global token and 64 square tokens, processes them with geometry-biased self-attention and predicts a legal-move distribution together with a scalar evaluation. Search evaluates every leaf from its exact chess state.

## 1. Notation

- $x$ denotes a complete game state maintained by the chess rules.
- $s=\phi_M(x)$ denotes the 67-integer network input encoded from $x$.
- $\mathcal A(x)$ denotes the set of legal moves in $x$.
- $\mathcal I_M=\{0,\ldots,4671\}$ denotes the complete Melano action-index set.
- $i_M(a)\in\mathcal I_M$ denotes the action index assigned to move $a$.
- $\theta$ denotes the trainable model parameters.
- $\ell_\theta(s,i)$ denotes the Policy logit assigned to action index $i$.
- $P_\theta(a\mid s)$ denotes the Policy probability of legal move $a$.
- $V_\theta(s)\in[-1,1]$ denotes the position evaluation from the perspective of the side to move.
- $C$ denotes the embedding width selected by `--channels`.
- $B$ denotes the number of geometry-attention blocks selected by `--blocks`.

## 2. State and Move Encoding

### 2.1 State Encoding

The state codec `melano_square_tokens` stores each position as 67 unsigned integers:

$$
s=(p_0,\ldots,p_{63},t,c,e).
$$

Each $p_q\in\{0,\ldots,12\}$ identifies the piece on square $q$, where 0 represents an empty square. The value $t\in\{0,1\}$ identifies the side to move, $c\in\{0,\ldots,15\}$ encodes the four castling rights as a bit mask and $e\in\{0,\ldots,8\}$ identifies the en-passant file or the absence of an en-passant square.

The chess library retains the complete game state used by search, including the half-move counter and repetition history. The 67-integer network input contains the position information consumed by the neural model.

### 2.2 Move Encoding

The move codec `sd_64x64_underpromo9` contains 4096 source-destination indices and 576 underpromotion indices:

$$
|\mathcal I_M|=64\times64+64\times9=4672.
$$

Ordinary moves, queen promotions and castling use

$$
i_M(a)=64f(a)+t(a),
$$

where $f(a)$ and $t(a)$ are the source and destination squares. Knight, bishop and rook underpromotions use the final 576 indices. Their nine planes represent three promotion pieces and three pawn directions for each source square.

Move decoding receives both an action index and the current game state. It resolves the index against $\mathcal A(x)$ and rejects an index that has no unique legal interpretation.

## 3. Network

### 3.1 Exact-State Geometry-Attention Encoder

The state embedding converts a batch of inputs with shape $[N,67]$ into a token tensor with shape $[N,65,C]$. Let $h_0(s)$ denote this initial token sequence. Its first element is a learned global token, while the remaining 64 elements correspond to board squares.

For square $q$, the initial square token is the sum of a piece embedding, an absolute-square embedding and the shared rule-context embedding:

$$
h_{0,q+1}(s)=E_{\mathrm{piece}}(p_q)+E_{\mathrm{square}}(q)+r(s).
$$

The rule-context embedding is

$$
r(s)=E_{\mathrm{side}}(t)+E_{\mathrm{castling}}(c)+E_{\mathrm{ep}}(e),
$$

and the initial global token is

$$
h_{0,0}(s)=g+r(s),
$$

where $g\in\mathbb R^C$ is trainable.

Each geometry-attention block uses the largest value in $\{8,4,2,1\}$ that divides $C$ as its attention-head count. Let $H$ be this count and let $d=C/H$. For block $b$, pre-normalized linear projections produce queries, keys and values:

$$
(Q_b,K_b,U_b)=\mathrm{Linear}_{qkv,b}(\mathrm{LN}_{b,1}(h_b+Z_b)),
$$

where $Z_b\in\mathbb R^{65\times C}$ is a learned token-position tensor.

Every ordered token pair $(u,v)$ has a static relation id $\rho(u,v)\in\{0,\ldots,28\}$. Square-pair relations distinguish identity, rank distance, file distance, diagonal distance, knight movement and residual distance classes. Pairs involving the global token use relation id 0. A learned table supplies the static per-head bias $\beta_{b,h,\rho(u,v)}$.

The global token also produces a position-dependent bias value for every head and relation class. Denote this value by $\gamma_{b,h,\rho(u,v)}(h_b)$. The attention score is

$$
S_{b,h,u,v}=
\frac{Q_{b,h,u}\cdot K_{b,h,v}}{\sqrt d}
+\beta_{b,h,\rho(u,v)}
+\gamma_{b,h,\rho(u,v)}(h_b).
$$

The block applies row-wise SoftMax to these scores and uses two residual updates:

$$
y_b=h_b+Z_b+\mathrm{Linear}_{o,b}
\left(\mathrm{SoftMax}(S_b)U_b\right),
$$

$$
h_{b+1}=y_b+\mathrm{FFN}_b(\mathrm{LN}_{b,2}(y_b)).
$$

The feed-forward network has dimensions $C\rightarrow4C\rightarrow C$ with GELU between the two linear maps. Applying all $B$ blocks defines the shared exact-state representation

$$
E_\theta(s)=h_B(s)\in\mathbb R^{65\times C}.
$$

### 3.2 Policy Head

Let $z_q=E_\theta(s)_{q+1}$ denote the transformed token for square $q$. After LayerNorm, separate linear maps produce source and destination vectors:

$$
u_q=W_F\mathrm{LN}(z_q)+b_F,
\qquad
v_q=W_T\mathrm{LN}(z_q)+b_T.
$$

The source-destination logit for index $64q+r$ is

$$
\ell_\theta(s,64q+r)=\frac{u_q\cdot v_r}{\sqrt C}.
$$

A separate linear map produces nine underpromotion logits from every normalized source token. Concatenating the 4096 source-destination logits and the 576 underpromotion logits gives

$$
\ell_\theta(s)\in\mathbb R^{4672}.
$$

For inference and search, legal-move masking defines the Policy distribution

$$
P_\theta(a\mid s)=
\frac{\exp\ell_\theta(s,i_M(a))}
{\displaystyle\sum_{b\in\mathcal A(x)}\exp\ell_\theta(s,i_M(b))}.
$$

### 3.3 Value Head

The Value head reads the transformed global token $E_\theta(s)_0$. It applies LayerNorm followed by linear maps with dimensions $C\rightarrow256\rightarrow1$, ReLU between the maps and hyperbolic tangent at the output:

$$
V_\theta(s)=
\tanh\left(W_2\mathrm{ReLU}
\left(W_1\mathrm{LN}(E_\theta(s)_0)+b_1\right)+b_2\right).
$$

The state embedding and all geometry-attention blocks form the backbone shared by the Policy and Value heads. Gradients from both supervised losses therefore update this shared representation.

## 4. Supervised Data

### 4.1 Policy and Value Targets

For each parseable PGN move, preprocessing records the state immediately before the move as $s$, the played move as $a^*$ and its action index as $i^*=i_M(a^*)$.

When `--has-cmt 1` is active, a signed pawn-unit comment evaluates the position produced by the move to which the comment is attached. Consequently, the comment on the previous move supplies the Value target for the current pre-move state. Let $c_W$ be that White-perspective score. Preprocessing converts it to the side-to-move perspective and maps it to $[-1,1]$:

$$
c_{\mathrm{stm}}=
\begin{cases}
c_W,&\text{White to move},\\
-c_W,&\text{Black to move},
\end{cases}
\qquad
V_{\mathrm{target}}(s)=\tanh\left(\frac{c_{\mathrm{stm}}}{3}\right).
$$

The first position of a game has no preceding move comment and therefore receives target 0. Other positions without a parseable preceding score also receive target 0. A game must contain at least one parseable evaluation to enter a comment-enabled dataset.

When `--has-cmt 0` is active, the game result supplies the target. A win for the side to move gives $+1$, a loss gives $-1$ and a draw or unknown result gives 0.

### 4.2 HDF5 Schema

A Melano HDF5 file has the following identifying attributes:

```text
arch_type=melano
state_encoding=melano_square_tokens
move_encoding=sd_64x64_underpromo9
target_schema=melano_policy_value
value_perspective=side_to_move
```

Its row-aligned datasets are:

- `states`, an unsigned-byte array with shape $[N,67]$
- `moves`, an unsigned-16 array with shape $[N]$
- `values`, a float-32 array with shape $[N]$

The reader validates every identifying attribute and dataset shape before exposing training rows. The schema is specific to this Melano architecture.

## 5. Supervised Training

For a training row $(s,i^*,V_{\mathrm{target}})$, supervised Policy training applies SoftMax over all 4672 logits:

$$
R_\theta(i\mid s)=
\frac{\exp\ell_\theta(s,i)}
{\displaystyle\sum_{j\in\mathcal I_M}\exp\ell_\theta(s,j)}.
$$

Let $\delta_{i^*}$ be the one-hot distribution concentrated at $i^*$. The Policy loss is

$$
L_{P,\mathrm{sup}}=
L_{\mathrm{CE}}\left(R_\theta(\cdot\mid s),\delta_{i^*}\right),
$$

and the Value loss is

$$
L_{V,\mathrm{sup}}=
\mathrm{MSE}\left(V_\theta(s),V_{\mathrm{target}}(s)\right).
$$

With the nonnegative coefficient $w_V$ supplied by `--value-weight`, the complete objective is

$$
L_{\mathrm{sup}}=L_{P,\mathrm{sup}}+w_VL_{V,\mathrm{sup}}.
$$

Training uses AdamW. HDF5 chunks are shuffled, read through a one-chunk asynchronous prefetch pipeline and shuffled again before minibatch construction. CUDA batches use pinned host memory and nonblocking transfers. The learning rate rises linearly during the warmup interval and then remains at the requested peak value. A positive `--grad-clip` applies global gradient-norm clipping before each optimizer step.

`--max-steps 0` leaves the optimizer-step count unrestricted and lets `--epochs` determine training length. A positive `--max-steps` stops training after the smaller of the requested step count and the number of batches available across all epochs.

## 6. Checkpoints

A Melano checkpoint contains the model archive and an architecture descriptor. The descriptor stores Melano type id 2, $C$, $B$ and action size 4672. Loading validates the type and action size before constructing the model.

Training writes checkpoints through a temporary file and atomically replaces the destination after serialization succeeds. The saved archive contains model parameters and architecture metadata. Optimizer state and training counters are outside the checkpoint format.

This architecture uses newly generated `melano_policy_value` HDF5 data and newly trained checkpoints.

## 7. Search

### 7.1 Closed Policy

Search type `closed` evaluates the root once and ranks legal moves by $P_\theta(a\mid s)$. It constructs no search tree, so the returned root distribution is the model's normalized legal-move Policy.

### 7.2 Tree Statistics

Search type `only-mcts` builds a separate exact-state tree for every input game. Each node retains the complete chess state reached by its path. Every edge $(s,a)$ stores:

- the Policy prior $P(s,a)$ assigned when its parent was expanded
- the completed visit count $N(s,a)$
- the accumulated backed-up return $W(s,a)$
- the temporary virtual-visit count $N_v(s,a)$

For an edge with at least one completed visit, its empirical mean from the child side-to-move perspective is

$$
Q_{\mathrm{child}}(s,a)=\frac{W(s,a)}{N(s,a)}.
$$

Selection occurs at the parent, so the corresponding exploitation value is

$$
Q(s,a)=-Q_{\mathrm{child}}(s,a).
$$

### 7.3 FPU and PUCT

Let

$$
M(s)=\sum_{b:N(s,b)>0}P(s,b)
$$

be the prior mass of visited edges. The first-play urgency value for an unvisited edge is

$$
Q_{\mathrm{FPU}}(s)=
\mathrm{clip}_{[-1,1]}
\left(Q_{\mathrm{parent}}(s)-r_{\mathrm{FPU}}\sqrt{M(s)}\right),
$$

where $r_{\mathrm{FPU}}$ is `--fpu-reduction` and $Q_{\mathrm{parent}}(s)$ is the parent node's empirical mean when available or 0 otherwise. Selection uses $Q(s,a)$ for visited edges and $Q_{\mathrm{FPU}}(s)$ for unvisited edges.

The exploration coefficient grows logarithmically with the parent visit count:

$$
c(s)=c_0+c_f\log
\left(\frac{N(s)+N_v(s)+c_b+1}{c_b}\right),
$$

where $c_0$, $c_b$ and $c_f$ are supplied by `--c-puct`, `--c-puct-base` and `--c-puct-factor`. PUCT selects the edge that maximizes

$$
U(s,a)=Q_{\mathrm{sel}}(s,a)
+c(s)P(s,a)
\frac{\sqrt{N(s)+N_v(s)+1}}
{1+N(s,a)+N_v(s,a)}
-\lambda_vN_v(s,a),
$$

where $Q_{\mathrm{sel}}$ is the visited-edge or FPU value described above and $\lambda_v$ is the nonnegative `--virtual-loss` coefficient. Equal scores are ordered by prior and then by $Q_{\mathrm{sel}}$.

### 7.4 Batched Exact-State Evaluation

Tree selection reserves each chosen path by increasing its virtual-visit counts. These reservations steer other selections in the same batch toward different leaves. The searcher then encodes the complete chess state of every nonterminal leaf and evaluates all selected leaves in one neural batch.

Expansion masks the leaf logits to its legal moves and stores the resulting Policy as child priors. If the leaf has network evaluation $v$, backup adds $v$ to the leaf and reverses its sign at every preceding ply:

$$
v_{d-1}=-v_d.
$$

Terminal leaves use the exact rule outcome $-1$, 0 or $+1$ from the side-to-move perspective. They require no neural evaluation.

### 7.5 Budget and Root Distribution

`--mcts-sims` is the hard simulation cap. Search first reaches a minimum budget and then estimates root uncertainty from visit entropy, the proximity of the two largest visit counts and the proximity of their empirical values. This uncertainty interpolates the dynamic target between the minimum and the hard cap. A positive `--movetime-ms` supplies an additional wall-clock deadline.

After search, the unnormalized weight of root move $a$ is

$$
\omega(a)=N(s,a)+P(s,a).
$$

Normalizing $\omega$ over legal moves produces the root Policy reported by `only-mcts`. Adding the prior keeps an unvisited legal move representable when the simulation budget is small.

## 8. Decision Components

The final decision layer receives either the closed Policy or the MCTS root Policy. Its optional components alter move ranking after neural evaluation and tree search, leaving model logits, priors and tree statistics unchanged.

Instant Mate First detects legal moves that immediately checkmate the opponent. When such moves exist, it raises the highest-ranked mating move's decision score to 1.

Repetition Policy Penalty examines whether a legal move immediately completes a threefold repetition or allows the opponent to complete one on the next ply. For a positive root evaluation, it subtracts

$$
d=r_{\mathrm{RPP}}\max(0,V_{\mathrm{root}})
$$

from the affected move's decision score, where $r_{\mathrm{RPP}}\in[0,1]$ is `--repetition-policy-penalty`. The score is clipped at 0. Final ordering uses the decision score, then the unmodified root Policy and finally the UCI move string.

## 9. Arena

Melano Arena loads a candidate and a baseline checkpoint once, assigns the same search options to both and evaluates games in batches. Paired openings give each model both colors from the same starting position. Standard initial positions are used when the opening-book path is empty.

The chess rules determine checkmate, stalemate, insufficient material, the fifty-move rule and threefold repetition. A game that reaches `--max-plies` is recorded as a draw by truncation. Arena reports wins, draws, losses, net wins, score, Elo difference and confidence intervals. The result gate accepts a candidate when its net wins reach `--min-net-wins`. Arena reports the gate result but does not write a promoted checkpoint.

## 10. UCI

The UCI executable loads one Melano checkpoint and exposes search controls through standard UCI options. `go` combines the GUI-supplied clock state with `MoveOverheadMS`, `MinMoveTimeMS`, `MaxMoveTimeMS`, `TimeDivisor` and `IncrementFraction` to derive the move deadline. An explicit `MoveTimeMS` overrides that allocation.

Periodic `info` records report depth-like tree diagnostics, completed simulations as nodes, elapsed time, a side-to-move `score cp` derived from the search value and the principal variation. The final `bestmove` is emitted after the search budget or deadline is reached.

## 11. Verification

The Melano CTest executable verifies:

- state encoding and all reachable geometry-relation ids
- move-index round trips for ordinary moves, castling, en passant and promotions
- terminal rules including repetition, the fifty-move rule and insufficient material
- annotated-PGN preprocessing and the `melano_policy_value` HDF5 schema
- Policy and Value tensor shapes, finite outputs and finite gradients
- atomic checkpoint round trips
- compact closed evaluation and exact-state batched PUCT search

The project build runs this test before publishing Melano binaries.
