# Melano

Melano is a geometry-aware transformer for chess. Its exact-state network predicts a move distribution, a scalar position evaluation and nonpositive move advantages. Given the encoded position and a move, Melano's action-conditioned dynamics predicts a latent representation of the successor position.

## 1. Notation and Abbreviations

>  Throughout this document, $V$, $Q$ and terminal outcomes use the perspective of the side to move. A definition states its perspective explicitly when it uses a different one.

- $\mathcal X$ is the set of complete game states. A state $x\in\mathcal X$ contains the board, side to move, castling rights, en passant state, halfmove clock and repetition history.
- $\mathcal A(x)$ is the set of legal actions in $x$. An action is written as $a\in\mathcal A(x)$, and $T(x,a)$ is the deterministic successor state produced by the chess rules.
- $z(x)\in\lbrace -1,0,1\rbrace$ is the outcome of a terminal state $x$ from the perspective of the side to move in $x$.
- $\mathcal S_M$ is the Melano encoded-state space. The map $\phi_M:\mathcal X\rightarrow\mathcal S_M$ is the Melano state encoder, and $s=\phi_M(x)$ is the network input corresponding to $x$.
- $\mathcal I_M$ is the Melano action-index set, and $i_M(a)\in\mathcal I_M$ is the index assigned to action $a$.
- $\theta$ denotes trainable network parameters. The Policy head produces a raw logit $\ell_\theta(s,i)$ for each action index $i$, and normalizing its legal-move logits gives the Policy probability $P_\theta(a\mid s)$. The scalar output of the Value head for exact-state input $s$ is denoted by $V_\theta(s)$, and the bounded output of the Advantage head for move $a$ is denoted by $A_\theta(s,a)$.
- $Q_\theta(s,a)$ combines $V_\theta(s)$ and $A_\theta(s,a)$ according to the definition in Section 3.2. Search uses the separately defined quantities $Q_{\mathrm{prior}}$, $Q_{\mathrm{MCTS}}$ and $Q_{\mathrm{edge}}$.
- $E_\theta$ is the exact-state encoder, $D_\theta$ is the action-conditioned latent transition, and $\mathcal V_\theta$ is the value head applied to a latent token sequence.
- A hat, as in $\widehat y$, denotes a prediction or estimate. A bar, as in $\overline y$, denotes a stopped-gradient target, a backed-up result or an aggregate. The surrounding definition specifies which meaning applies.
- A target is a fixed scalar, tensor or distribution used as the comparison value in a loss. Automatic differentiation treats targets as constants.
- $L_{\mathrm{CE}}(p,q)=-\sum_iq_i\log p_i$ is the cross-entropy from target distribution $q$ to predicted distribution $p$. $\mathrm{MSE}(u,v)$ is the mean squared error between corresponding elements.

The following abbreviations are used throughout this document.

- **PGN**: Portable Game Notation.
- **FEN**: Forsyth-Edwards Notation.
- **UCI**: Universal Chess Interface.
- **HDF5**: Hierarchical Data Format 5.
- **MCTS**: Monte Carlo Tree Search.
- **PUCT**: Predictor plus Upper Confidence bounds applied to Trees.
- **FPU**: First Play Urgency.
- **CE**: cross-entropy.
- **MSE**: mean squared error.
- **EMA**: exponential moving average.
- **LN**: layer normalization.
- **MLP**: multilayer perceptron.
- **IMF**: Instant Mate First.
- **RPP**: Repetition Policy Penalty.
- **PV**: principal variation. `MultiPV` denotes multiple reported principal variations.
- **NPS**: nodes per second.
- **CPU**: central processing unit.
- **GPU**: graphics processing unit.
- **CUDA**: NVIDIA's GPU computing platform.
- **FP32**: IEEE 754 single-precision floating point.
- **BF16**: bfloat16 floating point.
- **CI**: confidence interval.
- **JSON**: JavaScript Object Notation.

## 2. State and Action Encoding

The `melano_square_tokens` encoding represents a game state with 67 integers: one piece category for each of the 64 squares, followed by the side to move, castling-right bitmask and en passant file. Category 0 denotes an empty square, categories 1 through 6 denote the White pieces and categories 7 through 12 denote the Black pieces.

Let $e_{piece}$, $e_{square}$, $e_{side}$, $e_{castle}$ and $e_{ep}$ denote the learned embeddings, and let $g_0$ be a learned global token. For side-to-move category $u$, castling-right mask $c$ and en passant category $e$, define the shared rule-context embedding

$$
\chi=e_{side}(u)+e_{castle}(c)+e_{ep}(e).
$$

If $\tau_i$ is the piece category on square $i$, the input sequence consists of global token $h_0$ and square tokens $h_1,\ldots,h_{64}$:

$$
h_0=g_0+\chi,
\qquad
h_{i+1}=e_{piece}(\tau_i)+e_{square}(i)+\chi,
\quad 0\leq i<64.
$$

The `sd_64x64_underpromo9` action space contains

$$
|\mathcal I_M|=64\times64+64\times9=4672.
$$

Ordinary moves and queen promotions are indexed by source and destination square. Knight, bishop and rook underpromotions use nine additional indices per source square, corresponding to three pawn directions and three promotion pieces.

## 3. Network

### 3.1 Geometry Attention Encoder

Let $C$ be the hidden width selected by `--channels`. Each geometry-attention block contains pre-norm multi-head self-attention and a pre-norm feed-forward network, both with residual connections. Attention uses 32 ordered geometric relation classes and a dynamic relation bias conditioned on the global token. The number of heads $n_h$ is the largest member of $\lbrace 8,4,2,1\rbrace$ that divides $C$, giving head dimension $d_h=C/n_h$. The feed-forward sublayer maps $C\rightarrow4C\rightarrow C$ with a GELU activation.

For input token $h_i$ and learned positional vector $\zeta_i$, write $\widetilde h_i=h_i+\zeta_i$. Relation index $\kappa_{ij}\in\lbrace 0,\ldots,31\rbrace$ encodes the geometry of ordered token pair $(i,j)$. Class 0 is reserved for pairs involving the global token. Relations between square tokens distinguish identity, common rank or file, diagonals, knight moves, adjacency and distance.

For head $m$, let $b^{(m)}_{\kappa_{ij}}$ be the learned static bias for relation $\kappa_{ij}$. A two-layer MLP applied after LayerNorm to global token $\widetilde h_0$ produces position-dependent bias $d^{(m)}_{\kappa_{ij}}(\widetilde h_0)$. The attention weights are

$$
\alpha^{(m)}_{ij}=\mathrm{softmax}_j\left(
\frac{q^{(m)}_i\left(k^{(m)}_j\right)^{\mathsf T}}{\sqrt{d_h}}
+b^{(m)}_{\kappa_{ij}}
+d^{(m)}_{\kappa_{ij}}(\widetilde h_0)
\right).
$$

Here $q^{(m)}_i$, $k^{(m)}_j$ and $v^{(m)}_j$ are the attention query, key and value projections of $\mathrm{LN}(\widetilde h)$. Head $m$ returns $\sum_j\alpha^{(m)}_{ij}v^{(m)}_j$. The block concatenates the head outputs, applies an output projection, adds the attention residual and then applies the pre-norm feed-forward sublayer with a second residual. Lowercase $v^{(m)}_j$ denotes an attention value vector. Section 3.2 defines uppercase $V_\theta(s)$ as the scalar position evaluation.

`--blocks` sets the number of geometry-attention blocks.

### 3.2 Policy, Value and Advantage Heads

For an exact-state input $s$, write the encoded latent sequence as

$$
z=E_\theta(s)=(z_0,z_1,\ldots,z_{64}),
$$

where $z_0$ is the global token and $z_1,\ldots,z_{64}$ are the square tokens. Denote the Policy head's learned LayerNorm by $\mathrm{LN}_P$. Its source and destination projections have trainable parameters $W_{\mathrm{from}},W_{\mathrm{to}}\in\mathbb R^{C\times C}$ and $b_{\mathrm{from}},b_{\mathrm{to}}\in\mathbb R^C$. For $1\leq i\leq64$, define

$$
r_i=\mathrm{LN}_P(z_i),
\qquad
f_i=W_{\mathrm{from}}r_i+b_{\mathrm{from}},
\qquad
t_i=W_{\mathrm{to}}r_i+b_{\mathrm{to}},
$$

with $r_i,f_i,t_i\in\mathbb R^C$. The scaled source-destination logits are

$$
L_{\mathrm{sd}}(i,j)=\frac{f_i^{\mathsf T}t_j}{\sqrt C},
\qquad 1\leq i,j\leq64.
$$

An independent linear projection with trainable parameters $W_{\mathrm{up}}\in\mathbb R^{9\times C}$ and $b_{\mathrm{up}}\in\mathbb R^9$ produces nine underpromotion logits for each source square:

$$
L_{\mathrm{up}}(i,k)=
\left(W_{\mathrm{up}}r_i+b_{\mathrm{up}}\right)_k,
\qquad 1\leq i\leq64,\quad 1\leq k\leq9.
$$

Flattening $L_{\mathrm{sd}}$ and $L_{\mathrm{up}}$ in the order specified by `sd_64x64_underpromo9` and concatenating them defines the Policy head output

$$
\ell_\theta(s)=
\mathrm{concat}\left(\mathrm{vec}(L_{\mathrm{sd}}),
\mathrm{vec}(L_{\mathrm{up}})\right)
\in\mathbb R^{4672}.
$$

For game state $x$ with $s=\phi_M(x)$, normalizing the logits associated with legal moves defines the model Policy:

$$
P_\theta(a\mid s)=
\frac{\exp\ell_\theta(s,i_M(a))}
{\displaystyle\sum_{b\in\mathcal A(x)}
\exp\ell_\theta(s,i_M(b))},
\qquad a\in\mathcal A(x).
$$

The Value head applies its LayerNorm to the global token $z_0$, followed by $C\rightarrow256$ and $256\rightarrow1$ linear maps with an intermediate ReLU and a final hyperbolic tangent. Let $\mathcal V_\theta(z)$ denote the scalar produced by this head from latent sequence $z$. The symbol $V_\theta(s)$ denotes the Value head output for exact-state representation $E_\theta(s)$:

$$
V_\theta(s)=\mathcal V_\theta(E_\theta(s))\in[-1,1].
$$

The Advantage head has the same LayerNorm-and-projection structure as the Policy head, with a separate set of trainable parameters. Applied to $z_1,\ldots,z_{64}$, it produces an unbounded raw-logit vector $u_{A,\theta}(s)\in\mathbb R^{4672}$. The symbol $A_\theta(s,a)$ denotes the Advantage head output for legal move $a$, obtained by mapping the corresponding raw logit to $[-2,0]$:

$$
A_\theta(s,a)=-2\tanh^2\left(u_{A,\theta}(s,i_M(a))\right)
\in[-2,0].
$$

Melano combines $V_\theta(s)$ and $A_\theta(s,a)$ as

$$
Q_\theta(s,a)=
\mathrm{clip}\left(V_\theta(s)+A_\theta(s,a),-1,1\right).
$$

### 3.3 Action-Conditioned Latent Dynamics

The dynamics module predicts the successor latent from the current latent and the selected action. It conditions one residual geometry-attention block $\mathcal B$ on action embedding $c(a)$ and controls the residual update with channel-wise gate logits $\gamma(a)$. Both action-dependent vectors are broadcast across the token dimension. Let $h=E_\theta(s)$ be the exactly encoded latent, let $\sigma$ denote the sigmoid function and let $\odot$ denote element-wise multiplication. The predicted successor latent is

$$
\widehat h'=D_\theta(h,a)=
\mathrm{LN}\left(
h+\sigma(\gamma(a))\odot
\left[\mathcal B(h+c(a))-h\right]
\right).
$$

## 4. PGN Annotation and Preprocessing

### 4.1 Evaluation Comments

Melano preprocessing reads White-perspective pawn-unit evaluations from PGN comments of the form `{+x}` and `{-x}`. A comment attached to a move evaluates the position produced by that move. These evaluations determine the Value and Advantage targets defined in Section 4.2.

### 4.2 HDF5 Schema and Targets

Melano preprocessing writes the following HDF5 schema:

```text
states:      uint8,  (N, 67)
next_states: uint8,  (N, 67)
moves:       uint16, (N,)
values:      float32, (N,)
next_values: float32, (N,)
adv_moves:   uint16, (N,)
adv_values:  float32, (N,)

arch_type=melano
state_encoding=melano_square_tokens
move_encoding=sd_64x64_underpromo9
target_schema=pva_latent_dynamics
value_perspective=side_to_move
has_cmt=0|1
```

`--has-cmt` defaults to `1`. For recorded move $a_t$, let $x_t$ and $x_{t+1}=T(x_t,a_t)$ be the positions before and after the move, and define $s_t=\phi_M(x_t)$ and $s_{t+1}=\phi_M(x_{t+1})$.

Comment $c_{t-1}$ is attached to the preceding move and therefore evaluates $x_t$, whereas comment $c_t$ evaluates $x_{t+1}$. When comment $c$ contains a parseable evaluation, let $q(c)$ be that White-perspective pawn-unit value. Define $\rho(x)=1$ when White is to move and $\rho(x)=-1$ when Black is to move. The corresponding side-to-move evaluation is

$$
v(c,x)=\tanh\left(\frac{\rho(x)q(c)}{3}\right).
$$

In `--has-cmt 1` mode, the value targets for the current and successor states are

$$
V_{\mathrm{target}}(s_t)=
\begin{cases}
v(c_{t-1},x_t),&c_{t-1}\text{ contains a parseable evaluation},\\
0,&\text{otherwise},
\end{cases}
$$

$$
V_{\mathrm{target}}(s_{t+1})=
\begin{cases}
v(c_t,x_{t+1}),&c_t\text{ contains a parseable evaluation},\\
0,&\text{otherwise}.
\end{cases}
$$

When both comments contain evaluations, preprocessing expresses the post-move evaluation from the perspective of the player who moved in $x_t$ and defines

$$
A_{\mathrm{target}}(s_t,a_t)=
\mathrm{clip}\left(
\tanh\left(\frac{\rho(x_t)q(c_t)}{3}\right)
-v(c_{t-1},x_t),-2,0
\right).
$$

All other comment combinations receive $A_{\mathrm{target}}(s_t,a_t)=0$. In `--has-cmt 1` mode, preprocessing accepts games with at least one parseable evaluation. In `--has-cmt 0` mode, the final game result supplies $V_{\mathrm{target}}(s_t)$ and $V_{\mathrm{target}}(s_{t+1})$ from their respective side-to-move perspectives, and every $A_{\mathrm{target}}$ is zero.

`next_values` stores $V_{\mathrm{target}}(s_{t+1})$ from the perspective of the side to move in $x_{t+1}$. Comment $c_t$ supplies this target, so the first parseable comment in a game can supervise the predicted successor latent.

## 5. Supervised Training

### 5.1 Training Targets

For a training sample derived from game state $x$, let $s=\phi_M(x)$ and let $a^{\ast}$ be the recorded move. The predicted action evaluation is $Q_\theta(s,a^{\ast})$ as defined in Section 3.2. The fixed training target for this quantity is

$$
Q_{\mathrm{target}}(s,a^{\ast})=
\mathrm{clip}\left(
V_{\mathrm{target}}(s)+A_{\mathrm{target}}(s,a^{\ast}),-1,1
\right).
$$

The chess rules determine the successor input $s'=\phi_M(T(x,a^{\ast}))$. Let $\theta_E$ and $\bar\theta_E$ denote the online and target encoder parameters. Define $\mathrm{stopgrad}$ as the identity operation in the forward pass with a zero derivative in backpropagation. The predicted successor latent and its fixed reference are

$$
\widehat h'=D_\theta(E_{\theta_E}(s),a^{\ast}),
\qquad
\overline h'=\mathrm{stopgrad}(E_{\bar\theta_E}(s')).
$$

After each optimizer step, the target encoder follows the online encoder by exponential moving average. With $m$ set by `--target-decay`, the update is

$$
\bar\theta_E\leftarrow
m\bar\theta_E+(1-m)\theta_E.
$$

Let $\epsilon_D=10^{-4}$ and define the stabilized unit vector

$$
\mathrm{unit}_{\epsilon_D}(y)=
\frac{y}{\max(\lVert y\rVert_2,\epsilon_D)}.
$$

The latent-consistency loss averages the corresponding cosine distance over the global token and all 64 square tokens:

$$
L_D=1-\frac{1}{65}\sum_{i=0}^{64}
\mathrm{unit}_{\epsilon_D}(\widehat h'_i)^{\mathsf T}
\mathrm{unit}_{\epsilon_D}(\overline h'_i).
$$

The imagined-value loss compares the value predicted from the successor latent with $V_{\mathrm{target}}(s')$:

$$
L_I=\mathrm{MSE}
\left(\mathcal V_\theta(\widehat h'),V_{\mathrm{target}}(s')\right).
$$

The supervised policy distribution is a softmax over all 4672 action indices:

$$
R_\theta(i\mid s)=
\frac{\exp\ell_\theta(s,i)}
{\sum_{j\in\mathcal I_M}\exp\ell_\theta(s,j)}.
$$

$R_\theta$ and $P_\theta$ use the same Policy head logits. $R_\theta$ normalizes over the complete action-index set for supervised training, whereas $P_\theta$ normalizes over the legal moves of the current game state for inference and search.

Let $\delta_{i_M(a^{\ast})}$ be the one-hot distribution concentrated at the action index of $a^{\ast}$. Parameters `--value-weight`, `--dueling-q-weight`, `--dynamics-weight` and `--imagined-value-weight` define coefficients $\lambda_V$, $\lambda_Q$, $\lambda_D$ and $\lambda_I$. Dataset indicator $m_Q$ equals 1 when `has_cmt=1` and 0 when `has_cmt=0`. The complete supervised objective is

$$
L_{\mathrm{sup}}=
L_{\mathrm{CE}}\left(
R_\theta(\cdot\mid s),\delta_{i_M(a^{\ast})}
\right)
+\lambda_V\mathrm{MSE}
\left(V_\theta(s),V_{\mathrm{target}}(s)\right)
+\lambda_Q m_Q\mathrm{MSE}
\left(Q_\theta(s,a^{\ast}),Q_{\mathrm{target}}(s,a^{\ast})\right)
+\lambda_D L_D+\lambda_I L_I.
$$

Let $\mathbf g_\theta=\nabla_\theta L_{\mathrm{sup}}$. When `--grad-clip` supplies a limit $c>0$, gradient clipping applies the update

$$
\mathbf g_\theta\leftarrow
\begin{cases}
\mathbf g_\theta,&\lVert\mathbf g_\theta\rVert_2\leq c,\\[4pt]
\dfrac{c}{\lVert\mathbf g_\theta\rVert_2}\mathbf g_\theta,
&\lVert\mathbf g_\theta\rVert_2>c.
\end{cases}
$$

The metric `grad_norm_before_clip` reports $\lVert\mathbf g_\theta\rVert_2$ before clipping. A nonfinite norm raises an error and terminates training before the optimizer step.

Checkpoints serialize the online model used for inference. Training reconstructs the EMA target encoder from the online model and updates it after each optimizer step.

Each training invocation initializes a new Melano model. `--channels` and `--blocks` determine its width and depth. Checkpoints are written atomically by renaming a completed temporary file. Each checkpoint contains top-level keys `model` and `arch`. The `arch` entry records the architecture identifier, channel count, block count and action-space size.

`--precision` accepts `fp32` or `bf16` and defaults to `fp32`. CUDA forward computation uses BF16 in `bf16` mode. The policy softmax, all loss calculations, metric accumulation and checkpoint parameters use FP32. CUDA training batches use pinned host memory.

### 5.2 Learning-Rate Schedule

Let $\eta_{\max}$ be the peak AdamW learning rate selected by `--lr`, and let $T$ be the planned number of optimizer steps. The warmup length is

$$
T_w=\min\left(
T,\ 2000,\ \max\left(100,\left\lfloor\frac{T}{100}\right\rfloor\right)
\right).
$$

At optimizer step $t$, the learning rate is

$$
\eta_t=
\begin{cases}
\eta_{\max}\dfrac{t}{T_w},&t\leq T_w,\\
\eta_{\max}\sqrt{\dfrac{T_w}{t}},&t>T_w.
\end{cases}
$$

Warmup increases the learning rate linearly while the AdamW moment estimates form. After warmup, inverse-square-root decay progressively reduces the learning rate.

## 6. Search

### 6.1 Direct Policy and Anchored Latent MCTS

Search mode `closed` derives its initial move ranking from the model Policy $P_\theta$ defined in Section 3.2.

Search mode `only-mcts` uses anchored latent MCTS with anchor period $K=2$. Each node retains the complete game state, allowing the chess rules to determine legal moves, checks, terminal outcomes, repetitions and the fifty-move rule exactly. At search depth $d$, let $x_d$ be the complete game state, let $s_d=\phi_M(x_d)$ be its network input and let $a_{d-1}$ be the move from its parent when $d>0$. Neural evaluation alternates between exact encoding at even ply depths and one learned latent transition at odd ply depths:

$$
h_d=
\begin{cases}
E_\theta(s_d),&d\bmod2=0,\\
D_\theta(h_{d-1},a_{d-1}),&d\bmod2=1.
\end{cases}
$$

The three output heads applied to $h_d$ yield $V_d$, $A_d$ and the Policy logits. Normalizing those logits over $\mathcal A(x_d)$ gives $P_d$. Denote the combined operation by

$$
(P_d,V_d,A_d)=\mathcal H_\theta(h_d;x_d).
$$

Every predicted latent is therefore one move beyond the most recent exact encoding. Even-depth nodes cache $E_\theta(s_d)$, whereas odd-depth latents are temporary results of batched evaluation. Search reports these paths separately as `exact_evaluations` and `latent_evaluations`.

Expanding edge $(s_d,a)$ stores $P_d(a)$ as its fixed prior, which subsequent formulas denote by $P(s_d,a)$.

### 6.2 Pseudovisit $Q$

Fix depth $d$ and abbreviate $s_d$ as $s$ throughout this subsection. A **pseudovisit** is one unit of fixed statistical weight attached to the action-value estimate

$$
Q_{\mathrm{prior}}(s,a)=
\mathrm{clip}\left(V_d+A_d(a),-1,1\right).
$$

Completed MCTS visits supply the empirical contribution to the same edge statistic.

Let $N(s,a)$ be the number of completed visits to edge $(s,a)$, let $N(s)=\sum_aN(s,a)$, and let $Q_{\mathrm{MCTS}}(s,a)$ be the mean backed-up return from the perspective of the player at the parent node. For a visited edge, the pseudovisit and empirical returns are combined as

$$
Q_{\mathrm{edge}}(s,a)=
\frac{N(s,a)Q_{\mathrm{MCTS}}(s,a)+Q_{\mathrm{prior}}(s,a)}
{N(s,a)+1},
\qquad N(s,a)>0.
$$

For an unvisited edge, FPU subtracts an exploration-dependent reduction from $Q_{\mathrm{prior}}(s,a)$. With $r_{\mathrm{FPU}}\geq0$ set by `--fpu-reduction`,

$$
Q_{\mathrm{edge}}(s,a)=
\mathrm{clip}\left(
Q_{\mathrm{prior}}(s,a)
-r_{\mathrm{FPU}}\sqrt{\sum_{b:N(s,b)>0}P(s,b)},-1,1
\right),
\qquad N(s,a)=0.
$$

### 6.3 PUCT Selection

During batched selection, virtual visits prevent concurrent paths from repeatedly choosing the same edge. Let $N_v(s,a)$ be the number reserved on edge $(s,a)$, and let $N_v(s)=\sum_aN_v(s,a)$. Selection uses augmented counts

$$
\widetilde N(s,a)=N(s,a)+N_v(s,a),
\qquad
\widetilde N(s)=N(s)+N_v(s).
$$

Let $c_0$, $b_0$ and $f_0$ be the values supplied by `--c-puct`, `--c-puct-base` and `--c-puct-factor`. Define $b=\max(1,b_0)$ and $f=\max(0,f_0)$. The visit-dependent exploration coefficient is

$$
c_{\mathrm{puct}}(\widetilde N)=
\max\left(0,c_0+f\log\left(\frac{\widetilde N+b+1}{b}\right)\right).
$$

Let $l_v=\max(0,\texttt{--virtual-loss})$. The PUCT selection score is

$$
S(s,a)=Q_{\mathrm{edge}}(s,a)
+c_{\mathrm{puct}}(\widetilde N(s))P(s,a)
\frac{\sqrt{\widetilde N(s)+1}}{1+\widetilde N(s,a)}
-l_vN_v(s,a).
$$

Equal selection scores are ordered by descending $P(s,a)$ and then by descending $Q_{\mathrm{edge}}(s,a)$.

Let $V_{\mathrm{leaf}}$ be the scalar evaluation assigned to an evaluated leaf. MCTS backs up $V_{\mathrm{leaf}}$ and reverses its sign at every ply. This backup increments $N(s,a)$ and updates $Q_{\mathrm{MCTS}}(s,a)$ along the selected path.

When search ends, each legal root move receives weight $N(s,a)+P(s,a)$. Their normalized weights form the root move distribution

$$
P_{\mathrm{root}}(a\mid s)=
\frac{N(s,a)+P(s,a)}
{\sum_{b\in\mathcal A(x)}\left(N(s,b)+P(s,b)\right)}.
$$

### 6.4 Dynamic Simulation Budget

Let $N_{\mathrm{cap}}=\max(0,\texttt{--mcts-sims})$, $B_{\mathrm{batch}}=\max(1,\texttt{--mcts-batch-size})$ and $N_{\mathrm{floor}}=\max(0,\texttt{--mcts-min-sims})$. The nominal minimum simulation count is

$$
N_{\min}=
\begin{cases}
0,&N_{\mathrm{cap}}=0,\\
\max\!\left(1,\min\!\left(N_{\mathrm{cap}},N_{\mathrm{floor}}\right)\right),
&N_{\mathrm{cap}}>0\ \text{and}\ N_{\mathrm{floor}}>0,\\
\max\!\left(1,\min\!\left(N_{\mathrm{cap}},
\max\!\left(B_{\mathrm{batch}},\left\lfloor\dfrac{N_{\mathrm{cap}}}{4}\right\rfloor\right)
\right)\right),
&N_{\mathrm{cap}}>0\ \text{and}\ N_{\mathrm{floor}}=0.
\end{cases}
$$

A `movetime` deadline or UCI `stop` request may end search before $N_{\min}$ simulations. After $N_{\min}$ simulations have completed, a root with at least two legal actions uses the empirical visit distribution

$$
v_a=\frac{N(s,a)}{\sum_{b\in\mathcal A(x)}N(s,b)}.
$$

Let $a_1$ and $a_2$ be the two actions with the largest completed visit counts, and write $N_i=N(s,a_i)$. Define $Q_i=Q_{\mathrm{MCTS}}(s,a_i)$ when $N_i>0$ and $Q_i=0$ when $N_i=0$. The normalized visit entropy $H_N$, visit closeness $U_N$ and $Q_{\mathrm{MCTS}}$ closeness $U_Q$ are

$$
H_N=-\frac{\sum_{a\in\mathcal A(x)}v_a\log v_a}
{\log|\mathcal A(x)|},
$$

$$
U_N=1-\frac{|N_1-N_2|}{\max(1,N_1+N_2)},
\qquad
U_Q=1-\min\left(1,\frac{|Q_1-Q_2|}{0.5}\right).
$$

The combined uncertainty and resulting simulation target are

$$
u=\mathrm{clip}(0.5H_N+0.35U_N+0.15U_Q,0,1),
$$

$$
N_{\mathrm{target}}=
N_{\min}+\left\lceil u(N_{\mathrm{cap}}-N_{\min})\right\rceil.
$$

A root with one legal action uses $u=0$ and $N_{\mathrm{target}}=N_{\min}$.

### 6.5 Final Decision Components

The optional IMF and RPP rules operate on the final ranking rather than on the search tree. Before either rule is applied, the ranking score is

$$
D_0(a)=
\begin{cases}
P_\theta(a\mid s),&\texttt{closed},\\
P_{\mathrm{root}}(a\mid s),&\texttt{only-mcts}.
\end{cases}
$$

Let $\mathcal M(x)$ be the set of legal actions that immediately checkmate the opponent. If this set is nonempty, IMF selects

$$
a_M=\arg\max_{a\in\mathcal M(x)}D_0(a).
$$

IMF then defines

$$
D_I(a)=
\begin{cases}
1,&a=a_M,\\
D_0(a),&a\in\mathcal A(x)\setminus\lbrace a_M\rbrace.
\end{cases}
$$

When $\mathcal M(x)$ is empty, $D_I(a)=D_0(a)$ for every legal action.

Let $\lambda_R\in[0,1]$ be `--repetition-policy-penalty`, and let $V_R$ be the $V$ returned for the root by search. The set $\mathcal R_3(x)$ contains legal moves that either make a threefold-repetition claim available immediately or allow the opponent to do so with one reply. RPP computes

$$
d_R=\lambda_R\mathrm{clip}(V_R,0,1).
$$

RPP then produces the final decision score

$$
D(a)=
\begin{cases}
\max(0,D_I(a)-d_R),&a\in\mathcal R_3(x),\\
D_I(a),&a\in\mathcal A(x)\setminus\mathcal R_3(x).
\end{cases}
$$

The decision layer sorts legal moves by descending $D(a)$, then by descending $D_0(a)$ and finally by descending UCI notation. The first move in this order is selected.

## 7. Arena

The Melano arena executable loads both checkpoints once and advances several games concurrently. Positions evaluated by the same checkpoint are combined into inference batches. `--games` must be a positive even number because each sampled opening is played once with the candidate as White and once with the candidate as Black. `--games-in-flight` limits the number of active games.

Let $N_W$, $N_D$ and $N_L$ be the candidate's win, draw and loss counts, and let $N_{\mathrm{games}}=N_W+N_D+N_L$. The candidate score and net wins are

$$
score=\frac{N_W+\frac12N_D}{N_{\mathrm{games}}},
\qquad
net\_wins=N_W-N_L.
$$

Let $x_i\in\lbrace 0,\frac12,1\rbrace$ be the candidate score in game $i$. The population variance of these scores is

$$
\sigma^2=\frac1{N_{\mathrm{games}}}\sum_{i=1}^{N_{\mathrm{games}}}(x_i-score)^2.
$$

The implementation reports the clipped 95% normal-approximation interval

$$
CI_{95\%}=\mathrm{clip}\left(
score\pm1.96\sqrt{\frac{\sigma^2}{N_{\mathrm{games}}}},0,1
\right).
$$

For display, define $score_b=\mathrm{clip}(score,10^{-6},1-10^{-6})$. The reported Elo difference is

$$
\Delta Elo=400\log_{10}\left(\frac{score_b}{1-score_b}\right).
$$

Let $M_{\mathrm{gate}}$ be `--min-net-wins`. The arena result gate accepts the candidate when

$$
N_W-N_L\geq M_{\mathrm{gate}}.
$$

## 8. Arena Opening Book

Paired arena evaluation uses positions sampled at a fixed ply and filtered by a UCI evaluation bound. Each selected position is played once with each color assignment.

## 9. UCI Engine

### 9.1 Evaluation Output

The engine reports MultiPV lines, side-to-move `score cp`, nodes, NPS, elapsed time and a one-move PV. For a visited MCTS root edge, the line evaluation $q_{line}$ is $Q_{\mathrm{edge}}(s,a)$. For `closed` search or an unvisited MCTS root edge, $q_{line}=V_\theta(s)$. Let $c_s$ be `ScoreScale`. The displayed score is

$$
score\_cp=\mathrm{round}\left(
c_s\mathrm{clip}(q_{line},-0.999,0.999)
\right).
$$

### 9.2 Options

Melano exposes the following UCI options.

- `ModelPath` selects a Melano checkpoint. A packaged engine defaults to `melano.pth` in the executable directory.
- `Device` selects `auto`, `cpu` or `cuda`.
- `SearchType` selects direct policy ranking with `closed` or anchored latent MCTS with `only-mcts`. Both modes apply the enabled final decision components.
- `MCTSSims` sets the MCTS simulation cap and defaults to `100`. A UCI command `go nodes <n>` uses `<n>` as the current cap.
- `MCTSMinSims` sets the nominal minimum simulation count before dynamic budgeting and defaults to `0`. A zero value activates the formula derived from the cap and batch size. A time limit or UCI `stop` command may end search before this count is reached.
- `MCTSBatchSize` sets the neural leaf batch size and defaults to `32`.
- `MoveTimeMS` sets the fixed thinking time used when `go` supplies neither `movetime` nor a clock for the side to move. Its default is `0`.
- `MoveOverheadMS` reserves communication and move-submission time from a clock allocation. Its default is `50`.
- `MinMoveTimeMS` and `MaxMoveTimeMS` bound clock-based thinking time and default to `50` and `10000`.
- `TimeDivisor` allocates a fraction of remaining time through division and defaults to `30.0`.
- `IncrementFraction` allocates a fraction of the increment and defaults to `0.75`.
- `CPuct`, `CPuctBase` and `CPuctFactor` set the PUCT exploration schedule and default to `0.5`, `19652` and `1.0`.
- `FPUReduction` sets the FPU reduction and defaults to `0.15`.
- `VirtualLoss` sets the repeated-path penalty within one batched selection and defaults to `0.0`.
- `RepetitionPolicyPenalty` sets $\lambda_R$ in the RPP decision component, accepts $[0,1]$ and defaults to `0.0`.
- `InstantMateFirst` enables IMF and defaults to `false`.
- `ProgressIntervalMS` sets the interval between intermediate UCI `info` reports and defaults to `750`. A zero interval suppresses intermediate reports.
- `MultiPV` sets the number of reported analysis lines and defaults to `5`.
- `ScoreScale` sets $c_s$ for converting the engine's dimensionless evaluation to the displayed centipawn scale and defaults to `1000`.

The engine publishes its direct policy ranking when search begins and publishes intermediate MCTS results at intervals of `ProgressIntervalMS`. Search limits and the UCI `stop` command determine the final result.

## 10. Implementation Tests

The build process runs the Melano CTest executable before publishing binaries. The test suite covers `melano_square_tokens`, ordinary move and promotion encoding, terminal-state detection, policy, value and advantage output shapes, the advantage range, action-conditioned latent successors, anchored latent MCTS with $K=2$, finite numerical results, backward propagation and checkpoint round trips.
