# Gadus

Gadus is a residual policy-value network for chess with an AlphaZero-style action space.

## 1. Notation and Abbreviations

> Throughout this document, $V$, $Q$ and terminal outcomes use the perspective of the side to move. A definition states its perspective explicitly when it uses a different one.

- $\mathcal X$ is the set of complete game states. A state $x\in\mathcal X$ contains the board, side to move, castling rights, en passant state, halfmove clock and repetition history.
- $\mathcal A(x)$ is the set of legal actions in $x$. An action is written as $a\in\mathcal A(x)$, and $T(x,a)$ is the deterministic successor state produced by the chess rules.
- $z(x)\in\lbrace -1,0,1\rbrace$ is the outcome of a terminal state $x$ from the perspective of the side to move in $x$.
- $\mathcal S_G$ is the Gadus encoded-state space. The map $\phi_G:\mathcal X\rightarrow\mathcal S_G$ is the Gadus state encoder, and $s=\phi_G(x)$ is the network input corresponding to $x$.
- $\mathcal I_G$ is the Gadus action-index set, and $i_G(a)\in\mathcal I_G$ is the index assigned to action $a$.
- $\theta$ denotes trainable network parameters. $\ell_\theta(s,i)$ is the policy logit for action index $i$, $P_\theta(a\mid s)$ is the policy probability assigned to legal move $a$, and $V_\theta(s)$ is the scalar evaluation that the model assigns to $s$.
- $Q(s,a)$ denotes a scalar evaluation of legal move $a$ in encoded position $s$. A subscript on $Q$ identifies the estimator that supplies the evaluation.
- A hat, as in $\widehat y$, denotes an estimate produced by finite data or finite search. A bar, as in $\overline y$, denotes a backed-up or aggregated quantity.
- A target is a fixed scalar or distribution used as the comparison value in a loss. Automatic differentiation treats targets as constants.
- $L_{\mathrm{CE}}(p,q)=-\sum_iq_i\log p_i$ is the cross-entropy from target distribution $q$ to predicted distribution $p$. $\mathrm{MSE}(u,v)$ is the mean squared error between corresponding elements.
- $D_{\mathrm{KL}}(p\,\|\,q)=\sum_ip_i\log(p_i/q_i)$ is the Kullback-Leibler divergence from distribution $p$ to distribution $q$.
- Subscripts `old` and `new` identify the frozen source model and the trainable candidate within one FCPI iteration. A superscript `+` identifies a policy target produced by local policy improvement.

The following abbreviations are used throughout this document.

- **PGN**: Portable Game Notation.
- **FEN**: Forsyth-Edwards Notation.
- **UCI**: Universal Chess Interface.
- **HDF5**: Hierarchical Data Format 5.
- **ResNet**: residual network.
- **MLP**: multilayer perceptron.
- **MCTS**: Monte Carlo Tree Search.
- **PUCT**: Predictor plus Upper Confidence bounds applied to Trees.
- **FPU**: First Play Urgency.
- **MC**: Monte Carlo. In the FCPI sections, MC denotes a terminal return propagated along one completed trajectory.
- **FCPI**: Folded Counterfactual Policy Iteration.
- **KL**: Kullback-Leibler divergence.
- **CE**: cross-entropy.
- **MSE**: mean squared error.
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

The `gadus_18_planes` state encoding contains 12 piece planes, one side-to-move plane, four castling-right planes and one en passant file plane. Bit-packing each rank into one byte gives an HDF5 state size of $18\times8$ bytes.

The `alphazero_64x73` action encoding assigns 73 action planes to each source square. The planes contain 56 sliding moves in eight directions, eight knight moves and nine underpromotion moves. The action-space size is

$$
|\mathcal I_G|=64\times(56+8+9)=4672.
$$

`PackedState` is the fixed byte representation produced by `gadus_18_planes`. FCPI uses it as the grouping key for positions that have identical network inputs.

## 3. Network

The Gadus network contains a ResNet trunk, a linear policy head and an MLP value head. Let $C$ be the channel count supplied by `--channels`, and let $B$ be the residual-block count supplied by `--blocks`. For encoded input $s\in\mathbb R^{18\times8\times8}$, the stem applies a bias-free $3\times3$ convolution from 18 input planes to $C$ channels, followed by BatchNorm and ReLU:

$$
h_0=\mathrm{ReLU}\left(
\mathrm{BN}_{\mathrm{stem}}\left(
\mathrm{Conv}^{18\rightarrow C}_{3\times3,\mathrm{stem}}(s)
\right)\right)
\in\mathbb R^{C\times8\times8}.
$$

The tensor $h_0$ is the input to residual block 0. The residual transform in block $j$ is

$$
F_j(h)=\mathrm{BN}_{j,2}\left(
\mathrm{Conv}^{C\rightarrow C}_{3\times3,j,2}\left(
\mathrm{ReLU}\left(
\mathrm{BN}_{j,1}\left(
\mathrm{Conv}^{C\rightarrow C}_{3\times3,j,1}(h)
\right)\right)\right)\right),
$$

Residual block $j$ then produces

$$
h_{j+1}=\mathrm{ReLU}(h_j+F_j(h_j)),\qquad 0\leq j<B.
$$

For $j=0$, the recurrence computes $h_1$ from $h_0$. Repeating the recurrence through block $B-1$ produces the trunk output $h_B$. Every $3\times3$ convolution uses one-square padding and preserves the $8\times8$ spatial dimensions, so each tensor $h_j$ lies in $\mathbb R^{C\times8\times8}$.

$h_B$ is used as the common input to both output heads. The policy head applies a bias-free $1\times1$ convolution from $C$ channels to 32 channels, followed by BatchNorm, ReLU, flattening and a linear map from 2048 features to 4672 logits. The value head has an independent $1\times1$ convolution from $C$ channels to 32 channels, followed by BatchNorm, ReLU, flattening, a 2048-to-256 linear layer, ReLU, a scalar output layer and a hyperbolic tangent.

Writing $f_\theta$ for the complete network, its forward pass is

$$
(\ell_\theta(s),V_\theta(s))=f_\theta(s),\qquad
\ell_\theta(s)\in\mathbb R^{4672},\quad V_\theta(s)\in[-1,1].
$$

For game state $x$ with $s=\phi_G(x)$, legal-move inference selects the corresponding logits and normalizes them with a softmax:

$$
P_\theta(a\mid s)=
\frac{\exp \ell_\theta(s,i_G(a))}
{\displaystyle \sum_{b\in\mathcal A(x)}\exp \ell_\theta(s,i_G(b))}.
$$

## 4. Preprocessing

Gadus preprocessing writes the following HDF5 schema:

```text
states: uint8,  (N, 18, 8)
moves:  uint16, (N,)
values: float32, (N,)

arch_type=gadus
state_encoding=gadus_18_planes
move_encoding=alphazero_64x73
target_schema=pv_supervised
has_cmt=0|1
```

`--has-cmt` defaults to `1`. In this mode, the preprocessor reads White-perspective pawn-unit evaluations from comments of the form `{+x}` and `{-x}`. At ply $t$, let $x_t$ be the game state before the recorded move and let $s_t=\phi_G(x_t)$. A comment attached to a PGN move describes the position produced by that move, so the target for $s_t$ comes from the comment attached to the preceding move.

Let $q_t$ be the White-perspective evaluation assigned to $x_t$ by the preceding move's comment. Define $\rho(x_t)=1$ when White is to move and $\rho(x_t)=-1$ when Black is to move. The corresponding side-to-move evaluation and value target are

$$
q_{\mathrm{stm}}(s_t)=\rho(x_t)q_t,
\qquad
V_{\mathrm{target}}(s_t)=\tanh\left(\frac{q_{\mathrm{stm}}(s_t)}{3}\right).
$$

If the preceding comment contains no parseable evaluation, preprocessing sets $q_{\mathrm{stm}}=0$. In `--has-cmt 1` mode, a game is included when at least one comment contains a parseable evaluation. In `--has-cmt 0` mode, $V_{\mathrm{target}}\in\lbrace -1,0,1\rbrace$ is derived from the final result and expressed from the side-to-move perspective.

## 5. Supervised Training

Let $a^{\ast}$ be the move recorded in the PGN, and let $i^{\ast}=i_G(a^{\ast})$ be its action index. For a network input $s$, the Policy head produces one logit $\ell_\theta(s,i)$ for every action index $i\in\mathcal I_G$. Applying softmax to all 4672 logits gives the supervised Policy distribution $R_\theta$:

$$
R_\theta(i\mid s)=
\frac{\exp\ell_\theta(s,i)}
{\displaystyle\sum_{j\in\mathcal I_G}\exp\ell_\theta(s,j)}.
$$

$R_\theta$ and the legal-move distribution $P_\theta$ defined in Section 3 use the same logits. $R_\theta$ normalizes over the complete action-index set during supervised training, whereas $P_\theta$ normalizes over the legal moves of the current game state during inference.

Let $\delta_{i^{\ast}}$ be the one-hot target distribution that assigns probability 1 to $i^{\ast}$ and probability 0 to every other action index. The supervised Policy loss is

$$
L_{P,\mathrm{sup}}=
L_{\mathrm{CE}}\left(R_\theta(\cdot\mid s),\delta_{i^{\ast}}\right)
=-\log R_\theta(i^{\ast}\mid s).
$$

The Policy loss $L_{P,\mathrm{sup}}$ decreases as the probability assigned to the recorded move increases.

The Value head predicts $V_\theta(s)$, and the preprocessing stage supplies $V_{\mathrm{target}}(s)$. Their mean-squared error defines the supervised Value loss:

$$
L_{V,\mathrm{sup}}=
\mathrm{MSE}\left(V_\theta(s),V_{\mathrm{target}}(s)\right).
$$

Let $w_V$ be the nonnegative loss coefficient given by `--value-weight`. The complete supervised objective is

$$
L_{\mathrm{sup}}=
L_{P,\mathrm{sup}}+w_VL_{V,\mathrm{sup}}.
$$

Each training invocation initializes a new Gadus model. `--channels` and `--blocks` determine its width and depth. `--max-steps` limits the number of optimizer steps, and `--save-every` sets the interval between atomic checkpoint writes. `--seed` controls parameter initialization and dataset shuffling.

`--precision` accepts `fp32` or `bf16` and defaults to `fp32`. CUDA forward computation uses BF16 in `bf16` mode. The Policy softmax, loss calculations, metric accumulation and checkpoint parameters remain in FP32. CUDA training batches use pinned host memory.

Each checkpoint contains exactly two top-level keys:

```text
model #stores the network parameters
arch  #stores the Gadus identifier and the dimensions required to reconstruct the network
```

## 6. Search

### 6.1 Search Modes

Search mode `closed` derives its initial move ranking directly from the model policy. Search mode `only-mcts` evaluates leaf nodes in neural batches and derives its initial ranking from the resulting MCTS root distribution. Both modes then apply the enabled IMF and RPP decision components.

Within `only-mcts` mode, expanding node $s$ creates one outgoing edge $(s,a)$ for each legal action $a$ and stores policy probability $P(s,a)$ as that edge's prior. Each prior remains fixed throughout the search. The completed visit counts of an edge and its parent node are denoted by $N(s,a)$ and $N(s)=\sum_aN(s,a)$. During batched selection, virtual visits reserve edges for concurrent paths and reduce repeated selection of the same edge. Their corresponding counts are $N_v(s,a)$ and $N_v(s)=\sum_aN_v(s,a)$. Selection combines completed and virtual visits into the augmented counts:

$$
\widetilde N(s,a)=N(s,a)+N_v(s,a),
\qquad
\widetilde N(s)=N(s)+N_v(s).
$$

### 6.2 PUCT Selection

Let $c_0$, $b_0$ and $f_0$ be the values supplied by `--c-puct`, `--c-puct-base` and `--c-puct-factor`. Define $b=\max(1,b_0)$ and $f=\max(0,f_0)$. The visit-dependent exploration coefficient is

$$
c_{\mathrm{puct}}(\widetilde N)=
\max\left(0,c_0+f\log\left(\frac{\widetilde N+b+1}{b}\right)\right).
$$

An evaluated leaf supplies the scalar value $V_{\mathrm{leaf}}$. MCTS backs up this value along the selected path and reverses its sign at every ply. Each completed backup increments $N(s,a)$ and updates the mean return $Q(s,a)$ from the perspective of the player at the parent node $s$.

The selection estimate $Q_{\mathrm{sel}}(s,a)$ equals $Q(s,a)$ when $N(s,a)>0$. For an unvisited edge, $Q(s)$ denotes the mean return backed up to node $s$ from the perspective of the player to move at that node. First Play Urgency combines $Q(s)$ with the explored prior mass. With $r_{\mathrm{FPU}}=\max(0,\texttt{--fpu-reduction})$, the two cases are

$$
Q_{\mathrm{sel}}(s,a)=
\begin{cases}
Q(s,a), & N(s,a)>0,\\[4pt]
\mathrm{clip}\left(
Q(s)-r_{\mathrm{FPU}}\sqrt{\displaystyle\sum_{a':N(s,a')>0}P(s,a')},-1,1
\right), & N(s,a)=0.
\end{cases}
$$

With $l_v=\max(0,\texttt{--virtual-loss})$, the PUCT selection score is

$$
S(s,a)=Q_{\mathrm{sel}}(s,a)
+c_{\mathrm{puct}}(\widetilde N(s))P(s,a)
\frac{\sqrt{\widetilde N(s)+1}}{1+\widetilde N(s,a)}
-l_vN_v(s,a).
$$

Equal selection scores are ordered by descending $P(s,a)$ and then by descending $Q_{\mathrm{sel}}(s,a)$.

When search ends, each legal root move receives weight $N(s,a)+P(s,a)$. Their normalized weights form the root move distribution

$$
P_{\mathrm{root}}(a\mid s)=
\frac{N(s,a)+P(s,a)}
{\displaystyle\sum_{a'\in\mathcal A(x)}\left(N(s,a')+P(s,a')\right)}.
$$

### 6.3 Dynamic Simulation Budget

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
v_a=\frac{N(s,a)}{\displaystyle\sum_{b\in\mathcal A(x)}N(s,b)}.
$$

Let $a_1$ and $a_2$ be the two actions with the largest completed visit counts, and write $N_i=N(s,a_i)$. Define $Q_i=Q(s,a_i)$ when $N_i>0$ and $Q_i=0$ when $N_i=0$. The normalized visit entropy $H_N$, visit closeness $U_N$ and $Q$ closeness $U_Q$ are

$$
H_N=-\frac{\sum_{a\in\mathcal A(x)}v_a\log v_a}
{\log|\mathcal A(x)|},
$$

$$
U_N=1-\frac{|N_1-N_2|}{\max(1,N_1+N_2)},
$$

$$
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

### 6.4 Final Decision Components

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

When $\mathcal M(x)$ is empty, $D_I(a)$ equals $D_0(a)$ for every legal action.

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

The Gadus arena executable loads both checkpoints once and advances several games concurrently. Positions evaluated by the same checkpoint are combined into inference batches. `--games` must be a positive even number because each sampled opening is played once with the candidate as White and once with the candidate as Black. `--games-in-flight` limits the number of active games.

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

Let $M_{\mathrm{gate}}$ be the minimum net wins supplied by `--min-net-wins`. The arena result gate accepts the candidate when

$$
N_W-N_L\geq M_{\mathrm{gate}}.
$$

## 8. Folded Counterfactual Policy Iteration

### 8.1 Source Model, Candidate and Training Targets

Let $C_r$ be the current model at the start of FCPI iteration $r$, with parameters $\theta_{old}$. FCPI freezes $\theta_{old}$ while constructing the training set. For encoded position $s$, the source model returns $P_{old}(\cdot\mid s)$ and $V_{old}(s)$.

The candidate begins as a copy of the source model:

$$
\theta_{new}^{(0)}=\theta_{old}.
$$

The trainable candidate parameters are denoted by $\theta_{new}$, and the corresponding outputs are denoted by $P_{new}(\cdot\mid s)$ and $V_{new}(s)$.

FCPI constructs targets from completed self-play trajectories and finite counterfactual trees. A completed trajectory assigns terminal return $G_t$ to each recorded position and MC advantage coefficient $A_{\mathrm{MC}}(s_t,a_t)$ to the move selected there. A counterfactual tree computes $Q_{\mathrm{CF}}(s,a)$ for legal moves and then derives $\pi^+(\cdot\mid s)$ and $\overline V_{\mathrm{CF}}(s)$.

After aggregating these targets, FCPI trains the candidate with the objective defined in Section 8.7. AdamW updates $\theta_{new}$ from the gradients of that objective. A paired-game arena then compares the trained candidate with $C_r$ and promotes the candidate by atomic checkpoint replacement when it meets the acceptance threshold.

### 8.2 Starting Positions and Behavior Policy

When `--opening-book` names a sampling book, FCPI draws `--games-per-iter` distinct nonterminal positions without replacement from a pool of at most `--max-book-positions` FENs. The pool may include positions from any ply, including the standard initial position. With an empty book path, every game begins from the standard initial position. Iteration $r$ shuffles the pool with seed `--seed + r`.

Self-play uses the frozen `closed` policy of $C_r$. Let $T_b$ be `--behavior-temperature`, and define the effective temperature

$$
\widetilde T_b=\max(T_b,10^{-4}).
$$

The behavior distribution over legal actions is

$$
\mu_{old}(a\mid s)=
\frac{P_{old}(a\mid s)^{1/\widetilde T_b}}
{\sum_{b\in\mathcal A(x)}P_{old}(b\mid s)^{1/\widetilde T_b}}.
$$

Values $T_b<1$ concentrate probability on moves favored by the source policy, while values $T_b>1$ flatten the distribution.

FCPI tracks cumulative move counts for each Gadus `PackedState` throughout an iteration. Suppose encoded position $s$ is encountered for the $n$th time, and let $N_{n-1}(s,a)$ count how often move $a$ was chosen during the preceding $n-1$ encounters. The behavior distribution assigns desired cumulative count $n\mu_{old}(a\mid s)$ to move $a$. FCPI chooses the move with the largest deficit between desired and observed counts:

$$
a_n=\arg\max_{a\in\mathcal A(x)}
\left[n\mu_{old}(a\mid s)-N_{n-1}(s,a)\right].
$$

The legal-move array determines the tie-break order. Before playing $a_t$, FCPI records the FEN of $x_t$, encoded position $s_t$, $V_{old}(s_t)$, legal action indices, $P_{old}(\cdot\mid s_t)$ and selected move $a_t$. The trajectory ends at a terminal state or after `--max-plies` recorded plies.

### 8.3 Terminal Monte Carlo Targets

Consider a completed trajectory containing pre-move states $x_0,\ldots,x_{T_{\mathrm{traj}}-1}$. Its final move reaches terminal state $x_{T_{\mathrm{traj}}}$. Starting from terminal outcome $z(x_{T_{\mathrm{traj}}})$, FCPI reverses perspective at each ply:

$$
G_{T_{\mathrm{traj}}-1}=-z(x_{T_{\mathrm{traj}}}),
\qquad
G_t=-G_{t+1}\quad(0\leq t<T_{\mathrm{traj}}-1).
$$

Thus $G_t\in\lbrace -1,0,1\rbrace$ is the game outcome from the perspective of the player to move in $x_t$. Completed trajectories assign each recorded state weight

$$
w_{\mathrm{MC}}(s_t)=1.
$$

Each state in a truncated trajectory receives $w_{\mathrm{MC}}(s_t)=0$ because the trajectory supplies no terminal outcome.

For a completed trajectory, the selected move receives MC advantage coefficient

$$
A_{\mathrm{MC}}(s_t,a_t)=
\mathrm{clip}_{[-1,1]}
\left(\frac{G_t-V_{old}(s_t)}{2}\right).
$$

The selected move $a_t$ is the sole action with MC policy weight one in this record, and its signed coefficient is $A_{\mathrm{MC}}(s_t,a_t)$.

### 8.4 Finite Counterfactual Trees

Before constructing counterfactual trees, FCPI deduplicates each trajectory by Gadus `PackedState` and keeps the first occurrence of each encoded position. Occurrences with the same `PackedState` produce the same network input, so this step prevents that input from generating multiple trees and receiving repeated weight within one trajectory. Each retained occurrence supplies the FEN that initializes the root of a separate tree, regardless of whether the trajectory ended naturally or reached the ply limit. The FEN restores the board, side to move, castling rights, en passant state and move counters. It does not contain positions visited earlier in the trajectory, so the reconstructed root begins with an empty repetition history. The tree records every move played after that root and applies the chess termination rules to the resulting descendants. It can therefore only detect a threefold repetition when all three occurrences of the repeated position lie on the path from the root to the current node.

Let $B_{\mathrm{CF}}$ denote `--counterfactual-budget`. This budget limits the number of edges evaluated below the root of each tree. An edge evaluation plays one legal move and assigns the successor either its exact terminal outcome or $V_{old}$ from the frozen source model.

FCPI evaluates every legal root move outside this budget. Consequently, $B_{\mathrm{CF}}=0$ still yields a complete one-ply evaluation at the root, while $B_{\mathrm{CF}}>0$ permits further expansion of selected descendants.

Let $B_{\mathrm{rem}}$ be the number of evaluations from $B_{\mathrm{CF}}$ that remain when FCPI expands non-root node $x$. For $B_{\mathrm{rem}}>0$, the expansion width is

$$
w(x)=\min\left(
|\mathcal A(x)|,
B_{\mathrm{rem}},
\max\left(2,\left\lceil\sqrt{B_{\mathrm{rem}}}\right\rceil\right)
\right).
$$

The expansion set always includes the move with highest source-policy probability. Gumbel top-k samples the remaining $w(x)-1$ moves without replacement. For each remaining legal move, define

$$
k(a)=\log\left(
\mathrm{clip}(P_{old}(a\mid s),10^{-12},1)
\right)+g_a,
\qquad
g_a=-\log(-\log u_a),
\quad u_a\sim U(0,1).
$$

FCPI selects the highest $k(a)$ values until the expansion set reaches width $w(x)$. The Gumbel generator uses seed `--seed + 3000017`.

Each node in a counterfactual tree stores one environment state. For a given tree, let $\mathcal X_T$ be the set of terminal states stored in its nodes and let $\mathcal X_O$ be the set of stored nonterminal states that have legal moves. The leaf-evaluation function assigns the rule outcome to a state in $\mathcal X_T$ and the frozen source-model evaluation to a state in $\mathcal X_O$:

$$
v_{\mathrm{leaf}}(x)=
\begin{cases}
z(x),&x\in\mathcal X_T,\\
V_{old}(\phi_G(x)),&x\in\mathcal X_O.
\end{cases}
$$

The root has reach probability one. If edge $(x,a)$ leads to $x'=T(x,a)$, its reach probability is

$$
p_{\mathrm{reach}}(x')=
p_{\mathrm{reach}}(x)P_{old}(a\mid\phi_G(x)).
$$

Let $d(x')$ be the ply depth of $x'$ relative to the root. A newly created nonterminal child receives priority

$$
priority(x')=p_{\mathrm{reach}}(x')
\left(
\left|-v_{\mathrm{leaf}}(x')-V_{old}(\phi_G(x))\right|
+\frac{1}{\sqrt{2+d(x')}}
\right).
$$

The frontier contains the nonterminal nodes eligible for further expansion. FCPI repeatedly expands the highest-priority node until the tree has spent $B_{\mathrm{CF}}$ edge evaluations or the frontier is empty.

Backup jointly computes edge values $Q_{\mathrm{CF}}$ and node values $\overline V_{\mathrm{CF}}$ from deeper nodes toward the root. A nonterminal frontier node $x'$ without evaluated descendants is initialized by the source model:

$$
\overline V_{\mathrm{CF}}(\phi_G(x'))=V_{old}(\phi_G(x')).
$$

Let $E(x)\subseteq\mathcal A(x)$ be the moves evaluated explicitly at node $x$. Once the backed-up values of its evaluated children are available, node $x$ uses $s=\phi_G(x)$ and $x_a'=T(x,a)$ to define

$$
Q_{\mathrm{CF}}(s,a)=
\begin{cases}
-z(x_a'),&a\in E(x),\ x_a'\in\mathcal X_T,\\
-\overline V_{\mathrm{CF}}(\phi_G(x_a')),
&a\in E(x),\ x_a'\in\mathcal X_O,\\
V_{old}(s),&a\in\mathcal A(x)\setminus E(x).
\end{cases}
$$

An evaluated move that ends the game receives the exact terminal outcome from the parent's perspective. An evaluated nonterminal move receives the negated value backed up from its child. The final branch of the definition assigns $Q_{\mathrm{CF}}(s,a)=V_{old}(s)$ to every unevaluated move. Section 8.5 defines the backed-up value of an expanded node from its completed set of edge values.

### 8.5 Counterfactual Policy and Value Targets

At an expanded node, the $P_{old}$-weighted mean of $Q_{\mathrm{CF}}$ is

$$
m(s)=\sum_{a\in\mathcal A(x)}
P_{old}(a\mid s)Q_{\mathrm{CF}}(s,a).
$$

The clipped source-policy weight is defined by

$$
\widetilde p(a\mid s)=
\mathrm{clip}(P_{old}(a\mid s),10^{-12},1).
$$

The implementation fixes the counterfactual-policy temperature at $T_{\mathrm{CF}}=1$. Thus the target distribution is

$$
\pi^+(a\mid s)=
\frac{\widetilde p(a\mid s)
\exp\left(Q_{\mathrm{CF}}(s,a)-m(s)\right)}
{\displaystyle\sum_{b\in\mathcal A(x)}\widetilde p(b\mid s)
\exp\left(Q_{\mathrm{CF}}(s,b)-m(s)\right)}.
$$

The common factor $\exp[-m(s)]$ cancels during normalization. The normalized source-policy weights define the reference distribution

$$
p_\varepsilon(a\mid s)=
\frac{\widetilde p(a\mid s)}
{\displaystyle\sum_{b\in\mathcal A(x)}\widetilde p(b\mid s)}.
$$

Among all probability distributions $\pi(\cdot\mid s)$ over $\mathcal A(x)$, the same $\pi^+$ uniquely maximizes the following local objective with KL-regularization coefficient equal to 1:

$$
\pi^+=\arg\max_\pi
\left[
\sum_a\pi(a\mid s)Q_{\mathrm{CF}}(s,a)
-D_{\mathrm{KL}}\left(
\pi(\cdot\mid s)\,\|\,p_\varepsilon(\cdot\mid s)
\right)
\right].
$$

$\overline V_{\mathrm{CF}}(s)$ is the expectation of $Q_{\mathrm{CF}}(s,a)$ under $\pi^+(\cdot\mid s)$:

$$
\overline V_{\mathrm{CF}}(s)=
\mathrm{clip}_{[-1,1]}
\left(
\sum_{a\in\mathcal A(x)}
\pi^+(a\mid s)Q_{\mathrm{CF}}(s,a)
\right).
$$

Define the correction relative to $V_{old}(s)$ as

$$
\delta_{\mathrm{CF}}(s)=
\overline V_{\mathrm{CF}}(s)-V_{old}(s).
$$

Every unevaluated move satisfies $Q_{\mathrm{CF}}(s,a)=V_{old}(s)$, so the correction reduces to a sum over explicitly evaluated moves:

$$
\delta_{\mathrm{CF}}(s)=
\sum_{a\in E(x)}
\pi^+(a\mid s)
\left(Q_{\mathrm{CF}}(s,a)-V_{old}(s)\right).
$$

Substituting $\delta_{\mathrm{CF}}(s)$ gives

$$
\overline V_{\mathrm{CF}}(s)=V_{old}(s)+\delta_{\mathrm{CF}}(s).
$$

Let $\mathcal T_{\mathrm{exp}}$ contain every expanded decision node in a tree, including its root. Suppose the tree evaluates $N_E$ edges in total, of which $n(x)=|E(x)|$ originate at node $x\in\mathcal T_{\mathrm{exp}}$. FCPI assigns weights $w_P(x)$ and $w_T(x)$ to that node:

$$
w_P(x)=w_T(x)=\frac{n(x)}{N_E},
\qquad
N_E=\sum_{u\in\mathcal T_{\mathrm{exp}}}n(u).
$$

Root and descendant edges both contribute to $N_E$, and the weights within each tree sum to one:

$$
\sum_{x\in\mathcal T_{\mathrm{exp}}}w_P(x)
=\sum_{x\in\mathcal T_{\mathrm{exp}}}w_T(x)=1.
$$

### 8.6 Encoded-State Aggregation

Each expanded decision node produces one training record. Records at tree roots include the MC targets inherited from self-play, and records below the root contain counterfactual targets. FCPI groups records with identical Gadus `PackedState` encodings across the iteration and verifies that every member of a group has the same legal-move list.

Let $\mathcal S_{\mathrm{agg}}$ be the set of encoded positions represented after grouping, let $\mathcal R(s)$ be the group associated with $s\in\mathcal S_{\mathrm{agg}}$ and let $\mathcal A_s$ be its legal-move set. Record $i\in\mathcal R(s)$ carries counterfactual policy weight $w_{P,i}$, MC value weight $w_{\mathrm{MC},i}$, counterfactual value weight $w_{T,i}$ and corresponding targets $\pi_i^+(\cdot\mid s)$, $G_i$ and $\overline V_{\mathrm{CF},i}(s)$.

The total counterfactual policy weight is

$$
W_P(s)=\sum_{i\in\mathcal R(s)}w_{P,i}.
$$

For $W_P(s)>0$, the aggregated counterfactual policy target is

$$
\Pi^+(a\mid s)=
\frac{\sum_{i\in\mathcal R(s)}w_{P,i}\pi_i^+(a\mid s)}{W_P(s)}.
$$

$\Pi^+(\cdot\mid s)$ is the weighted mean of the counterfactual policy targets associated with encoded position $s$. FCPI renormalizes this distribution after aggregation to correct floating-point error in the probability sum.

The total MC value weight is

$$
W_{\mathrm{MC}}(s)=\sum_{i\in\mathcal R(s)}w_{\mathrm{MC},i}.
$$

For $W_{\mathrm{MC}}(s)>0$, the aggregated MC value target is

$$
\overline G(s)=
\frac{\sum_{i\in\mathcal R(s)}w_{\mathrm{MC},i}G_i}{W_{\mathrm{MC}}(s)}.
$$

$\overline G(s)$ is the weighted mean of the terminal-return targets associated with encoded position $s$.

The total counterfactual value weight is

$$
W_T(s)=\sum_{i\in\mathcal R(s)}w_{T,i}.
$$

For $W_T(s)>0$, the aggregated counterfactual value target is

$$
\overline V_T(s)=
\frac{\sum_{i\in\mathcal R(s)}w_{T,i}\overline V_{\mathrm{CF},i}(s)}{W_T(s)}.
$$

$\overline V_T(s)$ is the weighted mean of the counterfactual value targets associated with encoded position $s$.

The support sets for the three aggregated targets are

$$
\mathcal S_P=\lbrace s\in\mathcal S_{\mathrm{agg}}:W_P(s)>0\rbrace,
\qquad
\mathcal S_{\mathrm{MC}}=\lbrace s\in\mathcal S_{\mathrm{agg}}:W_{\mathrm{MC}}(s)>0\rbrace,
$$

$$
\mathcal S_T=\lbrace s\in\mathcal S_{\mathrm{agg}}:W_T(s)>0\rbrace.
$$

All three aggregate weights are nonnegative. Each target enters its corresponding loss on the support set where that target is defined.

The three weighted means above aggregate targets defined for an entire encoded position. The MC policy data are action-specific because each trajectory selects one move at that position. Let $a_i$ and $A_{\mathrm{MC},i}$ denote the selected move and its MC advantage coefficient in record $i$. The indicator $\mathbf 1[a_i=a]$ selects records associated with move $a$. The aggregate signed coefficient and sample weight for pair $(s,a)$ are

$$
S_A(s,a)=\sum_{i\in\mathcal R(s)}
w_{\mathrm{MC},i}\mathbf 1[a_i=a]A_{\mathrm{MC},i},
$$

$$
W_A(s,a)=\sum_{i\in\mathcal R(s)}
w_{\mathrm{MC},i}\mathbf 1[a_i=a].
$$

$S_A(s,a)$ is the weighted sum of MC advantage coefficients for selected move $a$ at encoded position $s$. $W_A(s,a)$ is the corresponding sample weight. The HDF5 datasets `mc_policy_advantage_sums` and `mc_policy_weights` store these two quantities.

### 8.7 Candidate Loss

Let $\ell_{new}(s,i_G(a))$ be the candidate logit for legal move $a$. The resulting legal-move policy is

$$
P_{new}(a\mid s)=
\frac{\exp\ell_{new}(s,i_G(a))}
{\sum_{b\in\mathcal A_s}\exp\ell_{new}(s,i_G(b))}.
$$

The distribution $\mu_{new}(a\mid s)$ used by $L_{P,\mathrm{MC}}^{(\mathcal B)}$ applies the same behavior temperature as self-play:

$$
\mu_{new}(a\mid s)=
\frac{\exp(\ell_{new}(s,i_G(a))/\widetilde T_b)}
{\sum_{b\in\mathcal A_s}\exp(\ell_{new}(s,i_G(b))/\widetilde T_b)}.
$$

At one optimizer step, let $\mathcal B\subseteq\mathcal S_{\mathrm{agg}}$ be the current minibatch. The counterfactual Policy loss normalizes its weights within $\mathcal B$:

$$
L_{P,\mathrm{CF}}^{(\mathcal B)}=
\frac{
\sum_{s\in\mathcal B}W_P(s)L_{\mathrm{CE}}
\left(P_{new}(\cdot\mid s),\Pi^+(\cdot\mid s)\right)
}
{\max\left(\sum_{s\in\mathcal B}W_P(s),10^{-8}\right)}.
$$

The MC Policy loss uses the same minibatch and normalizes by the corresponding action weights:

$$
L_{P,\mathrm{MC}}^{(\mathcal B)}=
-\frac{
\sum_{s\in\mathcal B}\sum_{a\in\mathcal A_s}
S_A(s,a)\log\mu_{new}(a\mid s)
}
{\max\left(
\sum_{s\in\mathcal B}\sum_{a\in\mathcal A_s}W_A(s,a),1
\right)}.
$$

For a single unmerged record from a completed trajectory, $S_A(s_t,a_t)=A_{\mathrm{MC}}(s_t,a_t)$ and $W_A(s_t,a_t)=1$. Gradient descent therefore raises the probability of a move with positive MC advantage and lowers the probability of a move with negative MC advantage. The two policy terms contribute additively:

$$
\nabla_{\theta_{new}}
\left(L_{P,\mathrm{CF}}^{(\mathcal B)}+L_{P,\mathrm{MC}}^{(\mathcal B)}\right)
=
\nabla_{\theta_{new}}L_{P,\mathrm{CF}}^{(\mathcal B)}
+\nabla_{\theta_{new}}L_{P,\mathrm{MC}}^{(\mathcal B)}.
$$

For scalar prediction error $e$, the terms involving $\overline G(s)$ and $\overline V_T(s)$ use the SmoothL1 penalty with threshold one:

$$
\mathrm{SL1}(e)=
\begin{cases}
\frac12e^2,&|e|<1,\\
|e|-\frac12,&|e|\geq1.
\end{cases}
$$

The Value loss combines the two weighted error sums within the same minibatch:

$$
L_V^{(\mathcal B)}=
\frac{
\sum_{s\in\mathcal B}W_{\mathrm{MC}}(s)
\mathrm{SL1}\left(V_{new}(s)-\overline G(s)\right)
+\sum_{s\in\mathcal B}W_T(s)
\mathrm{SL1}\left(V_{new}(s)-\overline V_T(s)\right)
}
{\max\left(
\sum_{s\in\mathcal B}W_{\mathrm{MC}}(s)
+\sum_{s\in\mathcal B}W_T(s),1
\right)}.
$$

The objective evaluated for minibatch $\mathcal B$ is the unweighted sum of these three components:

$$
L^{(\mathcal B)}=
L_{P,\mathrm{CF}}^{(\mathcal B)}+
L_{P,\mathrm{MC}}^{(\mathcal B)}+
L_V^{(\mathcal B)}.
$$

Automatic differentiation computes $\nabla_{\theta_{new}}L^{(\mathcal B)}$, and AdamW updates the policy head, value head and shared ResNet trunk. Every minibatch supplies its own weight denominators, so minibatches with different total weights are normalized independently.

### 8.8 Local Policy Update Property

Let the clipping floor $10^{-12}$ tend to zero, and suppose the candidate fits $\pi^+$ exactly. For two actions $a$ and $b$ at the same state,

$$
\log\frac{\pi^+(a\mid s)}{\pi^+(b\mid s)}
=
\log\frac{P_{old}(a\mid s)}{P_{old}(b\mid s)}
+Q_{\mathrm{CF}}(s,a)-Q_{\mathrm{CF}}(s,b).
$$

Assume repeated local updates preserve the fixed positive difference

$$
\Delta=Q_{\mathrm{CF}}(s,a)-Q_{\mathrm{CF}}(s,b)>0.
$$

If each ideal fit becomes the reference policy for the next update, then after $k$ fits the log odds of $a$ relative to $b$ have increased by $k\Delta$. Starting from policy $P_0$, the action order reverses when

$$
k>
\frac{\log P_0(b\mid s)-\log P_0(a\mid s)}{\Delta}.
$$

This statement applies to one fixed state, one fixed action pair and a constant $\Delta$.

### 8.9 Optimization, Artifacts and Promotion

Each iteration initializes the candidate from that iteration's `current.pth` and creates a new AdamW optimizer with weight decay $10^{-4}$. The training process initializes its random generator from `--seed` and uses that generator to shuffle records before every epoch. Training stops at the first limit reached by `--epochs` or `--train-max-steps`.

The model remains in `eval()` mode during FCPI training, which keeps BatchNorm running statistics fixed while preserving parameter gradients. Global gradient-norm clipping uses threshold one. CUDA forward computation uses BF16 when `--precision bf16` is selected. The policy logits, $V$ outputs, loss calculations and checkpoint parameters use FP32.

Each iteration writes an HDF5 target file containing `policy_targets`, `policy_weights`, `mc_policy_advantage_sums`, `mc_policy_weights`, `mc_value_targets`, `mc_value_weights`, `tree_value_targets` and `tree_value_weights`. The file is a persistent record of the aggregated targets. Candidate training consumes the same aggregated records from memory.

For reporting, the metrics `value_mc` and `value_tree` normalize each value-loss component by its own weight. Backpropagation uses the minibatch loss $L_V^{(\mathcal B)}$ defined above.

The iteration arena compares the candidate with the `current.pth` that generated its targets. If the candidate records $N_W$ wins, $N_D$ draws and $N_L$ losses, promotion occurs when

$$
N_W-N_L\geq\texttt{--eval-min-net-wins}.
$$

An accepted candidate atomically replaces the run's `current.pth`. A rejected candidate leaves that file unchanged, so the same current model generates the next iteration's targets.

Every FCPI invocation assigns a run identifier of the form `fcpi_YYYYMMDD_HHMMSS_id` and creates the following artifacts:

```text
data/runs/<run-id>/
	fcpi_iter_001.h5
	summary.json
models/runs/<run-id>/
	initial.pth
	current.pth
	candidate_iter_001.pth
```

## 9. Opening Books

FCPI sampling books contain reachable nonterminal states from arbitrary plies, covering both balanced and unbalanced positions. FCPI selects this position pool with `--opening-book` and limits the loaded pool with `--max-book-positions`.

Paired arena evaluation uses positions sampled at a fixed ply and filtered by a UCI evaluation bound. Each selected position is played once with each color assignment.

## 10. UCI Engine

### 10.1 Evaluation Output

The engine reports MultiPV lines, side-to-move `score cp`, nodes, NPS, elapsed time and a one-move PV. Let $V_{root}$ denote the root evaluation returned by search. In `closed` mode, $V_{root}=V_\theta(s)$. After MCTS completes at least one simulation, $V_{root}$ is the mean return backed up to the root. A visited MCTS root edge reports $q_{line}=Q(s,a)$, while an unvisited root edge reports $q_{line}=V_{root}$. Let $c_s$ be `ScoreScale`. The displayed score is

$$
score\_cp=\mathrm{round}\left(
c_s\mathrm{clip}(q_{line},-0.999,0.999)
\right).
$$

### 10.2 Options

Gadus exposes the following UCI options.

- `ModelPath` selects a Gadus checkpoint. A packaged engine defaults to `gadus.pth` in the executable directory.
- `Device` selects `auto`, `cpu` or `cuda`.
- `SearchType` selects direct policy ranking with `closed` or Gadus MCTS with `only-mcts`. Both modes apply the enabled final decision components.
- `MCTSSims` sets the MCTS simulation cap and defaults to `100`. A UCI command `go nodes <n>` uses `<n>` as the current cap.
- `MCTSMinSims` sets the nominal minimum simulation count before dynamic budgeting and defaults to `0`. A zero value activates the formula derived from the cap and batch size. A time limit or UCI `stop` command may end search before this count is reached.
- `MCTSBatchSize` sets the neural leaf batch size and defaults to `32`.
- `MoveOverheadMS` reserves communication and move-submission time from a clock allocation. Its default is `50`.
- `CPuct`, `CPuctBase` and `CPuctFactor` set the PUCT exploration schedule and default to `0.5`, `19652` and `1.0`.
- `FPUReduction` sets the FPU reduction and defaults to `0.15`.
- `VirtualLoss` sets the repeated-path penalty within one batched selection and defaults to `0.0`.
- `RepetitionPolicyPenalty` sets $\lambda_R$ in the RPP decision component, accepts $[0,1]$ and defaults to `0.0`.
- `InstantMateFirst` enables IMF and defaults to `false`.
- `ProgressIntervalMS` sets the interval between intermediate UCI `info` reports and defaults to `750`. A zero interval suppresses intermediate reports.
- `MultiPV` sets the number of reported analysis lines and defaults to `5`.
- `ScoreScale` sets $c_s$ for converting the engine's dimensionless evaluation to the displayed centipawn scale and defaults to `1000`.

The `go movetime` command supplies a fixed time budget directly. When `go` supplies the active side's remaining time and increment, the engine subtracts `MoveOverheadMS` from the allocation $t_{remain}/30+0.75t_{inc}$, bounds the result between 50 and 10000 milliseconds and then limits it to the usable remaining time. A `go` command without `movetime` or clock fields imposes no time limit.

The engine publishes its direct policy ranking when search begins and publishes intermediate MCTS results at intervals of `ProgressIntervalMS`. Search limits and the UCI `stop` command determine the final result.

## 11. Implementation Tests

The build process runs the Gadus CTest executable before publishing binaries. The test suite covers `gadus_18_planes`, ordinary and special action encoding, terminal-state detection, policy and value output shapes, finite numerical results, backward propagation, checkpoint round trips, `closed` search and batched MCTS.
