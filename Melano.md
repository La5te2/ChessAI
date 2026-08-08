# Melano

Melano is a geometry-aware Transformer chess architecture with Policy and Value outputs.

## 1. Notations

- $x$ denotes a complete game state maintained by the chess rules engine.
- $s=\phi_M(x)$ denotes the 67-integer network input encoded from $x$.
- $\mathcal A(x)$ denotes the set of legal moves in $x$.
- $\mathcal I_M=\{0,\ldots,4671\}$ denotes the complete Melano action-index set.
- $i_M(a)\in\mathcal I_M$ denotes the action index assigned to move $a$.
- $\theta$ denotes the trainable model parameters.
- $\ell_\theta(s,i)$ denotes the Policy logit assigned to action index $i$.
- $P_\theta(a\mid s)$ denotes the Policy probability assigned to legal move $a$.
- $V_\theta(s)\in[-1,1]$ denotes the evaluation of $s$ from the perspective of the side to move.
- $C$ denotes the embedding width.
- $B$ denotes the number of geometry-attention blocks.

## 2. State and Action Encoding

### 2.1 State Encoding

The rules engine represents the complete state $x$, including move counters and repetition history. The state codec `melano_states` maps $x$ to a 67-integer network input by retaining its board occupancy, side to move, castling rights and en-passant file:

$$
s=(p_0,\ldots,p_{63},t,c,e).
$$

Squares are indexed from 0 at `a1` to 63 at `h8`. For each square $q$, the token $p_q$ belongs to $\{0,\ldots,12\}$. Token 0 represents an empty square. White pawn, knight, bishop, rook, queen and king use tokens 1 through 6, while the corresponding Black pieces use tokens 7 through 12.

The side token is

$$
t=
\begin{cases}
1,&\text{White to move},\\
0,&\text{Black to move}.
\end{cases}
$$

The castling token $c\in\{0,\ldots,15\}$ is a four-bit mask. Its bits, from least to most significant, represent White kingside, White queenside, Black kingside and Black queenside castling rights. The en-passant token is 0 when no en-passant square exists and equals $1+f_{\mathrm{ep}}$ otherwise, where $f_{\mathrm{ep}}\in\{0,\ldots,7\}$ is the en-passant file.

### 2.2 Action Encoding

The move codec `sd_64x64_underpromo9` contains 4096 source-destination indices and 576 underpromotion indices:

$$
|\mathcal I_M|=64\times64+64\times9=4672.
$$

Let $f(a)$ denote the source square of move $a$, and let $t(a)$ denote the destination used by the codec. Ordinary moves and queen promotions use

$$
i_M(a)=64f(a)+t(a).
$$

The chess library represents castling internally as a king-to-rook move. The codec replaces that destination with the king's final square, `g1`, `c1`, `g8` or `c8`, before applying the source-destination formula.

Knight, bishop and rook underpromotions occupy the final 576 indices. Let $\Delta_f(a)\in\{-1,0,1\}$ be the destination-file displacement and let

$$
r(a)=
\begin{cases}
0,&\text{knight promotion},\\
1,&\text{bishop promotion},\\
2,&\text{rook promotion}.
\end{cases}
$$

Their action index is

$$
i_M(a)=4096+9f(a)+3\left(\Delta_f(a)+1\right)+r(a).
$$

Decoding compares an action index with every legal move in the current game state and returns the legal move whose encoded index matches. This round trip preserves castling, en-passant and promotion flags supplied by the rules engine.

## 3. Network

### 3.1 Geometry-Attention Encoder

For a batch of $n$ states, the state embedding produces a tensor $h_0\in\mathbb R^{n\times65\times C}$. Token 0 is a learned global token and tokens 1 through 64 correspond to board squares.

The rule-context embedding is

$$
r(s)=E_{\mathrm{side}}(t)+E_{\mathrm{castling}}(c)+E_{\mathrm{ep}}(e).
$$

For square $q$, the initial square token is

$$
h_{0,q+1}(s)=E_{\mathrm{piece}}(p_q)+E_{\mathrm{square}}(q)+r(s).
$$

The initial global token is

$$
h_{0,0}(s)=g+r(s),
$$

where $g\in\mathbb R^C$ is trainable. The piece, square, side, castling and en-passant embedding tables contain 13, 64, 2, 16 and 9 entries respectively.

Each attention block uses the largest value in $\{8,4,2,1\}$ that divides $C$ as its head count $H$. The dimension of each head is $d=C/H$.

Melano assigns one of 29 relation identifiers to every pair of tokens. Relation 0 applies whenever either token is the global token. For two square tokens, let $d_r$ and $d_f$ be the absolute rank and file differences. Their relation identifier is

$$
\rho(d_r,d_f)=
\begin{cases}
1,&d_r=0\ \text{and}\ d_f=0,\\
1+d_f,&d_r=0\ \text{and}\ 1\leq d_f\leq7,\\
8+d_r,&d_f=0\ \text{and}\ 1\leq d_r\leq7,\\
15+d_r,&d_r=d_f\ \text{and}\ 1\leq d_r\leq7,\\
23,&\{d_r,d_f\}=\{1,2\},\\
24+\min(4,d_r+d_f-4),&\text{otherwise}.
\end{cases}
$$

These cases produce identity, same-rank distance, same-file distance, diagonal distance, knight-move and five residual distance classes. Every identifier in $\{0,\ldots,28\}$ is reachable.

For block $b\in\{0,\ldots,B-1\}$, let $Z_b\in\mathbb R^{65\times C}$ be its learned token-position tensor and define

$$
\widetilde h_b=h_b+Z_b.
$$

A pre-normalized linear projection produces $3C$ features for every token. Splitting the last dimension into three parts and reshaping each part to $H$ heads gives

$$
(Q_b,K_b,U_b)=
\mathrm{split}_{3}\left(
\mathrm{reshape}_{3,H,d}
\left(W_{qkv,b}\mathrm{LN}_{b,1}(\widetilde h_b)+b_{qkv,b}\right)
\right).
$$

Each block contains a learned static bias table $\beta_b\in\mathbb R^{29\times H}$. It also derives a state-dependent bias table from the position-adjusted global token:

$$
\Gamma_b(s)=
\mathrm{reshape}_{H,29}\left(
W_{\gamma,b,2}\mathrm{GELU}
\left(W_{\gamma,b,1}\mathrm{LN}(\widetilde h_{b,0})+b_{\gamma,b,1}\right)
+b_{\gamma,b,2}
\right).
$$

For head $h$, source token $u$ and target token $v$, the attention score is

$$
S_{b,h,u,v}=
\frac{Q_{b,h,u}\cdot K_{b,h,v}}{\sqrt d}
+\beta_{b,\rho(u,v),h}
+\Gamma_b(s)_{h,\rho(u,v)}.
$$

The softmax function normalizes $S_{b,h,u,v}$ over target token $v$. The output of head $h$ for source token $u$ is

$$
o_{b,h,u}=\sum_v\mathrm{softmax}_v(S_{b,h,u,v})U_{b,h,v}.
$$

Concatenating the head outputs and applying the output projection gives the first residual update:

$$
y_{b,u}=\widetilde h_{b,u}+
W_{o,b}\mathrm{Concat}_{h=1}^{H}(o_{b,h,u})+b_{o,b}.
$$

After attention has exchanged information among the tokens, a pre-normalized feed-forward network transforms each token independently and forms the second residual update:

$$
h_{b+1}=y_b+
W_{f,b,2}\mathrm{GELU}
\left(W_{f,b,1}\mathrm{LN}_{b,2}(y_b)+b_{f,b,1}\right)+b_{f,b,2}.
$$

The feed-forward network expands each token from $C$ to $4C$ features and projects it back to $C$. Applying all $B$ geometry-attention blocks defines the shared representation

$$
E_\theta(s)=h_B(s)\in\mathbb R^{65\times C}.
$$

The global token $g$ and every position tensor $Z_b$ are initialized to zero. Embedding tables and linear maps use the default LibTorch initialization. LayerNorm uses trainable affine parameters, unit scale, zero bias and epsilon $10^{-5}$. The encoder contains no dropout.

### 3.2 Policy Head

Let $z_q=E_\theta(s)_{q+1}$ be the transformed token for square $q$. The Policy head applies a shared LayerNorm to each square token and then uses separate source and destination projections:

$$
u_q=W_F\mathrm{LN}(z_q)+b_F,
\qquad
v_k=W_T\mathrm{LN}(z_k)+b_T.
$$

The source-destination logit for action index $64q+k$ is

$$
\ell_\theta(s,64q+k)=\frac{u_q\cdot v_k}{\sqrt C}.
$$

A third linear map produces nine underpromotion logits from each normalized source token:

$$
c_q=W_U\mathrm{LN}(z_q)+b_U\in\mathbb R^9.
$$

For $m\in\{0,\ldots,8\}$, the underpromotion logit is

$$
\ell_\theta(s,4096+9q+m)=c_{q,m}.
$$

The component order of $c_q$ matches the underpromotion index formula in Section 2.2. Flattening the $64\times64$ source-destination matrix and the $64\times9$ underpromotion matrix gives

$$
\ell_\theta(s)\in\mathbb R^{4672}.
$$

Search supplies an encoded-state batch $\mathbf s=(s_1,\ldots,s_n)$ and a matrix $J\in\mathcal I_M^{n\times L}$ of requested legal-action indices. For an ordinary index $J_{rj}=64q+k$, the legal-action forward pass computes only $u_q\cdot v_k/\sqrt C$. For an underpromotion index $J_{rj}=4096+9q+m$, it applies the underpromotion map to source token $q$ and selects component $m$. The resulting tensor

$$
\Lambda_\theta(\mathbf s,J)_{rj}=\ell_\theta(s_r,J_{rj})
$$

contains the same requested logits as the complete Policy vector without constructing logits for unrequested actions.

For a game state $x$ with $s=\phi_M(x)$, legal-move inference normalizes the logits over $\mathcal A(x)$:

$$
P_\theta(a\mid s)=
\frac{\exp\ell_\theta(s,i_M(a))}
{\displaystyle\sum_{b\in\mathcal A(x)}\exp\ell_\theta(s,i_M(b))}.
$$

### 3.3 Value Head

The Value head reads the transformed global token $E_\theta(s)_0$. It applies LayerNorm, a $C\rightarrow256$ linear map, ReLU, a $256\rightarrow1$ linear map and a final hyperbolic tangent:

$$
V_\theta(s)=
\tanh\left(
W_{V,2}\mathrm{ReLU}
\left(W_{V,1}\mathrm{LN}(E_\theta(s)_0)+b_{V,1}\right)+b_{V,2}
\right).
$$

The complete forward pass is

$$
f_\theta(s)=\left(\ell_\theta(s),V_\theta(s)\right),
\qquad
\ell_\theta(s)\in\mathbb R^{4672},\quad V_\theta(s)\in[-1,1].
$$

The state embedding and the $B$ geometry-attention blocks form the backbone shared by the Policy and Value heads.

## 4. Preprocessing

### 4.1 Policy and Value Targets

Preprocessing follows each PGN game from its standard initial position or from its `FEN` header. For every parseable SAN move, it records the state immediately before the move as $s$, the played move as $a^*$ and the action index $i^*=i_M(a^*)$.

Preprocessing can derive Value targets either from numerical PGN comments or from the final game result. When comments provide the targets, signed pawn-unit evaluations such as `+0.60`, `-.25` and `+0.60/12` are accepted. The optional suffix after `/` does not affect the target. Each comment evaluates the state reached after its associated move, so the comment on the preceding move supplies the Value target for the current pre-move state.

Let $c_W$ be the parsed White-perspective score and let $c_{\mathrm{stm}}$ be the same score from the perspective of the side to move:

$$
c_{\mathrm{stm}}=
\begin{cases}
c_W,&\text{White to move},\\
-c_W,&\text{Black to move}.
\end{cases}
$$

The comment-derived target is

$$
V_{\mathrm{target}}(s)=\tanh\left(\frac{c_{\mathrm{stm}}}{3}\right).
$$

The initial state has no preceding move comment and receives target 0. Any later state whose preceding comment contains no parseable signed score also receives target 0. When comments provide the targets, preprocessing admits a game if at least one of its move comments contains a parseable signed score.

When the final game result provides the targets, a win from the perspective of the side to move gives $+1$, a loss gives $-1$ and a draw or unknown result gives 0.

### 4.2 HDF5 Schema

Preprocessing creates an HDF5 file with the following identifying attributes:

```text
arch_type=melano
state_encoding=melano_states
move_encoding=sd_64x64_underpromo9
target_schema=melano_policy_value
value_perspective=side_to_move
has_cmt=<0 or 1>
```

When `has_cmt=1`, the file also contains

```text
comment_eval_perspective=white
comment_value_transform=tanh(side_to_move_pawn_score/3)
```

The row-aligned datasets are

- `states`, a little-endian unsigned-byte array with shape $[N,67]$
- `moves`, a little-endian unsigned-16 array with shape $[N]$
- `values`, a little-endian float-32 array with shape $[N]$

All three datasets are extensible along their first dimension and use a common positive row count per HDF5 chunk. Optional compression applies the HDF5 shuffle filter followed by deflate. Final file attributes record the number of accepted games, written positions, skipped SAN moves and games rejected for lacking a parseable evaluation when comments provide the targets.

A valid Melano dataset has the identifying attributes above, a nonempty state array of width 67 and equal row counts across all three datasets.

## 5. Supervised Training

For a training row $(s,i^*,V_{\mathrm{target}})$, the model produces all 4672 Policy logits. Supervised training applies softmax over the complete action-index set:

$$
R_\theta(i\mid s)=
\frac{\exp\ell_\theta(s,i)}
{\displaystyle\sum_{j\in\mathcal I_M}\exp\ell_\theta(s,j)}.
$$

The same Policy logits determine the supervised distribution $R_\theta$ and the legal-move distribution $P_\theta$ defined in Section 3.2. The normalization for $R_\theta$ covers all action indices during training, whereas the normalization for $P_\theta$ covers the legal moves of the current game state during inference.

Let $\delta_{i^*}$ be the one-hot distribution that assigns probability 1 to $i^*$ and 0 to every other action index. The supervised Policy loss is

$$
L_{P,\mathrm{sup}}=
L_{\mathrm{CE}}\left(R_\theta(\cdot\mid s),\delta_{i^*}\right)
=-\log R_\theta(i^*\mid s).
$$

The supervised Value loss is

$$
L_{V,\mathrm{sup}}=
\mathrm{MSE}\left(V_\theta(s),V_{\mathrm{target}}(s)\right).
$$

Let $w_V\geq0$ be the Value-loss coefficient. The minibatch objective is

$$
L_{\mathrm{sup}}=L_{P,\mathrm{sup}}+w_VL_{V,\mathrm{sup}}.
$$

Because the Policy and Value heads receive the shared representation $E_\theta$, gradients from both loss terms contribute to updates of the state embedding and geometry-attention blocks.

Training initializes a model with channel count $C$ and block count $B$, then optimizes all parameters with AdamW. The loader shuffles HDF5 chunks, reads the next chunk asynchronously while processing another chunk and independently shuffles rows within each loaded chunk.

Let $N$ be the number of HDF5 rows and let $m$ be the minibatch size. The number of optimizer steps in one epoch is

$$
S_{\mathrm{epoch}}=\left\lceil\frac Nm\right\rceil
$$

Let $E$ be the positive epoch count and let $K_{\max}\in\mathbb N\cup\{\infty\}$ be the optimizer-step cap. The resulting number of steps and the warmup length are

$$
T=\min\left(K_{\max},E S_{\mathrm{epoch}}\right),
$$

$$
T_W=\min\left(T,2000,\max\left(100,\left\lfloor\frac{T}{100}\right\rfloor\right)\right).
$$

For peak learning rate $\eta_{\max}$, step $k\geq1$ uses

$$
\eta_k=
\begin{cases}
\eta_{\max}\dfrac{k}{T_W},&k\leq T_W,\\[6pt]
\eta_{\max}\sqrt{\dfrac{T_W}{k}},&k>T_W.
\end{cases}
$$

Thus the schedule uses linear warmup followed by inverse-square-root decay. A deterministic seed controls parameter initialization and both shuffle stages. Let $g_{\max}>0$ be the gradient-norm limit. Before each AdamW step, training clips the global gradient norm to $g_{\max}$ and terminates when that norm is nonfinite.

Let $K_{\mathrm{save}}>0$ be the periodic checkpoint interval. Training writes a checkpoint whenever the optimizer-step count is divisible by $K_{\mathrm{save}}$, after every processed epoch pass and after the final partial pass imposed by $K_{\max}$. The top-level archive contains `model` and `arch`. The architecture archive stores type identifier 2, $C$, $B$ and action size 4672 as integer tensors. Saving first writes a sibling `.tmp` file and then atomically replaces the destination. Loading validates every architecture field before constructing the model and restoring its parameters.

## 6. Search

### 6.1 Search Modes

In `closed` mode, search evaluates the root once and derives its initial move ranking from $P_\theta(a\mid s)$. In `only-mcts` mode with a positive simulation cap, search evaluates exact-state leaves in neural batches and derives its initial ranking from the resulting MCTS root distribution. An `only-mcts` search with a zero simulation cap returns the same legal Policy distribution as `closed`. In either mode, search subsequently applies the enabled IMF and RPP decision components defined in Section 6.6.

### 6.2 Neural Evaluation Representation

For a neural batch $\mathbf s=(s_1,\ldots,s_n)$, let $L$ be the largest legal-move count in the batch. Search constructs the legal-action matrix $J\in\mathcal I_M^{n\times L}$ and a mask $M\in\{0,1\}^{n\times L}$. Entries with $M_{rj}=1$ identify legal actions, while entries with $M_{rj}=0$ pad shorter rows. The legal-action forward pass defined in Section 3.2 returns $\Lambda_\theta(\mathbf s,J)$ and $(V_\theta(s_1),\ldots,V_\theta(s_n))$. Search replaces padded logits with $-\infty$ and applies softmax across each row, so the resulting probabilities are normalized over the legal actions of each state.

Selections from independent roots share neural batches. Duplicate selections of the same nonterminal leaf within one root batch are discarded before neural evaluation. The remaining requests are grouped by their 67-token encoded state, and all requests in one group share one network evaluation.

Each evaluated state produces a compact record containing its legal moves, action indices, legal-move probabilities and $V_\theta(s)$. Root initialization and leaf expansion create one edge from each aligned legal-move entry without constructing a 4672-entry Policy vector.

Let $p(a)$ denote the distribution supplied by direct Policy inference or by the MCTS root visit calculation. A final search result or progress snapshot materializes the public action-space representation

$$
p_{\mathrm{dense}}(i\mid s)=
\begin{cases}
p(a),&i=i_M(a)\text{ for }a\in\mathcal A(x),\\
0,&i\notin\{i_M(a):a\in\mathcal A(x)\}.
\end{cases}
$$

### 6.3 Evaluation Reuse

Let $k(s)$ be the 67-byte state encoding defined in Section 2.1. A cached evaluation associates $k(s)$ with the compact record defined in Section 6.2. Equality of cache keys therefore means equality of every feature supplied to the network. Rule-terminal detection precedes neural evaluation, so a terminal leaf contributes its rule outcome without consulting the cache.

Each search call creates a local evaluation cache. This cache removes repeated neural evaluations across batches and independent roots within that call. A positive cross-search capacity selects a persistent TLRU (trajectory-aware least-recently-used) cache instead. Setting the capacity to zero selects the local cache and discards its records when the search call returns.

An exact lookup moves its entry to the most-recent end of the LRU order. When an evaluated transition connects cached parent key $k_p$ to cached child key $k_c$, TLRU records the directed relation $k_p\rightarrow k_c$ without changing the recency of either entry. For the root key $k_r$ of a search call, define

$$
\mathcal N_2(k_r)=
\left\{k:\ d_C(k_r,k)\leq2\right\},
$$

where $d_C$ is the shortest directed-path length through recorded relations. Before root evaluation, TLRU moves entries in $\mathcal N_2(k_r)$ to the most-recent end in descending distance order. The resulting order places the root first, followed by its one-ply descendants and then its two-ply descendants. When the approximate byte capacity is exceeded, eviction removes entries from the least-recent end.

The cache stores network Policy and Value outputs. Each search call creates an independent MCTS tree with zero visits, zero virtual visits and zero accumulated returns.

### 6.4 PUCT Selection

In `only-mcts` mode, search initializes the root with one outgoing edge $(s,a)$ for every legal action and stores $P_\theta(a\mid s)$ as the edge prior $P(s,a)$. The child edges of one node occupy one contiguous array whose capacity equals the legal-action count at expansion. The prior remains fixed throughout the search. Let $N(s,a)$ be the completed visit count of edge $(s,a)$ and let $N(s)=\sum_aN(s,a)$ be the completed visit count of node $s$.

During batched selection, a virtual visit reserves every node on a selected path until the corresponding terminal or neural evaluation has been backed up. Let $N_v(s,a)$ be the virtual-visit count of edge $(s,a)$ and let $N_v(s)=\sum_aN_v(s,a)$. Selection uses the augmented counts

$$
\widetilde N(s,a)=N(s,a)+N_v(s,a),
\qquad
\widetilde N(s)=N(s)+N_v(s).
$$

Let $c_0$ be the initial exploration coefficient, let $b_0$ be its schedule base and let $f_0$ be its schedule factor. Define $b=\max(1,b_0)$ and $f=\max(0,f_0)$. The visit-dependent exploration coefficient is

$$
c_{\mathrm{puct}}(\widetilde N)=
\max\left(0,c_0+f\log\left(\frac{\widetilde N+b+1}{b}\right)\right).
$$

Each tree node stores the mean backed-up return $Q(s)$ from the perspective of the player to move in state $s$. A nonterminal leaf contributes $V_\theta(s)$, while a terminal leaf contributes the exact side-to-move result in $\{-1,0,1\}$. Backup reverses the sign at every ply. For a visited edge that reaches child state $s_a$, the parent-perspective action value is

$$
Q(s,a)=-Q(s_a).
$$

For an unvisited edge, First Play Urgency uses the parent value and the prior mass already visited. A node initializes this mass to zero and adds $P(s,a)$ when edge $(s,a)$ receives its first completed visit. Let $r_{\mathrm{FPU}}\geq0$ be the FPU reduction coefficient. The selection estimate is

$$
Q_{\mathrm{sel}}(s,a)=
\begin{cases}
Q(s,a),&N(s,a)>0,\\[4pt]
\mathrm{clip}\left(
Q(s)-r_{\mathrm{FPU}}
\sqrt{\displaystyle\sum_{a':N(s,a')>0}P(s,a')},-1,1
\right),&N(s,a)=0.
\end{cases}
$$

When node $s$ has no completed visit, its FPU baseline $Q(s)$ is 0. Let $l_v\geq0$ be the virtual-loss coefficient. The PUCT selection score is

$$
S(s,a)=Q_{\mathrm{sel}}(s,a)
+c_{\mathrm{puct}}(\widetilde N(s))P(s,a)
\frac{\sqrt{\widetilde N(s)+1}}{1+\widetilde N(s,a)}
-l_vN_v(s,a).
$$

Equal selection scores are ordered by descending $P(s,a)$ and then by descending $Q_{\mathrm{sel}}(s,a)$.

When search ends, each legal root move receives weight $N(s,a)+P(s,a)$. Normalizing these weights gives

$$
P_{\mathrm{root}}(a\mid s)=
\frac{N(s,a)+P(s,a)}
{\displaystyle\sum_{a'\in\mathcal A(x)}\left(N(s,a')+P(s,a')\right)}.
$$

### 6.5 Dynamic Simulation Budget

Let $N_{\mathrm{cap}}\geq0$ be the simulation cap, let $B_{\mathrm{batch}}\geq1$ be the neural batch capacity and let $N_{\mathrm{floor}}\geq0$ be the simulation-floor parameter. The nominal minimum simulation count is

$$
N_{\min}=
\begin{cases}
0,&N_{\mathrm{cap}}=0,\\
\max\left(1,\min\left(N_{\mathrm{cap}},N_{\mathrm{floor}}\right)\right),
&N_{\mathrm{cap}}>0\ \text{and}\ N_{\mathrm{floor}}>0,\\
\max\left(1,\min\left(N_{\mathrm{cap}},
\max\left(B_{\mathrm{batch}},\left\lfloor\dfrac{N_{\mathrm{cap}}}{4}\right\rfloor\right)
\right)\right),
&N_{\mathrm{cap}}>0\ \text{and}\ N_{\mathrm{floor}}=0.
\end{cases}
$$

An external deadline or cancellation signal may terminate search with fewer than $N_{\min}$ completed simulations. After $N_{\min}$ simulations have completed at a root with at least two legal actions, search forms the completed-visit distribution

$$
v_a=\frac{N(s,a)}{\displaystyle\sum_{b\in\mathcal A(x)}N(s,b)}.
$$

Let $a_1$ and $a_2$ be the two actions with the largest completed visit counts, using prior as their tie breaker. Write $N_i=N(s,a_i)$. Define $Q_i=Q(s,a_i)$ when $N_i>0$ and $Q_i=0$ otherwise. The normalized visit entropy, visit closeness and action-value closeness are

$$
H_N=-\frac{\displaystyle\sum_{a\in\mathcal A(x)}v_a\log v_a}
{\log|\mathcal A(x)|},
$$

where a term with $v_a=0$ contributes 0 to the entropy sum.

$$
U_N=1-\frac{|N_1-N_2|}{\max(1,N_1+N_2)},
$$

$$
U_Q=1-\min\left(1,\frac{|Q_1-Q_2|}{0.5}\right).
$$

The combined uncertainty and dynamic target are

$$
u=\mathrm{clip}(0.5H_N+0.35U_N+0.15U_Q,0,1),
$$

$$
N_{\mathrm{target}}=
N_{\min}+\left\lceil u(N_{\mathrm{cap}}-N_{\min})\right\rceil.
$$

For a root with one legal action, search sets $u=0$ and $N_{\mathrm{target}}=N_{\min}$. The search loop checks $N_{\mathrm{cap}}$ and $N_{\mathrm{target}}$ between batch-construction passes. Work completed inside the current pass is included in the final statistics, which makes the reported simulation limit a batched soft cap.

### 6.6 Final Decision Components

The optional IMF, or Instant Mate First, and RPP, or Repetition Policy Penalty, components modify the final ranking after Policy evaluation or MCTS. Before either component is applied, the ranking score is

$$
D_0(a)=
\begin{cases}
P_\theta(a\mid s),&\texttt{closed},\\
P_{\mathrm{root}}(a\mid s),&\texttt{only-mcts}.
\end{cases}
$$

Let $\mathcal M(x)$ be the set of legal actions that immediately checkmate the opponent. When this set is nonempty, IMF selects

$$
a_M=\arg\max_{a\in\mathcal M(x)}D_0(a)
$$

and defines

$$
D_I(a)=
\begin{cases}
1,&a=a_M,\\
D_0(a),&a\in\mathcal A(x)\setminus\{a_M\}.
\end{cases}
$$

When $\mathcal M(x)$ is empty, $D_I(a)=D_0(a)$ for every legal action.

Let $\lambda_R\in[0,1]$ be the repetition-penalty coefficient, and let $V_R$ be the root evaluation returned by search. The set $\mathcal R_3(x)$ contains legal moves that either make a threefold-repetition claim available immediately or allow the opponent to make one with a single reply. RPP computes

$$
d_R=\lambda_R\mathrm{clip}(V_R,0,1)
$$

and produces

$$
D(a)=
\begin{cases}
\max(0,D_I(a)-d_R),&a\in\mathcal R_3(x),\\
D_I(a),&a\in\mathcal A(x)\setminus\mathcal R_3(x).
\end{cases}
$$

The decision layer orders legal moves by descending $D(a)$, then by descending $D_0(a)$ and finally by descending coordinate move string. The first move in this order is selected. These components leave network logits, edge priors, visit counts and backed-up values unchanged.

## 7. Arena

The Melano arena loads the candidate and baseline checkpoints once and creates one searcher for each model. Both searchers receive the same search configuration. Let $N_G$ be the positive even number of games and let $K_G\geq1$ be the maximum number of active games. The arena advances at most $K_G$ games together, collects all current positions assigned to each model and calls that model's batched search once per turn.

Each selected opening is played twice, once with the candidate as White and once with the candidate as Black. Let $T_G$ be the per-game ply cap. The rules engine adjudicates checkmate, stalemate, insufficient material, the fifty-move rule and threefold repetition. A game that reaches $T_G$ is recorded as a draw with termination `max plies`.

Let $N_W$, $N_D$ and $N_L$ be the candidate's win, draw and loss counts, and let $N_G=N_W+N_D+N_L$. The candidate score and net wins are

$$
score=\frac{N_W+\frac12N_D}{N_G},
\qquad
net\_wins=N_W-N_L.
$$

Let $x_i\in\{0,\frac12,1\}$ be the candidate score in game $i$. The population variance is

$$
\sigma^2=\frac1{N_G}\sum_{i=1}^{N_G}(x_i-score)^2.
$$

The reported 95% normal-approximation interval is

$$
CI_{95\%}=\mathrm{clip}\left(
score\pm1.96\sqrt{\frac{\sigma^2}{N_G}},0,1
\right).
$$

For display, let $score_b=\mathrm{clip}(score,10^{-6},1-10^{-6})$. The reported Elo difference is

$$
\Delta Elo=400\log_{10}\left(\frac{score_b}{1-score_b}\right).
$$

Let $M_{\mathrm{gate}}$ be the minimum required net-win margin. The arena accepts the candidate when

$$
N_W-N_L\geq M_{\mathrm{gate}}.
$$

The arena returns the gate result and statistics as JSON. It can also write all games as 80-column SAN PGN with the starting FEN and termination reason. Checkpoint promotion belongs to the training procedure that invokes the arena, so arena evaluation itself does not replace either checkpoint.
