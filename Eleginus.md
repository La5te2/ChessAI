# Eleginus

Eleginus is a sparse chess architecture in which independent Policy and Value networks use king-conditioned features and incrementally maintained accumulators.

## 1. Notation

- $\mathcal X$ is the set of complete chess states maintained by the rules engine.
- $x\in\mathcal X$ is a complete chess state containing piece placement, side to move, castling rights, en passant state, move counters and repetition history.
- $\mathcal A(x)$ is the set of legal actions in complete state $x$.
- $T(x,a)$ is the complete state reached by applying legal action $a\in\mathcal A(x)$ to complete state $x$.
- $c_x$ is the side to move in complete state $x$, and $\bar c_x$ is the opposing color.
- $g(x,c)\in\lbrace0,\frac12,1\rbrace$ is the exact score of color $c$ in terminal state $x$, with $1$, $\frac12$ and $0$ representing a win, draw and loss.
- $\mathbf 1[A]$ equals $1$ when statement $A$ is true and $0$ otherwise.
- $\phi_E$ is the Eleginus state encoder that maps a complete chess state to a sparse network input.
- $s=\phi_E(x)$ is the Eleginus network input obtained from complete state $x$.
- $\mathcal I_E=\lbrace0,\ldots,4671\rbrace$ is the fixed set of Eleginus action indices.
- $i_E(x,a)\in\mathcal I_E$ is the action index assigned to legal action $a$ after orienting complete state $x$ for its side to move.
- $\text{P}$, which stands for Policy, is the network output that assigns a probability distribution over the legal actions available in a complete state.
- $\text{V}$, which stands for Value, is a scalar network output in $[0,1]$ that estimates the expected game score from the perspective of the side to move.
- $\theta_P$ is the parameter set of the Eleginus Policy network.
- $\theta_V$ is the parameter set of the Eleginus Value network.
- $\ell_{\theta_P}(s,i)$ is the Policy logit assigned to action index $i\in\mathcal I_E$ for network input $s$.
- $P_{\theta_P}(a\mid s)$ is the Policy probability assigned to legal action $a$ for network input $s$.
- $V_{\theta_V}(s)\in[0,1]$ is the Value prediction for network input $s$.

## 2. State and Action Encoding

The Eleginus state encoder and action codec both express board coordinates from the perspective of the side to move. The state encoder maps a complete state to the sparse network input $s=\phi_E(x)$, and the action codec maps each legal action to the fixed action-index set $\mathcal I_E$.

### 2.1 State Encoding

Eleginus first assigns an absolute index to every board square. Let $r,f\in\lbrace0,\ldots,7\rbrace$ be the rank and file coordinates, with $(r,f)=(0,0)$ representing `a1` and $(7,7)$ representing `h8`. The absolute square index is

$$
q(r,f)=8r+f.
$$

The feature representation describes the board separately from the perspective of each color $c\in\lbrace\mathrm W,\mathrm B\rbrace$. Its orientation map is

$$
\omega_c(q(r,f))=
\begin{cases}
q(r,f),&c=\mathrm W,\\
q(7-r,f),&c=\mathrm B.
\end{cases}
$$

White-oriented features retain the absolute square indices, whereas Black-oriented features reflect the board across its horizontal centre and preserve the file coordinates. In either orientation, the perspective color is treated as the friendly side.

The six piece types use the order pawn, knight, bishop, rook, queen and king. Let $\tau(p)\in\lbrace0,\ldots,5\rbrace$ be the type index of piece $p$. Relative to perspective $c$, the piece-category index is

$$
\rho_c(p)=
\tau(p)+6\mathbf 1[\mathrm{color}(p)\ne c].
$$

The range $\rho_c(p)\in\lbrace0,\ldots,11\rbrace$ therefore contains six friendly categories followed by six opposing categories. Let $\kappa_c(x)$ be the oriented square occupied by the king of color $c$. A piece $p$ on absolute square $q$ activates the king-conditioned feature identifier

$$
\iota_c(x,p,q)=
768\kappa_c(x)+64\rho_c(p)+\omega_c(q).
$$

The three terms distinguish 64 possible king squares, 12 perspective-relative piece categories and 64 possible piece squares. Consequently, $\iota_c(x,p,q)$ belongs to $\lbrace0,\ldots,49151\rbrace$. Changing the king square changes every piece identifier in that color perspective, which allows the sparse features to condition the entire piece configuration on king location.

Piece identifiers describe board occupancy but do not encode castling rights or en passant availability. Eleginus therefore adds one castling identifier and one en passant identifier to each color perspective.

For perspective $c$, the castling mask is

$$
m_c(x)=
\mathbf 1[c\text{ has kingside castling rights}]
+2\mathbf 1[c\text{ has queenside castling rights}]
+4\mathbf 1[\bar c\text{ has kingside castling rights}]
+8\mathbf 1[\bar c\text{ has queenside castling rights}].
$$

The 16 possible masks activate identifiers $49152+m_c(x)$ in the range $49152$ through $49167$. The en passant code is

$$
e(x)=
\begin{cases}
0,&\text{no en passant square exists},\\
1+f_{\mathrm{ep}},&\text{the en passant square lies on file }f_{\mathrm{ep}},
\end{cases}
$$

where $f_{\mathrm{ep}}\in\lbrace0,\ldots,7\rbrace$. The nine possible codes activate identifiers $49168+e(x)$ in the range $49168$ through $49176$. Because the orientation map preserves files, the same en passant code applies to both color perspectives.

The feature vocabulary contains 49,152 king-conditioned piece identifiers, 16 castling identifiers and 9 en passant identifiers. Identifier $49177$ is reserved for padding, giving a total vocabulary size of 49,178.

For perspective $c$, let $\mathcal F_c(x)$ contain the active piece identifiers together with the castling and en passant identifiers. Eleginus orders the piece identifiers by ascending absolute square, appends the two rule-context identifiers and pads the resulting list to 34 entries with identifier $49177$. Denote this fixed-length sequence by $F_c(x)$. The state encoder places the side-to-move perspective first:

$$
s=\phi_E(x)=
\bigl(F_{c_x}(x),F_{\bar c_x}(x)\bigr)
\in\lbrace0,\ldots,49177\rbrace^{2\times34}.
$$

Both networks receive this encoded state. The complete state $x$ additionally contains move counters and repetition history, so complete states that differ only in those fields produce the same $s$ while retaining their distinct rule histories.

### 2.2 Action Encoding

The action codec uses the orientation map $\omega_{c_x}$ to express every legal action from the perspective of the side to move. Let $q_{\mathrm{from}}(a)$ be the absolute source square of action $a$. Let $q_{\mathrm{to}}(a)$ be its absolute destination square, except that castling uses the king destination `g1`, `c1`, `g8` or `c8` when the rules library represents the internal move with the rook square. The oriented source and destination indices are

$$
\widetilde q_{\mathrm{from}}(x,a)=
\omega_{c_x}\left(q_{\mathrm{from}}(a)\right),
\qquad
\widetilde q_{\mathrm{to}}(x,a)=
\omega_{c_x}\left(q_{\mathrm{to}}(a)\right).
$$

Ordinary moves and queen promotions use one index for each oriented source-destination pair:

$$
i_E(x,a)=
64\widetilde q_{\mathrm{from}}(x,a)
+\widetilde q_{\mathrm{to}}(x,a).
$$

These actions occupy indices $0$ through $4095$. An underpromotion additionally distinguishes its destination-file displacement and promoted piece. Let $\Delta_f(a)\in\lbrace-1,0,1\rbrace$ be the destination file minus the source file, and define

$$
r(a)=
\begin{cases}
0,&a\text{ promotes to a knight},\\
1,&a\text{ promotes to a bishop},\\
2,&a\text{ promotes to a rook}.
\end{cases}
$$

The underpromotion index is

$$
i_E(x,a)=
4096
+9\widetilde q_{\mathrm{from}}(x,a)
+3\bigl(\Delta_f(a)+1\bigr)
+r(a).
$$

The source-destination block contains $64\times64=4096$ indices, and the underpromotion block contains $64\times9=576$ indices. Their union is the fixed action-index set

$$
\mathcal I_E=\lbrace0,\ldots,4671\rbrace,
\qquad
|\mathcal I_E|=4096+576=4672.
$$

To decode an index for complete state $x$, the action codec generates $\mathcal A(x)$ and returns the legal action whose side-relative encoding equals that index. The returned legal action retains the castling, en passant or promotion information required by the rules engine.

## 3. Network

Eleginus comprises two neural networks that receive the same encoded state $s$. The Policy network uses parameters $\theta_P$ to produce $\text{P}$, and the Value network uses parameters $\theta_V$ to produce $\text{V}$. Their feature tables, accumulator biases and dense layers are separate, so $\theta_P\cap\theta_V=\varnothing$. Both networks begin by summing the embeddings of active sparse features, after which different dense mappings produce their respective outputs.

### 3.1 Sparse Accumulators

Let $H\in\lbrace\mathrm P,\mathrm V\rbrace$ identify one of the two networks, and let $d_H$ be its accumulator width:

$$
d_{\mathrm P}=128,
\qquad
d_{\mathrm V}=256.
$$

Network $H$ has a feature table $E_H\in\mathbb R^{49178\times d_H}$ and an accumulator bias $\beta_H\in\mathbb R^{d_H}$. For color perspective $c$, summing the table rows selected by the active feature set $\mathcal F_c(x)$ gives the unclipped accumulator

$$
u_{H,c}(x)=
\beta_H+
\sum_{j\in\mathcal F_c(x)}E_{H,j}.
$$

The padding row $E_{H,49177}$ is the zero vector, so summing the fixed-length sequence $F_c(x)$ produces the same accumulator. Each coordinate is then restricted to $[0,1]$:

$$
z_{H,c}(x)=
\mathrm{clip}_{[0,1]}\left(u_{H,c}(x)\right).
$$

The dense portion of network $H$ receives the side-to-move perspective followed by the opposing perspective:

$$
h_H(s)=
z_{H,c_x}(x)\mathbin\Vert z_{H,\bar c_x}(x)
\in\mathbb R^{2d_H},
\qquad s=\phi_E(x).
$$

### 3.2 Policy Network

The Policy network applies a rectified affine map to its $256$-dimensional accumulator input:

$$
y_P(s)=
\mathrm{ReLU}\left(W_{P,1}h_{\mathrm P}(s)+b_{P,1}\right)
\in\mathbb R^{128},
$$

where $W_{P,1}\in\mathbb R^{128\times256}$, $b_{P,1}\in\mathbb R^{128}$ and $\mathrm{ReLU}(z)=\max(0,z)$ for each coordinate of $z$. A second affine map produces one logit for every action index:

$$
\ell_{\theta_P}(s)=
W_{P,2}y_P(s)+b_{P,2}
\in\mathbb R^{4672},
$$

where $W_{P,2}\in\mathbb R^{4672\times128}$ and $b_{P,2}\in\mathbb R^{4672}$. The component at index $i\in\mathcal I_E$ is the logit $\ell_{\theta_P}(s,i)$ defined in Section 1.

For an ongoing complete state $x$ with $s=\phi_E(x)$, normalizing the logits assigned to its legal actions gives

$$
P_{\theta_P}(a\mid s)=
\frac{\exp\ell_{\theta_P}\left(s,i_E(x,a)\right)}
{\displaystyle
\sum_{b\in\mathcal A(x)}
\exp\ell_{\theta_P}\left(s,i_E(x,b)\right)},
\qquad a\in\mathcal A(x).
$$

The output affine map mathematically defines all 4672 logits. During legal-action evaluation, the network evaluates only the rows of $W_{P,2}$ and $b_{P,2}$ selected by $\lbrace i_E(x,a):a\in\mathcal A(x)\rbrace$. This restricted projection produces the same legal-action logits and probabilities as the complete affine map.

### 3.3 Value Network

The Value network applies a rectified affine map to its $512$-dimensional accumulator input:

$$
y_V(s)=
\mathrm{ReLU}\left(W_{V,1}h_{\mathrm V}(s)+b_{V,1}\right)
\in\mathbb R^{64},
$$

where $W_{V,1}\in\mathbb R^{64\times512}$ and $b_{V,1}\in\mathbb R^{64}$. A second rectified affine map produces a $32$-dimensional bottleneck:

$$
r_V(s)=
\mathrm{ReLU}\left(W_{V,2}y_V(s)+b_{V,2}\right)
\in\mathbb R^{32},
$$

where $W_{V,2}\in\mathbb R^{32\times64}$ and $b_{V,2}\in\mathbb R^{32}$. An affine scalar output followed by the sigmoid function gives

$$
V_{\theta_V}(s)=
\sigma\left(w_V^\top r_V(s)+b_V\right)
\in[0,1],
\qquad
\sigma(z)=\frac{1}{1+\exp(-z)},
$$

with $w_V\in\mathbb R^{32}$ and $b_V\in\mathbb R$. The resulting scalar estimates the expected game score from the perspective of the side to move represented in $s$.

### 3.4 Incremental Evaluation

A full refresh computes the two unclipped accumulators of network $H$ from the active feature sets in Section 3.1. When legal action $a\in\mathcal A(x)$ produces the successor $x'=T(x,a)$, the accumulator for color perspective $c$ is updated by

$$
u_{H,c}(x')=
u_{H,c}(x)
-\sum_{j\in\mathcal F_c(x)\setminus\mathcal F_c(x')}E_{H,j}
+\sum_{j\in\mathcal F_c(x')\setminus\mathcal F_c(x)}E_{H,j}.
$$

The first sum removes features that cease to be active, and the second adds features that become active. Every feature shared by $\mathcal F_c(x)$ and $\mathcal F_c(x')$ remains in the accumulator, so the update gives the same $u_{H,c}(x')$ as a full refresh.

Most actions alter only a few feature identifiers. For a king move by color $c$, $\kappa_c(x')\ne\kappa_c(x)$, so every king-conditioned piece identifier in perspective $c$ is replaced. After either a full refresh or an incremental update, clipping and side-to-move ordering produce $h_H\left(\phi_E(x')\right)$ for the dense layers.

## 4. Supervised Training

### 4.1 Supervised Data

Let $\mathcal D_{\mathrm{sup}}$ be a supervised dataset containing $N$ records:

$$
\mathcal D_{\mathrm{sup}}=
\lbrace\xi_n\rbrace_{n=1}^{N}.
$$

Each record is associated with a complete pre-move state $x_n$ and a selected legal action $a_n\in\mathcal A(x_n)$. The record is

$$
\xi_n=(s_n,i_n,y_n),
$$

where

$$
s_n=\phi_E(x_n),
\qquad
i_n=i_E(x_n,a_n),
\qquad
y_n\in[0,1].
$$

The encoded state $s_n$ is the common input to the two networks, and the action index $i_n$ is the Policy target. The scalar $y_n$ is the Value target, expressed as an estimate of the expected game score from the perspective of the side to move in $x_n$. On this scale, $0$ denotes a loss, $\frac12$ denotes a draw and $1$ denotes a win, while intermediate values express expectations between these outcomes.

### 4.2 Supervised Objective

For network input $s$, softmax over the complete action-index set converts the Policy logit vector into the supervised action-index distribution

$$
R_{\theta_P}(i\mid s)=
\frac{\exp\ell_{\theta_P}(s,i)}
{\displaystyle
\sum_{j\in\mathcal I_E}
\exp\ell_{\theta_P}(s,j)},
\qquad i\in\mathcal I_E.
$$

The supervised distribution $R_{\theta_P}(\cdot\mid s)$ and the legal-action distribution $P_{\theta_P}(\cdot\mid s)$ are derived from the same logits. The former normalizes all 4672 components, whereas the latter selects and normalizes the components indexed by the legal-action set of a complete state.

For minibatch $\mathcal B\subseteq\mathcal D_{\mathrm{sup}}$, the supervised Policy loss is the mean negative log-probability assigned to the target action indices:

$$
L_{P,\mathrm{sup}}^{(\mathcal B)}=
-\frac{1}{|\mathcal B|}
\sum_{(s,i,y)\in\mathcal B}
\log R_{\theta_P}(i\mid s).
$$

The supervised Value loss is the mean squared difference between the Value predictions and targets in the same minibatch:

$$
L_{V,\mathrm{sup}}^{(\mathcal B)}=
\frac{1}{|\mathcal B|}
\sum_{(s,i,y)\in\mathcal B}
\left(V_{\theta_V}(s)-y\right)^2.
$$

Their sum defines the complete supervised objective:

$$
L_{\mathrm{sup}}^{(\mathcal B)}=
L_{P,\mathrm{sup}}^{(\mathcal B)}
+
L_{V,\mathrm{sup}}^{(\mathcal B)}.
$$

Because $\theta_P$ and $\theta_V$ are disjoint, each loss contributes gradients only to its corresponding network:

$$
\nabla_{\theta_P}
L_{\mathrm{sup}}^{(\mathcal B)}
=
\nabla_{\theta_P}
L_{P,\mathrm{sup}}^{(\mathcal B)},
\qquad
\nabla_{\theta_V}
L_{\mathrm{sup}}^{(\mathcal B)}
=
\nabla_{\theta_V}
L_{V,\mathrm{sup}}^{(\mathcal B)}.
$$

For a fixed distribution of supervised records, the population minimizers are

$$
R^*(i\mid s)=
\Pr\left(i_n=i\mid s_n=s\right),
$$

$$
V^*(s)=
\mathbb E\left(y_n\mid s_n=s\right).
$$

The Policy objective therefore fits the conditional distribution of supervised action indices, and the Value objective fits the conditional mean expected score. Records whose complete states share the same encoding contribute to the same conditional quantities, including records that differ only in move counters or repetition history.

### 4.3 Parameter Optimization

For minibatch size $B$, the data loader reads $\mathcal D_{\mathrm{sup}}$ in contiguous chunks of

$$
C=\max(4096,16B)
$$

records. A completed epoch visits every record once after randomizing the chunk order and then the record order within each loaded chunk. This two-level shuffle prevents the fixed storage order from repeatedly concentrating related positions in consecutive minibatches, a pattern that may create excessive correlation between successive gradient estimates. The resulting minibatches are indexed by optimizer step as $\mathcal B_1,\mathcal B_2,\ldots$.

For each network label $H\in\lbrace\mathrm P,\mathrm V\rbrace$, let $\theta_H^{(0)}$ denote its parameters before the first optimizer step. In a newly constructed network, every nonpadding row of $E_H$ is sampled independently from $\mathcal N(0,0.01^2)$, the padding row is set to zero and every coordinate of $\beta_H$ is set to $\frac12$. The weights and biases of each dense layer with fan-in $n$ are sampled from $U(-n^{-1/2},n^{-1/2})$. The final Value weight $w_V$ and bias $b_V$ are set to zero, so a newly constructed Value network satisfies $V_{\theta_V^{(0)}}(s)=\frac12$ for every input $s$.

At optimizer step $k\geq1$, automatic differentiation computes the raw gradient of the corresponding supervised loss:

$$
g_{H,k}=
\nabla_{\theta_H^{(k-1)}}
L_{H,\mathrm{sup}}^{(\mathcal B_k)}.
$$

The Policy and Value gradients are clipped independently to a maximum Euclidean norm of $1$. With $\epsilon_c=10^{-6}$, the gradient passed to optimizer $H$ is

$$
\overline g_{H,k}=
\alpha_{H,k}g_{H,k},
\qquad
\alpha_{H,k}=
\min\left(
1,
\frac{1}{\lVert g_{H,k}\rVert_2+\epsilon_c}
\right).
$$

The two networks use separate AdamW optimizers with a common learning rate $\eta$ and weight-decay coefficient $\lambda$. For each optimizer, the first-moment estimate $m_{H,k}$ and second-moment estimate $v_{H,k}$ begin at zero. With $\beta_1=0.9$, $\beta_2=0.999$ and $\epsilon_A=10^{-8}$, update $k$ computes

$$
m_{H,k}=
\beta_1m_{H,k-1}
+(1-\beta_1)\overline g_{H,k},
\qquad
v_{H,k}=
\beta_2v_{H,k-1}
+(1-\beta_2)\overline g_{H,k}^2,
$$

$$
\widehat m_{H,k}=
\frac{m_{H,k}}{1-\beta_1^k},
\qquad
\widehat v_{H,k}=
\frac{v_{H,k}}{1-\beta_2^k}.
$$

The bias-corrected moments determine the parameter update

$$
\theta_H^{(k)}=
(1-\eta\lambda)\theta_H^{(k-1)}
-\eta
\frac{\widehat m_{H,k}}
{\sqrt{\widehat v_{H,k}}+\epsilon_A}.
$$

The square, square root and quotient in these equations act coordinatewise on tensors with the same shapes as $\theta_H$. Optimization proceeds until the requested epochs are complete or the number of updates reaches a positive step limit. When the step limit is zero, the epoch count is the sole stopping condition. A common random seed controls the initialization of newly constructed networks and both levels of data shuffling.

## 5. Search

Eleginus combines its two networks through Policy-guided best-first minimax (BFM). The Policy network determines the order in which boundary nodes are expanded, and the Value network evaluates ongoing boundary states. Exact terminal scores and minimax backup propagate the resulting evaluations through the explicit game tree.

### 5.1 Tree Nodes

For a terminal root state $x_0$, the rules engine returns $g(x_0,c_{x_0})$. For an ongoing root state, the BFM procedure constructs a tree rooted at $x_0$. Each node corresponds to one complete state $x$, and an edge labelled by legal action $a\in\mathcal A(x)$ leads to a child corresponding to $T(x,a)$. The tree creates a separate child node for every reached path, and the complete state stored in that node retains the path's move counters and repetition history. Nodes may therefore share the same network encoding while retaining distinct rule histories.

A node carries separate Policy and Value accumulators. The root accumulators are computed by a full refresh, while each child obtains its accumulators from the incremental update in Section 3.4. The boundary value of a node is

$$
v_0(x)=
\begin{cases}
g(x,c_x),&x\text{ is terminal},\\
V_{\theta_V}\left(\phi_E(x)\right),&x\text{ is ongoing}.
\end{cases}
$$

The rules engine supplies $g(x,c_x)$ for checkmate, stalemate, insufficient material, the fifty-move rule and threefold repetition. When a node is created, its backed value is initialized as

$$
\overline v(x)=v_0(x).
$$

### 5.2 Expansion

Expanding an ongoing node corresponding to state $x$ evaluates $P_{\theta_P}\left(\cdot\mid\phi_E(x)\right)$ over $\mathcal A(x)$ and creates one child for every legal action. For child state $x'=T(x,a)$, the Policy probability

$$
p(x,a)=
P_{\theta_P}\left(a\mid\phi_E(x)\right)
$$

is stored on the connecting edge. The child receives incrementally updated Policy and Value accumulators, the boundary value $v_0(x')$ and its initial backed value $\overline v(x')=v_0(x')$. Each ongoing child below the depth limit then enters the global frontier.

The root is refreshed and expanded before any node is removed from the frontier. Every later expansion follows the same procedure.

### 5.3 Frontier Priority

Let $\mathcal P(x)$ be the sequence of state-action pairs on the tree path from the root to node $x$. Its frontier priority is the logarithm of the Policy probability of that path:

$$
F(x)=
\sum_{(y,a)\in\mathcal P(x)}
\log\max\left(p(y,a),10^{-12}\right),
\qquad
F(x_0)=0.
$$

A maximum-priority queue contains the ongoing, unexpanded nodes whose depths are below the depth limit. The node with the greatest $F$ receives the next expansion, so a path with greater Policy probability is explored earlier. Equal priorities are resolved in favor of the node created earlier. Frontier order is determined by Policy path probabilities, while Value enters the procedure through the minimax backup defined next.

### 5.4 Minimax Backup

After a node is expanded, its backed value and the backed values of its ancestors are recomputed by

$$
\overline v(x)=
\begin{cases}
v_0(x),
&x\text{ is terminal or unexpanded},\\[4pt]
\displaystyle
\max_{a\in\mathcal A(x)}
\left(1-\overline v\left(T(x,a)\right)\right),
&x\text{ is expanded}.
\end{cases}
$$

Each child value uses the perspective of the side to move in the child state. The complement $1-\overline v(T(x,a))$ converts that value to the perspective of the player choosing action $a$ in state $x$. Replacing a boundary estimate with backed values from deeper nodes may therefore raise or lower the backed values of its ancestors.

### 5.5 Root Decision

For root state $x_0$ and legal action $a$, define the backed action value

$$
Q_B(x_0,a)=
1-\overline v\left(T(x_0,a)\right).
$$

The selected action maximizes $Q_B(x_0,a)$. Equal backed action values are resolved by the larger edge probability $p(x_0,a)$ and then by the lexicographically smaller coordinate move string. The reported root evaluation is $\overline v(x_0)$.

### 5.6 Search Limits

One expansion generates the complete legal child set of one parent node. The expansion count therefore increases by one, while the evaluated-node count increases by the number of generated children. The BFM procedure stops when it reaches the expansion limit or when the frontier is empty. The frontier admits newly created nodes only when their depths are smaller than the depth limit.

Because the root is expanded first, an expansion limit of one evaluates every legal successor of the root. For fixed network parameters, root state, expansion limit and depth limit, the frontier priorities and all tie-breaking rules determine a unique result.
