# Eleginus

> **Development status:** Eleginus is under active architectural, training and search experimentation. This document records the current implementation at a practical level and may change with debugging results. It is not yet a stable technical specification.

Eleginus combines an independently trained sparse Policy network with an NNUE-style Value network, using Policy to order a Value-driven principal-variation search.

## 1. Notation

- $\mathcal X$ is the set of complete chess states maintained by the rules engine.
- $x\in\mathcal X$ is a complete state containing piece placement, side to move, castling rights, en passant state, move counters and repetition history.
- $\mathcal A(x)$ is the set of legal actions in state $x$.
- $T(x,a)$ is the state reached by applying legal action $a\in\mathcal A(x)$ to state $x$.
- $c_x$ is the side to move in state $x$, and $\bar c_x$ is the opposing color.
- $\mathbf 1[A]$ equals $1$ when statement $A$ is true and $0$ otherwise.
- $\phi_E$ is the Eleginus state encoder, and $s=\phi_E(x)$ is the encoded state supplied to both neural networks.
- $\mathcal I_E=\lbrace0,\ldots,4671\rbrace$ is the fixed set of Eleginus action indices.
- $i_E(x,a)\in\mathcal I_E$ is the index assigned to legal action $a$ in state $x$.
- $\text{P}$, which stands for Policy, is a probability distribution over the legal actions of a state.
- $\text{V}$, which stands for Value, is the expected game score in $[0,1]$ from the perspective of the side to move.
- $\theta_P$ and $\theta_V$ are the disjoint parameter sets of the Policy and Value networks.
- $\ell_{\theta_P}(s,i)$ is the Policy logit assigned to action index $i$ for encoded state $s$.
- $P_{\theta_P}(a\mid s)$ is the Policy probability assigned to legal action $a$.
- $v_{\theta_V}(s)\in\mathbb R$ is the raw Value score used by tree search.
- $V_{\theta_V}(s)=\sigma(v_{\theta_V}(s))$ is the expected score used by supervised Value training, where $\sigma$ is the sigmoid function.

## 2. State and Action Encoding

The state encoder describes a position with sparse features from both color perspectives. The action codec uses a fixed side-relative index space, allowing the same Policy output layer to represent legal actions in every state.

### 2.1 State Encoding

Let $r,f\in\lbrace0,\ldots,7\rbrace$ denote rank and file coordinates, with $(0,0)$ representing `a1` and $(7,7)$ representing `h8`. Their absolute square index is

$$
q(r,f)=8r+f.
$$

For color perspective $c\in\lbrace\mathrm W,\mathrm B\rbrace$, the vertical orientation map is

$$
\omega_c(q(r,f))=
\begin{cases}
q(r,f),&c=\mathrm W,\\
q(7-r,f),&c=\mathrm B.
\end{cases}
$$

This transformation places the home rank of the perspective color at the bottom of the oriented board. Let $\kappa_c(x)\in\lbrace0,\ldots,63\rbrace$ be the oriented square occupied by its king.

The six piece types use the order pawn, knight, bishop, rook, queen and king. Let $\tau(p)\in\lbrace0,\ldots,5\rbrace$ be the type index of piece $p$. Relative to perspective $c$, its category is

$$
\rho_c(p)=
\tau(p)+6\mathbf 1[\mathrm{color}(p)\ne c].
$$

The first six categories contain friendly pieces, and the remaining six contain opposing pieces. A piece $p$ on absolute square $q$ activates the identifier

$$
\iota_c(x,p,q)=
768\kappa_c(x)+64\rho_c(p)+\omega_c(q).
$$

The three terms distinguish 64 oriented king squares, 12 perspective-relative piece categories and 64 oriented piece squares. Piece identifiers consequently occupy the range $0$ through $49151$.

Castling rights and en passant availability use additional sparse identifiers. For perspective $c$, let $K_c$ and $Q_c$ denote its kingside and queenside castling indicators, and let $K_{\bar c}$ and $Q_{\bar c}$ denote those of the opponent. The castling mask is

$$
m_c(x)=K_c+2Q_c+4K_{\bar c}+8Q_{\bar c},
$$

which activates identifier $49152+m_c(x)$ in the range $49152$ through $49167$. The en passant code is

$$
e(x)=
\begin{cases}
0,&\text{when no en passant square exists},\\
1+f_{\mathrm{ep}},&\text{otherwise},
\end{cases}
$$

where $f_{\mathrm{ep}}$ is the absolute en passant file. Vertical orientation preserves file coordinates, so both perspectives use the same code. It activates identifier $49168+e(x)$ in the range $49168$ through $49176$.

For perspective $c$, let $\mathcal F_c(x)$ contain the active piece identifiers and the two rule-context identifiers. Eleginus orders the piece identifiers by absolute square, appends the castling and en passant identifiers and pads the sequence to 34 entries with identifier $49177$. Denote the resulting sequence by $F_c(x)$. The state encoder places the side-to-move perspective first:

$$
s=\phi_E(x)=
\bigl(F_{c_x}(x),F_{\bar c_x}(x)\bigr)
\in\lbrace0,\ldots,49177\rbrace^{2\times34}.
$$

The complete state $x$ also retains move counters and repetition history. These rule fields can distinguish complete states that share the same encoded state $s$.

### 2.2 Action Encoding

The action codec applies the vertical orientation $\omega_{c_x}$ to express every legal action from the perspective of the side to move. Its fixed action space preserves distinct kingside and queenside moves.

Let $q_{\mathrm{from}}(a)$ be the absolute source square of action $a$. Let $q_{\mathrm{to}}(a)$ be its absolute destination square, except that castling uses the king destination `g1`, `c1`, `g8` or `c8` when the rules library internally stores the rook square. The oriented squares are

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

The source-destination block contains $64\times64=4096$ indices, and the underpromotion block contains $64\times9=576$ indices. Their union is

$$
\mathcal I_E=\lbrace0,\ldots,4671\rbrace,
\qquad
|\mathcal I_E|=4672.
$$

To decode an index for state $x$, the action codec generates $\mathcal A(x)$ and returns the legal action whose side-relative encoding equals that index. The returned action retains the castling, en passant or promotion information required by the rules engine.

## 3. Network

The Policy and Value networks receive the same encoded state and use separate parameters, allowing either objective to improve its own network without changing the other. Both networks begin with incrementally maintained sparse accumulators, after which their dense layers serve different roles.

### 3.1 Sparse Accumulators

Before table lookup, Eleginus maps each 64-king-square feature sequence to a horizontally canonical 32-bucket sequence. Write the oriented king square as $\kappa_c(x)=q(r_c,f_c)$, and define

$$
\chi_c(q(r,f))=
\begin{cases}
q(r,7-f),&f_c<4,\\
q(r,f),&f_c\geq4.
\end{cases}
$$

The canonical king lies on files `e` through `h`, giving bucket

$$
\widehat\kappa_c(x)=
4r_c+\bigl(\chi_c(\kappa_c(x))\bmod8\bigr)-4
\in\lbrace0,\ldots,31\rbrace.
$$

The corresponding piece feature is

$$
\widehat\iota_c(x,p,q)=
768\widehat\kappa_c(x)
+64\rho_c(p)
+\chi_c\left(\omega_c(q)\right),
$$

which occupies the range $0$ through $24575$. When $f_c<4$, the same reflection exchanges the kingside and queenside castling indicators and maps en passant file $f_{\mathrm{ep}}$ to $7-f_{\mathrm{ep}}$. The transformed castling mask activates identifiers $24576$ through $24591$, and the transformed en passant code activates identifiers $24592$ through $24600$. Raw padding identifier $49177$ maps to network padding identifier $24601$.

Let $\widehat{\mathcal F}_c(x)$ contain the resulting network identifiers. This deterministic mapping allows the stable encoded state in Section 2.1 to feed the smaller mirrored feature tables and makes horizontally reflected positions share their sparse network representation.

The Policy feature table is $E_P\in\mathbb R^{24602\times128}$, and its accumulator bias is $\beta_P\in\mathbb R^{128}$. For perspective $c$, the unclipped Policy accumulator is

$$
u_{P,c}(x)=
\beta_P+
\sum_{j\in\widehat{\mathcal F}_c(x)}E_{P,j}.
$$

After clipping each coordinate to $[0,1]$, the side-to-move accumulator and the opposing accumulator are concatenated:

$$
h_P(s)=
\mathrm{clip}_{[0,1]}\left(u_{P,c_x}(x)\right)
\mathbin\Vert
\mathrm{clip}_{[0,1]}\left(u_{P,\bar c_x}(x)\right)
\in\mathbb R^{256}.
$$

The Value feature table is $E_V\in\mathbb R^{24602\times520}$, and its accumulator bias is $\beta_V\in\mathbb R^{520}$. Its unclipped accumulator is

$$
u_{V,c}(x)=
\beta_V+
\sum_{j\in\widehat{\mathcal F}_c(x)}E_{V,j}.
$$

The first 512 coordinates supply the dense Value network. Eleginus applies the squared clipped rectifier

$$
\mathrm{SCReLU}(z)=\mathrm{clip}_{[0,1]}(z)^2
$$

coordinatewise and forms

$$
h_V(s)=
\mathrm{SCReLU}\left(u_{V,c_x}(x)_{0:512}\right)
\mathbin\Vert
\mathrm{SCReLU}\left(u_{V,\bar c_x}(x)_{0:512}\right)
\in\mathbb R^{1024}.
$$

The final eight coordinates of each Value accumulator provide one direct sparse score for each material bucket. Section 3.3 combines the selected direct score with the dense Value output.

### 3.2 Policy Network

The Policy network applies a rectified affine map to its accumulator input:

$$
y_P(s)=
\mathrm{ReLU}\left(W_{P,1}h_P(s)+b_{P,1}\right)
\in\mathbb R^{128},
$$

where $W_{P,1}\in\mathbb R^{128\times256}$ and $b_{P,1}\in\mathbb R^{128}$. A second affine map produces the complete logit vector

$$
\ell_{\theta_P}(s)=
W_{P,2}y_P(s)+b_{P,2}
\in\mathbb R^{4672}.
$$

For an ongoing state $x$ with $s=\phi_E(x)$, normalizing the logits indexed by legal actions gives

$$
P_{\theta_P}(a\mid s)=
\frac{\exp\ell_{\theta_P}\left(s,i_E(x,a)\right)}
{\displaystyle
\sum_{b\in\mathcal A(x)}
\exp\ell_{\theta_P}\left(s,i_E(x,b)\right)},
\qquad a\in\mathcal A(x).
$$

Legal-action inference evaluates only the rows of $W_{P,2}$ and $b_{P,2}$ selected by the current legal-action indices. This restricted projection produces the same logits and probabilities as selecting those entries from the complete 4672-dimensional output.

### 3.3 Value Network

Let $n(x)$ be the number of pieces on the board. The material bucket is

$$
b(x)=
\min\left(
7,
\max\left(0,\left\lfloor\frac{n(x)-1}{4}\right\rfloor\right)
\right).
$$

Each bucket $b$ has its own two-layer dense mapping and scalar output. For the selected bucket,

$$
y_{V,b}(s)=
\mathrm{ReLU}\left(W_{V,1}^{(b)}h_V(s)+b_{V,1}^{(b)}\right)
\in\mathbb R^{32},
$$

$$
r_{V,b}(s)=
\mathrm{ReLU}\left(W_{V,2}^{(b)}y_{V,b}(s)+b_{V,2}^{(b)}\right)
\in\mathbb R^{32}.
$$

The direct sparse score for the same bucket is the difference between the two perspective accumulators:

$$
d_{V,b}(s)=
\frac12\left(
u_{V,c_x}(x)_{512+b}
-u_{V,\bar c_x}(x)_{512+b}
\right).
$$

The raw Value score combines the dense and direct paths:

$$
v_{\theta_V}(s)=
\left(w_V^{(b)}\right)^\top r_{V,b}(s)
+b_V^{(b)}
+d_{V,b}(s),
\qquad b=b(x).
$$

Supervised training maps this score to an expected game score:

$$
V_{\theta_V}(s)=
\sigma\left(v_{\theta_V}(s)\right)
=
\frac{1}{1+\exp\left(-v_{\theta_V}(s)\right)}
\in[0,1].
$$

The eight material-dependent mappings allow the Value network to assign different dense transformations to crowded middlegames and sparse endgames. The direct path gives every sparse feature an additive route to the final score, while the dense path models interactions among the active features.

### 3.4 Incremental Evaluation

A full refresh sums the active feature rows for both perspectives. When legal action $a\in\mathcal A(x)$ produces $x'=T(x,a)$, the accumulator of network $H\in\lbrace P,V\rbrace$ and perspective $c$ is updated by

$$
u_{H,c}(x')=
u_{H,c}(x)
-\sum_{j\in\widehat{\mathcal F}_c(x)\setminus\widehat{\mathcal F}_c(x')}E_{H,j}
+\sum_{j\in\widehat{\mathcal F}_c(x')\setminus\widehat{\mathcal F}_c(x)}E_{H,j}.
$$

The first sum removes inactive features, and the second adds newly active features. Features shared by the two states remain in the accumulator, so the update produces the same result as a full refresh.

Most actions change only a small number of identifiers. A king move can change the king bucket or horizontal reflection of its color perspective, in which case every piece identifier in that perspective is replaced. The dense layers receive the clipped or squared-clipped representation constructed from the updated accumulators.

## 4. Supervised Training

### 4.1 Supervised Data

Let $\mathcal D_{\mathrm{sup}}$ contain $N$ records:

$$
\mathcal D_{\mathrm{sup}}=
\lbrace\xi_n\rbrace_{n=1}^{N}.
$$

Each record is associated with a complete pre-move state $x_n$ and a selected legal action $a_n\in\mathcal A(x_n)$. Its stored form is

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

The action index $i_n$ is the Policy target. The scalar $y_n$ is the expected-score target from the perspective of the side to move in $x_n$, with $0$, $\frac12$ and $1$ representing a loss, draw and win. Intermediate values represent expectations between these outcomes.

### 4.2 Supervised Objective

Softmax over the complete action-index set converts the Policy logit vector into the supervised distribution

$$
R_{\theta_P}(i\mid s)=
\frac{\exp\ell_{\theta_P}(s,i)}
{\displaystyle
\sum_{j\in\mathcal I_E}
\exp\ell_{\theta_P}(s,j)},
\qquad i\in\mathcal I_E.
$$

The supervised distribution $R_{\theta_P}$ and the legal-action distribution $P_{\theta_P}$ are derived from the same logits, but they use different normalization domains. The former normalizes all 4672 action indices during training, whereas the latter normalizes the legal-action indices of the current state during inference.

For minibatch $\mathcal B\subseteq\mathcal D_{\mathrm{sup}}$, the supervised Policy loss is

$$
L_{P,\mathrm{sup}}^{(\mathcal B)}=
-\frac{1}{|\mathcal B|}
\sum_{(s,i,y)\in\mathcal B}
\log R_{\theta_P}(i\mid s).
$$

The supervised Value loss treats the expected-score target as a soft Bernoulli label and applies binary cross-entropy to the raw Value logit:

$$
L_{V,\mathrm{sup}}^{(\mathcal B)}=
-\frac{1}{|\mathcal B|}
\sum_{(s,i,y)\in\mathcal B}
\left[
y\log V_{\theta_V}(s)
+(1-y)\log\left(1-V_{\theta_V}(s)\right)
\right].
$$

The implementation evaluates this expression with a numerically stable binary-cross-entropy-with-logits operator. Its derivative with respect to the raw logit is proportional to $V_{\theta_V}(s)-y$, so corrections near either end of the expected-score range retain their gradient strength.

Their sum defines the complete supervised objective:

$$
L_{\mathrm{sup}}^{(\mathcal B)}=
L_{P,\mathrm{sup}}^{(\mathcal B)}
+L_{V,\mathrm{sup}}^{(\mathcal B)}.
$$

The disjoint parameter sets separate the two gradient paths:

$$
\nabla_{\theta_P}L_{\mathrm{sup}}^{(\mathcal B)}
=
\nabla_{\theta_P}L_{P,\mathrm{sup}}^{(\mathcal B)},
\qquad
\nabla_{\theta_V}L_{\mathrm{sup}}^{(\mathcal B)}
=
\nabla_{\theta_V}L_{V,\mathrm{sup}}^{(\mathcal B)}.
$$

For a fixed distribution of supervised records, the population minimizers are

$$
R^*(i\mid s)=
\Pr(i_n=i\mid s_n=s),
\qquad
V^*(s)=
\mathbb E(y_n\mid s_n=s).
$$

The Policy objective fits the conditional distribution of recorded action indices, and the Value objective fits the conditional mean expected score.

### 4.3 Parameter Optimization

For minibatch size $B$, the data loader reads contiguous chunks of

$$
C=\max(4096,16B)
$$

records. Each epoch visits every record once after randomizing the chunk order and the record order within each loaded chunk. This two-level shuffle prevents the fixed storage order from repeatedly concentrating related states in consecutive minibatches, which could increase correlation between successive gradient estimates.

The two networks use separate AdamW optimizers. For each network label $H\in\lbrace P,V\rbrace$, automatic differentiation computes

$$
g_{H,k}=
\nabla_{\theta_H^{(k-1)}}
L_{H,\mathrm{sup}}^{(\mathcal B_k)}
$$

at optimizer step $k$. Each gradient is independently rescaled to a maximum Euclidean norm of $1$ before its optimizer updates the corresponding parameter set. With learning rate $\eta$, weight-decay coefficient $\lambda$, $\beta_1=0.9$, $\beta_2=0.999$ and $\epsilon_A=10^{-8}$, AdamW maintains

$$
m_{H,k}=
\beta_1m_{H,k-1}+(1-\beta_1)\overline g_{H,k},
$$

$$
n_{H,k}=
\beta_2n_{H,k-1}+(1-\beta_2)\overline g_{H,k}^{,2},
$$

where $\overline g_{H,k}$ is the clipped gradient. The bias-corrected moments are

$$
\widehat m_{H,k}=
\frac{m_{H,k}}{1-\beta_1^k},
\qquad
\widehat n_{H,k}=
\frac{n_{H,k}}{1-\beta_2^k},
$$

and the parameter update is

$$
\theta_H^{(k)}=
(1-\eta\lambda)\theta_H^{(k-1)}
-\eta
\frac{\widehat m_{H,k}}
{\sqrt{\widehat n_{H,k}}+\epsilon_A}.
$$

All squares, square roots and divisions in these equations act coordinatewise. Optimization ends after the requested epochs or a positive optimizer-step limit. A zero step limit allows the epoch count to determine the training length.

The nonpadding feature rows begin with independent samples from $\mathcal N(0,0.01^2)$, and the accumulator biases begin at $\frac12$. The final eight coordinates of the Value feature rows and bias begin at zero, as do the dense Value output weights and biases. A newly initialized Value network therefore satisfies $v_{\theta_V}(s)=0$ and $V_{\theta_V}(s)=\frac12$ for every encoded state.

## 5. Search

Eleginus applies iterative deepening and principal-variation search to a negamax tree. The Value network supplies static leaf scores, while the Policy network orders legal actions so that promising branches establish alpha-beta bounds early.

### 5.1 Scores and Move Order

For an ongoing state $x$, the static score in centipawns is

$$
E_{\theta_V}(x)=
\mathrm{clip}_{[-25000,25000]}
\left(
\mathrm{round}\left(150v_{\theta_V}(\phi_E(x))\right)
\right).
$$

The scale follows the annotation transformation used by supervised Value targets. If a pawn-unit annotation $z$ produces target $y=(\tanh(z/3)+1)/2$, then the ideal raw score is $v=2z/3$, and $150v$ equals $100z$ centipawns.

The rules engine evaluates terminal states before neural evaluation. A draw receives score $0$. At search ply $p$, a win for the side to move receives $30000-p$, and a loss receives $-30000+p$. The ply adjustment prefers faster wins and slower losses.

At every ongoing state, legal actions are sorted by decreasing $P_{\theta_P}(a\mid\phi_E(x))$. Equal probabilities are ordered by their coordinate move strings. When a shallower completed iteration has identified a best action for the same board hash, that action is moved to the front of the ordered list. Policy and iterative-deepening information therefore determine search order, while returned action scores come from the Value-driven negamax tree.

### 5.2 Quiescence Search

At the nominal depth boundary, quiescence search replaces an immediate static evaluation. In a state where the side to move is not in check, the static score acts as the stand-pat value. A stand-pat score at least $\beta$ produces a beta cutoff, and a larger stand-pat score raises $\alpha$. The node returns after this update when its remaining quiescence depth is zero.

Quiescence search otherwise examines captures and promotions in decreasing order of captured-piece gain, with promotion value included in the ordering score. A checked state examines every legal evasion because stand pat is invalid while the king is in check. The default quiescence allowance is eight plies, and checked continuations may extend by four additional plies before static evaluation terminates the branch. A nonchecking capture is skipped when its stand-pat score plus the captured-piece value and a 120-centipawn margin remains below $\alpha$. Every searched child uses the negated window $[-\beta,-\alpha]$ and returns the negative of its child score.

### 5.3 Principal-Variation Search

Let the ordered legal actions at state $x$ be $a_1,\ldots,a_m$. At remaining principal depth $d>0$, the first action is searched with the full negated alpha-beta window:

$$
q_1=-\mathrm{PVS}\left(T(x,a_1),d-1,-\beta,-\alpha\right).
$$

After updating $\alpha$, each later action is first tested with a null window of width one centipawn:

$$
q_j=-\mathrm{PVS}\left(T(x,a_j),d-1,-\alpha-1,-\alpha\right),
\qquad j>1.
$$

A null-window result satisfying $\alpha<q_j<\beta$ proves that the action improves the current bound but does not determine its exact score. The procedure then repeats that child with the full window $[-\beta,-\alpha]$. A score satisfying $q_j\geq\beta$ produces a beta cutoff, and the remaining actions at that node do not affect the current bound.

Late-move reduction applies to a quiet, nonchecking action after the first three ordered actions when the remaining depth is at least three. The reduced null-window search removes one ply, with one further ply removed at depth six and another after the first eight actions. A reduced result above $\alpha$ is repeated at full depth before it can raise the node bound.

Depth zero invokes the quiescence procedure in Section 5.2. Terminal scores and static scores use the perspective of the side to move at their own nodes, and negation converts each child result to the parent perspective.

### 5.4 Transposition Table

A direct-mapped transposition table stores the position key, searched depth, score bound and best action. A stored entry may return an exact score or tighten $\alpha$ or $\beta$ when its depth covers the current request. Shallower entries still supply their best actions for move ordering. Rule-terminal states are resolved before table lookup, and the table key combines the board hash with the halfmove clock and current repetition status.

### 5.5 Iterative Deepening and Root Decision

For requested principal depth $D$, Eleginus completes searches at depths $1,2,\ldots,D$. Each iteration starts with the full root window $[-32000,32000]$. Best actions recorded at internal states become ordering hints for later iterations, allowing shallow tactical information to supplement the learned Policy order.

The final iteration determines the selected root action. Actions are ranked first by their PVS scores, then by exact bounds, Policy probabilities and coordinate move strings. The reported root score is the selected action's exact PVS score. Node counts include principal and quiescence nodes visited across all completed iterations, while the selective depth is the greatest ply reached by either procedure.
