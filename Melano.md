# Melano

Melano is a geometry-aware Transformer chess network that jointly predicts a move policy and a side-to-move position evaluation.

## 1. Notation

- $\mathcal X$ is the set of complete chess states maintained by the rules engine.
- $x\in\mathcal X$ is a complete chess state containing the board, side to move, castling rights, en passant state, move counters and repetition history.
- $\mathcal A(x)$ is the set of legal actions in state $x$.
- $T(x,a)$ is the complete state reached by applying legal action $a\in\mathcal A(x)$ to state $x$.
- $z(x)\in\lbrace-1,0,1\rbrace$ is the exact outcome of terminal state $x$ from the perspective of its side to move, with $1$, $0$ and $-1$ representing a win, draw and loss.
- $\phi_M$ is the Melano state encoder that maps a complete chess state to a network input.
- $s=\phi_M(x)$ is the Melano network input obtained from complete state $x$.
- $\mathcal I_M=\lbrace0,\ldots,4671\rbrace$ is the fixed set of Melano action indices.
- $i_M(a)\in\mathcal I_M$ is the action index assigned to legal action $a$.
- $\theta$ denotes the trainable network parameters.
- $\ell_\theta(s)\in\mathbb R^{4672}$ is the complete vector of Policy logits produced by the network with parameters $\theta$ and $\ell_\theta(s,i)$ is its scalar component for action index $i\in\mathcal I_M$.
- $\text{P}$, which stands for Policy, is the network output that assigns a probability distribution over the legal actions available in each encoded state.
- $\text{V}$, which stands for Value, is a scalar network output in $[-1,1]$ that estimates the expected game result from the perspective of the side to move.
- $Q$ denotes a state or action evaluation defined by a particular procedure. Each definition specifies its arguments and observation perspective.
- $\mathrm{clip}_{[l,u]}(y)=\min(u,\max(l,y))$ restricts scalar $y$ to the closed interval $[l,u]$.

## 2. State and Action Encoding

### 2.1 State Encoding

The network requires a fixed numerical representation of each complete chess state. Melano represents a state with 64 categorical square entries followed by three entries for rule information. The state encoder $\phi_M$ produces

$$
\phi_M(x)=s=(p_0,\ldots,p_{63},t,c,e),
$$

where each $p_q$ describes one board square, $t$ describes the side to move, $c$ describes the castling rights and $e$ describes the en passant file.

To order the square entries, let $r\in\lbrace0,\ldots,7\rbrace$ be a rank coordinate and let $f\in\lbrace0,\ldots,7\rbrace$ be a file coordinate. Rank 1 has coordinate $r=0$, and the coordinate increases toward rank 8. File `a` has coordinate $f=0$, and the coordinate increases toward file `h`. The square index is

$$
q=8r+f.
$$

For each $q\in\lbrace0,\ldots,63\rbrace$, the categorical entry $p_q\in\lbrace0,\ldots,12\rbrace$ records the occupant of that square. The value 0 represents an empty square. White pawn, knight, bishop, rook, queen and king use values 1 through 6, and the corresponding Black pieces use values 7 through 12.

The side-to-move entry is

$$
t=
\begin{cases}
1,&\text{White to move},\\
0,&\text{Black to move}.
\end{cases}
$$

The castling entry $c\in\lbrace0,\ldots,15\rbrace$ is a four-bit mask. From least to most significant, its bits represent White kingside, White queenside, Black kingside and Black queenside castling rights. The en passant entry is

$$
e=
\begin{cases}
0,&\text{no en passant square exists},\\
1+f_{\mathrm{ep}},&\text{an en passant square exists on file }f_{\mathrm{ep}},
\end{cases}
$$

where $f_{\mathrm{ep}}\in\lbrace0,\ldots,7\rbrace$ uses the same file coordinates as the square entries.

Each of the 67 entries fits in one unsigned byte. `PackedState` denotes their 67-byte storage representation in the order shown in $s$, and converting those bytes to integer embedding indices reproduces $\phi_M(x)$ exactly.

The encoded state $\phi_M(x)$ records piece placement, side to move, castling rights and the en passant file. The complete state $x$ additionally records move counters and repetition history. Since $\phi_M$ omits those two fields, complete states that differ only in move counters or repetition history produce the same network input.

### 2.2 Action Encoding

The legal-action set $\mathcal A(x)$ varies with the complete state $x$. To give actions a state-independent numerical representation, Melano assigns each legal action $a\in\mathcal A(x)$ an index $i_M(a)$ in the fixed set $\mathcal I_M$.

Melano divides $\mathcal I_M$ into 4096 source-destination indices and 576 underpromotion indices. Let $q\in\lbrace0,\ldots,63\rbrace$ be the source-square index of action $a$, and let $k\in\lbrace0,\ldots,63\rbrace$ be its destination-square index. An ordinary move or a promotion to a queen uses

$$
i_M(a)=64q+k.
$$

This formula assigns indices 0 through 4095 to all $64\times64$ source-destination pairs. The rules library represents castling internally with the rook square as the destination of the king move. Before computing $k$, Melano replaces that internal destination with the king destination `g1`, `c1`, `g8` or `c8`.

An underpromotion is determined by its source square, destination-file displacement and promoted piece. Let $\Delta f\in\lbrace-1,0,1\rbrace$ be the destination file minus the source file, and let

$$
u=
\begin{cases}
0,&\text{promotion to a knight},\\
1,&\text{promotion to a bishop},\\
2,&\text{promotion to a rook}.
\end{cases}
$$

The underpromotion index is

$$
i_M(a)=4096+9q+3(\Delta f+1)+u.
$$

This formula assigns indices 4096 through 4671 to the $64\times3\times3=576$ combinations. Together, the two regions give

$$
|\mathcal I_M|=64\times64+64\times9=4672.
$$

For complete state $x$, the available action indices are

$$
\mathcal I_M(x)=
\lbrace i_M(a)\mid a\in\mathcal A(x)\rbrace.
$$

To decode an available index, Melano generates $\mathcal A(x)$ and selects the legal action whose encoding equals that index. The selected action contains the castling, en passant or promotion information required by the rules engine.

## 3. Network

### 3.1 Geometry-Attention Encoder

The Melano network derives its $\text{P}$ and $\text{V}$ from a shared geometry-attention encoder. The state embedding converts the encoded state $s$ into one global token and 64 square tokens, and a sequence of geometry-attention blocks transforms these tokens into the shared representation used by both output heads.

Let $C\geq1$ be the feature width of every token and let $B\geq1$ be the number of geometry-attention blocks. The state embedding produces

$$
h_0(s)\in\mathbb R^{65\times C},
$$

where token 0 is global and token $q+1$ corresponds to square index $q\in\lbrace0,\ldots,63\rbrace$.

The side-to-move, castling and en passant entries first form the rule-context embedding

$$
r(s)=E_{\mathrm{side}}(t)+E_{\mathrm{castling}}(c)+E_{\mathrm{ep}}(e).
$$

The rule-context embedding contributes to all 65 initial tokens. For square $q$, it is added to the embeddings of the square occupant and the absolute square index:

$$
h_{0,q+1}(s)=E_{\mathrm{piece}}(p_q)+E_{\mathrm{square}}(q)+r(s).
$$

The global token combines the same context with a trainable vector $g\in\mathbb R^C$:

$$
h_{0,0}(s)=g+r(s).
$$

The piece, square, side-to-move, castling and en passant embedding tables contain 13, 64, 2, 16 and 9 vectors, respectively.

Each geometry-attention block uses the largest member of $\lbrace8,4,2,1\rbrace$ that divides $C$ as its head count $H$. The resulting feature width of each head is

$$
d=\frac{C}{H}.
$$

Attention scores combine learned token features with relations derived from chessboard geometry. Let $\rho(u,v)\in\lbrace0,\ldots,28\rbrace$ be the relation identifier assigned to source token $u$ and target token $v$. The global token has index 0, and every pair containing that token uses

$$
\rho(u,v)=0.
$$

When $u$ and $v$ are square-token indices, let $d_r$ and $d_f$ be the absolute rank and file differences between squares $u-1$ and $v-1$. Their relation identifier is

$$
\rho(u,v)=
\begin{cases}
1,&d_r=0\ \text{and}\ d_f=0,\\
1+d_f,&d_r=0\ \text{and}\ 1\leq d_f\leq7,\\
8+d_r,&d_f=0\ \text{and}\ 1\leq d_r\leq7,\\
15+d_r,&d_r=d_f\ \text{and}\ 1\leq d_r\leq7,\\
23,&\lbrace d_r,d_f\rbrace=\lbrace1,2\rbrace,\\
24+\min(4,d_r+d_f-4),&\text{otherwise}.
\end{cases}
$$

The square-token cases distinguish identical squares, distances along a rank, distances along a file, diagonal distances, knight-move geometry and five classes for the remaining relative positions. Together with relation 0, these cases use every identifier in $\lbrace0,\ldots,28\rbrace$.

For block $b\in\lbrace0,\ldots,B-1\rbrace$, let $Z_b\in\mathbb R^{65\times C}$ be its trainable position tensor. Adding this tensor to the block input gives

$$
\widetilde h_b=h_b+Z_b.
$$

The first LayerNorm in the block normalizes each token across its $C$ features. A linear map then produces the query, key and value features, which are reshaped into $H$ heads:

$$
(Q_b,K_b,U_b)=
\mathrm{Split}_{3}\left(
\mathrm{Reshape}_{3,H,d}
\left(W_{qkv,b}\mathrm{LN}_{b,1}(\widetilde h_b)+b_{qkv,b}\right)
\right).
$$

Each block contains a static geometry-bias table $\beta_b\in\mathbb R^{29\times H}$. The block also derives a state-dependent geometry-bias table from its position-adjusted global token:

$$
\Gamma_b(s)=
\mathrm{Reshape}_{H,29}\left(
W_{\gamma,b,2}\mathrm{GELU}
\left(W_{\gamma,b,1}\mathrm{LN}(\widetilde h_{b,0})+b_{\gamma,b,1}\right)
+b_{\gamma,b,2}
\right).
$$

For attention head $h$, source token $u$ and target token $v$, the query-key similarity, static geometry bias and state-dependent geometry bias define the attention score

$$
S_{b,h,u,v}=
\frac{Q_{b,h,u}\cdot K_{b,h,v}}{\sqrt d}
+\beta_{b,\rho(u,v),h}
+\Gamma_b(s)_{h,\rho(u,v)}.
$$

SoftMax normalizes these scores over all 65 possible target tokens for each fixed block, head and source token. The resulting output of head $h$ for source token $u$ is

$$
o_{b,h,u}=
\sum_{v=0}^{64}
\frac{\exp S_{b,h,u,v}}
{\displaystyle\sum_{w=0}^{64}\exp S_{b,h,u,w}}
U_{b,h,v}.
$$

Concatenating the $H$ head outputs, applying the output projection and adding the block input produces the attention residual output

$$
y_{b,u}=\widetilde h_{b,u}+
W_{o,b}\mathrm{Concat}_{h=1}^{H}(o_{b,h,u})+b_{o,b}.
$$

A second LayerNorm supplies each token to a feed-forward network whose hidden width is $4C$. The feed-forward result is added to $y_b$ to produce the block output

$$
h_{b+1}=y_b+
W_{f,b,2}\mathrm{GELU}
\left(W_{f,b,1}\mathrm{LN}_{b,2}(y_b)+b_{f,b,1}\right)+b_{f,b,2}.
$$

Applying all $B$ blocks defines the shared geometry-aware representation

$$
E_\theta(s)=h_B(s)\in\mathbb R^{65\times C}.
$$

The complete encoder path is

$$
s\longrightarrow\text{state embedding}\longrightarrow h_0
\longrightarrow\text{$B$ geometry-attention blocks}
\longrightarrow E_\theta(s).
$$

The trainable global token $g$ and every position tensor $Z_b$ begin at zero. Embedding tables and linear maps use the default LibTorch initialization. Each LayerNorm has trainable scale and bias, uses epsilon $10^{-5}$ and begins with unit scale and zero bias. The encoder applies no dropout.

### 3.2 Policy and Value Heads

The Policy head derives action logits from the 64 transformed square tokens in $E_\theta(s)$. Let

$$
z_q=E_\theta(s)_{q+1},
\qquad q\in\lbrace0,\ldots,63\rbrace,
$$

and let $\overline z_q=\mathrm{LN}_P(z_q)$ be the normalized token for square $q$. Separate linear maps produce its source and destination features:

$$
u_q=W_F\overline z_q+b_F,
\qquad
v_q=W_T\overline z_q+b_T.
$$

For source square $q$ and destination square $k$, their scaled dot product is the logit of source-destination index $64q+k$:

$$
\ell_\theta(s,64q+k)=\frac{u_q\cdot v_k}{\sqrt C}.
$$

A third linear map produces the nine underpromotion logits associated with source square $q$:

$$
c_q=W_U\overline z_q+b_U\in\mathbb R^9.
$$

For $m\in\lbrace0,\ldots,8\rbrace$, the corresponding logit is

$$
\ell_\theta(s,4096+9q+m)=c_{q,m}.
$$

The component index $m=3(\Delta f+1)+u$ uses the destination-file displacement $\Delta f$ and promoted-piece index $u$ defined in Section 2.2. Flattening the $64\times64$ source-destination matrix and appending the $64\times9$ underpromotion matrix produces the complete Policy-logit vector

$$
\ell_\theta(s)\in\mathbb R^{4672}.
$$

For complete state $x$ with $s=\phi_M(x)$, selecting the logits indexed by legal actions and normalizing them with softmax produces the legal-move Policy:

$$
P_\theta(a\mid s)=
\frac{\exp\ell_\theta(s,i_M(a))}
{\displaystyle\sum_{b\in\mathcal A(x)}\exp\ell_\theta(s,i_M(b))},
\qquad a\in\mathcal A(x).
$$

The denominator ranges over $\mathcal A(x)$, so $P_\theta(\cdot\mid s)$ is a probability distribution over the legal actions in complete state $x$.

The Value head derives its scalar output from the transformed global token $E_\theta(s)_0$. LayerNorm supplies that token to a $C\rightarrow256$ linear map and ReLU, and a $256\rightarrow1$ linear map followed by a hyperbolic tangent produces

$$
V_\theta(s)=
\tanh\left(
W_{V,2}\mathrm{ReLU}
\left(W_{V,1}\mathrm{LN}_V(E_\theta(s)_0)+b_{V,1}\right)+b_{V,2}
\right)
\in[-1,1].
$$

Writing $f_\theta$ for the complete network gives

$$
f_\theta(s)=\left(\ell_\theta(s),V_\theta(s)\right).
$$

The shared encoder and the two output heads form the path

$$
s\longrightarrow E_\theta(s)\longrightarrow
\begin{cases}
\text{Policy head}\longrightarrow\ell_\theta(s)
\longrightarrow P_\theta(\cdot\mid s),\\
\text{Value head}\longrightarrow V_\theta(s).
\end{cases}
$$

### 3.3 Inference Evaluation

While the Policy head defines one logit for every action index, forming the complete vector $\ell_\theta(s)$, legal-move inference requires only the components indexed by actions in $\mathcal A(x)$. The inference path therefore reuses the square-token projections defined in Section 3.2 and directly evaluates the corresponding source-destination dot products and underpromotion logits.

For a batch of encoded states $\mathbf s=(s_1,\ldots,s_n)$, let $L$ be the largest legal-action count in the batch. A matrix $J\in\mathcal I_M^{n\times L}$ stores the requested action indices, and a mask $M\in\lbrace0,1\rbrace^{n\times L}$ marks the entries that represent legal actions rather than padding. The selected-logit tensor is

$$
\Lambda_\theta(\mathbf s,J)_{rj}
=
\ell_\theta(s_r,J_{rj}).
$$

For an index $J_{rj}=64q+k$, Melano computes the source-destination dot product $u_q\cdot v_k/\sqrt C$. For an index $J_{rj}=4096+9q+m$, it selects component $m$ from $c_q$. Padded positions are excluded before softmax, so each row is normalized over the legal actions of its own complete state. This selected-logit evaluation produces the same legal-action logits and the same $P_\theta(\cdot\mid s_r)$ as the corresponding components of the complete vector.

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
s_n=\phi_M(x_n),
\qquad
i_n=i_M(a_n),
\qquad
y_n\in[-1,1].
$$

The encoded state $s_n$ is the network input, and the action index $i_n$ is the Policy target. The scalar $y_n$ is the Value target, expressed as an estimate of the expected game result from the perspective of the side to move in $x_n$. On this scale, $-1$ denotes a loss, $0$ denotes a draw and $1$ denotes a win, and intermediate values express expectations between these outcomes.

### 4.2 Supervised Objective

For network input $s$, the Policy head produces the complete logit vector

$$
\ell_\theta(s)=
\left(\ell_\theta(s,i)\right)_{i\in\mathcal I_M}
\in\mathbb R^{4672}.
$$

Applying softmax to all 4672 components produces the supervised action-index distribution $R_\theta$:

$$
R_\theta(i\mid s)=
\frac{\exp\ell_\theta(s,i)}
{\displaystyle\sum_{j\in\mathcal I_M}\exp\ell_\theta(s,j)},
\qquad i\in\mathcal I_M.
$$

Both $R_\theta(\cdot\mid s)$ and the legal-move distribution $P_\theta(\cdot\mid s)$ are derived from $\ell_\theta(s)$, but they differ in normalization domain. $R_\theta$ normalizes all 4672 components for supervised learning, whereas $P_\theta$ selects the components indexed by actions in $\mathcal A(x)$ and normalizes those components for legal-move inference.

For minibatch $\mathcal B\subseteq\mathcal D_{\mathrm{sup}}$, the supervised Policy loss is the mean negative log-probability assigned to the target action indices:

$$
L_{P,\mathrm{sup}}^{(\mathcal B)}=
-\frac{1}{|\mathcal B|}
\sum_{(s,i,y)\in\mathcal B}
\log R_\theta(i\mid s).
$$

The supervised Value loss is the mean squared difference between the predicted and target expected results in the same minibatch:

$$
L_{V,\mathrm{sup}}^{(\mathcal B)}=
\frac{1}{|\mathcal B|}
\sum_{(s,i,y)\in\mathcal B}
\left(V_\theta(s)-y\right)^2.
$$

The nonnegative coefficient $w_V$ sets the contribution of the Value loss to the complete minibatch objective:

$$
L_{\mathrm{sup}}^{(\mathcal B)}=
L_{P,\mathrm{sup}}^{(\mathcal B)}
+w_VL_{V,\mathrm{sup}}^{(\mathcal B)}.
$$

To specify how the two losses affect the shared network, partition the parameters as

$$
\theta=(\theta_E,\theta_P,\theta_V),
$$

where $\theta_E$ contains the state-embedding and geometry-attention parameters, while $\theta_P$ and $\theta_V$ contain the parameters of the Policy and Value heads. The gradients of the complete objective satisfy

$$
\nabla_{\theta_P}L_{\mathrm{sup}}^{(\mathcal B)}
=
\nabla_{\theta_P}L_{P,\mathrm{sup}}^{(\mathcal B)},
$$

$$
\nabla_{\theta_V}L_{\mathrm{sup}}^{(\mathcal B)}
=
w_V\nabla_{\theta_V}L_{V,\mathrm{sup}}^{(\mathcal B)},
$$

$$
\nabla_{\theta_E}L_{\mathrm{sup}}^{(\mathcal B)}
=
\nabla_{\theta_E}L_{P,\mathrm{sup}}^{(\mathcal B)}
+w_V\nabla_{\theta_E}L_{V,\mathrm{sup}}^{(\mathcal B)}.
$$

The Policy loss therefore updates the Policy head and shared encoder, and the Value loss updates the Value head and shared encoder. Their encoder gradients add according to the final equation.

### 4.3 Parameter Optimization

Parameter optimization arranges $\mathcal D_{\mathrm{sup}}$ into epochs, and each completed epoch uses every record exactly once. At the start of an epoch, the data loader randomizes the order of the storage chunks and then randomizes the records within each chunk. This two-level shuffle changes both the order and composition of successive minibatches, reducing the correlation that could otherwise arise from the fixed storage order.

Let $B_{\mathrm{opt}}\geq1$ be the minibatch size, and suppose the stored dataset consists of $J$ chunks containing $n_1,\ldots,n_J$ records, where $\sum_{j=1}^{J}n_j=N$. Since each chunk is partitioned independently into minibatches, one complete epoch contains

$$
K_{\mathrm{epoch}}=
\sum_{j=1}^{J}
\left\lceil\frac{n_j}{B_{\mathrm{opt}}}\right\rceil
$$

optimizer steps. Let $E\geq1$ be the epoch count, and let $K_{\max}\in\mathbb N\cup\lbrace\infty\rbrace$ be the optimizer-step limit, with a nonpositive configured limit represented by $K_{\max}=\infty$. The complete optimization sequence contains

$$
K=\min\left(K_{\max},E K_{\mathrm{epoch}}\right)
$$

steps, whose minibatches are denoted by $\mathcal B_1,\ldots,\mathcal B_K$. Let $\theta^{(0)}$ be the initial parameters of a newly initialized Melano network with feature width $C$ and block count $B$, and let $\theta^{(k)}$ denote the parameters produced by optimizer step $k$. For $k<K$, $\theta^{(k)}$ is used to process $\mathcal B_{k+1}$, while $\theta^{(K)}$ is the final trained parameter set.

The learning-rate schedule uses the nominal step horizon

$$
K_{\mathrm{sched}}=
\min\left(
K_{\max},
E\left\lceil\frac{N}{B_{\mathrm{opt}}}\right\rceil
\right).
$$

This horizon treats the $N$ records as one sequence of minibatches, while $K_{\mathrm{epoch}}$ counts the minibatches formed independently inside the storage chunks. The warmup length is

$$
K_W=
\min\left(
K_{\mathrm{sched}},
2000,
\max\left(
100,
\left\lfloor\frac{K_{\mathrm{sched}}}{100}\right\rfloor
\right)
\right).
$$

At optimizer step $k\in\lbrace1,\ldots,K\rbrace$, automatic differentiation computes

$$
g_k=
\nabla_{\theta^{(k-1)}}
L_{\mathrm{sup}}^{(\mathcal B_k)}.
$$

A positive global gradient-norm limit $g_{\max}$ rescales $g_k$ when its Euclidean norm exceeds $g_{\max}$. Denote the resulting gradient by $\overline g_k$. A nonpositive limit gives $\overline g_k=g_k$, while a positive limit also requires the measured norm to be finite.

For peak learning rate $\eta_{\max}$, optimizer step $k\in\lbrace1,\ldots,K\rbrace$ uses

$$
\eta_k=
\begin{cases}
\eta_{\max}\dfrac{k}{K_W},&k\leq K_W,\\[6pt]
\eta_{\max}\sqrt{\dfrac{K_W}{k}},&k>K_W.
\end{cases}
$$

The schedule therefore increases linearly to $\eta_{\max}$ during the warmup and then decays in inverse proportion to the square root of the step index.

AdamW maintains a first-moment estimate $u_k$ and a second-moment estimate $v_k$ of the gradients $\overline g_k$ supplied by the global-norm transformation. Both estimates have the same dimensions as $\theta$ and begin with $u_0=v_0=0$. With $\beta_1=0.9$ and $\beta_2=0.999$, their updates and bias corrections are

$$
u_k=\beta_1u_{k-1}+(1-\beta_1)\overline g_k,
\qquad
v_k=\beta_2v_{k-1}+(1-\beta_2)\overline g_k^2,
$$

$$
\widehat u_k=\frac{u_k}{1-\beta_1^k},
\qquad
\widehat v_k=\frac{v_k}{1-\beta_2^k}.
$$

Let $\lambda\geq0$ be the weight-decay coefficient and let $\epsilon_A=10^{-8}$ prevent division by zero. AdamW updates the network parameters according to

$$
\theta^{(k)}=
(1-\eta_k\lambda)\theta^{(k-1)}
-\eta_k
\frac{\widehat u_k}
{\sqrt{\widehat v_k}+\epsilon_A}.
$$

The square in $\overline g_k^2$, the square root in $\sqrt{\widehat v_k}$ and the quotient involving $\widehat u_k$ are evaluated elementwise. In the update for $\theta^{(k)}$, the factor $(1-\eta_k\lambda)$ applies decoupled weight decay to $\theta^{(k-1)}$, and the remaining term applies the adaptive gradient step determined by $\widehat u_k$ and $\widehat v_k$.

## 5. Search

### 5.1 Root Initialization

For a nonterminal complete state $x_0$, the MCTS procedure initializes a tree whose root corresponds to $x_0$. Each node corresponds to a complete state, and each edge leaving a node corresponding to state $x$ records a legal action $a\in\mathcal A(x)$ and leads to a child node corresponding to $T(x,a)$. A simulation follows selected edges from the root to a leaf, determines an evaluation for the leaf state and propagates that evaluation back along the selected path.

The evaluator then obtains $P_\theta(\cdot\mid s_0)$ and $V_\theta(s_0)$ for $s_0=\phi_M(x_0)$. The scalar $V_\theta(s_0)$ provides the reported root evaluation when no simulation completes, while the number of completed visits and the sum of backed-up evaluations both begin at zero. For every legal action $a\in\mathcal A(x_0)$, root expansion creates an outgoing edge and a child node, and the Policy probability assigned to $a$ defines the prior of that edge:

$$
P(x_0,a)=P_\theta(a\mid s_0).
$$

A nonterminal node is unexpanded while it has no outgoing edges. When a simulation reaches an unexpanded node corresponding to state $x$, the evaluator obtains $P_\theta(\cdot\mid\phi_M(x))$ and $V_\theta(\phi_M(x))$. Node expansion uses the Policy distribution to create one outgoing edge and child node for every action $a\in\mathcal A(x)$, assigning $P_\theta(a\mid\phi_M(x))$ to the edge prior $P(x,a)$. The backup procedure defined in Section 5.4 propagates $V_\theta(\phi_M(x))$ along the selected path.

### 5.2 Tree Statistics

Every node maintains a completed-visit count and the sum of the evaluations propagated to that node. For a node representing $x$, let $N(x)$ denote its completed-visit count, let $W(x)$ denote its accumulated evaluation and define its empirical mean by

$$
Q(x)=
\begin{cases}
\dfrac{W(x)}{N(x)},&N(x)>0,\\[6pt]
0,&N(x)=0.
\end{cases}
$$

The value $Q(x)$ uses the perspective of the side to move in $x$. For an edge that applies action $a$ at $x$, the child node represents $T(x,a)$ and therefore uses the opponent's perspective. Once that child has been visited, the action evaluation in the parent perspective is

$$
Q(x,a)=-Q(T(x,a)).
$$

Let $N(x,a)$ denote the completed-visit count of the child reached through action $a$. This count equals the number of completed simulations that traversed the edge from $x$ to $T(x,a)$. A node can be evaluated before any simulation traverses one of its outgoing edges, so $N(x)$ can exceed $\sum_{a\in\mathcal A(x)}N(x,a)$.

### 5.3 PUCT Selection

Each simulation uses Predictor + Upper Confidence bounds applied to Trees (PUCT) to descend through expanded nodes. The PUCT score combines the empirical evaluation of an action with an exploration term derived from its prior and visit count. Because an unvisited edge has no empirical action evaluation, First Play Urgency (FPU) supplies its initial selection value.

The explored prior mass at node $x$ is

$$
M_P(x)=
\sum_{b\in\mathcal A(x):N(x,b)>0}P(x,b).
$$

For FPU reduction coefficient $r_{\mathrm{FPU}}\geq0$, the action evaluation used during selection is

$$
Q_{\mathrm{sel}}(x,a)=
\begin{cases}
Q(x,a),&N(x,a)>0,\\[4pt]
\mathrm{clip}_{[-1,1]}\left(
Q(x)-r_{\mathrm{FPU}}\sqrt{M_P(x)}
\right),&N(x,a)=0.
\end{cases}
$$

FPU starts from the empirical mean of the parent and subtracts a reduction proportional to the square root of its explored prior mass. An unvisited edge therefore receives a more conservative initial evaluation after the node has explored actions that the network considered probable.

Several paths may be selected before their leaf states are evaluated together. A temporary virtual visit reserves every node on each selected path so that later selections in the same batch account for pending work. Let $N_v(x)$ be the number of active reservations through node $x$, and let $N_v(x,a)$ be the number through the child edge for action $a$. The augmented counts are

$$
\widetilde N(x)=N(x)+N_v(x),
\qquad
\widetilde N(x,a)=N(x,a)+N_v(x,a).
$$

The exploration coefficient follows a logarithmic schedule based on the augmented parent count. Let $c_0$ be its initial coefficient, let $b_0$ be its schedule base and let $f_0$ be its schedule factor. After defining $b=\max(1,b_0)$ and $f=\max(0,f_0)$, Melano computes

$$
c_{\mathrm{puct}}(n)=
\max\left(
0,
c_0+f\log\left(\frac{n+b+1}{b}\right)
\right).
$$

For virtual-loss coefficient $l_v\geq0$, the complete selection score is

$$
S(x,a)=
Q_{\mathrm{sel}}(x,a)
+c_{\mathrm{puct}}\left(\widetilde N(x)\right)P(x,a)
\frac{\sqrt{\widetilde N(x)+1}}
{1+\widetilde N(x,a)}
-l_vN_v(x,a).
$$

At each expanded node, the selector follows the action with the largest $S(x,a)$. Equal scores are resolved first by the larger prior $P(x,a)$ and then by the larger $Q_{\mathrm{sel}}(x,a)$. If all three quantities are equal, the selector follows the action that the rules engine enumerated first when the node was expanded. The selected path ends at a terminal state or at an unexpanded nonterminal node.

### 5.4 Leaf Evaluation and Backup

The rules engine determines whether the selected leaf state $x_d$ is terminal before neural evaluation. A terminal leaf receives the exact outcome $z(x_d)$. An unexpanded nonterminal leaf receives the network evaluation $V_\theta(\phi_M(x_d))$ together with the Policy probabilities required for expansion. Both scalar evaluations use the perspective of the side to move at the leaf:

$$
\rho_d=
\begin{cases}
z(x_d),&x_d\text{ is terminal},\\
V_\theta(\phi_M(x_d)),&x_d\text{ is nonterminal}.
\end{cases}
$$

Denote the selected path by $(x_0,x_1,\ldots,x_d)$, and let $a_k$ be the action that leads from $x_{k-1}$ to $x_k$. Alternating sides to move gives the evaluation at every node on the path:

$$
\rho_k=(-1)^{d-k}\rho_d.
$$

The backup operation removes the virtual reservation from the path and updates each selected node by

$$
N(x_k)\leftarrow N(x_k)+1,
\qquad
W(x_k)\leftarrow W(x_k)+\rho_k.
$$

These updates increase the completed count of every traversed child and therefore increase the corresponding edge count $N(x_{k-1},a_k)$. The empirical means $Q(x_k)$ and parent-perspective action evaluations $Q(x_{k-1},a_k)$ then follow from Section 5.2.

### 5.5 Simulation Budget

The simulation budget uses root uncertainty to choose a target between a required minimum and a configured soft cap. Let $N_{\mathrm{cap}}\geq0$ be the soft simulation cap, let $B_{\mathrm{batch}}\geq1$ be the neural batch capacity and let $N_{\mathrm{floor}}\geq0$ be an optional explicit minimum. The minimum number of completed simulations is

$$
N_{\min}=
\begin{cases}
0,&N_{\mathrm{cap}}=0,\\
\max\left(1,\min\left(N_{\mathrm{cap}},N_{\mathrm{floor}}\right)\right),
&N_{\mathrm{cap}}>0\ \text{and}\ N_{\mathrm{floor}}>0,\\
\max\left(1,\min\left(
N_{\mathrm{cap}},
\max\left(
B_{\mathrm{batch}},
\left\lfloor\dfrac{N_{\mathrm{cap}}}{4}\right\rfloor
\right)
\right)\right),
&N_{\mathrm{cap}}>0\ \text{and}\ N_{\mathrm{floor}}=0.
\end{cases}
$$

When $N_{\mathrm{cap}}>0$, selection-and-evaluation cycles continue until the root has completed at least $N_{\min}$ simulations. Its root-edge counts then define the empirical visit distribution

$$
\widehat P_N(a\mid x_0)=
\frac{N(x_0,a)}
{\displaystyle\sum_{b\in\mathcal A(x_0)}N(x_0,b)}.
$$

For a root with at least two legal actions, sort the actions first by decreasing visit count $N(x_0,a)$. Among actions with equal visit counts, place the action with the larger prior $P(x_0,a)$ first. Denote the first two actions in this order by $a_1$ and $a_2$, and define $N_i=N(x_0,a_i)$. For each $a_i$, define $Q_i=Q(x_0,a_i)$ when $N_i>0$ and $Q_i=0$ when $N_i=0$. The normalized visit entropy $H_N$, visit-count proximity $U_N$ and action-evaluation proximity $U_Q$ are

$$
H_N=-
\frac{\displaystyle\sum_{a\in\mathcal A(x_0)}
\widehat P_N(a\mid x_0)\log\widehat P_N(a\mid x_0)}
{\log|\mathcal A(x_0)|},
$$

$$
U_N=
1-\frac{|N_1-N_2|}{\max(1,N_1+N_2)},
$$

$$
U_Q=
1-\min\left(1,\frac{|Q_1-Q_2|}{0.5}\right).
$$

The entropy summand for $\widehat P_N(a\mid x_0)=0$ equals zero by the limit $\lim_{p\to0^+}p\log p=0$. The three statistics lie in $[0,1]$, and their weighted sum defines the root uncertainty

$$
u=
\mathrm{clip}_{[0,1]}
\left(0.5H_N+0.35U_N+0.15U_Q\right).
$$

The current uncertainty determines the simulation target

$$
N_{\mathrm{target}}=
N_{\min}
+\left\lceil
u(N_{\mathrm{cap}}-N_{\min})
\right\rceil.
$$

The MCTS procedure recalculates $N_{\mathrm{target}}$ after each selection-and-evaluation cycle once the root has reached $N_{\min}$. A new cycle starts for a root only while its completed count is smaller than both the current target and the soft cap. Terminal backups and neural evaluations already selected in that cycle are completed before the next stopping check, so they may carry the final count beyond either threshold. A root with one legal action uses $u=0$ and admits no new cycle after reaching $N_{\min}$. A zero cap sets both $N_{\min}$ and $N_{\mathrm{target}}$ to zero. A cancellation signal or an expired execution deadline stops further leaf selection, while leaves already selected still produce their completed backups.

### 5.6 Evaluation Reuse

Repeated neural evaluation of the same `PackedState` produces the same compact Policy and Value record, so Melano stores each completed network evaluation in a cache indexed by its 67-byte `PackedState`. For a requested complete state, the rules engine checks for a terminal outcome before the cache is consulted. This order is required because `PackedState` omits move counters and repetition history. Exact rule outcomes therefore depend on the complete state, whereas cached network outputs depend only on the encoded state.

One MCTS invocation receives one or more root states and constructs a separate tree for each root. All trees created by that invocation access the same evaluation cache, which allows simulations within one tree and simulations from different trees to reuse completed network records. Let $M_C\geq0$ be the configured memory capacity for records retained across invocations. When $M_C=0$, the cache exists only for the current invocation. When $M_C>0$, the cache persists across invocations and uses TLRU (trajectory-aware least-recently-used) to order its records. A successful lookup moves the accessed record to the most-recent end of this order, and inserting a new record places it at the same end.

TLRU records a directed link when a cached nonterminal child is reached from a cached parent. Let $\mathcal C$ be the set of retained entries and let $E_C\subseteq\mathcal C\times\mathcal C$ be the set of recorded parent-child links. For a retained root entry $r\in\mathcal C$, the trajectory neighborhood of radius two is

$$
\mathcal N_2(r)=
\lbrace y\in\mathcal C:d_C(r,y)\leq2\rbrace,
$$

where $d_C(r,y)$ is the length of the shortest directed path from $r$ to $y$ in $(\mathcal C,E_C)$. Before evaluating a new root, TLRU touches the retained entries in $\mathcal N_2(r)$ in decreasing order of $d_C(r,y)$. Two-ply descendants are touched first, one-ply descendants next and the root last, leaving the root as the most-recent entry. Ordinary lookups and insertions then continue to update the same order.

When the approximate memory use exceeds $M_C$, TLRU removes entries from the least-recent end until the retained records fit within the capacity. TLRU stores compact network records, whereas tree nodes and MCTS statistics belong to one invocation. Every invocation constructs fresh nodes with zero visits, zero accumulated evaluations and zero virtual reservations, initializes their network fields from cached records when available and computes their tree statistics through MCTS.

### 5.7 Batched Evaluation

When one invocation receives several root states, their initial evaluations from Section 5.1 pass through the evaluation-reuse mechanism together. Cached roots are resolved first, unresolved roots with the same `PackedState` share one request and the remaining unique roots are evaluated in one neural batch.

After root initialization, one selection-and-evaluation cycle processes each tree whose root count $N(x_0)$ is smaller than both $N_{\mathrm{target}}$ and $N_{\mathrm{cap}}$. For each such tree, the requested number of distinct nonterminal leaves is

$$
m=
\min\left(
B_{\mathrm{batch}},
N_{\mathrm{target}}-N(x_0),
N_{\mathrm{cap}}-N(x_0)
\right).
$$

The three terms limit the request by the neural batch capacity, the remaining count to the current target and the remaining count to the simulation cap, respectively.

Each selection attempt starts at the root and uses PUCT to descend through expanded nodes until it reaches a terminal node or an unexpanded nonterminal node. A terminal node is evaluated and backed up immediately according to Section 5.4, which completes one simulation without adding a neural-evaluation request. An unexpanded nonterminal node enters the request list when the same tree has not already selected that node during the current cycle, and its virtual visits remain on the selected path until evaluation finishes. A repeated selection of an already reserved node releases the temporary virtual visits and contributes no request. The tree makes at most $\max(5m,m+8)$ selection attempts while collecting up to $m$ distinct nonterminal leaves.

The requests collected from all trees are concatenated and partitioned into neural batches no larger than $B_{\mathrm{batch}}$. Within each batch, the evaluation-reuse mechanism described in Section 5.6 first resolves requests that already have cached records. The evaluator then groups the unresolved requests by `PackedState`. Every request in one group has the same network input, so one neural evaluation supplies the result for the entire group.

For a representative leaf state $x$, let $L=|\mathcal A(x)|$, and write its ordered legal actions as $a_1,\ldots,a_L$. The compact evaluation record contains the three sequences

$$
(a_j)_{j=1}^{L},
\qquad
(i_M(a_j))_{j=1}^{L},
\qquad
(P_\theta(a_j\mid\phi_M(x)))_{j=1}^{L},
$$

together with the scalar $V_\theta(\phi_M(x))$. Entries with the same index $j$ describe the same legal action, which defines the alignment among the action, action-index and Policy sequences. Section 3.3 supplies these Policy probabilities by evaluating the selected legal-action logits.

After neural evaluation, the evaluator stores the completed record through the evaluation-reuse mechanism and assigns it to every request in the matching `PackedState` group. For each requested leaf, tree expansion creates one outgoing edge for every $a_j$ and assigns $P_\theta(a_j\mid\phi_M(x))$ as that edge's prior. The scalar $V_\theta(\phi_M(x))$ is then backed up along the leaf's reserved path. The trees therefore share network records while retaining separate nodes, paths and MCTS statistics.

### 5.8 Root Evaluation and Policy

After the simulation phase, the root evaluation uses the network estimate when no simulation has completed and the empirical root mean otherwise:

$$
V_{\mathrm{root}}(x_0)=
\begin{cases}
V_\theta(s_0),&N(x_0)=0,\\
Q(x_0),&N(x_0)>0.
\end{cases}
$$

The completed root-edge visits and their priors define the root Policy distribution

$$
P_{\mathrm{root}}(a\mid s_0)=
\frac{N(x_0,a)+P(x_0,a)}
{\displaystyle\sum_{b\in\mathcal A(x_0)}
\left(N(x_0,b)+P(x_0,b)\right)}.
$$

The priors sum to one over $\mathcal A(x_0)$, so the added prior terms contribute total weight one to the root distribution. When $N_{\mathrm{cap}}=0$, every root-edge visit count remains zero and the formula reduces to

$$
P_{\mathrm{root}}(a\mid s_0)=P_\theta(a\mid s_0).
$$

The MCTS root Policy therefore coincides with the network Policy when the simulation cap is zero.

When an output requires one probability for every index in $\mathcal I_M$, Melano expands the compact root distribution into

$$
P_{\mathrm{dense}}(i\mid s_0)=
\begin{cases}
P_{\mathrm{root}}(a\mid s_0),
&i=i_M(a)\text{ for }a\in\mathcal A(x_0),\\
0,&i\notin\lbrace i_M(a):a\in\mathcal A(x_0)\rbrace.
\end{cases}
$$

This conversion is performed only when an output requires the fixed action-index set $\mathcal I_M$.

### 5.9 Decision Components

The decision layer can apply two optional transformations to the final move ordering. It begins with a copy of the root Policy distribution:

$$
D_0(a)=P_{\mathrm{root}}(a\mid s_0).
$$

IMF (Instant Mate First) examines the set $\mathcal M(x_0)$ of legal actions that immediately checkmate the opponent. When IMF is enabled and this set contains at least one action, it selects an action with the largest base score

$$
a_M=
\arg\max_{a\in\mathcal M(x_0)}D_0(a).
$$

If several actions attain this maximum, IMF chooses the action that the rules engine enumerates first. Its output is

$$
D_I(a)=
\begin{cases}
1,&a=a_M,\\
D_0(a),&a\in\mathcal A(x_0)\setminus\lbrace a_M\rbrace.
\end{cases}
$$

If IMF is disabled or $\mathcal M(x_0)$ is empty, its output is $D_I(a)=D_0(a)$ for every legal action.

RPP (Repetition Policy Penalty) applies when the root evaluation favors the side to move. Let $\lambda_R\in[0,1]$ be its penalty coefficient, where $\lambda_R=0$ disables the transformation, and let $\mathcal R_3(x_0)$ contain the legal actions that make a threefold-repetition claim available immediately or allow the opponent to make such a claim after one reply. RPP computes

$$
d_R=
\lambda_R
\mathrm{clip}_{[0,1]}
\left(V_{\mathrm{root}}(x_0)\right)
$$

and produces the final decision score

$$
D(a)=
\begin{cases}
\max(0,D_I(a)-d_R),
&a\in\mathcal R_3(x_0),\\
D_I(a),
&a\in\mathcal A(x_0)\setminus\mathcal R_3(x_0).
\end{cases}
$$

IMF and RPP transform a copy of $P_{\mathrm{root}}$, so the network probabilities, edge priors and tree statistics keep their computed values. The decision scores are ordering quantities rather than a probability distribution. Melano orders legal actions by decreasing $D(a)$, then by decreasing $D_0(a)$ and finally by decreasing coordinate move string in Universal Chess Interface (UCI) notation. The first action in this deterministic ordering becomes the selected move.
