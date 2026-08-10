# Gadus

Gadus is a residual convolutional chess network that jointly predicts a move policy and a side-to-move position evaluation.

## 1. Notation

- $\mathcal X$ is the set of complete chess states maintained by the rules engine.
- $x\in\mathcal X$ is a complete chess state containing the board, side to move, castling rights, en passant state, move counters and repetition history.
- $\mathcal A(x)$ is the set of legal actions in state $x$.
- $T(x,a)$ is the complete state reached by applying legal action $a\in\mathcal A(x)$ to state $x$.
- $z(x)\in\lbrace-1,0,1\rbrace$ is the exact outcome of terminal state $x$ from the perspective of its side to move, with $1$, $0$ and $-1$ representing a win, draw and loss.
- $\phi_G$ is the Gadus state encoder that maps a complete chess state to a network input.
- $s=\phi_G(x)$ is the Gadus network input obtained from complete state $x$.
- $\mathcal I_G=\lbrace0,\ldots,4671\rbrace$ is the fixed set of Gadus action indices.
- $i_G(a)\in\mathcal I_G$ is the action index assigned to legal action $a$.
- $\theta$ denotes the trainable network parameters.
- $\ell_\theta(s)\in\mathbb R^{4672}$ is the complete vector of Policy logits produced by the network with parameters $\theta$ and $\ell_\theta(s,i)$ is its scalar component for action index $i\in\mathcal I_G$.
- $\text{P}$, which stands for Policy, is the network output that assigns a probability distribution over the legal actions available in each encoded state.
- $\text{V}$, which stands for Value, is a scalar network output in $[-1,1]$ that estimates the expected game result from the perspective of the side to move.
- $Q$ denotes a state or action evaluation defined by a particular procedure. Each definition specifies its arguments and observation perspective.
- $\mathrm{clip}_{[l,u]}(y)=\min(u,\max(l,y))$ restricts scalar $y$ to the closed interval $[l,u]$.

## 2. State and Action Encoding

### 2.1 State Encoding

The network requires a fixed numerical representation of each complete chess state. Gadus represents a state with 18 binary feature maps called planes. Each plane is an $8\times8$ grid aligned with the chessboard, and each square corresponds to one scalar entry in that grid. An entry equals $1$ when its plane marks the corresponding square and $0$ otherwise. The state encoder $\phi_G$ stacks the 18 planes into one tensor:

$$
\phi_G:\mathcal X\rightarrow\lbrace0,1\rbrace^{18\times8\times8}.
$$

The first tensor dimension identifies a plane, the second identifies a rank and the third identifies a file. Planes 0 through 5 represent White pawn, knight, bishop, rook, queen and king occupancy. Each entry in a piece plane equals $1$ exactly when the corresponding square contains the piece represented by that plane. For example, the `f3` entry in the White-knight plane equals $1$ exactly when a White knight occupies `f3`. Planes 6 through 11 represent the six Black piece types in the same order.

The remaining six planes represent features that apply to the complete state or to one file. Plane 12 contains ones in all 64 entries when White is to move and zeros when Black is to move. Planes 13 through 16 represent White kingside, White queenside, Black kingside and Black queenside castling rights. Each castling plane contains ones in all 64 entries when its right is available and zeros otherwise. When an en passant square exists, plane 17 contains ones in the eight entries belonging to its file and zeros in all other entries. When no en passant square exists, plane 17 contains only zeros.

For storage, Gadus packs the eight entries on one rank of one plane into one byte. Within each plane, the bytes appear in order from rank 1 through rank 8, and the planes appear in numerical order from 0 through 17. Within each byte, the bits represent files `a` through `h`, with file `a` in the most significant bit and file `h` in the least significant bit. One plane therefore requires eight bytes, and all 18 planes require $18\times8=144$ bytes. `PackedState` denotes this 144-byte representation, whose unpacked bits reproduce $\phi_G(x)$ exactly.

The encoded tensor $\phi_G(x)$ records piece placement, side to move, castling rights and the en passant file. The complete state $x$ additionally records move counters and repetition history. Since $\phi_G$ omits move counters and repetition history, complete states that differ only in those fields produce the same network input.

### 2.2 Action Encoding

The legal-action set $\mathcal A(x)$ varies with the complete state $x$. To give actions a state-independent numerical representation, Gadus assigns each legal action $a\in\mathcal A(x)$ an index $i_G(a)$ in the fixed set $\mathcal I_G$.

Every action index has two components. The source-square component $q$ identifies the square from which the move begins. The motion-pattern component $p$ describes the displacement from the source square to the destination square and, for an underpromotion, also identifies the promoted piece.

The $8\times8$ board gives $q$ 64 possible values. The 73 possible values of $p$ consist of 56 patterns for motion along a rank, file or diagonal, 8 patterns for knight motion and 9 patterns for underpromotion. These counts satisfy

$$
73=(8\times7)+8+(3\times3).
$$

The following definitions specify the ordering of the 64 source squares and the 73 motion patterns.

For the source square, let $r\in\lbrace0,\ldots,7\rbrace$ be its rank coordinate and let $f\in\lbrace0,\ldots,7\rbrace$ be its file coordinate. Rank 1 has coordinate $r=0$, and the coordinate increases toward rank 8. File `a` has coordinate $f=0$, and the coordinate increases toward file `h`. The source-square index is

$$
q=8r+f.
$$

To describe motion along a rank, file or diagonal, let $(r,f)$ and $(r',f')$ be the coordinates of the source and destination squares. Their displacement is

$$
(\Delta r,\Delta f)=(r'-r,f'-f).
$$

Gadus orders the eight possible directions as

$$
(-1,-1),\ (-1,0),\ (-1,1),\ (0,-1),\
(0,1),\ (1,-1),\ (1,0),\ (1,1),
$$

Let $(u_d,v_d)$ be the direction at position $d\in\lbrace0,\ldots,7\rbrace$ in this list. If the move travels $m\in\lbrace1,\ldots,7\rbrace$ squares in that direction, then

$$
(\Delta r,\Delta f)=m(u_d,v_d),
$$

and its motion-pattern index is

$$
p=7d+(m-1).
$$

This construction assigns indices 0 through 55 to the $8\times7=56$ rank, file and diagonal patterns.

A knight move uses one of the following eight displacements:

$$
(-2,-1),\ (-2,1),\ (-1,-2),\ (-1,2),\
(1,-2),\ (1,2),\ (2,-1),\ (2,1).
$$

Let $k\in\lbrace0,\ldots,7\rbrace$ be the position of the knight displacement $(\Delta r,\Delta f)$ in this list. Its motion-pattern index is

$$
p=56+k.
$$

This construction assigns indices 56 through 63 to the eight knight patterns.

An underpromotion always moves a pawn to the final rank, so its remaining choices are the destination file and the promoted piece. Let $\Delta f\in\lbrace-1,0,1\rbrace$ be the destination-file offset relative to the source file. Let $u=0$, $u=1$ and $u=2$ represent promotion to a knight, bishop and rook. The motion-pattern index is

$$
p=64+3(\Delta f+1)+u.
$$

This construction assigns indices 64 through 72 to the $3\times3=9$ underpromotion patterns. A queen promotion uses the corresponding one-square rank, file or diagonal pattern because the legal move itself records promotion to a queen.

Castling also uses a rank displacement pattern. The rules library represents castling internally with the rook square as the destination of the king move. Before computing $p$, Gadus replaces that internal destination with the king destination `g1`, `c1`, `g8` or `c8`.

Once $q$ and $p$ have been determined, the action index is

$$
i_G(a)=73q+p.
$$

Combining 64 source-square indices with 73 motion-pattern indices gives

$$
|\mathcal I_G|=64\times73=4672.
$$

For complete state $x$, the available action indices are

$$
\mathcal I_G(x)=
\lbrace i_G(a)\mid a\in\mathcal A(x)\rbrace.
$$

To decode an available index, Gadus generates the legal-action set $\mathcal A(x)$ and selects the action whose encoding equals that index. The selected legal action contains the castling, en passant or promotion information required by the rules engine.

## 3. Network

### 3.1 Residual Trunk

The Gadus network derives its $\text{P}$ and $\text{V}$ from a shared residual trunk composed of a convolutional stem followed by a sequence of residual blocks. Within the trunk, the stem produces the initial feature tensor $h_0$, and each residual block $j$ combines its input $h_j$ with a learned transformation $F_j(h_j)$ to produce the next tensor $h_{j+1}$. The following formulas define these computations.

Let $C$ be the number of feature channels in the trunk and let $B$ be its number of residual blocks. The stem maps the 18 binary input planes to $C$ real-valued $8\times8$ feature maps. The symbol $\mathrm{Conv}^{C_{\mathrm{in}}\rightarrow C_{\mathrm{out}}}_{3\times3}$ denotes a bias-free convolution with $C_{\mathrm{in}}$ input channels, $C_{\mathrm{out}}$ output channels and one-square zero padding. The symbol $\mathrm{BN}$ denotes batch normalization applied independently to each output channel, and $\mathrm{ReLU}(z)=\max(0,z)$ is applied elementwise. The stem computes

$$
h_0=\mathrm{ReLU}\left(
\mathrm{BN}_{\mathrm{stem}}\left(
\mathrm{Conv}^{18\rightarrow C}_{3\times3,\mathrm{stem}}(s)
\right)\right)
\in\mathbb R^{C\times8\times8}.
$$

The first residual block receives the stem output $h_0$ as its input. The learned transformation in residual block $j$ is

$$
F_j(h)=\mathrm{BN}_{j,2}\left(
\mathrm{Conv}^{C\rightarrow C}_{3\times3,j,2}\left(
\mathrm{ReLU}\left(
\mathrm{BN}_{j,1}\left(
\mathrm{Conv}^{C\rightarrow C}_{3\times3,j,1}(h)
\right)\right)\right)\right).
$$

Residual block $j$ evaluates $F_j(h_j)$, adds the transformed tensor to its input $h_j$ and then applies ReLU:

$$
h_{j+1}=\mathrm{ReLU}\left(h_j+F_j(h_j)\right),
\qquad 0\leq j<B.
$$

The stem produces $C$ feature maps on the $8\times8$ board grid, and every residual block preserves both the channel count and the spatial dimensions. Consequently,

$$
h_j\in\mathbb R^{C\times8\times8},
\qquad 0\leq j\leq B.
$$

After $B$ residual blocks, $h_B$ is the output of the residual trunk and serves as the shared input to the two output paths:

$$
s\longrightarrow\text{residual trunk}\longrightarrow h_B\longrightarrow
\begin{cases}
\text{Policy head}\longrightarrow\ell_\theta(s)
\longrightarrow P_\theta(\cdot\mid s),\\
\text{Value head}\longrightarrow V_\theta(s).
\end{cases}
$$

Section 3.2 defines both heads and the legal-move normalization that converts $\ell_\theta(s)$ into $P_\theta(\cdot\mid s)$.

### 3.2 Policy and Value Heads

The Policy head assigns one unnormalized score, called a logit, to every action index in $\mathcal I_G$. It maps $h_B$ through a bias-free $1\times1$ convolution from $C$ channels to 32 channels, followed by batch normalization and ReLU. Flattening the resulting $32\times8\times8$ tensor produces a 2048-dimensional vector, and a linear map converts that vector into 4672 logits.

The Value head produces an estimate of the expected game result from the perspective of the player to move. It maps $h_B$ through a separate bias-free $1\times1$ convolution from $C$ channels to 32 channels, followed by batch normalization, ReLU and flattening. A $2048\rightarrow256$ linear map and ReLU produce an intermediate vector. A final linear map produces one scalar, and the hyperbolic tangent restricts that scalar to $[-1,1]$.

Writing $f_\theta$ for the complete network gives

$$
f_\theta(s)=\left(\ell_\theta(s),V_\theta(s)\right),
\qquad
\ell_\theta(s)\in\mathbb R^{4672},
\quad
V_\theta(s)\in[-1,1].
$$

The complete logit vector assigns a score to every index in $\mathcal I_G$. For $s=\phi_G(x)$, selecting the indices $i_G(a)$ for $a\in\mathcal A(x)$ and normalizing their logits with softmax produces the legal-move Policy:

$$
P_\theta(a\mid s)=
\frac{\exp\ell_\theta(s,i_G(a))}
{\displaystyle\sum_{b\in\mathcal A(x)}\exp\ell_\theta(s,i_G(b))},
\qquad a\in\mathcal A(x).
$$

The denominator ranges over $\mathcal A(x)$, so $P_\theta(\cdot\mid s)$ is a probability distribution over the legal actions in complete state $x$.

### 3.3 Inference Evaluation

The final Policy layer contains one weight row and one bias value for each action index in $\mathcal I_G$. For complete state $x$, the legal-move distribution $P_\theta(\cdot\mid s)$ uses only the rows indexed by $i_G(a)$ for actions $a\in\mathcal A(x)$. Let $h_P(s)\in\mathbb R^{2048}$ be the flattened Policy features, and let $W_{P,i}$ and $b_{P,i}$ be the weight row and bias associated with action index $i$. The logit of legal action $a$ is

$$
\ell_\theta(s,i_G(a))=W_{P,i_G(a)}h_P(s)+b_{P,i_G(a)}.
$$

Evaluating the selected rows produces the same legal-action logits as evaluating the complete 4672-row linear map. For a batch of complete states, the inference path pads every legal-index array to the largest legal-action count in that batch. It evaluates that many selected rows for each state and masks the padded positions before applying softmax. The resulting distribution for each state contains exactly the probabilities of its legal actions.

During inference, each batch-normalization layer uses a fixed running mean, running variance, learned scale and learned bias. These fixed quantities allow a convolution and its following batch-normalization layer to be replaced by one convolution with transformed weights and an added bias. For output channel $o$, let $W_o$ be the original bias-free convolution weights, $\mu_o$ and $\sigma_o^2$ be the stored batch-normalization mean and variance, $\gamma_o$ and $\beta_o$ be its learned scale and bias and let $\epsilon$ be its numerical constant. The equivalent convolution parameters are

$$
W'_o=\frac{\gamma_o}{\sqrt{\sigma_o^2+\epsilon}}W_o,
\qquad
b'_o=\beta_o-\frac{\gamma_o\mu_o}{\sqrt{\sigma_o^2+\epsilon}}.
$$

Substituting $W'_o$ and $b'_o$ preserves the evaluation-mode output of every convolution-batch-normalization pair. Batch-normalization fusion and selected-row Policy evaluation therefore reduce the computation performed during inference while preserving $P_\theta(\cdot\mid s)$ and $V_\theta(s)$.

## 4. Supervised Training

### 4.1 Supervised Data

Let $\mathcal D_{\mathrm{sup}}$ be a supervised dataset containing $N$ records:

$$
\mathcal D_{\mathrm{sup}}=
\lbrace \xi_n\rbrace_{n=1}^{N}.
$$

Each record is associated with a complete pre-move state $x_n$ and a selected legal action $a_n\in\mathcal A(x_n)$. The record is

$$
\xi_n=(s_n,i_n,y_n),
$$

where

$$
s_n=\phi_G(x_n),
\qquad
i_n=i_G(a_n),
\qquad
y_n\in[-1,1].
$$

The encoded state $s_n$ is the network input, and the action index $i_n$ is the Policy target. The scalar $y_n$ is the Value target, expressed as an estimate of the expected game result from the perspective of the side to move in $x_n$. On this scale, $-1$ denotes a loss, $0$ denotes a draw and $1$ denotes a win, while intermediate values express expectations between these outcomes.

### 4.2 Supervised Objective

For network input $s$, the Policy head produces the 4672-dimensional logit vector
$$
\ell_\theta(s)=\left(\ell_\theta(s,i)\right)_{i\in\mathcal I_G}\in\mathbb R^{4672}.
$$
Its component $\ell_\theta(s,i)$ is the logit assigned to action index $i$. Supervised training applies softmax to all 4672 components of this vector, producing the supervised action-index distribution $R_\theta$:
$$
R_\theta(i\mid s)=
\frac{\exp\ell_\theta(s,i)}
{\displaystyle\sum_{j\in\mathcal I_G}\exp\ell_\theta(s,j)},
\qquad i\in\mathcal I_G.
$$

Both $R_\theta(\cdot\mid s)$ and the legal-move distribution $P_\theta(\cdot\mid s)$ are derived from the same logit vector $\ell_\theta(s)$, but they differ in normalization domain. $R_\theta$ normalizes all 4672 components for supervised training, whereas $P_\theta$ selects the components indexed by legal actions in $\mathcal A(x)$ and normalizes those components during inference.

For minibatch $\mathcal B\subseteq\mathcal D_{\mathrm{sup}}$, the supervised Policy loss is the mean negative log-probability assigned to the target action indices:

$$
L_{P,\mathrm{sup}}^{(\mathcal B)}=
-\frac{1}{|\mathcal B|}
\sum_{(s,i,y)\in\mathcal B}\log R_\theta(i\mid s).
$$

The supervised Value loss is the mean squared difference between the predicted and target expected results in the same minibatch:

$$
L_{V,\mathrm{sup}}^{(\mathcal B)}=
\frac{1}{|\mathcal B|}
\sum_{(s,i,y)\in\mathcal B}
\left(V_\theta(s)-y\right)^2.
$$

The nonnegative coefficient $w_V$ sets the contribution of the supervised Value loss to the complete minibatch objective:

$$
L_{\mathrm{sup}}^{(\mathcal B)}=
L_{P,\mathrm{sup}}^{(\mathcal B)}+
w_VL_{V,\mathrm{sup}}^{(\mathcal B)}.
$$

To express how the two losses contribute gradients to the network, partition the network parameters as $\theta=(\theta_T,\theta_P,\theta_V)$, where $\theta_T$ contains the residual-trunk parameters and $\theta_P$ and $\theta_V$ contain the parameters of the two output heads. The gradient of the complete objective satisfies

$$
\nabla_{\theta_P}L_{\mathrm{sup}}^{(\mathcal B)}=
\nabla_{\theta_P}L_{P,\mathrm{sup}}^{(\mathcal B)},
$$

$$
\nabla_{\theta_V}L_{\mathrm{sup}}^{(\mathcal B)}=
w_V\nabla_{\theta_V}L_{V,\mathrm{sup}}^{(\mathcal B)},
$$

$$
\nabla_{\theta_T}L_{\mathrm{sup}}^{(\mathcal B)}=
\nabla_{\theta_T}L_{P,\mathrm{sup}}^{(\mathcal B)}+
w_V\nabla_{\theta_T}L_{V,\mathrm{sup}}^{(\mathcal B)}.
$$

The Policy loss contributes gradients to the Policy head and the residual trunk, whereas the Value loss contributes gradients to the Value head and the residual trunk. Their contributions add in the shared trunk according to the final equation above.

### 4.3 Parameter Optimization

Parameter optimization processes $\mathcal D_{\mathrm{sup}}$ in epochs, each of which uses every record exactly once. At the start of an epoch, the data loader first randomizes the order of the HDF5 chunks and then randomizes the records within each chunk before forming minibatches. This two-level shuffle prevents the fixed storage order from repeatedly placing related positions in consecutive minibatches, a pattern that may produce excessive correlation between successive gradient estimates. The resulting minibatches are indexed by optimizer step as $\mathcal B_1,\mathcal B_2,\ldots,\mathcal B_n$.

Let $\theta^{(0)}$ denote the initial parameters of a newly initialized network whose residual trunk has width $C$ and depth $B$. At optimizer step $k\geq1$, $\theta^{(k-1)}$ denotes the parameters available before the update. Automatic differentiation computes the gradient of the objective for $\mathcal B_k$ with respect to $\theta^{(k-1)}$:

$$
g_k=\nabla_{\theta^{(k-1)}}
L_{\mathrm{sup}}^{(\mathcal B_k)}.
$$

AdamW combines $g_k$ with information accumulated from earlier optimizer steps. Its first-moment estimate $u_k$ is an exponential moving average of the gradients, while its second-moment estimate $v_k$ is an exponential moving average of the squared gradients. Both estimates have the same dimensions as $\theta$ and begin with $u_0=v_0=0$. Because this initialization draws their early magnitudes toward zero, AdamW corrects the resulting bias. With $\beta_1=0.9$ and $\beta_2=0.999$, the moment estimates and their bias-corrected forms are

$$
u_k=\beta_1u_{k-1}+(1-\beta_1)g_k,
\qquad
v_k=\beta_2v_{k-1}+(1-\beta_2)g_k^2,
$$

$$
\widehat u_k=\frac{u_k}{1-\beta_1^k},
\qquad
\widehat v_k=\frac{v_k}{1-\beta_2^k}.
$$

Let $\eta$ be the learning rate, let $\lambda$ be the weight-decay coefficient and let $\epsilon_A=10^{-8}$ prevent division by zero. AdamW updates the parameters according to

$$
\theta^{(k)}=
(1-\eta\lambda)\theta^{(k-1)}-
\eta\frac{\widehat u_k}{\sqrt{\widehat v_k}+\epsilon_A}.
$$

The square, square root and quotient in these equations are evaluated separately for each scalar parameter. In the update equation, the first term applies decoupled weight decay to $\theta^{(k-1)}$, and the second term applies the adaptive step determined by the corrected moment estimates. The resulting parameters $\theta^{(k)}$ become the starting point for the next minibatch. Optimization proceeds until the requested epochs are complete or the number of updates reaches a positive optimizer-step limit. In the absence of a positive limit, the epoch count alone determines completion.

## 5. Search

### 5.1 Root Initialization

For a nonterminal complete state $x_0$, the MCTS procedure initializes a tree whose root corresponds to $x_0$. Each node corresponds to a complete state, and each edge leaving a node corresponding to state $x$ records a legal action $a\in\mathcal A(x)$ and leads to a child node corresponding to $T(x,a)$. A simulation follows selected edges from the root to a leaf, determines an evaluation for the leaf state and propagates that evaluation back along the selected path.

The evaluator then obtains $P_\theta(\cdot\mid s_0)$ and $V_\theta(s_0)$ for $s_0=\phi_G(x_0)$. The scalar $V_\theta(s_0)$ provides the reported root evaluation when no simulation completes, while the number of completed visits and the sum of backed-up evaluations both begin at zero. For every legal action $a\in\mathcal A(x_0)$, root expansion creates an outgoing edge and a child node, and the Policy probability assigned to $a$ defines the prior of that edge:

$$
P(x_0,a)=P_\theta(a\mid s_0).
$$

A nonterminal node is unexpanded while it has no outgoing edges. When a simulation reaches an unexpanded node corresponding to state $x$, the evaluator obtains $P_\theta(\cdot\mid\phi_G(x))$ and $V_\theta(\phi_G(x))$. Node expansion uses the Policy distribution to create one outgoing edge and child node for every action $a\in\mathcal A(x)$, assigning $P_\theta(a\mid\phi_G(x))$ to the edge prior $P(x,a)$. The backup procedure defined in Section 5.4 propagates $V_\theta(\phi_G(x))$ along the selected path.

### 5.2 Tree Statistics

Every node maintains a completed-visit count and the sum of the evaluations propagated to that node. For a node representing $x$, let $N(x)$ denote its completed-visit count, let $W(x)$ denote its accumulated evaluation and define its empirical mean by

$$
Q(x)=
\begin{cases}
\dfrac{W(x)}{N(x)},&N(x)>0,\\[6pt]
0,&N(x)=0.
\end{cases}
$$

The value $Q(x)$ uses the perspective of the side to move in $x$. For an edge that applies $a$ at $x$, the child node represents $T(x,a)$ and therefore uses the opponent's perspective. Once that child has been visited, the action evaluation in the parent perspective is

$$
Q(x,a)=-Q(T(x,a)).
$$

Let $N(x,a)$ denote the completed-visit count of the child reached through action $a$. This count equals the number of completed simulations that traversed the edge from $x$ to $T(x,a)$. A node can first be evaluated before any simulation traverses one of its outgoing edges, so its node count $N(x)$ can exceed $\sum_{a\in\mathcal A(x)}N(x,a)$.

### 5.3 PUCT Selection

Each simulation uses Predictor + Upper Confidence bounds applied to Trees (PUCT) to descend through expanded nodes. The PUCT score combines the empirical evaluation of an action with an exploration term derived from its prior and visit count. Because an unvisited edge has no empirical action evaluation, First Play Urgency (FPU) supplies its initial selection value.

Let the explored prior mass at node $x$ be

$$
M_P(x)=
\sum_{b\in\mathcal A(x):N(x,b)>0}P(x,b).
$$

With FPU reduction coefficient $r_{\mathrm{FPU}}\geq0$, the action value used during selection is

$$
Q_{\mathrm{sel}}(x,a)=
\begin{cases}
Q(x,a),&N(x,a)>0,\\[4pt]
\mathrm{clip}_{[-1,1]}\left(
Q(x)-r_{\mathrm{FPU}}\sqrt{M_P(x)}
\right),&N(x,a)=0.
\end{cases}
$$

FPU uses the empirical mean of the parent node and subtracts a reduction proportional to the square root of its explored prior mass. An unvisited edge therefore receives a more conservative initial evaluation after the node has already explored actions that the network considered probable.

Several paths may be selected before their leaf states are evaluated together. A temporary virtual visit reserves every node on each selected path so that later selections in the same batch account for the pending work. Let $N_v(x)$ be the number of active reservations through node $x$, and let $N_v(x,a)$ be the number through the child edge for action $a$. The augmented counts are

$$
\widetilde N(x)=N(x)+N_v(x),
\qquad
\widetilde N(x,a)=N(x,a)+N_v(x,a).
$$

The exploration coefficient follows a logarithmic schedule based on the augmented parent count. Let $c_0$ be its initial coefficient, let $b_0$ be its schedule base and let $f_0$ be its schedule factor. After defining $b=\max(1,b_0)$ and $f=\max(0,f_0)$, Gadus computes

$$
c_{\mathrm{puct}}(n)=
\max\left(0,c_0+f\log\left(\frac{n+b+1}{b}\right)\right).
$$

With virtual-loss coefficient $l_v\geq0$, the complete PUCT score is

$$
S(x,a)=Q_{\mathrm{sel}}(x,a)
+c_{\mathrm{puct}}\left(\widetilde N(x)\right)P(x,a)
\frac{\sqrt{\widetilde N(x)+1}}{1+\widetilde N(x,a)}
-l_vN_v(x,a).
$$

At each expanded node, the selector follows the action with the largest $S(x,a)$. Equal PUCT scores are resolved first by the larger prior $P(x,a)$ and then by the larger $Q_{\mathrm{sel}}(x,a)$. If all three quantities are equal, the selector follows the action that the rules engine enumerated first when the node was expanded. The resulting path ends at a terminal state or at a nonterminal node that has not yet been expanded.

### 5.4 Leaf Evaluation and Backup

The rules engine determines whether the selected leaf state $x_d$ is terminal before neural evaluation. A terminal leaf receives the exact outcome $z(x_d)$. For an unexpanded nonterminal leaf, the network supplies the scalar evaluation $V_\theta(\phi_G(x_d))$ together with the Policy probabilities required for expansion. Both $z(x_d)$ and $V_\theta(\phi_G(x_d))$ use the perspective of the side to move at the leaf, so the scalar leaf evaluation is

$$
\rho_d=
\begin{cases}
z(x_d),&x_d\text{ is terminal},\\
V_\theta(\phi_G(x_d)),&x_d\text{ is nonterminal}.
\end{cases}
$$

To define the backup operation, denote the selected path by $(x_0,x_1,\ldots,x_d)$ and let $a_k$ be the action that leads from $x_{k-1}$ to $x_k$. For each node $x_k$ on this path, the evaluation in its side-to-move perspective is

$$
\rho_k=(-1)^{d-k}\rho_d.
$$

The backup operation removes the virtual reservation from the path and updates every selected node by

$$
N(x_k)\leftarrow N(x_k)+1,
\qquad
W(x_k)\leftarrow W(x_k)+\rho_k.
$$

These updates increase the completed count of each traversed child and thereby increase the corresponding edge count $N(x_{k-1},a_k)$. The empirical means $Q(x_k)$ and the parent-perspective action evaluations $Q(x_{k-1},a_k)$ then follow from the definitions in Section 5.2.

### 5.5 Simulation Budget

The simulation budget uses root uncertainty to choose a target between a required minimum and a fixed cap. Let $N_{\mathrm{cap}}\geq0$ be the simulation cap, let $B_{\mathrm{batch}}\geq1$ be the neural batch capacity and let $N_{\mathrm{floor}}\geq0$ be an optional explicit minimum. The minimum number of completed simulations is

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

When $N_{\mathrm{cap}}>0$, the root first completes $N_{\min}$ simulations. Its root-edge counts then define the empirical visit distribution

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
U_N=1-
\frac{|N_1-N_2|}{\max(1,N_1+N_2)},
$$

$$
U_Q=1-
\min\left(1,\frac{|Q_1-Q_2|}{0.5}\right).
$$

The entropy summand for $\widehat P_N(a\mid x_0)=0$ equals zero by the limit $\lim_{p\to0^+}p\log p=0$. The three statistics lie in $[0,1]$, and their weighted sum defines the root uncertainty

$$
u=\mathrm{clip}_{[0,1]}\left(0.5H_N+0.35U_N+0.15U_Q\right).
$$

The current uncertainty determines the simulation target

$$
N_{\mathrm{target}}=
N_{\min}+\left\lceil u(N_{\mathrm{cap}}-N_{\min})\right\rceil.
$$

The MCTS procedure recalculates $N_{\mathrm{target}}$ after each selection-and-evaluation cycle once the root has reached $N_{\min}$. Simulations for that root end when its completed count reaches the current target or the cap. A root with one legal action uses $u=0$ and therefore stops at $N_{\min}$. A zero cap sets both $N_{\min}$ and $N_{\mathrm{target}}$ to zero. If the caller signals cancellation or the execution deadline expires, the procedure finishes the leaves already selected and returns the statistics produced by completed backups.

### 5.6 Evaluation Reuse

Repeated neural evaluation of the same `PackedState` produces the same compact Policy and Value record, so Gadus stores each completed network evaluation in a cache indexed by its `PackedState`. For a requested complete state, the rules engine checks for a terminal outcome before consulting this cache. This order is required because `PackedState` omits move counters and repetition history. Exact rule outcomes therefore depend on the complete state, whereas cached network outputs depend only on the encoded state.

One MCTS invocation receives one or more root states and constructs a separate search tree for each root. All trees created by that invocation access the same evaluation cache, which allows simulations within one tree and simulations from different trees to reuse completed network records. Let $M_C\geq0$ be the configured memory capacity for records retained across invocations. When $M_C=0$, the cache exists only for the current invocation and is discarded with its search trees. When $M_C>0$, the same cache persists across invocations and uses TLRU (trajectory-aware least-recently-used) to order its records. A successful lookup moves the accessed record to the most-recent end of this order, and inserting a new record places it at the same end.

TLRU records a directed link when a cached nonterminal child is reached from a cached parent. Let $\mathcal C$ be the set of retained entries and let $E_C\subseteq\mathcal C\times\mathcal C$ be the set of recorded parent-child links. For a retained root entry $r\in\mathcal C$, the trajectory neighborhood of radius two is

$$
\mathcal N_2(r)=
\lbrace y\in\mathcal C:d_C(r,y)\leq2\rbrace,
$$

where $d_C(r,y)$ is the length of the shortest directed path from $r$ to $y$ in $(\mathcal C,E_C)$. Before evaluating a new root, TLRU touches the retained entries in $\mathcal N_2(r)$ in decreasing order of $d_C(r,y)$. Two-ply descendants are touched first, one-ply descendants next and the root last, leaving the root as the most-recent entry. Ordinary lookups and insertions then continue to update the same order.

When the approximate memory use exceeds the configured capacity, TLRU removes entries from the least-recent end until the retained records fit within the limit. TLRU stores compact network records, whereas tree nodes and search statistics belong to one invocation. Every invocation therefore constructs fresh nodes with zero visits, zero accumulated evaluations and zero virtual reservations, initializes their network fields from cached records when available and computes their tree statistics through MCTS.

### 5.7 Batched Evaluation

During one selection-and-evaluation cycle, MCTS processes each search tree whose root count $N(x_0)$ is smaller than both $N_{\mathrm{target}}$ and $N_{\mathrm{cap}}$. For each such tree, the requested number of distinct nonterminal leaves is

$$
m=\min\left(
B_{\mathrm{batch}},
N_{\mathrm{target}}-N(x_0),
N_{\mathrm{cap}}-N(x_0)
\right).
$$

The three terms limit the request by the neural batch capacity, the remaining count to the current target and the remaining count to the simulation cap, respectively.

Each selection attempt starts at the root and uses PUCT to descend through expanded nodes until it reaches either a terminal node or an unexpanded nonterminal node. A terminal node is evaluated and backed up immediately according to Section 5.4, which completes one simulation without adding a neural-evaluation request. An unexpanded nonterminal node enters the request list when the same tree has not already selected that node during the current cycle, and its virtual visits remain on the selected path until evaluation finishes. A repeated selection of an already reserved node releases the temporary virtual visits and contributes no request. The tree makes at most $\max(5m,m+8)$ selection attempts while collecting up to $m$ distinct nonterminal leaves.

The requests collected from all search trees are concatenated and partitioned into neural batches no larger than the neural batch capacity. Within each batch, the evaluation-reuse mechanism described in Section 5.6 first resolves requests that already have cached records. The evaluator then groups the unresolved requests by `PackedState`. Every request in one group has the same network input, so one neural evaluation supplies the result for the entire group.

For a representative leaf state $x$, let $L=|\mathcal A(x)|$, and write its ordered legal actions as $a_1,\ldots,a_L$. The compact evaluation record contains the three sequences

$$
(a_j)_{j=1}^{L},\qquad
(i_G(a_j))_{j=1}^{L},\qquad
(P_\theta(a_j\mid\phi_G(x)))_{j=1}^{L},
$$

together with the scalar $V_\theta(\phi_G(x))$. Entries with the same index $j$ describe the same legal action, which defines the alignment among the action, action-index and Policy sequences.

After neural evaluation, the evaluator stores the completed record through the evaluation-reuse mechanism and assigns it to every request in the matching `PackedState` group. For each requested leaf, tree expansion creates one outgoing edge for every $a_j$ and assigns $P_\theta(a_j\mid\phi_G(x))$ as that edge's prior. The scalar $V_\theta(\phi_G(x))$ is then backed up along the leaf's reserved path. The trees therefore share network evaluation records while retaining separate nodes, paths and search statistics.

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

The MCTS root Policy therefore coincides with the direct network Policy when the simulation cap is zero.

When an output requires one probability for every index in $\mathcal I_G$, Gadus expands the compact root distribution into

$$
P_{\mathrm{dense}}(i\mid s_0)=
\begin{cases}
P_{\mathrm{root}}(a\mid s_0),
&i=i_G(a)\text{ for }a\in\mathcal A(x_0),\\
0,&i\notin\lbrace i_G(a):a\in\mathcal A(x_0)\rbrace.
\end{cases}
$$

This conversion is performed only for a final result or a progress report that requires the fixed action-index space $\mathcal I_G$.

### 5.9 Decision Components

The decision layer can apply two optional transformations to the final move ordering. It begins with a copy of the root Policy distribution:

$$
D_0(a)=P_{\mathrm{root}}(a\mid s_0).
$$

When enabled, IMF (Instant Mate First) examines the set $\mathcal M(x_0)$ of legal actions that immediately checkmate the opponent. When this set contains at least one action, IMF selects an action with the largest base score

$$
a_M=\arg\max_{a\in\mathcal M(x_0)}D_0(a)
$$

If several actions attain this maximum, IMF chooses the action that the rules engine enumerates first. IMF then defines

$$
D_I(a)=
\begin{cases}
1,&a=a_M,\\
D_0(a),&a\in\mathcal A(x_0)\setminus\lbrace a_M\rbrace.
\end{cases}
$$

If IMF is disabled or $\mathcal M(x_0)$ is empty, its output is $D_I(a)=D_0(a)$ for every legal action.

RPP (Repetition Policy Penalty) applies when the root evaluation favors the side to move. Let $\lambda_R\in[0,1]$ be its penalty coefficient, where $\lambda_R=0$ disables the transformation, and let $\mathcal R_3(x_0)$ contain the legal actions that make a threefold-repetition claim available immediately or allow the opponent to make such a claim after one reply. RPP computes the deduction

$$
d_R=\lambda_R\mathrm{clip}_{[0,1]}\left(V_{\mathrm{root}}(x_0)\right)
$$

and produces the final decision score

$$
D(a)=
\begin{cases}
\max(0,D_I(a)-d_R),&a\in\mathcal R_3(x_0),\\
D_I(a),&a\in\mathcal A(x_0)\setminus\mathcal R_3(x_0).
\end{cases}
$$

IMF and RPP transform a copy of $P_{\mathrm{root}}$, so the network probabilities, edge priors and tree statistics keep their computed values. The decision scores are ordering quantities rather than a probability distribution. Gadus orders legal actions by decreasing $D(a)$, then by decreasing $D_0(a)$ and finally by decreasing coordinate move string in Universal Chess Interface (UCI) notation. The first action in this deterministic ordering becomes the selected move.

## 6. FCPI

### 6.1 Iteration Framework

Folded Counterfactual Policy Iteration (FCPI) is the self-learning procedure used by Gadus. One iteration starts from a current model, uses that model to generate fixed training targets, optimizes a candidate from the same initial parameters and compares the candidate with the current model in Arena.

Let $C_r$ be the current model at the beginning of iteration $r$, and denote its parameters by $\theta_{old}$. FCPI holds $\theta_{old}$ fixed while generating targets, and the resulting source model outputs are denoted by $P_{old}(\cdot\mid s)$ and $V_{old}(s)$. Candidate optimization begins from the same parameters:

$$
\theta^{(0)}=\theta_{old}.
$$

At optimizer step $k$, the symbols $P_{new}$ and $V_{new}$ denote the outputs parameterized by $\theta^{(k-1)}$. Together with the fixed targets, these outputs determine the minibatch loss whose parameter update produces $\theta^{(k)}$. After $K$ updates, $\theta_{new}=\theta^{(K)}$ parameterizes the candidate.

Self-play controlled by the frozen source model produces the positions from which FCPI constructs two forms of supervision. For positions in completed trajectories, the terminal result determines the realized game returns $G_t$ and action coefficients $A_{\mathrm{MC}}(s_t,a_t)$. At recorded positions from completed and truncated trajectories, finite counterfactual trees evaluate played and alternative moves to obtain $Q_{\mathrm{CF}}(s,a)$, $\pi^+(\cdot\mid s)$ and $\overline V_{\mathrm{CF}}(s)$.

Several target records may share the same encoded input, although one deterministic network produces only one Policy and one Value for that input. Section 6.6 therefore aggregates records by encoded input before candidate optimization. Section 6.7 defines the resulting minibatch objective, and Section 6.8 defines the updates from $\theta^{(0)}$ to $\theta_{new}$. Arena then compares the candidate with $C_r$ under paired starting positions, and the promotion rule in Section 6.9 determines $C_{r+1}$.

### 6.2 Self-Play Trajectories

Self-play produces the trajectories from which both forms of FCPI supervision are derived. At every ply, the frozen source model controls the side to move. Let $\mathcal O_r$ be a supplied pool of reachable nonterminal starting states. FCPI assigns distinct pool states to the games in iteration $r$; these states may occur at any ply, including the standard initial position. If no pool is supplied, every game starts from the standard initial position. A seed derived from $r$ determines the pool order.

Move selection begins with the source model's legal-move Policy $P_{old}(\cdot\mid s)$. Let $T_b\geq0$ be the behavior temperature, and define its numerically bounded value by

$$
\widetilde T_b=\max(T_b,10^{-4}).
$$

The clipped behavior weight of legal move $a$ is

$$
\widetilde P_b(a\mid s)=
\mathrm{clip}_{[10^{-12},1]}(P_{old}(a\mid s)).
$$

Raising these weights to reciprocal temperature and renormalizing gives the behavior distribution

$$
\mu_{old}(a\mid s)=
\frac{\widetilde P_b(a\mid s)^{1/\widetilde T_b}}
{\sum_{b\in\mathcal A(x)}\widetilde P_b(b\mid s)^{1/\widetilde T_b}}.
$$

The resulting behavior distribution is more concentrated than $P_{old}$ when $T_b<1$ and flatter when $T_b>1$. At $T_b=1$, it equals $P_{old}$ up to the probability clipping used by the implementation.

FCPI realizes this distribution through cumulative probability deficits rather than independent random draws. For every Gadus `PackedState`, it counts the moves selected during the entire iteration. Suppose encoded position $s$ is encountered for the $n$-th time, and let $N_{n-1}(s,a)$ be the number of preceding encounters that selected $a$. The behavior distribution assigns desired cumulative count $n\mu_{old}(a\mid s)$ after the current encounter. The selected move has the largest deficit between desired and observed counts:

$$
a_n=\arg\max_{a\in\mathcal A(x)}
\left[n\mu_{old}(a\mid s)-N_{n-1}(s,a)\right].
$$

The order of the legal-action list resolves equal deficits. Before applying the selected action $a_t$ in complete state $x_t$, FCPI creates one trajectory record. Let $a_{t,1},\ldots,a_{t,L_t}$ be the ordered legal actions in $x_t$. The record contains the FEN serialization of $x_t$, the encoded input $s_t$, the source evaluation $V_{old}(s_t)$, the two aligned sequences

$$
\left(i_G(a_{t,j})\right)_{j=1}^{L_t}
\quad\text{and}\quad
\left(P_{old}(a_{t,j}\mid s_t)\right)_{j=1}^{L_t},
$$

and the selected action $a_t$. Entries with the same index $j$ refer to the same legal action.

Let $T_{\max}$ be the configured maximum number of pre-move records in one trajectory. After creating the record for $x_t$, the rules engine applies $a_t$ to obtain $x_{t+1}=T(x_t,a_t)$. A terminal successor completes the trajectory. If the successor is nonterminal and the number of records has reached or exceeded $T_{\max}$, the trajectory is truncated. Otherwise, $x_{t+1}$ becomes the next pre-move state.

### 6.3 Monte Carlo Targets

Monte Carlo (MC) supervision assigns the realized terminal result to every decision that preceded it. Consider a completed trajectory with pre-move states $x_0,\ldots,x_{T_{\mathrm{traj}}-1}$. Its final move reaches terminal state $x_{T_{\mathrm{traj}}}$. Starting from $z(x_{T_{\mathrm{traj}}})$, the recurrence reverses perspective at each ply:

$$
G_{T_{\mathrm{traj}}-1}=-z(x_{T_{\mathrm{traj}}}),
\qquad
G_t=-G_{t+1}\quad(0\leq t<T_{\mathrm{traj}}-1).
$$

The resulting $G_t\in\lbrace-1,0,1\rbrace$ is the final result from the perspective of the player to move in $x_t$. Every recorded state in a completed trajectory receives MC Value weight

$$
w_{\mathrm{MC}}(s_t)=1.
$$

A truncated trajectory ends at a nonterminal state and therefore supplies no realized terminal return. Each of its recorded states receives MC Value weight $0$, and every state-action pair from that trajectory receives MC Policy sample weight $0$.

For a completed trajectory, the discrepancy between $G_t$ and the source Value determines the coefficient assigned to selected move $a_t$:

$$
A_{\mathrm{MC}}(s_t,a_t)=
\mathrm{clip}_{[-1,1]}
\left(\frac{G_t-V_{old}(s_t)}{2}\right).
$$

Since both $G_t$ and $V_{old}(s_t)$ belong to $[-1,1]$, division by $2$ maps their difference to $[-1,1]$. A positive $A_{\mathrm{MC}}$ means that the realized result exceeded the source Value, and a negative value means that it fell below the source Value. The MC Policy record assigns this coefficient and sample weight $1$ to $a_t$. The other legal moves receive sample weight $0$.

### 6.4 Counterfactual Trees

Each recorded self-play decision identifies one legal move as the action that was actually played. A counterfactual tree rooted at such a position evaluates the played move together with alternative legal moves under a finite edge budget. These local evaluations produce counterfactual Policy and Value targets for both completed and truncated trajectories.

Before constructing the trees, FCPI deduplicates the records within each trajectory by Gadus `PackedState` and retains the first occurrence of each encoded position. Every retained record becomes one tree root. Since equal packed states produce equal network inputs, this rule prevents one trajectory from assigning repeated tree weight to the same input. Completed and truncated trajectories use the same root-selection rule.

The recorded FEN reconstructs the board, side to move, castling rights, en passant state and move counters at each selected root. Repetition history begins at this root because FEN contains no earlier positions. Tree transitions append later positions along each branch, so the rules engine detects a threefold repetition whose three occurrences all lie between the root and the current node. A repetition claim that also depends on an occurrence before the root lies outside this reconstructed history.

FCPI evaluates every legal edge from the root before spending any deeper-edge budget. Let $B_{\mathrm{CF}}\geq0$ be the additional edge budget for deeper nodes. A zero budget produces a complete one-ply tree, and a positive budget distributes deeper evaluations through the frontier procedure below.

Suppose $B_{\mathrm{rem}}>0$ deeper edge evaluations remain when non-root node $x$ is selected for expansion. The number of its legal moves to evaluate is

$$
w(x)=\min\left(
|\mathcal A(x)|,
B_{\mathrm{rem}},
\max\left(2,\left\lceil\sqrt{B_{\mathrm{rem}}}\right\rceil\right)
\right).
$$

The expansion set first receives the legal move with the largest source Policy probability. Removing that move from the sampling pool leaves the actions eligible for the remaining $w(x)-1$ places. For every action $a$ in this remaining pool, define its Gumbel sampling score by

$$
\kappa(a)=\log\left(
\mathrm{clip}_{[10^{-12},1]}(P_{old}(a\mid s))
\right)+g_a,
\qquad
g_a=-\log(-\log \xi_a),
\quad \xi_a\sim U(0,1).
$$

Here each $\xi_a$ is an independent sample from the uniform distribution on $(0,1)$, and $g_a$ is its Gumbel transform. The moves with the largest $\kappa(a)$ values fill the remaining slots. A separate seeded random stream supplies these samples, which makes the tree selection reproducible for a fixed seed.

Each evaluated edge from state $x$ through action $a$ creates the child state $x'=T(x,a)$. Let $\mathcal X_T$ contain the terminal child states reached by the tree, and let $\mathcal X_O$ contain its nonterminal child states. Before backward recursion, the initial evaluation assigned to a child state $y\in\mathcal X_T\cup\mathcal X_O$ is

$$
v_{\mathrm{leaf}}(y)=
\begin{cases}
z(y),&y\in\mathcal X_T,\\
V_{old}(\phi_G(y)),&y\in\mathcal X_O.
\end{cases}
$$

The frontier priority uses the probability that the source Policy would follow the branch. The root has reach probability $1$. For the child state $x'$, this probability is propagated by

$$
p_{\mathrm{reach}}(x')=
p_{\mathrm{reach}}(x)P_{old}(a\mid\phi_G(x)).
$$

Let $d(x')$ be the depth of $x'$ in plies from the root. Negating $v_{\mathrm{leaf}}(x')$ expresses the child evaluation from the parent perspective. The absolute difference between this value and $V_{old}(\phi_G(x))$ measures local disagreement with the source Value. A nonterminal child enters the frontier with priority

$$
priority(x')=p_{\mathrm{reach}}(x')
\left(
\left|-v_{\mathrm{leaf}}(x')-V_{old}(\phi_G(x))\right|
+\frac{1}{\sqrt{2+d(x')}}
\right).
$$

The reach factor favors branches that the source Policy considers plausible, the disagreement term favors branches that challenge the source Value and the depth term favors shorter branches when the other factors are similar. FCPI repeatedly expands the highest-priority frontier node until it spends $B_{\mathrm{CF}}$ deeper edge evaluations or exhausts the frontier.

### 6.5 Counterfactual Targets

After tree construction, a backward recursion converts leaf evaluations into action values and state targets. Processing nodes from greatest depth toward the root ensures that every evaluated child has a backed evaluation before its parent is processed. A terminal child contributes its exact rules outcome, while a nonterminal frontier node $x'$ with no evaluated children begins with

$$
\overline V_{\mathrm{CF}}(\phi_G(x'))=V_{old}(\phi_G(x')).
$$

For an expanded node $x$, let $\mathcal E(x)\subseteq\mathcal A(x)$ contain the moves evaluated by the tree. Set $s=\phi_G(x)$ and $x_a'=T(x,a)$. Once every child reached through an action in $\mathcal E(x)$ has either an exact terminal outcome or a backed-up nonterminal evaluation, define the counterfactual action value by

$$
Q_{\mathrm{CF}}(s,a)=
\begin{cases}
-z(x_a'),&a\in\mathcal E(x),\ x_a'\in\mathcal X_T,\\
-\overline V_{\mathrm{CF}}(\phi_G(x_a')),
&a\in\mathcal E(x),\ x_a'\in\mathcal X_O,\\
V_{old}(s),&a\in\mathcal A(x)\setminus\mathcal E(x).
\end{cases}
$$

An evaluated move receives the negated child result because the side to move changes after the action. An unevaluated move receives $V_{old}(s)$. This baseline confines the difference between the tree target and the source Value to the explicitly evaluated alternatives.

The local Policy update uses the source-Policy expectation of the counterfactual action values as its centering term:

$$
c_{\mathrm{CF}}(s)=\sum_{a\in\mathcal A(x)}
P_{old}(a\mid s)Q_{\mathrm{CF}}(s,a).
$$

The source probability of each legal move is also clipped to a positive lower bound:

$$
\widetilde p(a\mid s)=
\mathrm{clip}_{[10^{-12},1]}(P_{old}(a\mid s)).
$$

At unit temperature, the Policy target is obtained by multiplying each source weight by the exponentiated centered action value and normalizing the products:

$$
\pi^+(a\mid s)=
\frac{\widetilde p(a\mid s)
\exp\left(Q_{\mathrm{CF}}(s,a)-c_{\mathrm{CF}}(s)\right)}
{\displaystyle\sum_{b\in\mathcal A(x)}\widetilde p(b\mid s)
\exp\left(Q_{\mathrm{CF}}(s,b)-c_{\mathrm{CF}}(s)\right)}.
$$

Subtracting $c_{\mathrm{CF}}(s)$ improves numerical centering and leaves the normalized distribution unchanged because the common factor $\exp[-c_{\mathrm{CF}}(s)]$ cancels. Normalizing the clipped source weights alone gives the reference Policy

$$
p_\varepsilon(a\mid s)=
\frac{\widetilde p(a\mid s)}
{\displaystyle\sum_{b\in\mathcal A(x)}\widetilde p(b\mid s)}.
$$

For distributions $p$ and $q$ on the same finite action set, define

$$
D_{\mathrm{KL}}(p\,\|\,q)=
\sum_a p(a)\log\frac{p(a)}{q(a)}.
$$

The relation between the reference Policy and the target Policy can be stated as a regularized optimization problem. Among all probability distributions over $\mathcal A(x)$, $\pi^+$ uniquely maximizes

$$
\pi^+=\arg\max_\pi
\left[
\sum_a\pi(a\mid s)Q_{\mathrm{CF}}(s,a)
-D_{\mathrm{KL}}\left(
\pi(\cdot\mid s)\,\|\,p_\varepsilon(\cdot\mid s)
\right)
\right].
$$

The first term rewards probability assigned to larger counterfactual action values. The KL term has coefficient $1$ and penalizes departure from the reference Policy.

The same improved Policy combines the action values into the counterfactual Value target:

$$
\overline V_{\mathrm{CF}}(s)=
\mathrm{clip}_{[-1,1]}
\left(
\sum_{a\in\mathcal A(x)}
\pi^+(a\mid s)Q_{\mathrm{CF}}(s,a)
\right).
$$

Since every $Q_{\mathrm{CF}}(s,a)$ belongs to $[-1,1]$ and $\pi^+(\cdot\mid s)$ is a probability distribution, their weighted mean also belongs to $[-1,1]$. The clipping operation therefore leaves the exact mathematical value unchanged.

The difference between this target and the source Value is the counterfactual correction

$$
\delta_{\mathrm{CF}}(s)=
\overline V_{\mathrm{CF}}(s)-V_{old}(s).
$$

Every unevaluated move contributes zero after $V_{old}(s)$ is subtracted from its action value. The correction can consequently be expressed using the evaluated moves alone:

$$
\delta_{\mathrm{CF}}(s)=
\sum_{a\in\mathcal E(x)}
\pi^+(a\mid s)
\left(Q_{\mathrm{CF}}(s,a)-V_{old}(s)\right).
$$

Equivalently, the target adds this correction to the source Value:

$$
\overline V_{\mathrm{CF}}(s)=V_{old}(s)+\delta_{\mathrm{CF}}(s).
$$

One tree can produce targets at several expanded nodes. Let $\mathcal T_{\mathrm{exp}}$ contain these nodes, including the root, and let $n(x)=|\mathcal E(x)|$ be the number of evaluated edges originating at $x$. The total number of evaluated edges in the tree is

$$
N_{\mathcal E}=\sum_{u\in\mathcal T_{\mathrm{exp}}}n(u).
$$

The Policy and Value targets at node $x$ receive equal within-tree weights

$$
w_P(x)=w_T(x)=\frac{n(x)}{N_{\mathcal E}}.
$$

Thus a node receives weight proportional to the number of tree evaluations performed there, and each tree contributes one unit of total Policy weight and one unit of total Value weight:

$$
\sum_{x\in\mathcal T_{\mathrm{exp}}}w_P(x)
=\sum_{x\in\mathcal T_{\mathrm{exp}}}w_T(x)=1.
$$

### 6.6 Target Aggregation

Each expanded decision node produces one training record containing its counterfactual targets and within-tree weights. A root record associated with a completed trajectory position also contains its MC targets and MC weight. FCPI groups records with identical Gadus `PackedState` encodings across the iteration and verifies that the records in each group share one legal-move list.

Let $\mathcal S_{\mathrm{agg}}$ be the encoded positions that remain after grouping. For each $s\in\mathcal S_{\mathrm{agg}}$, let $\mathcal R(s)$ be its record group and let $\mathcal A_s$ be its legal-move set. Each record $i\in\mathcal R(s)$ associates the counterfactual Policy target $\pi_i^+(\cdot\mid s)$ with weight $w_{P,i}$, the MC Value target $G_i$ with weight $w_{\mathrm{MC},i}$ and the counterfactual Value target $\overline V_{\mathrm{CF},i}(s)$ with weight $w_{T,i}$.

The total counterfactual policy weight is

$$
W_P(s)=\sum_{i\in\mathcal R(s)}w_{P,i}.
$$

For $W_P(s)>0$, the aggregated counterfactual policy target is

$$
\Pi^+(a\mid s)=
\frac{\sum_{i\in\mathcal R(s)}w_{P,i}\pi_i^+(a\mid s)}{W_P(s)}.
$$

$\Pi^+(\cdot\mid s)$ is the weighted mean of the counterfactual Policy targets for $s$. A final normalization corrects floating-point error in its probability sum.

The total MC value weight is

$$
W_{\mathrm{MC}}(s)=\sum_{i\in\mathcal R(s)}w_{\mathrm{MC},i}.
$$

For $W_{\mathrm{MC}}(s)>0$, the aggregated MC value target is

$$
\overline G(s)=
\frac{\sum_{i\in\mathcal R(s)}w_{\mathrm{MC},i}G_i}{W_{\mathrm{MC}}(s)}.
$$

$\overline G(s)$ is the weighted mean of the terminal-return targets for $s$.

The total counterfactual value weight is

$$
W_T(s)=\sum_{i\in\mathcal R(s)}w_{T,i}.
$$

For $W_T(s)>0$, the aggregated counterfactual value target is

$$
\overline V_T(s)=
\frac{\sum_{i\in\mathcal R(s)}w_{T,i}\overline V_{\mathrm{CF},i}(s)}{W_T(s)}.
$$

$\overline V_T(s)$ is the weighted mean of the counterfactual Value targets for $s$.

The support sets for the three aggregated targets are

$$
\mathcal S_P=\lbrace s\in\mathcal S_{\mathrm{agg}}:W_P(s)>0\rbrace,
\qquad
\mathcal S_{\mathrm{MC}}=\lbrace s\in\mathcal S_{\mathrm{agg}}:W_{\mathrm{MC}}(s)>0\rbrace,
$$

$$
\mathcal S_T=\lbrace s\in\mathcal S_{\mathrm{agg}}:W_T(s)>0\rbrace.
$$

All three aggregate weights are nonnegative. The support sets identify the encoded positions on which each aggregated target contributes to its corresponding loss.

The counterfactual Policy, MC Value and counterfactual Value targets are aggregated by encoded state. MC Policy supervision requires state-action aggregation because each trajectory record assigns its signed coefficient to one selected action. Let $a_i$ and $A_{\mathrm{MC},i}$ be the selected action and its coefficient in record $i$. For each pair $(s,a)$, define the aggregated signed coefficient and sample weight by

$$
S_A(s,a)=\sum_{i\in\mathcal R(s)}
w_{\mathrm{MC},i}\mathbf 1[a_i=a]A_{\mathrm{MC},i},
$$

$$
W_A(s,a)=\sum_{i\in\mathcal R(s)}
w_{\mathrm{MC},i}\mathbf 1[a_i=a].
$$

$S_A(s,a)$ accumulates the signed MC coefficients assigned to move $a$ at $s$, and $W_A(s,a)$ records their total sample weight.

### 6.7 Training Objective

At optimizer step $k$, let $\ell_{new}(s,i_G(a))$ be the logit of legal move $a$ produced by the current parameters $\theta^{(k-1)}$. Normalizing these logits over $\mathcal A_s$ gives the current legal-move Policy:

$$
P_{new}(a\mid s)=
\frac{\exp\ell_{new}(s,i_G(a))}
{\sum_{b\in\mathcal A_s}\exp\ell_{new}(s,i_G(b))}.
$$

Applying the self-play behavior temperature to the same logits gives

$$
\mu_{new}(a\mid s)=
\frac{\exp(\ell_{new}(s,i_G(a))/\widetilde T_b)}
{\sum_{b\in\mathcal A_s}\exp(\ell_{new}(s,i_G(b))/\widetilde T_b)}.
$$

For distributions $p$ and $q$ on the same finite action set, define their cross-entropy as

$$
L_{\mathrm{CE}}(p,q)=-\sum_a q(a)\log p(a).
$$

Let $\mathcal B\subseteq\mathcal S_{\mathrm{agg}}$ be the minibatch used at this optimizer step. The counterfactual Policy loss is the weighted cross-entropy between $P_{new}$ and $\Pi^+$:

$$
L_{P,\mathrm{CF}}^{(\mathcal B)}=
\frac{
\sum_{s\in\mathcal B}W_P(s)L_{\mathrm{CE}}
\left(P_{new}(\cdot\mid s),\Pi^+(\cdot\mid s)\right)
}
{\max\left(\sum_{s\in\mathcal B}W_P(s),10^{-8}\right)}.
$$

The MC Policy loss applies the aggregated signed coefficients to the log probabilities under $\mu_{new}$:

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

For an unmerged record from a completed trajectory, $S_A(s_t,a_t)=A_{\mathrm{MC}}(s_t,a_t)$ and $W_A(s_t,a_t)=1$. A positive coefficient directs gradient descent toward a larger probability for $a_t$, and a negative coefficient directs it toward a smaller probability. At optimizer step $k$, both Policy losses act on the same logits and current parameters $\theta^{(k-1)}$, so their total gradient is

$$
\nabla_{\theta^{(k-1)}}
\left(L_{P,\mathrm{CF}}^{(\mathcal B)}+L_{P,\mathrm{MC}}^{(\mathcal B)}\right)
=
\nabla_{\theta^{(k-1)}}L_{P,\mathrm{CF}}^{(\mathcal B)}
+\nabla_{\theta^{(k-1)}}L_{P,\mathrm{MC}}^{(\mathcal B)}.
$$

For scalar prediction error $e$, define the SmoothL1 penalty with threshold $1$ by

$$
\mathrm{SL1}(e)=
\begin{cases}
\frac12e^2,&|e|<1,\\
|e|-\frac12,&|e|\geq1.
\end{cases}
$$

The Value loss combines terminal-return supervision and counterfactual-tree supervision:

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

The complete minibatch objective is

$$
L^{(\mathcal B)}=
L_{P,\mathrm{CF}}^{(\mathcal B)}+
L_{P,\mathrm{MC}}^{(\mathcal B)}+
L_V^{(\mathcal B)}.
$$

Each denominator is computed from the current minibatch. The resulting objective is therefore a minibatch-normalized estimator of the weighted training criterion.

### 6.8 Parameter Optimization

Let $\mathcal D=\mathcal S_{\mathrm{agg}}$ be the aggregated training set. Target generation has already completed with the fixed source parameters $\theta_{old}$, and optimization begins from $\theta^{(0)}=\theta_{old}$.

Let $E_{\mathrm{opt}}$ be the epoch count, $B_{\mathrm{opt}}$ the minibatch size and $K_{\max}\in\mathbb N\cup\lbrace\infty\rbrace$ the update cap. A seeded random generator permutes $\mathcal D$ at the beginning of each epoch, after which the records are partitioned into minibatches. Applying the update cap to the concatenated epoch sequence gives $\mathcal B_1,\ldots,\mathcal B_K$, where

$$
K=\min\left(K_{\max},E_{\mathrm{opt}}\left\lceil\frac{|\mathcal D|}{B_{\mathrm{opt}}}\right\rceil\right).
$$

For update $k$, the parameter gradient of the minibatch objective is

$$
g_k=\nabla_{\theta^{(k-1)}}L^{(\mathcal B_k)}.
$$

FCPI clips the global Euclidean norm of this gradient at $1$. With $\varepsilon_c=10^{-6}$, the gradient supplied to AdamW is

$$
\overline g_k=\alpha_k g_k,
\qquad
\alpha_k=\min\left(1,\frac{1}{\lVert g_k\rVert_2+\varepsilon_c}\right).
$$

Each BatchNorm layer separates fixed running statistics from trainable affine parameters. During the forward pass for update $k$, layer $j$ uses the running mean $\mu_{j,old}$ and running variance $\sigma^2_{j,old}$ stored by the source model. Its scale $\gamma_j^{(k-1)}$ and bias $\beta_j^{(k-1)}$ belong to $\theta^{(k-1)}$. With $\varepsilon_{\mathrm{BN}}=10^{-5}$, the layer maps activation $u$ to

$$
\mathrm{BN}_j^{(k-1)}(u)=
\gamma_j^{(k-1)}\odot
\frac{u-\mu_{j,old}}{\sqrt{\sigma^2_{j,old}+\varepsilon_{\mathrm{BN}}}}
+\beta_j^{(k-1)}.
$$

Fixing $\mu_{j,old}$ and $\sigma^2_{j,old}$ preserves the source model's normalization reference during candidate optimization. The clipped gradient $\overline g_k$ contains derivatives with respect to $\gamma_j^{(k-1)}$, $\beta_j^{(k-1)}$ and every other trainable component of $\theta^{(k-1)}$. The AdamW update below uses these derivatives to produce $\theta^{(k)}$.

AdamW converts $\overline g_k$ into the parameter update. Let $u_k$ and $v_k$ denote its first- and second-moment estimates. With $\beta_1=0.9$, $\beta_2=0.999$, $\varepsilon_A=10^{-8}$, weight-decay coefficient $\lambda=10^{-4}$ and initial values $u_0=v_0=0$, update $k$ computes

$$
u_k=\beta_1u_{k-1}+(1-\beta_1)\overline g_k,
\qquad
v_k=\beta_2v_{k-1}+(1-\beta_2)\overline g_k^2,
$$

$$
\widehat u_k=\frac{u_k}{1-\beta_1^k},
\qquad
\widehat v_k=\frac{v_k}{1-\beta_2^k},
$$

and, for learning rate $\eta$,

$$
\theta^{(k)}=
(1-\eta\lambda)\theta^{(k-1)}
-\eta\frac{\widehat u_k}{\sqrt{\widehat v_k}+\varepsilon_A}.
$$

The square, division and square root act component-wise. After $K$ updates, $\theta_{new}=\theta^{(K)}$ parameterizes the candidate.

### 6.9 Arena and Promotion

Arena compares two Gadus models through direct play. During iteration $r$, one model is the candidate with parameters $\theta_{new}$ and the other is the current model $C_r$ with parameters $\theta_{old}$. Both models use the same decision procedure, so the comparison isolates the difference between their network parameters.

Let $\mathcal O_A$ be a finite pool of reachable nonterminal starting states, and let $N_G$ be a positive even number of games. Arena selects $N_G/2$ states from $\mathcal O_A$ and plays two games from each selected state. The candidate controls White in one game and Black in the other, giving both models one game with each color from the same state.

Arena may advance at most $K_G\geq1$ games concurrently. At each batched step, positions awaiting a move from the candidate are evaluated together, and positions awaiting a move from the current model are evaluated together. This arrangement shares neural forward passes across games and keeps a separate complete state and outcome for every game.

Let $N_W$, $N_D$ and $N_L$ be the candidate's win, draw and loss counts, where $N_G=N_W+N_D+N_L$. Its mean score and net-win margin are

$$
S=\frac{N_W+\frac12N_D}{N_G},
\qquad
M=N_W-N_L.
$$

To quantify the sampling uncertainty in $S$, let $X_i\in\lbrace0,\frac12,1\rbrace$ be the candidate's score in game $i$. Arena computes the population variance

$$
\sigma^2=\frac1{N_G}\sum_{i=1}^{N_G}(X_i-S)^2
$$

and the clipped 95% normal-approximation interval

$$
CI_{95\%}=\mathrm{clip}_{[0,1]}\left(
S\pm1.96\sqrt{\frac{\sigma^2}{N_G}}
\right).
$$

Let $M_{\mathrm{gate}}$ be the required net-win margin for promotion. FCPI applies the criterion

$$
M\geq M_{\mathrm{gate}}.
$$

When the criterion is satisfied, FCPI sets $C_{r+1}$ to the candidate model. When the criterion is not satisfied, FCPI sets $C_{r+1}=C_r$.

### 6.10 Idealized Policy Shift

Consider one encoded state $s$ and two legal actions $a,b\in\mathcal A(x)$. The following idealized analysis examines how repeated exact fits to the local target $\pi^+(\cdot\mid s)$ change the Policy log odds between these actions. Let the clipping floor $10^{-12}$ tend to zero, and suppose one candidate update fits $\pi^+(\cdot\mid s)$ exactly. The resulting log-odds relation is

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

Let $P_0$ be the legal-move Policy before the first ideal fit, and let $P_k$ be the Policy after $k$ consecutive fits. If the target fitted at each step becomes the reference Policy for the next step, then

$$
\log\frac{P_k(a\mid s)}{P_k(b\mid s)}
=
\log\frac{P_0(a\mid s)}{P_0(b\mid s)}
+k\Delta.
$$

For $P_0(a\mid s)<P_0(b\mid s)$, action $a$ overtakes action $b$ when

$$
k>
\frac{\log P_0(b\mid s)-\log P_0(a\mid s)}{\Delta}.
$$

This conclusion applies to one fixed state, one fixed action pair and a constant positive $\Delta$.
