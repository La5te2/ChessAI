# Gadus

Gadus is a chess-structured residual network that predicts a legal-move Policy and a side-to-move Value from a canonical representation of the current position.

## 1. Notation

- $\mathcal X$ is the set of complete chess states maintained by the rules engine.
- $x\in\mathcal X$ is a complete chess state containing the board, side to move, castling rights, en passant state, move counters and repetition history.
- $\mathcal A(x)$ is the set of legal actions in state $x$.
- $T(x,a)$ is the complete state reached by applying legal action $a\in\mathcal A(x)$ to state $x$.
- $z(x)\in\lbrace-1,0,1\rbrace$ is the exact outcome of terminal state $x$ from the perspective of its side to move, with $1$, $0$ and $-1$ representing a win, draw and loss.
- $\phi_G$ is the canonical state encoder that converts a complete chess state to the side-to-move network representation.
- $s=\phi_G(x)$ is the canonical Gadus network input obtained from complete state $x$.
- $\mathcal I_G=\lbrace0,\ldots,4671\rbrace$ is the fixed set of Gadus action indices.
- $\widetilde i_G(a)\in\mathcal I_G$ is the physical-board index of legal action $a$.
- $i_G(x,a)\in\mathcal I_G$ is the canonical index of $a\in\mathcal A(x)$ after applying the side-to-move transformation associated with $x$.
- $\theta$ denotes the trainable network parameters.
- $\ell_\theta(s)\in\mathbb R^{4672}$ is the complete vector of Policy logits produced by the network with parameters $\theta$, and $\ell_\theta(s,i)$ is its scalar component for action index $i\in\mathcal I_G$.
- $\text{P}$, which stands for Policy, is the network output that assigns a probability distribution over the legal actions available in each encoded state.
- $\text{V}$, which stands for Value, is a scalar network output in $[-1,1]$ that estimates the expected game result from the perspective of the side to move.
- $Q$ denotes a state or action evaluation defined by a particular procedure. Each definition specifies its arguments and observation perspective.
- $\mathrm{clip}_{[l,u]}(y)=\min(u,\max(l,y))$ restricts scalar $y$ to the closed interval $[l,u]$.

## 2. State and Action Encoding

### 2.1 State Encoding

The state encoder uses file coordinates $0$ through $7$ for files `a` through `h` and rank coordinates $0$ through $7$ for ranks 1 through 8. It labels the pieces of the player to move as friendly and the other pieces as opposing. For a White-to-move state, canonical coordinates equal physical board coordinates. For a Black-to-move state, the encoder maps physical square $(r,f)$ to canonical square $(7-r,f)$.

The resulting network input contains 17 binary planes, each of which is an $8\times8$ grid aligned with the canonical board. One entry equals $1$ when the feature represented by its plane is present at the corresponding square and equals $0$ otherwise:

$$
\phi_G:\mathcal X\rightarrow\lbrace0,1\rbrace^{17\times8\times8}.
$$

Planes 0 through 5 represent friendly pawn, knight, bishop, rook, queen and king occupancy. Planes 6 through 11 represent the opposing piece types in the same order. Each entry in these planes indicates whether the represented piece occupies the corresponding canonical square.

Planes 12 through 15 represent friendly kingside, friendly queenside, opposing kingside and opposing queenside castling rights. Each castling plane contains ones at all 64 squares when its right is available and zeros at all 64 squares otherwise. Plane 16 contains ones at the eight squares of the canonical en passant file and zeros elsewhere. The complete plane is zero when no en passant square exists. The canonical transformation identifies the player to move through the friendly role, so the network input requires no separate side-to-move plane.

The canonical input records piece placement relative to the player to move, castling rights and the en passant file. The complete state $x$ additionally records move counters and repetition history. Complete states that differ only in these omitted fields therefore produce the same network input.

### 2.2 Action Encoding

The legal-action set $\mathcal A(x)$ varies with the complete state $x$. Gadus first assigns each legal action a state-independent physical index in $\mathcal I_G$, then transforms that index to the same canonical coordinates as the encoded state.

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
(0,1),\ (1,-1),\ (1,0),\ (1,1).
$$

Let $(u_d,v_d)$ be the direction at position $d\in\lbrace0,\ldots,7\rbrace$ in this list. For a move that travels $m\in\lbrace1,\ldots,7\rbrace$ squares in direction $(u_d,v_d)$, the motion-pattern index is

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
\widetilde i_G(a)=73q+p.
$$

Combining 64 source-square indices with 73 motion-pattern indices gives

$$
|\mathcal I_G|=64\times73=4672.
$$

The map $\widetilde i_G$ assigns an index in physical-board coordinates. For $a\in\mathcal A(x)$, the canonical map $i_G(x,a)$ applies $\widetilde i_G$ directly when White is to move. When Black is to move, it reflects the source and destination ranks by $r\mapsto7-r$ before applying the same $64\times73$ codec. This transformation changes rank-sensitive sliding and knight patterns but preserves the file offset and promoted piece of an underpromotion. The state and its legal actions consequently use the same canonical coordinates, which are shared by supervised targets, legal-action inference and Policy logits.

For complete state $x$, the available action indices are

$$
\mathcal I_G(x)=
\lbrace i_G(x,a)\mid a\in\mathcal A(x)\rbrace.
$$

To decode an available physical index, Gadus generates the legal-action set $\mathcal A(x)$ and selects the action whose $\widetilde i_G$ encoding equals that index. The selected legal action contains the castling, en passant or promotion information required by the rules engine. Canonical indices are used only at the network boundary, where each legal action remains paired with its physical move representation.

## 3. Network

### 3.1 Residual Trunk

The Gadus network derives its $\text{P}$ and $\text{V}$ from a shared chess-structured residual trunk. Let $C$ be the trunk width and let $B$ be its number of residual blocks. The stem maps the 17 canonical input planes to $C$ board-aligned feature channels, and a learned square embedding supplies an independent $C$-dimensional offset at each canonical square. With bias-free convolution, batch normalization and elementwise ReLU denoted by $\mathrm{Conv}$, $\mathrm{BN}$ and $\mathrm{ReLU}$, the initial feature tensor is

$$
h_0=
\mathrm{ReLU}\left(
\mathrm{BN}_{\mathrm{stem}}\left(
\mathrm{Conv}^{17\rightarrow C}_{3\times3,\mathrm{stem}}(s)
\right)\right)+E_S,
\qquad
E_S\in\mathbb R^{C\times8\times8}.
$$

Every residual block preserves the shape $C\times8\times8$ and combines a local board transformation with a full-board relation transformation. The block first applies a $1\times1$ projection and obtains

$$
b_j=
\mathrm{ReLU}\left(
\mathrm{BN}_{j,\downarrow}\left(
\mathrm{Conv}^{C\rightarrow C}_{1\times1,j,\downarrow}(h_j)
\right)\right).
$$

The local transformation contains parallel $3\times3$, $1\times1$ and identity branches. Each branch has its own batch normalization, and their outputs are added before the next nonlinearity:

$$
L_j(b_j)=
\mathrm{BN}_{j,3}\left(
\mathrm{Conv}^{C\rightarrow C}_{3\times3,j}(b_j)
\right)
+\mathrm{BN}_{j,1}\left(
\mathrm{Conv}^{C\rightarrow C}_{1\times1,j}(b_j)
\right)
+\mathrm{BN}_{j,I}(b_j).
$$

The full-board transformation acts on ordered pairs of the 64 canonical squares. Let $q=(r_q,f_q)$ be an output square and $p=(r_p,f_p)$ an input square. Their target-relative displacement is

$$
\delta(q,p)=(r_q-r_p,f_q-f_p)\in\mathcal D,
\qquad
\mathcal D=\{-7,\ldots,7\}^2.
$$

For every $(a,b)\in\mathcal D$, define

$$
T_{a,b}(q,p)=\mathbf 1[\delta(q,p)=(a,b)],
\qquad
n_{a,b}=\lVert T_{a,b}\rVert_F^2=(8-|a|)(8-|b|),
$$

and

$$
\widehat B_{a,b}=\frac{T_{a,b}}{\sqrt{n_{a,b}}}.
$$

The 225 matrices $\widehat B_{a,b}$ have disjoint support and therefore form an orthonormal basis for every square relation whose coefficient depends only on relative displacement. Each relation group learns one coefficient $\alpha_{j,g,a,b}$ for every member of this basis.

A static absolute-position residual $\Delta A^\perp_{j,g}$ supplies the complementary square-pair information. It obeys

$$
\sum_{\delta(q,p)=(a,b)}
\Delta A^\perp_{j,g}(q,p)=0
\qquad
\text{for every }(a,b)\in\mathcal D.
$$

The static relation is consequently

$$
A^{\mathrm{static}}_{j,g}=
\sum_{(a,b)\in\mathcal D}
\alpha_{j,g,a,b}\widehat B_{a,b}
+\Delta A^\perp_{j,g}.
$$

This decomposition is unique. Its relative component contains the common value of each displacement class, while the constrained residual contains only differences among square pairs in the same class. For each relation group, the implementation stores 225 coefficients indexed by the $15\times15$ displacement set and obtains the corresponding dense entries by lookup rather than storing 225 dense basis matrices.

Sliding-piece geometry also depends on occupied intermediate squares. Let $o_s(p)\in\{0,1\}$ indicate whether square $p$ is occupied in encoded state $s$, and let $I(q,p)$ be the squares strictly between two aligned endpoints. Define

$$
\begin{aligned}
V^R_s(q,p)&=
\mathbf 1[q\ne p\land(r_q=r_p\lor f_q=f_p)]
\prod_{u\in I(q,p)}(1-o_s(u)),\\
V^B_s(q,p)&=
\mathbf 1[q\ne p\land|r_q-r_p|=|f_q-f_p|]
\prod_{u\in I(q,p)}(1-o_s(u)).
\end{aligned}
$$

The empty product equals one. Thus adjacent aligned squares are visible, an occupied intermediate square blocks every longer ray through it, and distance does not weaken an otherwise unobstructed relation. The complete position-dependent matrix is

$$
A_{j,g}(s)=A^{\mathrm{static}}_{j,g}
+\beta^R_{j,g}V^R_s
+\beta^B_{j,g}V^B_s,
$$

where $\beta^R_{j,g}$ and $\beta^B_{j,g}$ are learned scalars. The visibility matrices $V^R_s$ and $V^B_s$ are constructed once for each encoded position and used by every residual block that processes that position.

During training, the implementation evaluates the products of the feature tensor with the static, rook-visibility and bishop-visibility matrices separately, then adds the three results. This algebraically equivalent form avoids retaining a batch-sized $N\times K\times64\times64$ relation tensor for every block during backpropagation. Inference retains no backward state, so its fused form first combines the three matrices and then performs one grouped relation multiplication.

Let

$$
K=\gcd(C,8),
\qquad
d=\frac{C}{K}.
$$

Here $K$ is the number of computational channel groups in one residual block. The channels of $b_j$ are divided into $K$ groups of width $d$, and each group has its own $A_{j,g}(s)\in\mathbb R^{64\times64}$.

After reshaping group $g$ to $b_{j,g}\in\mathbb R^{64\times d}$, the relation matrix acts on its square dimension:

$$
\widetilde R_{j,g}=A_{j,g}(s)b_{j,g}.
$$

Concatenating the groups restores a $C\times8\times8$ tensor. A non-affine batch-normalization layer then produces the relation output $R_j(b_j)$.

The learned vector $\lambda_j\in\mathbb R^C$ controls the relative contribution of the two transformations independently in every channel. Broadcasting it over the board gives

$$
c_j=
\mathrm{ReLU}\left(
\frac{L_j(b_j)+\lambda_j\odot R_j(b_j)}
{\sqrt{1+\lambda_j^2}}
\right).
$$

All operations involving $\lambda_j$ are applied independently to each channel and broadcast over its $8\times8$ feature map. A final $1\times1$ projection maps the combined representation back to the residual stream:

$$
h_{j+1}=
\mathrm{ReLU}\left(
h_j+
\mathrm{BN}_{j,\uparrow}\left(
\mathrm{Conv}^{C\rightarrow C}_{1\times1,j,\uparrow}(c_j)
\right)
\right),
\qquad 0\leq j<B.
$$

The eight available initialization modes are identity, king displacement, knight displacement, canonical friendly-pawn advance, canonical friendly-pawn capture, complete-board communication, unobstructed rook geometry and unobstructed bishop geometry. If $K<8$, the first $K$ modes are used. For a static offset set $E\subseteq\mathcal D$, let $M_E=\sum_{(a,b)\in E}T_{a,b}$. Its group begins from the unit-Frobenius matrix

$$
\overline M_E=\frac{M_E}{\lVert M_E\rVert_F},
\qquad
\alpha_{j,g,a,b}=
\begin{cases}
\sqrt{n_{a,b}}/\lVert M_E\rVert_F,&(a,b)\in E,\\
0,&(a,b)\notin E.
\end{cases}
$$

The static offset sets are

$$
\begin{aligned}
E_I&=\{(0,0)\},\\
E_K&=\{(a,b):\max(|a|,|b|)=1\},\\
E_N&=\{(a,b):(|a|,|b|)\in\{(1,2),(2,1)\}\},\\
E_{P,m}&=\{(1,0)\},\\
E_{P,c}&=\{(1,-1),(1,1)\},\\
E_G&=\mathcal D.
\end{aligned}
$$

The six static initialization modes use the canonical displacement coordinates defined above. In the friendly-pawn modes, canonical rank displacement $+1$ points forward for the friendly side. The remaining two modes initialize position-dependent visibility relations. If relation group $6$ exists, its rook-visibility coefficient $\beta^R_{j,6}$ is initialized to one. If relation group $7$ exists, its bishop-visibility coefficient $\beta^B_{j,7}$ is initialized to one. Every relation coefficient and absolute residual entry not assigned by these initialization modes is initialized to zero. Independently of the relation initialization, each block begins with $\lambda_j=\mathbf1$, which assigns the local and full-board transformations the same coefficient $1/\sqrt2$ in every channel.

The affine scales of the three local-branch normalization layers are initialized to $1/\sqrt3$, and their biases are initialized to zero. In the shared and Policy sequences, the affine scale of $\mathrm{BN}_{j,\uparrow}$ is initialized to $1/\sqrt D$ for a sequence of length $D$. The corresponding scale is initialized to one in the first Value block and to zero in the second. The second Value block is therefore an identity map at initialization. All corresponding biases are initialized to zero. The square embedding $E_S$ is also initialized to zero. The displacement atoms and ray geometry are fixed, whereas the square embedding and all block-specific convolutional, normalization, relation and path-balance parameters are trainable.

After the $B$ blocks, $h_B\in\mathbb R^{C\times8\times8}$ is the shared representation supplied to the Policy and Value heads defined below.

### 3.2 Policy and Value Heads

Let $C_P$, $C_V$ and $H_V$ be the positive integer widths of the Policy representation, the Value feature tensor and the hidden Value layer, respectively. Let $D_P$ and $D_V$ be the positive integer numbers of Policy-specific and Value-specific chess-structured residual blocks. The fixed head dimensions are

$$
D_P=D_V=2,
\qquad
C_P=128,
\qquad
C_V=48,
\qquad
H_V=512.
$$

The Policy head assigns one unnormalized score, called a logit, to every canonical action index in $\mathcal I_G$. It begins with a bias-free $1\times1$ convolution from $C$ channels to $C_P$ channels, followed by batch normalization and ReLU:

$$
u_0=\mathrm{ReLU}\left(
\mathrm{BN}_P\left(
\mathrm{Conv}^{C\rightarrow C_P}_{1\times1,P}(h_B)
\right)
\right).
$$

The Policy head passes $u_0$ through its chess-structured residual blocks in sequence, each following the construction in Section 3.1 with its own parameters. Let $F_{P,j}$ denote the input-output map of block $j$. The block sequence satisfies

$$
u_{j+1}=F_{P,j}(u_j),
\qquad 0\leq j<D_P.
$$

A bias-free $1\times1$ convolution maps $u_{D_P}$ to 73 action-pattern planes. The learned tensor $B_P\in\mathbb R^{73\times8\times8}$ supplies a separate bias for every combination of motion pattern and source square, giving

$$
L_\theta(s)=
\mathrm{Conv}^{C_P\rightarrow73}_{1\times1,\mathrm{out}}(u_{D_P})+B_P
\in\mathbb R^{73\times8\times8}.
$$

Let $(r,f)$ be the canonical coordinates of a source square and let $q=8r+f$ be its square index. For motion pattern $p$, the component of $L_\theta(s)$ at channel $p$, rank $r$ and file $f$ is the logit assigned to action index $73q+p$:

$$
\ell_\theta(s,73q+p)=L_\theta(s)_{p,r,f}.
$$

The Value head estimates the expected game result from the perspective of the player to move. It passes $h_B$ through its Value-specific chess-structured residual blocks, whose parameters are separate from those of the shared trunk and Policy head. Let $F_{V,j}$ denote the input-output map of Value block $j$. With $v_0=h_B$, the block sequence satisfies

$$
v_{j+1}=F_{V,j}(v_j),
\qquad 0\leq j<D_V.
$$

A bias-free $1\times1$ convolution maps $v_{D_V}$ from $C$ channels to $C_V$ channels. Batch normalization and ReLU then produce

$$
r_V=\mathrm{ReLU}\left(
\mathrm{BN}_V\left(
\mathrm{Conv}^{C\rightarrow C_V}_{1\times1,V}(v_{D_V})
\right)\right).
$$

Flattening $r_V$ in channel-rank-file order gives $\mathrm{vec}(r_V)\in\mathbb R^{64C_V}$. Let $W_{V,1}\in\mathbb R^{H_V\times64C_V}$ and $b_{V,1}\in\mathbb R^{H_V}$ be the parameters of the hidden Value layer, and let $W_{V,2}\in\mathbb R^{1\times H_V}$ and $b_{V,2}\in\mathbb R$ be the parameters of its output layer. The Value estimate is

$$
V_\theta(s)=
\tanh\left(
W_{V,2}\mathrm{ReLU}\left(
W_{V,1}\mathrm{vec}(r_V)+b_{V,1}
\right)+b_{V,2}
\right).
$$

Writing $f_\theta$ for the complete network gives

$$
f_\theta(s)=\left(\ell_\theta(s),V_\theta(s)\right),
\qquad
\ell_\theta(s)\in\mathbb R^{4672},
\quad
V_\theta(s)\in[-1,1].
$$

The complete logit vector assigns a score to every index in $\mathcal I_G$. For $s=\phi_G(x)$, selecting the indices $i_G(x,a)$ for $a\in\mathcal A(x)$ and normalizing their logits with softmax produces the legal-move Policy:

$$
P_\theta(a\mid s)=
\frac{\exp\ell_\theta(s,i_G(x,a))}
{\displaystyle\sum_{b\in\mathcal A(x)}\exp\ell_\theta(s,i_G(x,b))},
\qquad a\in\mathcal A(x).
$$

The denominator ranges over $\mathcal A(x)$, so $P_\theta(\cdot\mid s)$ is a probability distribution over the legal actions in complete state $x$.

The complete data flow is therefore

$$
s\longrightarrow\text{shared trunk}\longrightarrow h_B\longrightarrow
\begin{cases}
\text{Policy head}\longrightarrow\ell_\theta(s)
\longrightarrow P_\theta(\cdot\mid s),\\
\text{Value head}\longrightarrow V_\theta(s).
\end{cases}
$$

### 3.3 Inference Evaluation

The action-plane tensor stores the logit for source square $q=8r+f$ and motion pattern $p$ at component $(p,r,f)$. To construct the complete logit vector $\ell_\theta(s)$, the network permutes $L_\theta(s)$ from action-pattern-major order to source-square-major order and then flattens the permuted tensor. This permutation makes component $73q+p$ of the flattened vector equal to $L_\theta(s)_{p,r,f}$, as required by the action encoding in Section 2.2.

The final $1\times1$ Policy convolution associates action pattern $p$ with a weight vector $w_p\in\mathbb R^{C_P}$. For requested action index $i$, define

$$
q=\left\lfloor\frac{i}{73}\right\rfloor,
\qquad
p=i\bmod73,
\qquad
r=\left\lfloor\frac{q}{8}\right\rfloor,
\qquad
f=q\bmod8.
$$

The channel values of $u_{D_P}$ at source square $(r,f)$ form the Policy feature vector

$$
u_{D_P}(r,f)=\left(\left(u_{D_P}\right)_{c,r,f}\right)_{c=0}^{C_P-1}
\in\mathbb R^{C_P}.
$$

The logit for action index $i$ is obtained from this feature vector, the weight vector for pattern $p$ and the corresponding action-position bias:

$$
\ell_\theta(s,i)=w_p^{\mathsf T}u_{D_P}(r,f)+B_{P,p,r,f}.
$$

This expression equals $L_\theta(s)_{p,r,f}$ and computes the requested logit without materializing the other action-plane components. For a batch of complete states, inference first converts every legal action index to canonical coordinates. The legal-action arrays are padded to the largest legal-action count in the batch so that the requested logits can be evaluated together. The padded positions are masked before softmax, leaving each state with a distribution over exactly its legal actions.

In evaluation mode, every convolution followed directly by affine batch normalization is replaced by its equivalent biased convolution. Within each chess-structured block, the three local branches are combined into one biased $3\times3$ convolution by placing the $1\times1$ and normalized identity kernels at the center of the $3\times3$ kernel. Each $A^{\mathrm{static}}_{j,g}$ is computed once from its displacement coefficients and constrained residual. For each evaluated position, the network constructs $V^R_s$ and $V^B_s$ once and combines them with the rook and bishop visibility coefficients of every relation group in each block. Fusion also evaluates the fixed statistics of the non-affine relation normalization and the learned path-balance vector once. The resulting local coefficient and relation-centering term are absorbed into the fused local convolution, while the resulting relation coefficient becomes a fixed channelwise scale applied after grouped relation multiplication. These transformations preserve the network outputs while removing affine normalization attached to fused convolutions, local-branch summation, repeated construction of static relation matrices, relation normalization and path-balance square roots from inference.

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
i_n=i_G(x_n,a_n),
\qquad
y_n\in[-1,1].
$$

The encoded state $s_n$ is the canonical network input, and the action index $i_n$ is the corresponding canonical Policy target. The scalar $y_n$ is the Value target, expressed as an estimate of the expected game result from the perspective of the side to move in $x_n$. On this scale, $-1$ denotes a loss, $0$ denotes a draw and $1$ denotes a win, while intermediate values express expectations between these outcomes.

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

Let $w_{\min}=0.2$ and $w_{\max}=2$ be the lower and upper bounds of the Value-loss coefficient. The initial coefficient $w_{V,0}$ belongs to this range, and $w_{V,k}$ denotes the coefficient used at optimizer step $k$ ($k\geq1$). The complete objective for minibatch $\mathcal B_k$ is

$$
L_{\mathrm{sup}}^{(\mathcal B_k)}=
L_{P,\mathrm{sup}}^{(\mathcal B_k)}+
w_{V,k}L_{V,\mathrm{sup}}^{(\mathcal B_k)}.
$$

The controller adjusts a positive coefficient by comparing the Policy and Value gradients within the shared residual trunk. Define these gradient vectors by

$$
g_{P,k}=\nabla_{\theta_T}L_{P,\mathrm{sup}}^{(\mathcal B_k)},
\qquad
g_{V,k}=\nabla_{\theta_T}L_{V,\mathrm{sup}}^{(\mathcal B_k)}.
$$

Their norms and inner product are

$$
n_{P,k}=\lVert g_{P,k}\rVert_2,
\qquad
n_{V,k}=\lVert g_{V,k}\rVert_2,
\qquad
c_k=g_{P,k}^{\mathsf T}g_{V,k}.
$$

Let $\eta_g=0.5$ control the target ratio between the two gradient norms, and let $\epsilon_g=10^{-12}$ provide numerical stability. The norm-balanced proposal is

$$
\widetilde w_{V,k}=
\mathrm{clip}_{[w_{\min},w_{\max}]}
\left(\eta_g\frac{n_{P,k}}{n_{V,k}+\epsilon_g}\right).
$$

When $c_k<0$, a step along the negative combined gradient decreases both minibatch losses to first order precisely when

$$
\frac{-c_k}{n_{V,k}^2}<w_{V,k}<\frac{n_{P,k}^2}{-c_k}.
$$

The controller uses an interior margin $\delta_g=10^{-3}$ and intersects this interval with the admissible coefficient range:

$$
I_k=
\left[
\max\left(w_{\min},(1+\delta_g)\frac{-c_k}{n_{V,k}^2}\right),
\min\left(w_{\max},(1-\delta_g)\frac{n_{P,k}^2}{-c_k}\right)
\right].
$$

For $c_k\geq0$, define $I_k=[w_{\min},w_{\max}]$. Let $\mathrm{proj}_{I_k}(w)$ denote the nearest point to $w$ in the closed interval $I_k$. When $I_k$ is nonempty, the controller projects $\widetilde w_{V,k}$ into $I_k$ and smooths the projected value in logarithmic space with update rate $\gamma=0.08$:

$$
\widehat w_{V,k}=\mathrm{proj}_{I_k}(\widetilde w_{V,k}),
$$

$$
w_{V,k}=\mathrm{clip}_{[w_{\min},w_{\max}]}\left(
\exp\left((1-\gamma)\log w_{V,k-1}+\gamma\log\widehat w_{V,k}\right)
\right).
$$

At a periodic probe point, an empty $I_k$, a nonfinite gradient statistic or either squared gradient norm being no greater than $\epsilon_g$ causes the controller to set $w_{V,k}=w_{V,k-1}$. If none of these conditions holds, the controller applies the projection and logarithmic smoothing defined above. At optimizer steps without a probe, the coefficient also satisfies $w_{V,k}=w_{V,k-1}$.

To express how the resulting objective updates the complete network, partition the network parameters as $\theta=(\theta_T,\theta_P,\theta_V)$, where $\theta_T$ contains the residual-trunk parameters and $\theta_P$ and $\theta_V$ contain the parameters of the two output heads. The gradients of $L_{\mathrm{sup}}^{(\mathcal B_k)}$ with respect to these parameter groups satisfy

$$
\nabla_{\theta_P}L_{\mathrm{sup}}^{(\mathcal B_k)}=
\nabla_{\theta_P}L_{P,\mathrm{sup}}^{(\mathcal B_k)},
$$

$$
\nabla_{\theta_V}L_{\mathrm{sup}}^{(\mathcal B_k)}=
w_{V,k}\nabla_{\theta_V}L_{V,\mathrm{sup}}^{(\mathcal B_k)},
$$

$$
\nabla_{\theta_T}L_{\mathrm{sup}}^{(\mathcal B_k)}=
g_{P,k}+w_{V,k}g_{V,k}.
$$

The optimizer updates $\theta_P$ from the Policy-loss gradient and $\theta_V$ from the weighted Value-loss gradient. Its update to $\theta_T$ uses the combined gradient $g_{P,k}+w_{V,k}g_{V,k}$.

### 4.3 Parameter Optimization

Before the first parameter update, Gadus estimates output priors from a stratified sample of the supervised targets. Let $c_i$ be the number of sampled Policy targets with canonical action index $i$, let $\bar y$ be the mean sampled Value target and let $a_0=1$ be the smoothing pseudocount. The smoothed action prior is

$$
\pi_0(i)=
\frac{c_i+a_0}
{\displaystyle\sum_{j\in\mathcal I_G}c_j+a_0|\mathcal I_G|},
\qquad i\in\mathcal I_G.
$$

The action-position bias corresponding to index $i$ is initialized to $\log\pi_0(i)$. The Policy output weights and Value output weights are each scaled from their initial values by $\rho_0=0.1$, and the Value output bias is set to

$$
b_{V,2}=
\mathrm{atanh}\left(
\mathrm{clip}_{[-1+\epsilon_0,1-\epsilon_0]}(\bar y)
\right),
\qquad \epsilon_0=10^{-4}.
$$

The Policy bias places the sampled action frequencies in the initial logits, while the reduced output weights limit their random perturbation. The Value initialization similarly places the initial estimate near the sampled target mean.

After initialization, parameter optimization partitions each randomized traversal of $\mathcal D_{\mathrm{sup}}$ into minibatches. For minibatch $\mathcal B_k$, automatic differentiation computes the gradients of $L_{\mathrm{sup}}^{(\mathcal B_k)}$ given in Section 4.2. Before the AdamW update, Gadus limits the Euclidean norm of the Value-head gradient to the threshold $G_V=1$ while leaving the Policy-head and shared-trunk gradients unchanged. Let

$$
g_{V,k}^{\mathrm{head}}=
\nabla_{\theta_V}L_{\mathrm{sup}}^{(\mathcal B_k)}.
$$

The gradient supplied to AdamW for the Value-head parameters is

$$
\widetilde g_{V,k}^{\mathrm{head}}=
\begin{cases}
g_{V,k}^{\mathrm{head}},
&\lVert g_{V,k}^{\mathrm{head}}\rVert_2\leq G_V,\\[6pt]
G_V\dfrac{g_{V,k}^{\mathrm{head}}}
{\lVert g_{V,k}^{\mathrm{head}}\rVert_2},
&\lVert g_{V,k}^{\mathrm{head}}\rVert_2>G_V.
\end{cases}
$$

AdamW applies the gradients for $\theta_T$ and $\theta_P$ from Section 4.2 together with $\widetilde g_{V,k}^{\mathrm{head}}$ to obtain an intermediate parameter state. The residual matrix of every relation group is then projected onto the displacement-orthogonal subspace. For an intermediate residual $Z_{j,g}$, the stored residual for the next step is

$$
\Delta A^{\perp}_{j,g}(q,p)=Z_{j,g}(q,p)
-\frac{1}{n_{a,b}}
\sum_{\delta(u,v)=(a,b)}Z_{j,g}(u,v),
\qquad
(a,b)=\delta(q,p).
$$

This projected update preserves the zero-sum constraint in every displacement class and prevents the relative coefficients and absolute residual from optimizing duplicate directions. The resulting parameters form $\theta^{(k+1)}$.

## 5. Search

### 5.1 Root Initialization

For a nonterminal complete state $x_0$, MCTS initializes a tree rooted at $x_0$. Each node represents one complete state $x$, and each outgoing edge represents a legal action $a\in\mathcal A(x)$ that connects the node to the child state $T(x,a)$. A simulation follows a sequence of edges to a leaf, evaluates that leaf and propagates the result back along the same path.

Root initialization evaluates $s_0=\phi_G(x_0)$ to obtain $P_\theta(\cdot\mid s_0)$ and $V_\theta(s_0)$. Before any simulation completes, $V_\theta(s_0)$ supplies the reported root evaluation, and the root statistics satisfy $N(x_0)=W(x_0)=0$. The procedure then creates one outgoing edge and child node for every legal action $a\in\mathcal A(x_0)$, using its Policy probability as the edge prior:

$$
P(x_0,a)=P_\theta(a\mid s_0).
$$

A nonterminal node with no outgoing edges is called an unexpanded node. When a simulation reaches an unexpanded node representing state $x$, the evaluator obtains $P_\theta(\cdot\mid\phi_G(x))$ and $V_\theta(\phi_G(x))$. Node expansion uses the Policy distribution to create one outgoing edge and child node for every action $a\in\mathcal A(x)$, assigning $P_\theta(a\mid\phi_G(x))$ to the edge prior $P(x,a)$. The backup procedure defined in Section 5.4 propagates $V_\theta(\phi_G(x))$ along the selected path.

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

Let $N(x,a)$ denote the completed-visit count of the child reached through action $a$. This count equals the number of completed simulations that traversed the edge from $x$ to $T(x,a)$. A node receives its first evaluation before any simulation traverses an outgoing edge from that node. Consequently, its node count $N(x)$ may exceed $\sum_{a\in\mathcal A(x)}N(x,a)$.

### 5.3 Non-root PUCT Selection

After the root procedure in Section 5.5 selects the first action of a simulation, Predictor + Upper Confidence bounds applied to Trees (PUCT) selects each subsequent action at an expanded non-root node. Its score combines the empirical evaluation of an action with an exploration term derived from the edge prior and visit counts. First Play Urgency (FPU) supplies the selection value of an unvisited edge, which has no empirical action evaluation.

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

At each expanded non-root node, the selector follows the action with the largest $S(x,a)$. Equal PUCT scores are resolved first by the larger prior $P(x,a)$ and then by the larger $Q_{\mathrm{sel}}(x,a)$. If all three quantities are equal, the selector follows the action that the rules engine enumerated first when the node was expanded. The resulting path ends at a terminal state or at a nonterminal node that has not yet been expanded.

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

### 5.5 Root Allocation

Let $N_{\mathrm{ref}}\geq0$ be the reference budget used to determine the fair visit floor, let $N_{\mathrm{cap}}\in\mathbb N_0\cup\lbrace\infty\rbrace$ be the maximum number of simulations and let $M=|\mathcal A(x_0)|$ be the number of legal root actions. Bounded search uses $N_{\mathrm{ref}}=N_{\mathrm{cap}}$. Unbounded search retains a positive reference budget while setting $N_{\mathrm{cap}}=\infty$. A zero reference budget performs no simulation. For $N_{\mathrm{ref}}>0$, let $\kappa_{\mathrm{fair}}=10$ be the scale of the fair visit floor, which is

$$
m_{\mathrm{fair}}=
\max\left(
1,
\left\lfloor
\kappa_{\mathrm{fair}}\log\left(
1+\frac{N_{\mathrm{ref}}}{\kappa_{\mathrm{fair}}M}
\right)
\right\rfloor
\right).
$$

The scale $\kappa_{\mathrm{fair}}$ places the transition between two regimes. The floor is approximately linear in the reference budget per root action, $N_{\mathrm{ref}}/M$, when that quantity is small, while its growth becomes logarithmic as the reference budget increases. Every legal root action receives this floor, so a completed fair phase uses $Mm_{\mathrm{fair}}$ root visits.

Each fair-stage simulation first enters the root action selected by the allocation deficit. At every expanded nonterminal state reached afterward, all legal actions participate in the PUCT ordering from Section 5.3. The original Policy prior therefore governs the complete legal-action set throughout the fair phase. During post-fair allocation, root selection uses the tempered prior defined below, and expanded non-root nodes use the action-opening and verification rules defined in Section 5.6.

For $a\in\mathcal A(x_0)$, the fair-allocation deficit is

$$
d_{\mathrm{fair}}(x_0,a)=
\max\left(
0,
m_{\mathrm{fair}}-\widetilde N(x_0,a)
\right).
$$

While at least one legal root action has a positive deficit, the next simulation enters an action with the largest deficit. The PUCT ordering from Section 5.3 resolves equal deficits, and root expansion order resolves any remaining equality. Completed visits and virtual reservations both reduce the deficit, which distributes concurrent requests before their neural evaluations return. The scheduler begins post-fair allocation after every legal root action has completed $m_{\mathrm{fair}}$ visits, so pending virtual reservations contribute to batch formation but not to the fair-stage estimates.

Let

$$
Q_{\mathrm{fair}}(x_0,a)
$$

be the empirical action value at the end of the fair phase. Among unordered pairs of actions in $\mathcal A(x_0)$, let $C$ count pairs ordered identically by $P(x_0,a)$ and $Q_{\mathrm{fair}}(x_0,a)$, let $D$ count pairs ordered oppositely, let $T_P$ count pairs tied only in Policy and let $T_Q$ count pairs tied only in fair-stage value. Define

$$
Z=\sqrt{(C+D+T_P)(C+D+T_Q)}
$$

and

$$
\tau_{\mathrm{fair}}=
\begin{cases}
\dfrac{C-D}{Z},&Z>0,\\[6pt]
0,&Z=0.
\end{cases}
$$

The fixed root-prior exponent and its equivalent temperature are

$$
\gamma_{\mathrm{fair}}=
\frac{1+\tau_{\mathrm{fair}}}{2},
\qquad
\alpha_{\mathrm{fair}}=
\begin{cases}
1/\gamma_{\mathrm{fair}},&\gamma_{\mathrm{fair}}>0,\\
\infty,&\gamma_{\mathrm{fair}}=0.
\end{cases}
$$

The post-fair root prior is

$$
\widehat P_{\alpha}(x_0,a)=
\frac{P(x_0,a)^{\gamma_{\mathrm{fair}}}}
{\displaystyle\sum_{b\in\mathcal A(x_0)}
P(x_0,b)^{\gamma_{\mathrm{fair}}}},
\qquad a\in\mathcal A(x_0).
$$

When $\gamma_{\mathrm{fair}}=0$, every numerator in this expression is defined as one, producing the uniform distribution on $\mathcal A(x_0)$. The scheduler computes $\alpha_{\mathrm{fair}}$ once from the completed fair-stage values and uses the resulting prior throughout the remaining root allocation.

Every legal root action has a completed visit after the fair phase, so its post-fair action value is $Q(x_0,a)$. Root PUCT assigns

$$
S_{\mathrm{root}}(x_0,a)=
Q(x_0,a)
+c_{\mathrm{puct}}\left(\widetilde N(x_0)\right)
\widehat P_{\alpha}(x_0,a)
\frac{\sqrt{\widetilde N(x_0)+1}}
{1+\widetilde N(x_0,a)}
-l_vN_v(x_0,a).
$$

The scheduler enters the legal root action with the largest $S_{\mathrm{root}}$. Equal scores are resolved by the larger $\widehat P_{\alpha}$, the larger $Q$ and root expansion order, in that sequence. The common fair baseline cancels in pairwise differences between root-edge visit counts, while fixed-$\alpha$ PUCT creates the visit-count differences used by the final root Policy.

For bounded search with $N_{\mathrm{cap}}=N_{\mathrm{ref}}$ and fixed $M$,

$$
\lim_{N_{\mathrm{ref}}\rightarrow\infty}m_{\mathrm{fair}}=\infty,
\qquad
\lim_{N_{\mathrm{ref}}\rightarrow\infty}
\frac{Mm_{\mathrm{fair}}}{N_{\mathrm{ref}}}=0.
$$

The first limit increases the evidence collected for every legal root action, while the second leaves an asymptotically dominant fraction of the budget for fixed-$\alpha$ PUCT. If the procedure reaches its cap during the fair phase, the completed fair visits alone determine the root statistics and $\alpha_{\mathrm{fair}}$ remains undefined.

A configured deadline or a caller-supplied stop signal can end bounded search before it reaches $N_{\mathrm{cap}}$ or terminate an unbounded search. Every completed backup remains in the tree, and Section 5.8 specifies how unfinished reservations are released.

### 5.6 Internal Action Opening

For an expanded non-root state $x$, let $M_x=|\mathcal A(x)|$ and order its legal actions $a_{(1)},\ldots,a_{(M_x)}$ by decreasing original Policy prior $P(x,a)$. Actions with equal priors retain their order in the legal-action array. The selectable actions form a prefix of this ordering.

At the end of the fair phase, the calculation in Section 5.5 fixes $\gamma_{\mathrm{fair}}$ for the remaining simulations. Let $\beta_{\min}=1/2$ be the minimum internal-opening exponent. The exponent used at every expanded non-root state is

$$
\beta_{\mathrm{open}}=
\max\left(\beta_{\min},1-\gamma_{\mathrm{fair}}\right).
$$

Let $K(x)$ denote the active-prefix width and initialize it to zero when $x$ is expanded. Before each selection at $x$, the scheduler updates this width by

$$
K(x)\leftarrow
\max\left(
K(x),
\min\left(
M_x,
\left\lceil
\left(\widetilde N(x)+1\right)^{\beta_{\mathrm{open}}}
\right\rceil
\right)
\right).
$$

If the active prefix contains an action with $\widetilde N(x,a)=0$, the scheduler selects the first such action in Policy order. After every active action has a completed visit or virtual reservation, the allocation depends on whether the active prefix covers the complete legal-action set.

Once $K(x)=M_x$, the internal verification floor is

$$
m_{\mathrm{verify}}(x)=
\max\left(
1,
\left\lfloor
\frac{N(x)}{M_x\log(e+N(x))}
\right\rfloor
\right).
$$

For every action $a\in\mathcal A(x)$, its internal verification deficit is

$$
d_{\mathrm{verify}}(x,a)=
m_{\mathrm{verify}}(x)-\widetilde N(x,a).
$$

Whenever an action has a positive verification deficit, the scheduler selects an action with the largest deficit. The PUCT ordering from Section 5.3 resolves equal deficits. After all deficits become nonpositive, the same PUCT ordering allocates subsequent visits among the complete legal-action set.

In the active-width update, the outer maximum makes $K(x)$ nondecreasing, and the augmented count $\widetilde N(x)$ distributes concurrent reservations across newly active and under-verified actions. As $N(x)$ increases, the active prefix eventually reaches every legal action. Under a finite simulation budget, the visits accumulated at $x$ determine whether the prefix reaches full width and how far the subsequent verification proceeds. For fixed $M_x$,

$$
\lim_{N(x)\rightarrow\infty}m_{\mathrm{verify}}(x)=\infty,
\qquad
\lim_{N(x)\rightarrow\infty}
\frac{M_xm_{\mathrm{verify}}(x)}{N(x)}=0.
$$

The first limit increases the evidence collected for every legal action at a fully opened node, while the second leaves an asymptotically dominant fraction of the node visits for PUCT.

### 5.7 Evaluation Reuse

For cache keys, Gadus packs 18 physical-board binary planes into a 144-byte representation called `PackedState`: 12 piece planes, one side-to-move plane, four castling-right planes and one en-passant plane. It omits move counters and repetition history. The rules engine therefore checks whether a requested complete state is terminal before consulting the evaluation cache. Equal `PackedState` keys determine equal network inputs and can therefore share one compact Policy and Value record.

One MCTS invocation receives one or more root states and constructs a separate search tree for each root. These trees access one evaluation cache, allowing simulations within the same tree or across different trees to reuse completed network records. Let $M_C\geq0$ be the configured memory capacity for records retained across invocations. When $M_C=0$, the cache belongs only to the current invocation and is discarded with its search trees. When $M_C>0$, the cache persists across invocations and maintains a trajectory-aware least-recently-used (TLRU) order. A successful lookup moves the accessed record to the most-recent end of this order. An insertion places the new record at the same end.

TLRU records a directed link when a cached nonterminal child is reached from a cached parent. Let $\mathcal C$ be the set of retained entries and let $\mathcal T(x_0)$ be the node set of the completed tree rooted at $x_0$. For every visited node $v\in\mathcal T(x_0)$ whose network record remains in $\mathcal C$, write $\kappa(v)$ for the corresponding cache entry. When $N(x_0)>0$, the trajectory heat of entry $c$ is

$$
H_{x_0}(c)=
\sum_{\substack{v\in\mathcal T(x_0):\,N(v)>0\\\kappa(v)=c}}
\frac{N(v)}{N(x_0)}.
$$

The sum combines the contributions of distinct tree nodes whose encoded states map to the same cache entry. When one invocation searches several roots, TLRU adds the separately normalized heat contributions from their trees. This normalization gives each root one common visit-mass scale even when the trees complete different numbers of simulations.

After an invocation completes, TLRU assigns one common heat-generation identifier to the retained entries represented by its visited tree nodes and stores each entry's aggregated $H$. TLRU updates their positions in the recency order, processing lower heat before higher heat and, at equal heat, greater depth before lesser depth. High-heat entries and shallower entries at equal heat consequently finish nearer the most-recent end.

At the beginning of the next invocation, TLRU receives the complete states selected as its search roots. For every such root found in the cache, TLRU follows the retained parent-child links through entries that belong to the root's heat generation. It restricts the previous trajectory prediction to descendants of the actual root, combines the stored heat of entries reached from multiple roots and applies the same increasing-heat recency order before neural evaluation begins. Ordinary lookups and insertions then continue to update that order.

When the approximate retained-cache memory exceeds the configured capacity, TLRU removes entries from the least-recent end until the retained records fit within the limit. Its byte account includes the compact network arrays, fixed entry fields, a conservative allowance for container bookkeeping and the allocated capacity of recorded parent-child links. Tree nodes and search statistics belong to one invocation, so every invocation constructs fresh nodes with zero visits, zero accumulated evaluations and zero virtual reservations, initializes their network fields from cached records when available and computes their tree statistics through MCTS. Cache ordering affects the survival of exact network records under the memory limit, while root allocation, PUCT, backup and decision scores depend only on the current search tree.

### 5.8 Batched Evaluation

One MCTS invocation may receive several root states. The evaluator first assigns cached records to matching root nodes. It then groups the remaining roots by `PackedState`, evaluates the unique uncached states in one neural batch and assigns each resulting record to every matching root node.

The batching scheduler retains a latency estimate $\tau$, measured in milliseconds, from one invocation to the next. The estimate begins at zero. Let $\Delta t$ be the measured duration of a neural call that evaluates exactly one uncached state. Such a call supplies the sample

$$
\tau'=\max(0.05,\Delta t).
$$

The first sample sets $\tau=\tau'$, and each later sample updates the estimate by

$$
\tau\leftarrow\max\left(\tau',0.8\tau+0.2\tau'\right).
$$

For a capacity upper bound $B$ and $R$ milliseconds of remaining time, the deadline-limited capacity is

$$
D(R,B)=
\min\left(
B,
\max\left(
0,
\left\lfloor
\frac{R-2}{1.25\max(1,\tau)}
\right\rfloor
\right)
\right).
$$

The subtraction of 2 milliseconds and the factor 1.25 provide fixed and proportional timing margins. At the beginning of each selection-and-evaluation cycle, the available capacity is

$$
B_{\mathrm{cycle}}=
\begin{cases}
B_{\mathrm{batch}},&\text{if the invocation has no deadline},\\
D(R_{\mathrm{cycle}},B_{\mathrm{batch}}),&\text{if a deadline is active},
\end{cases}
$$

where $R_{\mathrm{cycle}}$ is the remaining time at the start of the cycle. If $B_{\mathrm{cycle}}=0$, the invocation ends before the cycle selects any leaves.

For $B_{\mathrm{cycle}}>0$, the cycle processes each search tree whose root count $N(x_0)$ is smaller than $N_{\mathrm{cap}}$. The maximum number of distinct nonterminal leaves requested from one such tree is

$$
m=\min\left(
B_{\mathrm{cycle}},
N_{\mathrm{cap}}-N(x_0)
\right).
$$

The two terms limit the request by the available cycle capacity and the remaining simulation budget.

To build the request set, the root scheduler first selects a legal root action with the largest positive fair-allocation deficit. After every legal root action completes the fair floor, it selects the action with the largest fixed-$\alpha$ root PUCT score defined in Section 5.5. During the fair phase, the selected root action is followed by full-width PUCT at every expanded non-root node. After the fair phase fixes $\gamma_{\mathrm{fair}}$, the scheduler instead updates the active prefix at each expanded non-root node according to Section 5.6. At each such node, an active action without a visit takes precedence. If every active action has been visited and the prefix has reached full width, the scheduler next considers positive verification deficits. When neither condition applies, it uses the PUCT ordering from Section 5.3. This descent ends at a terminal node or an unexpanded nonterminal node.

A terminal node receives its exact rule outcome, and immediate backup completes one simulation without adding a neural-evaluation request. An unexpanded nonterminal node enters the request set when that tree has not reserved the node earlier in the same cycle, and its selected path retains one virtual visit until the request is resolved. Immediate terminal backups and accepted neural-leaf reservations both count toward the limit $m$. If another attempt from the same tree reaches an already reserved node, the selector releases the virtual visits introduced by that attempt and ends request selection for that tree in the current cycle. The accepted requests are evaluated and backed up before the next cycle begins. The resulting updates prevent the selector from immediately reconstructing the same deterministic path after the temporary reservations are released. The tree performs at most $\max(5m,m+8)$ attempts while scheduling at most $m$ simulations, and a repeated reservation can end its current selection pass earlier.

The requests collected from all trees form one list. Before each evaluation submission, the scheduler computes

$$
B_{\mathrm{call}}=
\begin{cases}
B_{\mathrm{cycle}},&\text{if the invocation has no deadline},\\
D(R_{\mathrm{call}},B_{\mathrm{cycle}}),&\text{if a deadline is active},
\end{cases}
$$

where $R_{\mathrm{call}}$ is the remaining time before that submission. The next submission contains at most $B_{\mathrm{call}}$ requests. If this capacity is zero or a stop signal has been received, the scheduler releases the virtual visits attached to all remaining requests and ends batch submission.

For each submitted group, the evaluation-reuse mechanism from Section 5.7 resolves cache hits before the evaluator groups the unresolved requests by `PackedState`. When unresolved requests remain, their unique states form one neural batch, and one computed record serves every request in the corresponding group.

For a representative leaf state $x$, let $L=|\mathcal A(x)|$, and write its ordered legal actions as $a_1,\ldots,a_L$. The compact evaluation record contains the three sequences

$$
(a_j)_{j=1}^{L},\qquad
(\widetilde i_G(a_j))_{j=1}^{L},\qquad
(P_\theta(a_j\mid\phi_G(x)))_{j=1}^{L},
$$

together with the scalar $V_\theta(\phi_G(x))$. Entries with the same index $j$ describe the same legal action, which defines the alignment among the action, action-index and Policy sequences. Before neural evaluation, the evaluator converts each physical index $\widetilde i_G(a_j)$ to the canonical index $i_G(x,a_j)$ required by the network.

After the neural batch returns, each newly computed record enters the active cache. The evaluator then assigns a cached or computed record to every request with the matching `PackedState`. For each requested leaf, tree expansion creates one outgoing edge for every $a_j$ and assigns $P_\theta(a_j\mid\phi_G(x))$ as that edge's prior. The scalar $V_\theta(\phi_G(x))$ is then backed up along the leaf's reserved path, so different trees can share a network record while retaining separate nodes, paths and search statistics.

After the submitted leaves complete their backups, another cycle begins only when at least one tree remains below $N_{\mathrm{cap}}$ and the preceding cycle completed at least one backup. The updated completed-visit counts and action evaluations determine the fair-allocation deficits and PUCT scores used by the next cycle. The first cycle following completion of the fair floor computes $\alpha_{\mathrm{fair}}$, whose fixed value determines the root prior throughout the remainder of that invocation.

### 5.9 Root Evaluation and Policy

After the simulation phase, the root evaluation is determined by whether any simulation has completed:

$$
V_{\mathrm{root}}(x_0)=
\begin{cases}
V_\theta(s_0),&N(x_0)=0,\\
Q(x_0),&N(x_0)>0.
\end{cases}
$$

The completed root-edge visits and original priors define the root Policy distribution

$$
P_{\mathrm{root}}(a\mid s_0)=
\frac{N(x_0,a)+P(x_0,a)}
{\displaystyle\sum_{b\in\mathcal A(x_0)}
\left(N(x_0,b)+P(x_0,b)\right)},
\qquad a\in\mathcal A(x_0).
$$

The denominator normalizes the visit-plus-prior weights of all legal root actions. When $N(x_0)=0$, all visit terms vanish, and this distribution equals the legal-move Policy

$$
P_{\mathrm{root}}(a\mid s_0)=P_\theta(a\mid s_0).
$$

The visit-based distribution $P_{\mathrm{root}}$ records the allocation produced by the fair visit floor and subsequent fixed-$\alpha$ root PUCT. It also supplies the base ordering used by the decision components in Section 5.10. Every legal root action retains positive weight because its original prior is included alongside its completed visits.

When an output requires one probability for every physical-board action index, Gadus expands the compact root distribution into

$$
P_{\mathrm{dense}}(i\mid s_0)=
\begin{cases}
P_{\mathrm{root}}(a\mid s_0),
&i=\widetilde i_G(a)\text{ for }a\in\mathcal A(x_0),\\
0,&i\notin\lbrace \widetilde i_G(a):a\in\mathcal A(x_0)\rbrace.
\end{cases}
$$

This conversion is performed only for a final result or a progress report that requires the fixed action-index space $\mathcal I_G$.

### 5.10 Decision Components

The decision layer can apply two optional transformations to the final move ordering. It begins with a copy of the root Policy defined in Section 5.9:

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

IMF and RPP transform a copy of $D_0$, so the network probabilities, edge priors and tree statistics keep their computed values. The decision scores are ordering quantities rather than a probability distribution. Gadus orders legal actions by decreasing $D(a)$, then by decreasing $D_0(a)$ and finally by decreasing UCI move string. The first action in this deterministic ordering becomes the selected move.
