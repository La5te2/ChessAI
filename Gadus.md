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
- $\mathcal J_G=\lbrace0,\ldots,4671\rbrace$ is the expanded action-index set formed by 64 source squares and 73 motion patterns.
- $\widetilde j_G(a)\in\mathcal J_G$ is the expanded index of legal action $a$ in physical-board coordinates.
- $j_G(x,a)\in\mathcal J_G$ is the expanded index of $a$ after the side-to-move transformation of state $x$.
- $\mathcal J_G^\star\subset\mathcal J_G$ is the set of expanded indices whose source-pattern pairs have valid destinations in canonical board coordinates.
- $\kappa:\mathcal J_G^\star\rightarrow\mathcal I_G$ is the order-preserving bijection from the valid expanded indices to the compact action-index set.
- $\mathcal I_G=\lbrace0,\ldots,1857\rbrace$ is the compact action-index set.
- $i_G(x,a)=\kappa(j_G(x,a))$ is the compact canonical index of legal action $a\in\mathcal A(x)$.
- $\theta$ denotes the trainable network parameters.
- $\ell_\theta(s)\in\mathbb R^{1858}$ is the complete vector of Policy logits produced by the network with parameters $\theta$, and $\ell_\theta(s,i)$ is its scalar component for action index $i\in\mathcal I_G$.
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

The legal-action set $\mathcal A(x)$ varies with the complete state $x$, whereas the Policy head requires a fixed output domain. The action encoder first embeds each legal action in the expanded set $\mathcal J_G$, applies the same side-to-move transformation used by the state encoder and then maps the valid canonical source-pattern pairs bijectively to $\mathcal I_G$.

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

The eight sliding directions are ordered as

$$
(-1,-1),\ (-1,0),\ (-1,1),\ (0,-1),\
(0,1),\ (1,-1),\ (1,0),\ (1,1).
$$

Let $(u_d,v_d)$ denote the direction at position $d\in\lbrace0,\ldots,7\rbrace$ in this sequence. A move that travels $m\in\lbrace1,\ldots,7\rbrace$ squares in that direction has displacement

$$
(\Delta r,\Delta f)=m(u_d,v_d).
$$

Its motion-pattern index is

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

This construction assigns indices 64 through 72 to the $3\times3=9$ underpromotion patterns. A queen promotion uses the corresponding one-square rank, file or diagonal pattern because the action itself retains the promoted piece.

Castling is encoded by the king displacement from its source square to `g1`, `c1`, `g8` or `c8`. A move representation that denotes castling with the rook square as its destination is therefore converted to the corresponding king destination before the motion pattern is computed.

Once $q$ and $p$ have been determined, the expanded physical-board index is

$$
\widetilde j_G(a)=73q+p.
$$

The expanded action-index set therefore has cardinality

$$
|\mathcal J_G|=64\times73=4672.
$$

For $a\in\mathcal A(x)$, the canonical expanded index $j_G(x,a)$ equals $\widetilde j_G(a)$ when White is to move. When Black is to move, the source and destination ranks are reflected by $r\mapsto7-r$ before the same source-pattern encoding is applied. The reflection changes every rank-sensitive sliding or knight pattern while preserving the file displacement and promoted piece of an underpromotion.

An expanded index belongs to $\mathcal J_G^\star$ precisely when its source-pattern pair has an on-board destination in canonical coordinates. An underpromotion index additionally requires the source square to have canonical rank coordinate $6$. The valid set therefore has cardinality

$$
|\mathcal J_G^\star|
=1456+336+66
=1858.
$$

The three terms count the valid sliding, knight and underpromotion indices, respectively. Their inherited order defines the bijection $\kappa:\mathcal J_G^\star\rightarrow\mathcal I_G$ by

$$
\kappa(j)=
\left|
\left\{t\in\mathcal J_G^\star:t<j\right\}
\right|,
\qquad
j\in\mathcal J_G^\star.
$$

The compact canonical index of legal action $a$ is thus

$$
i_G(x,a)=\kappa(j_G(x,a))\in\mathcal I_G.
$$

For complete state $x$, the indices available to the legal-move Policy are

$$
\mathcal I_G(x)=
\lbrace i_G(x,a)\mid a\in\mathcal A(x)\rbrace.
$$

Decoding an available compact index in state $x$ selects the unique action $a\in\mathcal A(x)$ with that index. Because the selection is made within the legal-action set, the decoded action retains its castling, en passant and promotion semantics.

## 3. Network

### 3.1 Residual Trunk

A shared chess-structured residual trunk produces the representation used by both output heads. Let $C$ be the trunk width and let $B$ be its number of residual blocks. The stem maps the 17 canonical input planes to $C$ board-aligned feature channels, and a learned square embedding supplies an independent $C$-dimensional offset at each canonical square. With bias-free convolution, batch normalization and elementwise ReLU denoted by $\mathrm{Conv}$, $\mathrm{BN}$ and $\mathrm{ReLU}$, the initial feature tensor is

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

For every $(a,b)\in\mathcal D$, define the displacement indicator $T_{a,b}$ and its support size $n_{a,b}$ by

$$
T_{a,b}(q,p)=\mathbf 1[\delta(q,p)=(a,b)],
\qquad
n_{a,b}=\lVert T_{a,b}\rVert_F^2=(8-|a|)(8-|b|).
$$

The corresponding normalized displacement atom is

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

This decomposition is unique. The relative component assigns one shared coefficient to each displacement class, whereas the constrained residual represents only differences among square pairs within the same class. Consequently, the relative term is determined by 225 coefficients indexed by the $15\times15$ displacement set. Evaluating the coefficient associated with each square pair by its displacement avoids representing the 225 basis matrices explicitly.

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
+\beta^B_{j,g}V^B_s.
$$

The learned scalars $\beta^R_{j,g}$ and $\beta^B_{j,g}$ weight the two visibility relations. The matrices $V^R_s$ and $V^B_s$ are constructed once for each encoded position and shared by every residual block that processes that position.

By distributivity, applying $A_{j,g}(s)$ to $b_{j,g}$ is equivalent to applying its static, rook-visibility and bishop-visibility terms separately and then adding the three products. For a minibatch of $N_b$ positions, parameter optimization uses the separated form and thereby avoids forming an $N_b\times K\times64\times64$ position-dependent relation tensor for every block. Network evaluation first combines the three matrices and then performs one grouped relation multiplication. Both orders produce the same relation output.

Define the relation-group count $K$ and group width $d$ by

$$
K=\gcd(C,8),
\qquad
d=\frac{C}{K}.
$$

Thus each residual block contains $K$ computational channel groups. The channels of $b_j$ are divided into groups of width $d$, and each group has its own $A_{j,g}(s)\in\mathbb R^{64\times64}$.

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

Relation groups are initialized from the ordered sequence

$$
E_I,\ E_K,\ E_N,\ E_{P,m},\ E_{P,c},\ E_G,\ V^R,\ V^B.
$$

The eight members of this sequence represent identity, king displacement, knight displacement, canonical friendly-pawn advance, canonical friendly-pawn capture, complete-board communication, unobstructed rook geometry and unobstructed bishop geometry, respectively. The first $K$ members initialize the $K$ relation groups.

For a static offset set $E\subseteq\mathcal D$, define

$$
M_E=\sum_{(a,b)\in E}T_{a,b},
\qquad
\overline M_E=\frac{M_E}{\lVert M_E\rVert_F}.
$$

A group initialized from $E$ represents $\overline M_E$ by setting

$$
\alpha_{j,g,a,b}=
\begin{cases}
\sqrt{n_{a,b}}/\lVert M_E\rVert_F,&(a,b)\in E,\\
0,&(a,b)\notin E.
\end{cases}
$$

The six static offset sets are

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

The canonical displacement coordinates define all six sets. In the two pawn sets, rank displacement $+1$ points forward for the friendly side. Relation group $6$ begins with $\beta^R_{j,6}=1$ when $K>6$, and relation group $7$ begins with $\beta^B_{j,7}=1$ when $K>7$. Every coefficient and residual entry not assigned by these rules begins at zero.

Each block begins with $\lambda_j=\mathbf1$, so the local and full-board paths initially have the common channelwise coefficient $1/\sqrt2$. The affine scales of the three local-branch normalization layers begin at $1/\sqrt3$, and their biases begin at zero. The final normalization scale is initialized to $1/\sqrt B$ in every shared-trunk block and to $1/\sqrt{D_P}$ in every Policy block; the corresponding biases begin at zero. In the Value sequence, each nonfinal block begins with unit final scale, while the final block begins with zero final scale; all corresponding biases begin at zero. The final Value block is therefore an identity map at initialization.

The square embedding $E_S$ begins at zero. The displacement atoms and ray geometry are fixed, while the square embedding and all block-specific convolutional, normalization, relation and path-balance parameters are trainable.

After the $B$ blocks, $h_B\in\mathbb R^{C\times8\times8}$ is the shared representation supplied to the Policy and Value heads defined below.

### 3.2 Policy and Value Heads

Let $C_P,C_V,H_V\in\mathbb N_{>0}$ denote the Policy width, Value feature width and Value hidden width. Let $D_P,D_V\in\mathbb N_{>0}$ denote the numbers of Policy-specific and Value-specific chess-structured residual blocks.

The Policy head assigns one unnormalized score, called a logit, to every compact canonical action index in $\mathcal I_G$. It begins with a bias-free $1\times1$ convolution from $C$ channels to $C_P$ channels, followed by batch normalization and ReLU:

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

Flattening the board dimensions of $u_{D_P}$ gives a feature vector $u(q)\in\mathbb R^{C_P}$ at each canonical square $q$. Every compact action index $i\in\mathcal I_G$ has the unique expanded representation

$$
\kappa^{-1}(i)=73q_i+p_i.
$$

This representation identifies the source square $q_i$ and motion pattern $p_i$. Together they determine an on-board destination square $d_i$ because $\kappa^{-1}(i)\in\mathcal J_G^\star$.

Each motion pattern $p$ has source, destination and interaction vectors $w_p,v_p,g_p\in\mathbb R^{C_P}$. A shared matrix $M\in\mathbb R^{C_P\times C_P}$ transforms destination features. Its rows are normalized independently:

$$
\widehat M_{c,:}=
\frac{M_{c,:}}
{\max\left(\lVert M_{c,:}\rVert_2,10^{-12}\right)}.
$$

Let $b\in\mathbb R^{1858}$ be the raw compact-action bias. Removing its common component gives

$$
\widetilde b_i=b_i-\frac{1}{1858}\sum_{t\in\mathcal I_G}b_t.
$$

The resulting Policy logit is

$$
\ell_\theta(s,i)=
w_{p_i}^{\mathsf T}u(q_i)
+v_{p_i}^{\mathsf T}u(d_i)
+g_{p_i}^{\mathsf T}
\left[
u(q_i)\odot\left(\widehat M u(d_i)\right)
\right]
+\widetilde b_i.
$$

Here $\odot$ denotes elementwise multiplication. The source and destination terms score the two endpoint features separately. The interaction term scores their channelwise products after the shared destination transformation. The centered bias represents relative preferences among compact actions without introducing an unidentifiable common logit offset.

The destination and interaction vectors begin at zero, and $M$ begins as the identity matrix. Consequently, the initial readout consists of the source term and centered bias; training introduces the destination and interaction contributions.

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
\ell_\theta(s)\in\mathbb R^{1858},
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

The Policy readout can be evaluated over either the complete compact set $\mathcal I_G$ or the legal subset $\mathcal I_G(x)$. Evaluation over $\mathcal I_G$ first decomposes every compact index $i$ into the source square $q_i$, motion pattern $p_i$ and destination square $d_i$ defined in Section 3.2. Applying the Policy formula to these triples produces all 1858 compact logits without constructing the geometrically invalid members of $\mathcal J_G$.

Evaluation over $\mathcal I_G(x)$ applies the same formula only to the indices $i_G(x,a)$ of legal actions $a\in\mathcal A(x)$. When states with different legal-action counts are evaluated together, absent positions are excluded from the softmax normalization. Both domains therefore assign the same logit to every legal action, while the restricted domain omits actions unavailable in the corresponding state.

After the evaluation domain has been chosen, the remaining network operations admit exact algebraic simplifications. Once the batch-normalization statistics have been fixed, each convolution followed directly by affine batch normalization is replaced by the equivalent biased convolution. Within a chess-structured block, the $1\times1$ kernel and the normalized identity kernel are embedded at the center of a $3\times3$ kernel and added to the normalized $3\times3$ branch. This produces one biased local convolution with the same output as the three training branches.

Each static relation matrix $A^{\mathrm{static}}_{j,g}$ is then evaluated once from its displacement coefficients and constrained residual. For each position, $V^R_s$ and $V^B_s$ are constructed once and shared by all blocks. The fixed statistics of the non-affine relation normalization and the learned path-balance vector determine a local scale, a relation-centering term and a relation scale. The local scale and centering term are absorbed into the fused local convolution, while the relation scale is applied channelwise after grouped relation multiplication.

These algebraic substitutions preserve the network function. They remove repeated affine normalization, local-branch summation, static-relation reconstruction, relation normalization and path-balance square roots from evaluation.

## 4. Supervised Training

### 4.1 Supervised Data

Let $\mathcal D_{\mathrm{sup}}$ be a supervised dataset containing $N$ records:

$$
\mathcal D_{\mathrm{sup}}=
\lbrace \xi_n\rbrace_{n=1}^{N}.
$$

Each record is associated with a complete pre-move state $x_n$ and a selected legal action $a_n\in\mathcal A(x_n)$. Its three components are

$$
\xi_n=(s_n,i_n,y_n).
$$

The three components satisfy

$$
s_n=\phi_G(x_n),
\qquad
i_n=i_G(x_n,a_n),
\qquad
y_n\in[-1,1].
$$

The encoded state $s_n$ is the canonical network input, and the compact action index $i_n$ is the corresponding canonical Policy target. The scalar $y_n$ is the Value target, expressed as an estimate of the expected game result from the perspective of the side to move in $x_n$. On this scale, $-1$ denotes a loss, $0$ denotes a draw and $1$ denotes a win, while intermediate values express expectations between these outcomes.

### 4.2 Supervised Objective

For an encoded state $s$, the Policy head produces the compact logit vector

$$
\ell_\theta(s)=
\left(\ell_\theta(s,i)\right)_{i\in\mathcal I_G}
\in\mathbb R^{1858}.
$$

Supervised training normalizes this vector over the complete compact action set:

$$
R_\theta(i\mid s)=
\frac{\exp\ell_\theta(s,i)}
{\displaystyle\sum_{j\in\mathcal I_G}\exp\ell_\theta(s,j)},
\qquad i\in\mathcal I_G.
$$

The distributions $R_\theta$ and $P_\theta$ differ only in their normalization domains. The supervised distribution $R_\theta$ includes every compact action index, whereas the legal-move distribution $P_\theta$ includes only the indices $i_G(x,a)$ for $a\in\mathcal A(x)$. Both distributions are therefore derived from the same logits.

For a minibatch $\mathcal B\subseteq\mathcal D_{\mathrm{sup}}$, the Policy objective is the mean negative log-probability of the target compact indices:

$$
L_{P,\mathrm{sup}}^{(\mathcal B)}=
-\frac{1}{|\mathcal B|}
\sum_{(s,i,y)\in\mathcal B}\log R_\theta(i\mid s).
$$

The Value objective on the same minibatch is the mean squared error of the predicted expected results:

$$
L_{V,\mathrm{sup}}^{(\mathcal B)}=
\frac{1}{|\mathcal B|}
\sum_{(s,i,y)\in\mathcal B}
\left(V_\theta(s)-y\right)^2.
$$

The two objectives are combined through a positive Value coefficient. Its fixed bounds are

$$
w_{\min}=0.2,
\qquad
w_{\max}=2.
$$

The initial value satisfies $w_{V,0}\in[w_{\min},w_{\max}]$. At optimizer step $k\geq1$, the minibatch objective is

$$
L_{\mathrm{sup}}^{(\mathcal B_k)}=
L_{P,\mathrm{sup}}^{(\mathcal B_k)}+
w_{V,k}L_{V,\mathrm{sup}}^{(\mathcal B_k)}.
$$

The coefficient controller compares the two objective gradients within the shared trunk. With $\theta_T$ denoting all shared-trunk parameters, define

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

The norm-balanced proposal targets the fixed gradient-norm ratio $\eta_g=0.5$. With $\epsilon_g=10^{-12}$ preventing division by zero, it is

$$
\widetilde w_{V,k}=
\mathrm{clip}_{[w_{\min},w_{\max}]}
\left(\eta_g\frac{n_{P,k}}{n_{V,k}+\epsilon_g}\right).
$$

A negative inner product $c_k$ indicates first-order disagreement between the two objectives. In that case, a descent step along $g_{P,k}+w_{V,k}g_{V,k}$ decreases both minibatch losses to first order exactly when

$$
\frac{-c_k}{n_{V,k}^2}<w_{V,k}<\frac{n_{P,k}^2}{-c_k}.
$$

The controller retains an interior margin $\delta_g=10^{-3}$ and intersects the resulting interval with the admissible coefficient range:

$$
I_k=
\left[
\max\left(w_{\min},(1+\delta_g)\frac{-c_k}{n_{V,k}^2}\right),
\min\left(w_{\max},(1-\delta_g)\frac{n_{P,k}^2}{-c_k}\right)
\right].
$$

A nonnegative inner product imposes no additional first-order restriction; hence $I_k=[w_{\min},w_{\max}]$ when $c_k\geq0$. At a coefficient probe, the controller updates the coefficient only when the gradient statistics are finite, both squared norms exceed $\epsilon_g$ and $I_k$ is nonempty. It first projects the proposal onto $I_k$ by nearest-point projection:

$$
\widehat w_{V,k}=\operatorname{proj}_{I_k}(\widetilde w_{V,k}).
$$

The controller then smooths the projected value in logarithmic space with $\gamma=0.08$:

$$
w_{V,k}=
\mathrm{clip}_{[w_{\min},w_{\max}]}
\left[
\exp\left((1-\gamma)\log w_{V,k-1}+\gamma\log\widehat w_{V,k}\right)
\right].
$$

If any update condition fails, the probe leaves $w_{V,k}$ equal to $w_{V,k-1}$. Steps without a probe preserve the same equality.

Partition the network parameters as $\theta=(\theta_T,\theta_P,\theta_V)$, where $\theta_T$ belongs to the shared trunk and $\theta_P,\theta_V$ belong exclusively to the Policy and Value heads. The objective gradients for these three groups are

$$
\begin{aligned}
\nabla_{\theta_P}L_{\mathrm{sup}}^{(\mathcal B_k)}=
\nabla_{\theta_P}L_{P,\mathrm{sup}}^{(\mathcal B_k)},\\
\nabla_{\theta_V}L_{\mathrm{sup}}^{(\mathcal B_k)}=
w_{V,k}\nabla_{\theta_V}L_{V,\mathrm{sup}}^{(\mathcal B_k)},\\
\nabla_{\theta_T}L_{\mathrm{sup}}^{(\mathcal B_k)}=
g_{P,k}+w_{V,k}g_{V,k}.
\end{aligned}
$$

Thus the head-specific parameters receive only their corresponding objective gradients, while the shared trunk receives their weighted sum.

### 4.3 Parameter Optimization

Before optimization begins, a stratified target sample $\mathcal D_0\subseteq\mathcal D_{\mathrm{sup}}$ determines the output priors. Define

$$
c_i=
\sum_{(s,j,y)\in\mathcal D_0}\mathbf 1[j=i],
\qquad
\bar y=
\frac{1}{|\mathcal D_0|}
\sum_{(s,j,y)\in\mathcal D_0}y.
$$

With smoothing pseudocount $a_0=1$, the initial compact-action prior is

$$
\pi_0(i)=
\frac{c_i+a_0}
{\displaystyle\sum_{j\in\mathcal I_G}c_j+a_0|\mathcal I_G|},
\qquad i\in\mathcal I_G.
$$

The raw compact-action bias is initialized by

$$
b_i=\log\pi_0(i).
$$

The Policy readout uses the centered bias $\widetilde b_i$ defined in Section 3.2. Because subtracting a common scalar from every logit leaves softmax unchanged, this centering preserves the distribution $\pi_0$. After the remaining parameters have received their ordinary random initial values, the Policy source vectors $w_p$ and the Value output weights $W_{V,2}$ are multiplied by $\rho_0=0.1$. The Value output bias is initialized by

$$
b_{V,2}=
\operatorname{atanh}
\left(
\operatorname{clip}_{[-1+\epsilon_0,\,1-\epsilon_0]}(\bar y)
\right),
\qquad \epsilon_0=10^{-4}.
$$

These choices place the initial Policy logits near the sampled relative action frequencies and the initial Value estimate near the sampled target mean. Scaling the two output paths limits the random deviation from those priors.

Each randomized traversal of $\mathcal D_{\mathrm{sup}}$ is partitioned into minibatches. At step $k$, the gradients of the objective in Section 4.2 are computed for $\mathcal B_k$. Before the parameter update, the Value-head gradient is clipped to Euclidean norm $G_V=1$; the Policy-head and shared-trunk gradients are not clipped by this rule. Let

$$
g_{V,k}^{\mathrm{head}}=
\nabla_{\theta_V}
L_{\mathrm{sup}}^{(\mathcal B_k)}.
$$

The Value-head gradient supplied to AdamW is

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

AdamW applies the Policy-head and shared-trunk gradients from Section 4.2 together with $\widetilde g_{V,k}^{\mathrm{head}}$, producing an intermediate parameter state. The relation residuals in that state are then projected onto the displacement-orthogonal subspace. For intermediate residual $Z_{j,g}$, define

$$
\Delta A^{\perp}_{j,g}(q,p)=Z_{j,g}(q,p)
-\frac{1}{n_{a,b}}
\sum_{\delta(u,v)=(a,b)}Z_{j,g}(u,v),
\qquad
(a,b)=\delta(q,p).
$$

The projection restores the zero-sum constraint within every displacement class. It therefore prevents the displacement coefficients and the absolute residual from representing the same direction. The projected parameter state is $\theta^{(k+1)}$.

## 5. Search

### 5.1 Root Initialization

For a nonterminal complete state $x_0$, MCTS initializes a tree rooted at $x_0$. Each node represents one complete state $x$, and each outgoing edge represents a legal action $a\in\mathcal A(x)$ that connects the node to the child state $T(x,a)$. A simulation follows a sequence of edges to a leaf, evaluates that leaf and propagates the result back along the same path.

Root initialization evaluates $s_0=\phi_G(x_0)$ to obtain $P_\theta(\cdot\mid s_0)$ and $V_\theta(s_0)$. Before any simulation completes, $V_\theta(s_0)$ supplies the reported root evaluation, and the root statistics satisfy $N(x_0)=W(x_0)=0$. The procedure then creates one outgoing edge and child node for every legal action $a\in\mathcal A(x_0)$, using its Policy probability as the edge prior:

$$
P(x_0,a)=P_\theta(a\mid s_0).
$$

A nonterminal node with no outgoing edges is called an unexpanded node. Expansion of the node representing $x$ begins by evaluating $\phi_G(x)$, which yields $P_\theta(\cdot\mid\phi_G(x))$ and $V_\theta(\phi_G(x))$. The expansion then creates one edge and child node for every $a\in\mathcal A(x)$ and assigns

$$
P(x,a)=P_\theta(a\mid\phi_G(x)).
$$

The Value estimate $V_\theta(\phi_G(x))$ becomes the leaf evaluation propagated by the backup rule in Section 5.4.

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

The exploration coefficient follows a logarithmic schedule in the augmented parent count. Let $c_0$, $b_0$ and $f_0$ denote its initial coefficient, schedule base and schedule factor. Define the effective parameters by

$$
b=\max(1,b_0),
\qquad
f=\max(0,f_0).
$$

The exploration schedule is

$$
c_{\mathrm{puct}}(n)=
\max\left(
0,
c_0+f\log\left(\frac{n+b+1}{b}\right)
\right).
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

Define the legal root width by

$$
M_0=|\mathcal A(x_0)|.
$$

Let $N_{\mathrm{ref}}\geq0$ denote the reference budget and let $N_{\mathrm{cap}}\in\mathbb N_0\cup\lbrace\infty\rbrace$ denote the simulation cap. The reference budget determines the fair visit floor, whereas the cap determines when simulation must stop. Bounded search sets $N_{\mathrm{ref}}=N_{\mathrm{cap}}$. Unbounded search retains a positive $N_{\mathrm{ref}}$ and sets $N_{\mathrm{cap}}=\infty$. No simulation is performed when $N_{\mathrm{ref}}=0$.

For $N_{\mathrm{ref}}>0$, the scale $\kappa_{\mathrm{fair}}=10$ defines the fair visit floor:

$$
m_{\mathrm{fair}}=
\max\left[
1,
\left\lfloor
\kappa_{\mathrm{fair}}\log\left(
1+\frac{N_{\mathrm{ref}}}{\kappa_{\mathrm{fair}}M_0}
\right)
\right\rfloor
\right].
$$

The scale $\kappa_{\mathrm{fair}}$ places the transition between two regimes. The floor is approximately linear in the reference budget per root action, $N_{\mathrm{ref}}/M_0$, when that quantity is small, while its growth becomes logarithmic as the reference budget increases. Every legal root action receives this floor, so a completed fair phase uses $M_0m_{\mathrm{fair}}$ root visits.

Each fair-stage simulation first enters the root action selected by the allocation deficit. At every expanded nonterminal state reached afterward, all legal actions participate in the PUCT ordering from Section 5.3. The original Policy prior therefore governs the complete legal-action set throughout the fair phase. During post-fair allocation, root selection uses the tempered prior defined below, and expanded non-root nodes use the action-opening and verification rules defined in Section 5.6.

For $a\in\mathcal A(x_0)$, the fair-allocation deficit is

$$
d_{\mathrm{fair}}(x_0,a)=
\max\left(
0,
m_{\mathrm{fair}}-\widetilde N(x_0,a)
\right).
$$

While at least one legal root action has a positive deficit, the next simulation enters an action with the largest deficit. The PUCT ordering from Section 5.3 resolves equal deficits, and root expansion order resolves any remaining equality. Completed visits and virtual reservations both reduce the deficit. Counting both quantities distributes concurrent requests before their neural evaluations return. Post-fair allocation begins only after every legal root action has completed $m_{\mathrm{fair}}$ visits; pending virtual reservations therefore contribute to batch formation but not to the fair-stage estimates.

Let $Q_{\mathrm{fair}}(x_0,a)$ denote the empirical value of root action $a$ when the fair phase ends. Over all unordered pairs of legal root actions, let $n_C$ count pairs ordered alike by the Policy prior and fair-stage value, and let $n_D$ count pairs ordered oppositely. Let $n_P$ count pairs tied only in Policy, and let $n_Q$ count pairs tied only in fair-stage value. Their tie-corrected normalization is

$$
Z=\sqrt{(n_C+n_D+n_P)(n_C+n_D+n_Q)}.
$$

The fair-stage rank agreement is

$$
\tau_{\mathrm{fair}}=
\begin{cases}
\dfrac{n_C-n_D}{Z},&Z>0,\\[6pt]
0,&Z=0.
\end{cases}
$$

The rank agreement determines the root-prior exponent and its reciprocal temperature:

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

The exponent defines the post-fair root prior

$$
\widehat P_{\alpha}(x_0,a)=
\frac{P(x_0,a)^{\gamma_{\mathrm{fair}}}}
{\displaystyle\sum_{b\in\mathcal A(x_0)}
P(x_0,b)^{\gamma_{\mathrm{fair}}}},
\qquad a\in\mathcal A(x_0).
$$

At $\gamma_{\mathrm{fair}}=0$, every numerator is defined as one, so $\widehat P_{\alpha}$ is uniform. The completed fair-stage values fix $\gamma_{\mathrm{fair}}$, equivalently $\alpha_{\mathrm{fair}}$, once; the resulting prior remains unchanged during all subsequent root allocation.

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

For bounded search with $N_{\mathrm{cap}}=N_{\mathrm{ref}}$ and fixed $M_0$,

$$
\lim_{N_{\mathrm{ref}}\rightarrow\infty}m_{\mathrm{fair}}=\infty,
\qquad
\lim_{N_{\mathrm{ref}}\rightarrow\infty}
\frac{M_0m_{\mathrm{fair}}}{N_{\mathrm{ref}}}=0.
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

Selection within the active prefix follows three priorities. An action with $\widetilde N(x,a)=0$ takes precedence, with Policy order resolving multiple unvisited actions. Once every active action has a completed visit or virtual reservation, PUCT governs the active prefix until it spans all $M_x$ legal actions.

A fully opened node, characterized by $K(x)=M_x$, receives the verification floor

$$
m_{\mathrm{verify}}(x)=
\max\left[
1,
\left\lfloor
\frac{N(x)}{M_x\log(e+N(x))}
\right\rfloor
\right].
$$

The corresponding deficit of action $a$ is

$$
d_{\mathrm{verify}}(x,a)=
m_{\mathrm{verify}}(x)-\widetilde N(x,a).
$$

A positive verification deficit takes precedence over PUCT. The largest deficit is selected first, with the PUCT ordering from Section 5.3 resolving equal deficits. Once every deficit is nonpositive, PUCT allocates all later visits over the complete legal-action set.

In the active-width update, the outer maximum makes $K(x)$ nondecreasing, and the augmented count $\widetilde N(x)$ distributes concurrent reservations across newly active and under-verified actions. As $N(x)$ increases, the active prefix eventually reaches every legal action. Under a finite simulation budget, the visits accumulated at $x$ determine whether the prefix reaches full width and how far the subsequent verification proceeds. For fixed $M_x$,

$$
\lim_{N(x)\rightarrow\infty}m_{\mathrm{verify}}(x)=\infty,
\qquad
\lim_{N(x)\rightarrow\infty}
\frac{M_xm_{\mathrm{verify}}(x)}{N(x)}=0.
$$

The first limit increases the evidence collected for every legal action at a fully opened node, while the second leaves an asymptotically dominant fraction of the node visits for PUCT.

### 5.7 Evaluation Reuse

Evaluation reuse uses the physical-board key

$$
\chi(x)\in\{0,1\}^{18\times64}.
$$

The key contains 18 binary planes: 12 piece planes, one side-to-move plane, four castling-right planes and one en-passant plane. Bit packing represents this key in $18\cdot64=1152$ bits, or 144 bytes. Move counters and repetition history are absent because they do not enter $\phi_G(x)$. Although omitted from the key, these fields may affect terminal status, so the rules of chess determine whether $x$ is terminal before $\chi(x)$ is used for evaluation reuse. Equal keys determine equal network inputs and may share one exact Policy-Value record.

A search call may contain several root states. Each root has a separate MCTS tree, while all trees share one evaluation cache. Let $M_C\geq0$ denote the memory capacity retained between calls. At $M_C=0$, an unbounded local cache serves only the active call and is discarded with its trees. At $M_C>0$, a capacity-limited cache survives between calls and orders its entries by trajectory-aware least recent use (TLRU). Both a successful lookup and a new insertion move the affected entry to the most-recent end.

The persistent cache records a directed link from a cached parent entry to a cached nonterminal child entry whenever search traverses that evaluated transition. Let $\mathcal C$ be the retained entry set and $\mathcal T(x_0)$ the completed search tree rooted at $x_0$. For a visited tree node $v$ whose evaluation remains cached, let $e(v)\in\mathcal C$ denote its entry. Provided $N(x_0)>0$, the heat contributed by this root to entry $c\in\mathcal C$ is

$$
H_{x_0}(c)=
\sum_{\substack{
v\in\mathcal T(x_0)\\
N(v)>0,\ e(v)=c
}}
\frac{N(v)}{N(x_0)}.
$$

Distinct tree nodes that share one key contribute to the same entry. A call with several roots adds their separately normalized heat values, giving each root one unit-scale visit distribution regardless of its simulation count.

After a call, all retained entries represented by its visited nodes receive one common heat-generation label and their aggregated heat. Their recency positions are then updated in increasing heat order; equal heat is ordered by decreasing tree depth. Consequently, high-heat entries finish nearer the most-recent end, and a shallower entry follows a deeper entry when their heat is equal.

Before the next call begins, its requested roots condition the trajectory prediction on the states about to be searched. Starting from every requested root that remains cached, TLRU follows retained parent-child links only through entries carrying the same heat-generation label. Contributions reached from several roots are added, and the resulting entries are again promoted in increasing heat order with decreasing depth as the tie-break. This procedure favors the part of the previous trajectory reachable from the requested roots.

The memory charge of an entry includes its packed key, Policy-Value record, recency and trajectory metadata, and allocated parent-child links. Whenever the total charge exceeds $M_C$, entries are removed from the least-recent end until the capacity constraint is restored. Search-tree nodes and their visit statistics are never cached. Every call therefore begins with fresh node counts, accumulated values and virtual reservations, although its nodes may acquire exact network records from the cache. Cache ordering changes only which records survive the memory constraint; root allocation, PUCT, backup and final decision use statistics from the active trees alone.

### 5.8 Batched Evaluation

A search call may contain several root states. Before simulation begins, cache lookup resolves every root with a retained record. The remaining roots are grouped by their keys $\chi(x)$; each distinct uncached key is evaluated once, and the resulting record is assigned to every root in its group.

To avoid starting a neural call that is unlikely to finish before a deadline, the batching scheduler carries a single-state latency estimate $\tau$, measured in milliseconds, between search calls. The estimate begins at zero. A neural call that contains exactly one uncached key and lasts $\Delta t$ supplies the sample

$$
\tau'=\max(0.05,\Delta t).
$$

The first such sample sets $\tau=\tau'$. Every later sample updates it by

$$
\tau\leftarrow
\max\left(
\tau',
0.8\tau+0.2\tau'
\right).
$$

Let $B_{\mathrm{prop}}$ be a proposed batch capacity and let $R$ be the remaining time in milliseconds. The deadline-admissible capacity is

$$
D(R,B_{\mathrm{prop}})=
\min\left[
B_{\mathrm{prop}},
\max\left(
0,
\left\lfloor
\frac{R-2}{1.25\max(1,\tau)}
\right\rfloor
\right)
\right].
$$

The subtraction of 2 milliseconds and the factor $1.25$ provide fixed and proportional timing margins. Define the cycle capacity by

$$
B_{\mathrm{cycle}}=
\begin{cases}
B_{\mathrm{batch}},&\text{without a deadline},\\
D(R_{\mathrm{cycle}},B_{\mathrm{batch}}),&\text{with a deadline},
\end{cases}.
$$

Here $R_{\mathrm{cycle}}$ is the remaining time at the beginning of the cycle. A zero cycle capacity ends the search before any additional leaf is selected.

For every tree with $N(x_0)<N_{\mathrm{cap}}$, define the cycle contribution limit by

$$
m(x_0)=
\min\left(
B_{\mathrm{cycle}},
N_{\mathrm{cap}}-N(x_0)
\right).
$$

The tree may contribute at most $m(x_0)$ simulations to the cycle. Root selection first serves positive fair-allocation deficits and, after the fair phase, uses the fixed-$\alpha$ root score from Section 5.5. Descents during the fair phase use full-width PUCT at expanded non-root nodes. Later descents use the opening, verification and PUCT priorities from Section 5.6. Each descent ends at a terminal node or an unexpanded nonterminal node.

A terminal leaf receives its exact outcome and completes one simulation immediately. An unexpanded nonterminal leaf becomes a neural request, and its path retains one virtual visit until that request is resolved. Terminal backups and accepted neural requests both count toward $m(x_0)$. A tree does not reserve the same unexpanded node twice in one cycle: reaching an already reserved node releases the reservations introduced by that attempt and ends request selection for that tree. The selection pass accepts at most $m(x_0)$ simulations. Its attempt limit is

$$
A_{\max}(x_0)=
\max\left(
5m(x_0),
m(x_0)+8
\right).
$$

The requests from all trees form one ordered list. Before each neural submission, define the call capacity by

$$
B_{\mathrm{call}}=
\begin{cases}
B_{\mathrm{cycle}},&\text{without a deadline},\\
D(R_{\mathrm{call}},B_{\mathrm{cycle}}),&\text{with a deadline},
\end{cases}.
$$

Here $R_{\mathrm{call}}$ is measured immediately before submission. The next neural call contains at most $B_{\mathrm{call}}$ requests. A zero value or an external stop signal cancels the unsubmitted requests and releases all virtual visits attached to them.

Before neural evaluation, the cache resolves retained records and the remaining requests are grouped by $\chi(x)$. One network evaluation is performed for each distinct uncached key. For a representative nonterminal state $x$, let $a_1,\ldots,a_L$ be its ordered legal actions, where $L=|\mathcal A(x)|$. Its exact evaluation record is

$$
E_\theta(x)=
\left(
(a_j)_{j=1}^{L},
(i_G(x,a_j))_{j=1}^{L},
(P_\theta(a_j\mid\phi_G(x)))_{j=1}^{L},
V_\theta(\phi_G(x))
\right).
$$

Components carrying the same index $j$ refer to the same legal action. The record therefore preserves the alignment among legal actions, compact indices and Policy probabilities, and one copy may serve every request with key $\chi(x)$.

Each newly computed record enters the active cache. Every requesting tree expands its own leaf from the shared record and backs up the shared Value estimate along its own reserved path. The trees consequently reuse network computation without sharing nodes, paths or search statistics.

A further selection cycle requires both a tree below $N_{\mathrm{cap}}$ and at least one completed backup in the preceding cycle. The updated visits determine all later deficits and PUCT scores. The first post-fair cycle fixes $\gamma_{\mathrm{fair}}$, and this exponent determines the tempered root prior for the remainder of the search call.

### 5.9 Root Evaluation and Policy

The root evaluation uses the initial network estimate until at least one simulation has completed, after which it uses the empirical root mean:

$$
V_{\mathrm{root}}(x_0)=
\begin{cases}
V_\theta(s_0),&N(x_0)=0,\\
Q(x_0),&N(x_0)>0.
\end{cases}
$$

Completed root-edge visits and original priors define the legal root Policy:

$$
P_{\mathrm{root}}(a\mid s_0)=
\frac{N(x_0,a)+P(x_0,a)}
{\displaystyle\sum_{b\in\mathcal A(x_0)}
\left[N(x_0,b)+P(x_0,b)\right]},
\qquad a\in\mathcal A(x_0).
$$

With no completed simulation, every root-edge visit count is zero, so

$$
P_{\mathrm{root}}(a\mid s_0)=P_\theta(a\mid s_0).
$$

After simulation, $P_{\mathrm{root}}$ records the allocation produced by the fair floor and fixed-$\alpha$ PUCT. Adding the original prior keeps every legal root action at positive weight and preserves a Policy distribution when the simulation count is zero. The decision rules in Section 5.10 use this distribution as their base score.

The legal distribution has the following embedding in the complete compact action set:

$$
P_{\mathrm{dense}}(i\mid s_0)=
\begin{cases}
P_{\mathrm{root}}(a\mid s_0),
&i=i_G(x_0,a),\quad a\in\mathcal A(x_0),\\
0,&i\in\mathcal I_G\setminus\mathcal I_G(x_0).
\end{cases}
$$

Thus the embedding agrees with $P_{\mathrm{root}}$ on legal compact indices and vanishes on every other member of $\mathcal I_G$.

### 5.10 Decision Components

Two optional decision transformations may modify the final move ordering without changing the search tree. Their common base score is

$$
D_0(a)=P_{\mathrm{root}}(a\mid s_0).
$$

Let $\mathcal M(x_0)$ be the set of legal actions that immediately checkmate the opponent. IMF acts as the identity whenever it is disabled or $\mathcal M(x_0)$ is empty:

$$
D_I(a)=D_0(a).
$$

When IMF is enabled and $\mathcal M(x_0)$ is nonempty, the selected mating action is

$$
a_M\in
\operatorname*{arg\,max}_{a\in\mathcal M(x_0)}
D_0(a).
$$

Root expansion order resolves multiple maximizers. IMF then defines

$$
D_I(a)=
\begin{cases}
1,&a=a_M,\\
D_0(a),&a\in\mathcal A(x_0)\setminus\lbrace a_M\rbrace.
\end{cases}
$$

Repetition Policy Penalty (RPP) uses a coefficient $\lambda_R\in[0,1]$; $\lambda_R=0$ makes it the identity transformation. Let $\mathcal R_3(x_0)$ contain each legal action that makes a threefold-repetition claim available immediately or permits the opponent to make such a claim after one reply. The deduction is

$$
d_R=
\lambda_R
\operatorname{clip}_{[0,1]}
\left(V_{\mathrm{root}}(x_0)\right).
$$

Applying this deduction to the IMF output gives

$$
D(a)=
\begin{cases}
\max(0,D_I(a)-d_R),&a\in\mathcal R_3(x_0),\\
D_I(a),&a\in\mathcal A(x_0)\setminus\mathcal R_3(x_0).
\end{cases}
$$

Both transformations act only on a copy of the root Policy. Network probabilities, edge priors and tree statistics therefore remain unchanged, and $D$ need not sum to one. Legal actions are ordered first by decreasing $D(a)$, then by decreasing $D_0(a)$, and finally by decreasing lexicographic UCI notation. The first action in this deterministic order is selected.
