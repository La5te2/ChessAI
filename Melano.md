# Melano

Melano is a geometry-aware Transformer chess network that jointly predicts a move policy and a side-to-move position evaluation.

## 1. Notation

- $\mathcal X$ is the set of complete chess states maintained by the rules engine.
- $x\in\mathcal X$ is a complete chess state containing the board, side to move, castling rights, en passant state, move counters and repetition history.
- $\mathcal A(x)$ is the set of legal actions in state $x$.
- $T(x,a)$ is the complete state reached by applying legal action $a\in\mathcal A(x)$ to state $x$.
- $z(x)\in\lbrace-1,0,1\rbrace$ is the exact outcome of terminal state $x$ from the perspective of its side to move, with $1$, $0$ and $-1$ representing a win, draw and loss.
- $\phi_M$ is the Melano state encoder that maps a complete chess state to a side-to-move canonical network input.
- $s=\phi_M(x)$ is the Melano network input obtained from complete state $x$.
- $\mathcal I_M=\lbrace0,\ldots,1857\rbrace$ is the fixed set of Melano action indices.
- $i_M(x,a)\in\mathcal I_M$ is the side-to-move canonical index assigned to legal action $a\in\mathcal A(x)$.
- $\theta$ denotes the trainable network parameters.
- $\ell_\theta(s)\in\mathbb R^{1858}$ is the complete vector of Policy logits produced by the network with parameters $\theta$, and $\ell_\theta(s,i)$ is its scalar component for action index $i\in\mathcal I_M$.
- $\text{P}$, which stands for Policy, is the probability distribution that the network assigns to the legal actions of a specified complete state.
- $\text{V}$, which stands for Value, is a scalar network output in $[-1,1]$ that estimates the expected game result from the perspective of the side to move.
- $Q$ denotes a state or action evaluation defined by a particular procedure. Each definition specifies its arguments and observation perspective.
- $\mathrm{clip}_{[l,u]}(y)=\min(u,\max(l,y))$ restricts scalar $y$ to the closed interval $[l,u]$.

## 2. State and Action Encoding

### 2.1 State Encoding

The state encoder expresses every position from the perspective of the side to move. It maps the network-visible components of a complete chess state to 64 categorical square entries followed by castling and en passant entries:

$$
\phi_M(x)=s=(p_0,\ldots,p_{63},c,e).
$$

Let $r,f\in\lbrace0,\ldots,7\rbrace$ denote the physical rank and file coordinates of a square, where $r=0$ denotes rank 1 and $f=0$ denotes file `a`. Its physical square index is $q=8r+f$. The canonical square map for state $x$ is

$$
\chi_x(q)=
\begin{cases}
8r+f,&\text{if White is to move},\\
8(7-r)+f,&\text{if Black is to move}.
\end{cases}
$$

Thus, the friendly side always advances toward increasing canonical ranks, while the files retain their physical order.

For every physical square $q$, the entry $p_{\chi_x(q)}\in\lbrace0,\ldots,12\rbrace$ records its occupant. The value 0 denotes an empty square. Friendly pawns, knights, bishops, rooks, queens and kings use values 1 through 6, while the corresponding opposing pieces use values 7 through 12. Physical colors therefore do not appear as separate input categories.

The castling entry $c\in\lbrace0,\ldots,15\rbrace$ is a four-bit mask. From least to most significant, its bits represent friendly kingside, friendly queenside, opposing kingside and opposing queenside castling rights. The en passant entry is

$$
e=
\begin{cases}
0,&\text{if no en passant square exists},\\
1+f_{\mathrm{ep}},&\text{if an en passant square exists on file }f_{\mathrm{ep}},
\end{cases}
$$

where $f_{\mathrm{ep}}\in\lbrace0,\ldots,7\rbrace$ uses the unchanged file coordinate.

Each of the 66 categorical entries occupies one unsigned byte. `PackedState` denotes their 66-byte representation in the order shown in $s$, and widening these bytes to integer embedding indices reproduces $\phi_M(x)$ exactly.

The encoded state $\phi_M(x)$ records canonical piece placement, relative castling rights and the en passant file. The complete state $x$ additionally records move counters and repetition history. Because $\phi_M$ omits those fields, complete states that differ only in move counters or repetition history produce the same network input.

### 2.2 Action Encoding

The legal-action set $\mathcal A(x)$ varies with the complete state $x$. The action encoder first defines an expanded code space and retains the codes whose source-destination geometry can occur in a chess move. Each legal action is canonicalized into one retained code. This construction produces a state-independent compact action set without retaining impossible source-destination pairs.

For canonical square indices $q,k\in\lbrace0,\ldots,63\rbrace$, define

$$
\Delta r(q,k)=\left\lfloor\frac{k}{8}\right\rfloor-
\left\lfloor\frac{q}{8}\right\rfloor,
\qquad
\Delta f(q,k)=(k\bmod 8)-(q\bmod 8).
$$

An ordered square pair is geometrically admissible when its displacement is horizontal, vertical, diagonal or knight-shaped:

$$
G(q,k)\iff
q\ne k
\ \land\
\left(
\Delta r(q,k)=0
\ \lor\
\Delta f(q,k)=0
\ \lor\
|\Delta r(q,k)|=|\Delta f(q,k)|
\ \lor\
\lbrace|\Delta r(q,k)|,|\Delta f(q,k)|\rbrace=\lbrace1,2\rbrace
\right).
$$

The expanded ordinary-action set is

$$
\mathcal J_O=
\left\{
64q+k\ \middle|\ q,k\in\lbrace0,\ldots,63\rbrace,\ G(q,k)
\right\}.
$$

It contains 1792 source-destination codes. Ordinary moves and promotions to a queen use this region. Before constructing the code, the action encoder maps the source and destination through the canonical square map $\chi_x$. It also converts the rules engine's king-to-rook castling representation to the king's destination square.

An underpromotion is determined by a canonical source square $q$ on rank 7, a destination-file displacement $\Delta f\in\lbrace-1,0,1\rbrace$ and a promoted-piece index

$$
u=
\begin{cases}
0,&\text{for a knight},\\
1,&\text{for a bishop},\\
2,&\text{for a rook}.
\end{cases}
$$

Its expanded code is

$$
j=4096+9q+3(\Delta f+1)+u.
$$

The destination lies on canonical rank 8 and has file $(q\bmod8)+\Delta f$. Retaining only source squares with $\lfloor q/8\rfloor=6$ and destination files in $\lbrace0,\ldots,7\rbrace$ gives the 66-element set $\mathcal J_U$. The complete geometrically admissible set is

$$
\mathcal J_M=\mathcal J_O\cup\mathcal J_U,
\qquad
|\mathcal J_M|=1792+66=1858.
$$

Let $\mu_M:\mathcal J_M\to\mathcal I_M$ be the order-preserving bijection

$$
\mu_M(j)=
\left|
\left\{
h\in\mathcal J_M\mid h<j
\right\}
\right|.
$$

For $a\in\mathcal A(x)$, let $j_M(x,a)\in\mathcal J_M$ be its canonical expanded code. Its Melano action index is

$$
i_M(x,a)=\mu_M\!\left(j_M(x,a)\right).
$$

The action indices available in state $x$ are

$$
\mathcal I_M(x)=
\left\{
i_M(x,a)\mid a\in\mathcal A(x)
\right\}.
$$

Decoding remains state-dependent because geometric admissibility does not imply legality in a particular position. Given $i\in\mathcal I_M(x)$, the decoder generates $\mathcal A(x)$ and returns the legal action $a$ satisfying $i_M(x,a)=i$. This action retains the castling, en passant and promotion information required by the rules engine.

## 3. Network

### 3.1 Geometry-Attention Encoder

The network computes its $\text{P}$ and $\text{V}$ from a shared geometry-aware representation, denoted by $E_\theta(s)$. A state-embedding layer maps the encoded state $s$ to 64 square tokens, after which a sequence of geometry-attention blocks transforms those tokens into $E_\theta(s)$. Section 3.2 defines how the Policy and Value computations derive their outputs from this shared representation.

Let $C\geq1$ be the feature width of every token and let $B\geq1$ be the number of geometry-attention blocks. The state embedding produces

$$
h_0(s)\in\mathbb R^{64\times C},
$$

where token $q$ corresponds to square index $q\in\lbrace0,\ldots,63\rbrace$.

The castling and en passant entries form the rule-context embedding

$$
r(s)=E_{\mathrm{castling}}(c)+E_{\mathrm{ep}}(e).
$$

For square $q$, the rule-context embedding is added to the embeddings of the canonical square occupant and the canonical absolute square index:

$$
h_{0,q}(s)=E_{\mathrm{piece}}(p_q)+E_{\mathrm{square}}(q)+r(s).
$$

The piece, square, castling and en passant embedding tables contain 13, 64, 16 and 9 vectors, respectively.

Each geometry-attention block uses the largest member of $\lbrace8,4,2,1\rbrace$ that divides $C$ as its head count $H$. The resulting feature width of each head is

$$
d=\frac{C}{H}.
$$

The geometry-attention blocks share a trainable bank of $R=64$ full-board relation templates,

$$
T=(T_1,\ldots,T_R),
\qquad
T_r\in\mathbb R^{64\times64}.
$$

For every ordered pair of squares $(q,k)$, the entry $T_{r,q,k}$ is the contribution of template $r$ to attention from query square $q$ to key-value square $k$. The template bank is shared across all blocks, while each block has an independent coefficient generator.

For block $b\in\lbrace0,\ldots,B-1\rbrace$, write $h_b=h_b(s)\in\mathbb R^{64\times C}$ for the token sequence supplied to that block. The block first adds a trainable position tensor $Z_b\in\mathbb R^{64\times C}$ to $h_b$, producing

$$
\widetilde h_b=h_b+Z_b.
$$

The block applies its first LayerNorm to each token of $\widetilde h_b$, normalizing that token across its $C$ features. A linear projection then maps every normalized token to $3C$ features. Reshaping the $3C$ features into three groups of $H$ width-$d$ heads produces the query, key and value tensors $Q_b,K_b,U_b\in\mathbb R^{H\times64\times d}$:

$$
(Q_b,K_b,U_b)=
\mathrm{Split}_{3}\left(
\mathrm{Reshape}_{3,H,d}
\left(W_{qkv,b}\mathrm{LN}_{b,1}(\widetilde h_b)+b_{qkv,b}\right)
\right).
$$

The coefficient generator uses the position-adjusted tokens rather than a fixed displacement class. Its fixed intermediate dimensions are $C_G=8$ square features and $C_S=32$ state features. Let $L^{\mathrm{sq}}_b:\mathbb R^C\to\mathbb R^{C_G}$ act independently on each token. The flattened square features are

$$
a_b=\operatorname{vec}\!\left(L^{\mathrm{sq}}_b(\widetilde h_b)\right)
\in\mathbb R^{64C_G}.
$$

Two further affine maps, with LayerNorm applied after each GELU, produce a state feature and then one coefficient for each pair of an attention head and a relation template:

$$
g_b=\mathrm{LN}^{\mathrm{state}}_b\!\left(
\mathrm{GELU}\!\left(L^{\mathrm{state}}_b(a_b)\right)
\right)
\in\mathbb R^{C_S},
$$

$$
\alpha_b(s)=
\operatorname{Reshape}_{H,R}\!\left(
\mathrm{LN}^{\mathrm{coef}}_b\!\left(
\mathrm{GELU}\!\left(L^{\mathrm{coef}}_b(g_b)\right)
\right)
\right)
\in\mathbb R^{H\times R}.
$$

The coefficients combine the shared templates into the block- and state-dependent attention bias

$$
G_{b,h,q,k}(s)=
\sum_{r=1}^{R}\alpha_{b,h,r}(s)T_{r,q,k}.
$$

For block $b$, head $h$, query square $q$ and key-value square $k$, the query-key similarity and dynamic geometry bias define the attention score

$$
S_{b,h,q,k}=
\frac{Q_{b,h,q}\cdot K_{b,h,k}}{\sqrt d}
+G_{b,h,q,k}(s).
$$

Softmax normalizes these scores over all 64 key-value squares for each fixed block, head and query square. The resulting output of head $h$ for query square $q$ is

$$
o_{b,h,q}=
\sum_{k=0}^{63}
\frac{\exp S_{b,h,q,k}}
{\displaystyle\sum_{w=0}^{63}\exp S_{b,h,q,w}}
U_{b,h,k}.
$$

The attention sublayer concatenates the $H$ head outputs for query square $q$ and projects the resulting $C$-dimensional vector back to the token feature space. Adding the projected vector to the position-adjusted token $\widetilde h_{b,q}$ produces the attention residual output

$$
y_{b,q}=\widetilde h_{b,q}+
W_{o,b}\mathrm{Concat}_{h=1}^{H}(o_{b,h,q})+b_{o,b}.
$$

The feed-forward sublayer applies the second LayerNorm to each token in $y_b$, maps the normalized feature vector from $C$ to $2C$ features, applies GELU and maps the result back to $C$ features. Adding this result to $y_b$ produces the block output.

$$
h_{b+1}=y_b+
W_{f,b,2}\mathrm{GELU}
\left(W_{f,b,1}\mathrm{LN}_{b,2}(y_b)+b_{f,b,1}\right)+b_{f,b,2}.
$$

Applying all $B$ blocks defines the shared geometry-aware representation

$$
E_\theta(s)=h_B(s)\in\mathbb R^{64\times C}.
$$

The complete encoder path is

$$
s\longrightarrow\text{state embedding}\longrightarrow h_0
\longrightarrow\text{$B$ geometry-attention blocks}
\longrightarrow E_\theta(s).
$$

Every position tensor $Z_b$ begins at zero. For $A\in\mathbb R^{64\times64}$, define the row-centering operator

$$
\bigl(\Pi(A)\bigr)_{q,k}
=
A_{q,k}
-
\frac{1}{64}\sum_{w=0}^{63}A_{q,w}.
$$

Xavier normal initialization first produces an unconstrained matrix $\widetilde T_r$ for every relation template, after which the initialized template is $T_r=\Pi(\widetilde T_r)$. Consequently, every initialized template satisfies

$$
\sum_{k=0}^{63}T_{r,q,k}=0
\qquad
\text{for every }r\in\lbrace1,\ldots,R\rbrace
\text{ and }q\in\lbrace0,\ldots,63\rbrace.
$$

Each embedding entry is initialized independently from $\mathcal N(0,1)$. For an affine map with $n$ input features, each weight and bias entry is initialized independently from $\mathcal U(-n^{-1/2},n^{-1/2})$. Each LayerNorm has trainable scale and bias, uses epsilon $10^{-5}$ and begins with unit scale and zero bias. The encoder applies no dropout.

### 3.2 Policy and Value Heads

The Policy head derives action logits from the 64 transformed square tokens in $E_\theta(s)$. Let

$$
z_q=E_\theta(s)_q,
\qquad q\in\lbrace0,\ldots,63\rbrace,
$$

and let $\overline z_q=\mathrm{LN}_P(z_q)$ be the normalized token for square $q$. Separate linear maps produce its source and destination features:

$$
u_q=W_F\overline z_q+b_F,
\qquad
v_q=W_T\overline z_q+b_T.
$$

For every compact action index $i\in\mathcal I_M$, let

$$
j_i=\mu_M^{-1}(i)
$$

be its expanded code. An ordinary-action code satisfies $j_i<4096$ and determines

$$
q_i=\left\lfloor\frac{j_i}{64}\right\rfloor,
\qquad
k_i=j_i\bmod64.
$$

The corresponding Policy logit is

$$
\ell_\theta(s,i)=
\frac{u_{q_i}\cdot v_{k_i}}{\sqrt C}.
$$

A third linear map produces nine underpromotion scores for each canonical source square:

$$
c_q=W_U\overline z_q+b_U\in\mathbb R^9.
$$

An underpromotion code satisfies $j_i\geq4096$ and determines

$$
q_i=\left\lfloor\frac{j_i-4096}{9}\right\rfloor,
\qquad
m_i=(j_i-4096)\bmod9.
$$

Its Policy logit is

$$
\ell_\theta(s,i)=c_{q_i,m_i}.
$$

The component index $m_i=3(\Delta f+1)+u$ uses the destination-file displacement and promoted-piece index defined in Section 2.2. Applying these formulas to every $i\in\mathcal I_M$ produces

$$
\ell_\theta(s)\in\mathbb R^{1858}.
$$

For complete state $x$ with $s=\phi_M(x)$, selecting the logits indexed by legal actions and normalizing them with softmax produces the legal-move Policy:

$$
P_\theta(a\mid s)=
\frac{\exp\ell_\theta\bigl(s,i_M(x,a)\bigr)}
{\displaystyle\sum_{b\in\mathcal A(x)}
\exp\ell_\theta\bigl(s,i_M(x,b)\bigr)},
\qquad a\in\mathcal A(x).
$$

The denominator ranges over $\mathcal A(x)$, so $P_\theta(\cdot\mid s)$ is a probability distribution over the legal actions in complete state $x$.

The Value head first applies LayerNorm to each transformed square token:

$$
\overline z_q^{\,V}=\mathrm{LN}_V(z_q).
$$

A trainable query $g_V\in\mathbb R^C$ assigns a content-dependent score to every square,

$$
a_q=\frac{g_V^{\mathsf T}\overline z_q^{\,V}}{\sqrt C},
$$

and softmax converts the 64 scores into pooling weights:

$$
\alpha_q=
\frac{\exp a_q}
{\displaystyle\sum_{k=0}^{63}\exp a_k}.
$$

The weighted sum

$$
z_V=\sum_{q=0}^{63}\alpha_q\overline z_q^{\,V}
$$

is the Value representation. The Value head maps $z_V$ to 256 hidden features, applies ReLU and produces the unbounded scalar

$$
t_\theta(s)=
W_{V,2}\mathrm{ReLU}
\left(W_{V,1}z_V+b_{V,1}\right)+b_{V,2}.
$$

Applying the hyperbolic tangent gives the bounded Value output

$$
V_\theta(s)=\tanh\bigl(t_\theta(s)\bigr)\in[-1,1].
$$

The query $g_V$ begins at zero, so the initial pooling weights satisfy $\alpha_q=1/64$ for every square. The query is confined to the Value head: it reads the completed shared representation but does not participate in a geometry-attention block or in the Policy computation.

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

The complete Policy head can produce $\ell_\theta(s)$, whereas legal-move inference for a nonterminal complete state $x$ requires only the components indexed by actions in $\mathcal A(x)$. Before inference, the affine source and destination maps in Section 3.2 are combined into

$$
M_P=\frac{W_F^{\mathsf T}W_T}{\sqrt C},
\qquad
r_F=\frac{W_F^{\mathsf T}b_T}{\sqrt C},
\qquad
r_T=\frac{W_T^{\mathsf T}b_F}{\sqrt C},
\qquad
c_P=\frac{b_F^{\mathsf T}b_T}{\sqrt C}.
$$

For an ordinary action with source square $q_i$ and destination square $k_i$, the resulting identity is

$$
\ell_\theta(s,i)=
\overline z_{q_i}^{\mathsf T}M_P\overline z_{k_i}
+\overline z_{q_i}^{\mathsf T}r_F
+\overline z_{k_i}^{\mathsf T}r_T
+c_P.
$$

The four fused quantities depend only on the trained Policy parameters and are computed once before inference. The original form applies one $C\times C$ affine map to each source representation and one to each destination representation. The fused form applies the $C\times C$ matrix $M_P$ only to source representations and evaluates the remaining terms with vector inner products, thereby removing one of the two matrix transformations required by the ordinary-action readout.

The selected-logit computation gathers the source and destination representations required by the requested actions. When the requested width is less than 64, it applies $M_P$ only to the gathered source representations. Otherwise, it applies $M_P$ to all 64 source representations once and gathers the required results. In either case, it evaluates only the ordinary-action or underpromotion components indexed by the requested actions. The underpromotion readout remains unchanged.

For a batch of nonterminal complete states $\mathbf x=(x_1,\ldots,x_n)$, let $s_r=\phi_M(x_r)$ and $L_r=|\mathcal A(x_r)|$. Enumerate the legal actions as $(a_{rj})_{j=1}^{L_r}$ in the order produced by the rules engine, and define

$$
i_{rj}=i_M(x_r,a_{rj}).
$$

The selected Policy logits are

$$
\Lambda_\theta(s_r)_j=
\ell_\theta(s_r,i_{rj}),
\qquad j\in\lbrace1,\ldots,L_r\rbrace.
$$

For each selected index, let $\widehat i_{rj}=\mu_M^{-1}(i_{rj})$. When $\widehat i_{rj}<4096$, the expanded code determines a source square $q$ and destination square $k$, and the selected-logit computation evaluates the fused ordinary-action identity above. Otherwise, the expanded code determines an underpromotion source square $q$ and component $m$, and the computation selects $c_{q,m}$. These values equal the components of $\ell_\theta(s_r)$ indexed by the legal actions in $\mathcal A(x_r)$. Softmax normalizes only these $L_r$ components:

$$
P_\theta(a_{rj}\mid s_r)=
\frac{\exp\Lambda_\theta(s_r)_j}
{\displaystyle\sum_{t=1}^{L_r}\exp\Lambda_\theta(s_r)_t}.
$$

## 4. Supervised Training

### 4.1 Supervised Data

Let $\mathcal D_{\mathrm{sup}}$ be a supervised dataset containing $N$ records:

$$
\mathcal D_{\mathrm{sup}}=
\lbrace \xi_n\rbrace_{n=1}^{N}.
$$

Each record is associated with a complete pre-move state $x_n$ and a selected legal action $a_n\in\mathcal A(x_n)$. Its three components are

$$
\xi_n=(s_n,i_n,y_n),
$$

The three components satisfy

$$
s_n=\phi_M(x_n),
\qquad
i_n=i_M(x_n,a_n),
\qquad
y_n\in[-1,1].
$$

The encoded state $s_n$ is the canonical network input, and the compact action index $i_n$ is the corresponding canonical Policy target. The scalar $y_n$ is the Value target, expressed as an estimate of the expected game result from the perspective of the side to move in $x_n$. On this scale, $-1$ denotes a loss, $0$ denotes a draw and $1$ denotes a win, while intermediate values express expectations between these outcomes.

### 4.2 Supervised Objective

For network input $s$, the Policy head produces

$$
\ell_\theta(s)=
\left(\ell_\theta(s,i)\right)_{i\in\mathcal I_M}
\in\mathbb R^{1858}.
$$

Applying softmax to these 1858 components produces the supervised action-index distribution

$$
R_\theta(i\mid s)=
\frac{\exp\ell_\theta(s,i)}
{\displaystyle\sum_{j\in\mathcal I_M}\exp\ell_\theta(s,j)},
\qquad i\in\mathcal I_M.
$$

The supervised distribution $R_\theta(\cdot\mid s)$ and the legal-move distribution $P_\theta(\cdot\mid s)$ are derived from the same logit vector but use different normalization domains. Supervised learning normalizes all 1858 geometrically admissible action components, whereas legal-move inference selects and normalizes only the components indexed by $\mathcal A(x)$.

For minibatch $\mathcal B\subseteq\mathcal D_{\mathrm{sup}}$, the supervised Policy loss is the mean negative log-probability assigned to the target action indices:

$$
L_{P,\mathrm{sup}}^{(\mathcal B)}=
-\frac{1}{|\mathcal B|}
\sum_{(s,i,y)\in\mathcal B}
\log R_\theta(i\mid s).
$$

Map the target and predicted Value to the unit interval by

$$
p(y)=\frac{y+1}{2},
\qquad
\widehat p_\theta(s)
=
\frac{V_\theta(s)+1}{2}
=
\frac{1}{1+\exp\bigl(-2t_\theta(s)\bigr)}.
$$

The supervised Value loss is the mean binary cross-entropy between these quantities:

$$
L_{V,\mathrm{sup}}^{(\mathcal B)}
=
-\frac{1}{|\mathcal B|}
\sum_{(s,i,y)\in\mathcal B}
\left[
p(y)\log\widehat p_\theta(s)
+
\bigl(1-p(y)\bigr)
\log\bigl(1-\widehat p_\theta(s)\bigr)
\right].
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

The Policy-loss term determines $\nabla_{\theta_P}L_{\mathrm{sup}}^{(\mathcal B)}$, and the weighted Value-loss term determines $\nabla_{\theta_V}L_{\mathrm{sup}}^{(\mathcal B)}$. Both terms contribute to $\nabla_{\theta_E}L_{\mathrm{sup}}^{(\mathcal B)}$, where their gradients add as shown in the final equation.

### 4.3 Parameter Optimization

Training traverses $\mathcal D_{\mathrm{sup}}$ in epochs, with each completed epoch using every record exactly once. Before forming minibatches, each epoch randomizes the order of contiguous record blocks and then randomizes the records within each block. Minibatches do not cross block boundaries. This two-level shuffle varies the order and composition of successive minibatches across epochs, reducing correlations that a fixed record order could otherwise create between consecutive gradient estimates.

Let $B_{\mathrm{opt}}\geq1$ be the minibatch size, and suppose the dataset is partitioned into $J$ contiguous record blocks containing $n_1,\ldots,n_J$ records, where $\sum_{j=1}^{J}n_j=N$. Since each block is partitioned independently into minibatches, one complete epoch contains

$$
K_{\mathrm{epoch}}=
\sum_{j=1}^{J}
\left\lceil\frac{n_j}{B_{\mathrm{opt}}}\right\rceil
$$

optimizer steps. Let $E$ be the nonnegative integer number of requested epochs, and let $K_{\max}\in\mathbb N\cup\lbrace\infty\rbrace$ be the optimizer-step limit, with a nonpositive configured limit represented by $K_{\max}=\infty$. The number of optimizer steps is

$$
K=\min\left(K_{\max},E K_{\mathrm{epoch}}\right).
$$

Denote the corresponding minibatches by $\mathcal B_1,\ldots,\mathcal B_K$. A newly initialized Melano network with feature width $C$ and block count $B$ supplies the initial parameters $\theta^{(0)}$, and optimizer step $k$ produces $\theta^{(k)}$. For $k<K$, the parameters $\theta^{(k)}$ process $\mathcal B_{k+1}$, while $\theta^{(K)}$ is the final trained parameter set.

The learning-rate schedule uses the positive nominal step horizon

$$
K_{\mathrm{sched}}=
\max\left(
1,
\min\left(
K_{\max},
E\left\lceil\frac{N}{B_{\mathrm{opt}}}\right\rceil
\right)
\right).
$$

The lower bound keeps the warmup horizon defined when $E=0$, although no optimizer step evaluates the schedule in that case. For positive $E$, this horizon treats the $N$ records as one sequence of minibatches, while $K_{\mathrm{epoch}}$ counts the minibatches formed independently within the record blocks. The warmup length is

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

Let $g_{\max}\in\mathbb R$ be the global gradient-norm limit, and let $\epsilon_g=10^{-6}$. For $g_{\max}>0$, define

$$
\rho_k=
\min\left(
1,
\frac{g_{\max}}
{\lVert g_k\rVert_2+\epsilon_g}
\right).
$$

The gradient supplied to AdamW is

$$
\overline g_k=
\begin{cases}
g_k,&g_{\max}\leq0,\\
\rho_k g_k,&g_{\max}>0.
\end{cases}
$$

For $g_{\max}>0$, a finite gradient norm is a precondition for clipping. A nonfinite norm causes training to fail, so no valid parameter update is defined for that step.

For peak learning rate $\eta_{\max}$, optimizer step $k\in\lbrace1,\ldots,K\rbrace$ uses

$$
\eta_k=
\begin{cases}
\eta_{\max}\dfrac{k}{K_W},&k\leq K_W,\\[6pt]
\eta_{\max}\sqrt{\dfrac{K_W}{k}},&k>K_W.
\end{cases}
$$

The schedule therefore increases linearly to $\eta_{\max}$ during the warmup and then decays in inverse proportion to the square root of the step index.

AdamW forms a first-moment estimate $u_k$ and a second-moment estimate $v_k$ from $\overline g_k$. Both estimates have the same dimensions as $\theta$ and begin with $u_0=v_0=0$. With $\beta_1=0.9$ and $\beta_2=0.999$, their updates and bias corrections are

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

Let $\lambda\geq0$ be the weight-decay coefficient and let $\epsilon_A=10^{-8}$ prevent division by zero. AdamW first produces the provisional parameter set

$$
\widetilde\theta^{(k)}=
(1-\eta_k\lambda)\theta^{(k-1)}
-\eta_k
\frac{\widehat u_k}
{\sqrt{\widehat v_k}+\epsilon_A}.
$$

Let $\widetilde T_r^{(k)}$ denote the provisional value of relation template $r$. The final parameter set $\theta^{(k)}$ retains every non-template component of $\widetilde\theta^{(k)}$ and replaces each relation template with

$$
T_r^{(k)}=\Pi\left(\widetilde T_r^{(k)}\right).
$$

The square in $\overline g_k^2$, the square root in $\sqrt{\widehat v_k}$ and the quotient involving $\widehat u_k$ are evaluated elementwise. The factor $(1-\eta_k\lambda)$ applies decoupled weight decay, while the remaining term applies the adaptive gradient step determined by $\widehat u_k$ and $\widehat v_k$. The final projection restores the zero-mean row constraint of every relation template.

## 5. Search

### 5.1 Root Initialization

For a nonterminal complete state $x_0$, the MCTS procedure initializes a tree whose root corresponds to $x_0$. Each node corresponds to a complete state, and each edge leaving a node corresponding to state $x$ records a legal action $a\in\mathcal A(x)$ and leads to a child node corresponding to $T(x,a)$. A simulation follows selected edges from the root to a leaf, determines an evaluation for the leaf state and propagates that evaluation back along the selected path.

The evaluator obtains $P_\theta(\cdot\mid s_0)$ and $V_\theta(s_0)$ for $s_0=\phi_M(x_0)$. Before any simulation completes, $V_\theta(s_0)$ serves as the reported root evaluation, and the root's completed-visit count and accumulated evaluation are both zero. Root expansion then creates one outgoing edge and one child node for every legal action $a\in\mathcal A(x_0)$. The Policy probability assigned to $a$ becomes the prior of that edge:

$$
P(x_0,a)=P_\theta(a\mid s_0).
$$

A nonterminal node with no outgoing edges is called an unexpanded node. When a simulation reaches such a node for state $x$, the evaluator obtains $P_\theta(\cdot\mid\phi_M(x))$ and $V_\theta(\phi_M(x))$. The evaluator expands the node by creating one outgoing edge and one child node for every action $a\in\mathcal A(x)$, assigning $P_\theta(a\mid\phi_M(x))$ to the edge prior $P(x,a)$. The backup operation in Section 5.4 propagates $V_\theta(\phi_M(x))$ along the selected path.

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

The exploration coefficient follows a logarithmic schedule based on the augmented parent count. Let $c_0$ be the initial coefficient, let $b_0$ determine the scale of the count inside the logarithm and let $f_0$ scale the logarithmic increase. With $b=\max(1,b_0)$ and $f=\max(0,f_0)$, the coefficient at augmented parent count $n$ is

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
\nu_d=
\begin{cases}
z(x_d),&x_d\text{ is terminal},\\
V_\theta(\phi_M(x_d)),&x_d\text{ is nonterminal}.
\end{cases}
$$

Denote the selected path by $(x_0,x_1,\ldots,x_d)$, and let $a_k$ be the action that leads from $x_{k-1}$ to $x_k$. Since the side to move alternates after every action, the leaf evaluation propagated to node $x_k$ is

$$
\nu_k=(-1)^{d-k}\nu_d.
$$

The backup operation removes the virtual reservation from the path and updates each selected node by

$$
N(x_k)\leftarrow N(x_k)+1,
\qquad
W(x_k)\leftarrow W(x_k)+\nu_k.
$$

For each $k\geq1$, the edge count $N(x_{k-1},a_k)$ equals the completed-visit count $N(x_k)$ of its child, so the backup increments both quantities through the same update. Substituting the updated counts and evaluation sums into the definitions in Section 5.2 gives $Q(x_k)$ and $Q(x_{k-1},a_k)$.

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

The MCTS procedure recalculates $N_{\mathrm{target}}$ after each selection-and-evaluation cycle once the root has reached $N_{\min}$. A new cycle starts for a root only while its completed count is smaller than both the current target and the soft cap. The target and soft cap are checked between cycles, so backups completed within the final cycle may carry the final count beyond either threshold. A root with one legal action uses $u=0$ and admits no new cycle after reaching $N_{\min}$. A zero cap sets both $N_{\min}$ and $N_{\mathrm{target}}$ to zero.

An invocation configured with an unbounded simulation count bypasses $N_{\mathrm{target}}$ and $N_{\mathrm{cap}}$, so its selection-and-evaluation cycles continue until an execution deadline expires or the caller supplies a stop signal. A bounded invocation observes the same two stopping conditions together with $N_{\mathrm{target}}$ and $N_{\mathrm{cap}}$. When any applicable condition ends an invocation, the evaluator stops selecting leaves and submitting neural-evaluation requests. Completed backups remain in the tree, and Section 5.7 describes the release of virtual visits attached to unresolved requests.

### 5.6 Evaluation Reuse

Repeated network evaluation of the same `PackedState` produces the same compact Policy and Value record. The evaluator therefore stores each completed record in a cache keyed by its 66-byte `PackedState`. For a requested complete state, the rules engine determines whether the state is terminal before the evaluator queries this cache. This order preserves exact terminal detection because `PackedState` omits move counters and repetition history, whereas the cached network output depends only on the encoded state.

One MCTS invocation receives one or more root states and constructs a separate tree for each root. All trees created by that invocation access the same evaluation cache, which allows simulations within one tree and simulations from different trees to reuse completed network records. Let $M_C\geq0$ be the configured memory capacity for records retained across invocations. When $M_C=0$, the cache exists only for the current invocation. When $M_C>0$, the cache persists across invocations and uses TLRU (trajectory-aware least-recently-used) to order its records. A successful lookup moves the accessed record to the most-recent end of this order, and inserting a new record places it at the same end.

TLRU records a directed link whenever evaluation of a child state follows evaluation of its parent state. Let $\mathcal C$ be the set of retained entries, let $E_C\subseteq\mathcal C\times\mathcal C$ contain these links and let $\kappa(v)\in\mathcal C$ identify the cache entry used by tree node $v$. After an invocation completes, a root $x_0$ assigns the following heat to each retained entry $c$ reached by its tree:

$$
H_{x_0}(c)=
\sum_{\substack{v\text{ is a visited node}\\\kappa(v)=c}}
\frac{N(v)}{N(x_0)}.
$$

The visit ratio measures how strongly the completed tree used the trajectory through $c$. Contributions from different tree nodes that reuse the same encoded state, including nodes from several roots in one invocation, are added. TLRU assigns one generation identifier to the resulting heat values, then refreshes entries in increasing heat order. Entries with greater heat therefore finish nearer the most-recent end of the cache. Equal heat is ordered by decreasing tree depth before the refresh, so states nearer the root finish later in that group.

Before the next invocation evaluates its actual roots, TLRU follows the recorded links from each retained root through entries that belong to the same heat generation. The heat values of descendants reached from multiple roots are added, and the resulting entries are refreshed by the same heat and depth order. Ordinary cache hits and insertions continue to move their entries to the most-recent end.

When the approximate memory use exceeds $M_C$, TLRU removes entries from the least-recent end until the retained data fits within the capacity. The accounting includes compact evaluation arrays, fixed entry fields, trajectory metadata and the allocated capacity of recorded child links. Tree nodes and MCTS statistics remain local to one invocation, so each invocation starts with fresh visits, accumulated evaluations and virtual reservations. A cache hit supplies the Policy probabilities and Value needed to initialize or expand the requested tree node.

### 5.7 Batched Evaluation

One MCTS invocation may receive several root states. The evaluator first resolves every initial request whose `PackedState` has a cached record, then groups the remaining requests by `PackedState`. When uncached requests remain, their unique states form one neural batch. Each cached or computed record initializes every root tree with the corresponding `PackedState`.

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

For $B_{\mathrm{cycle}}>0$, a bounded invocation processes each search tree whose root count $N(x_0)$ is smaller than both $N_{\mathrm{target}}$ and $N_{\mathrm{cap}}$. The maximum number of distinct nonterminal leaves requested from one such tree is

$$
m=\min\left(
B_{\mathrm{cycle}},
N_{\mathrm{target}}-N(x_0),
N_{\mathrm{cap}}-N(x_0)
\right).
$$

The three terms limit the request by the available cycle capacity, the remaining count to the current target and the remaining count to the simulation cap, respectively. An unbounded invocation sets $m=B_{\mathrm{cycle}}$ and continues scheduling cycles until the stopping condition from Section 5.5 occurs.

To build this request set, each selection attempt starts at the root and follows PUCT through expanded nodes until it reaches a terminal node or an unexpanded nonterminal node. A terminal node receives its exact rule outcome, and immediate backup completes one simulation without adding a neural-evaluation request. An unexpanded nonterminal node enters the request set when that tree has not reserved the node earlier in the same cycle, and its selected path retains one virtual visit until the request is resolved. If another attempt from the same tree reaches an already reserved node, the selector removes the virtual visits introduced by that attempt and adds no request. The tree performs at most $\max(5m,m+8)$ attempts while collecting up to $m$ distinct nonterminal leaves.

The requests collected from all trees form one list. Before each evaluation submission, the scheduler computes

$$
B_{\mathrm{call}}=
\begin{cases}
B_{\mathrm{cycle}},&\text{if the invocation has no deadline},\\
D(R_{\mathrm{call}},B_{\mathrm{cycle}}),&\text{if a deadline is active},
\end{cases}
$$

where $R_{\mathrm{call}}$ is the remaining time before that submission. The next submission contains at most $B_{\mathrm{call}}$ requests. If this capacity is zero or cancellation has been requested, the scheduler releases the virtual visits attached to all remaining requests and ends batch submission.

For each submitted group, the evaluation-reuse mechanism from Section 5.6 resolves cache hits before the evaluator groups the unresolved requests by `PackedState`. When unresolved requests remain, their unique states form one neural batch, and one computed record serves every request in the corresponding group.

For a representative leaf state $x$, let $L=|\mathcal A(x)|$, and write its ordered legal actions as $a_1,\ldots,a_L$. The compact evaluation record contains the three sequences

$$
(a_j)_{j=1}^{L},
\qquad
(i_M(x,a_j))_{j=1}^{L},
\qquad
(P_\theta(a_j\mid\phi_M(x)))_{j=1}^{L},
$$

Together with the scalar $V_\theta(\phi_M(x))$, these sequences form the compact evaluation record. Entries with the same index $j$ describe the same legal action, which defines the alignment among the action, action-index and Policy sequences. The selected-logit computation defined in Section 3.3 evaluates the indexed logits and produces the Policy sequence.

After the neural batch returns, each newly computed record enters the active cache. The evaluator then assigns a cached or computed record to every request with the matching `PackedState`. For each requested leaf, tree expansion creates one outgoing edge for every $a_j$ and assigns $P_\theta(a_j\mid\phi_M(x))$ as that edge's prior. The scalar $V_\theta(\phi_M(x))$ is then backed up along the leaf's reserved path, so different trees can share a network record while retaining separate nodes, paths and search statistics.

After the submitted leaves complete their backups, every root that has reached $N_{\min}$ receives a recalculated dynamic target. Another cycle begins only when at least one tree remains below both its target and $N_{\mathrm{cap}}$ and the preceding cycle completed at least one backup.

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

When an output requires one probability for every index in $\mathcal I_M$, the compact root distribution expands to

$$
P_{\mathrm{dense}}(i\mid s_0)=
\begin{cases}
P_{\mathrm{root}}(a\mid s_0),
&i=i_M(x_0,a)\text{ for }a\in\mathcal A(x_0),\\
0,&i\notin\lbrace i_M(x_0,a):a\in\mathcal A(x_0)\rbrace.
\end{cases}
$$

The result-construction step performs this conversion for outputs defined over the fixed action-index set $\mathcal I_M$.

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

RPP (Repetition Policy Penalty) applies when the root evaluation favors the side to move. Let $\lambda_R\in[0,1]$ be its penalty coefficient, with $\lambda_R=0$ disabling the transformation. Let $\mathcal R_3(x_0)$ contain the legal actions that either make a threefold-repetition claim available immediately or allow the opponent to make such a claim after one reply. The effective repetition penalty is

$$
d_R=
\lambda_R
\mathrm{clip}_{[0,1]}
\left(V_{\mathrm{root}}(x_0)\right).
$$

Applying this penalty produces the final decision score

$$
D(a)=
\begin{cases}
\max(0,D_I(a)-d_R),
&a\in\mathcal R_3(x_0),\\
D_I(a),
&a\in\mathcal A(x_0)\setminus\mathcal R_3(x_0).
\end{cases}
$$

IMF and RPP transform a copy of $P_{\mathrm{root}}$ and produce the ordering score $D$. The network Policy, edge priors and tree statistics retain the values computed before the decision layer is applied. The decision layer compares the components of $D$ directly, ordering legal actions first by decreasing $D(a)$, then by decreasing $D_0(a)$ and finally by decreasing coordinate move string in Universal Chess Interface (UCI) notation. The first action in this deterministic ordering is the selected move.
