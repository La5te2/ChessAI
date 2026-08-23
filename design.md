# Gadus

This document specifies the Gadus network architecture under design. Gadus remains a spatial Policy-Value network over an $8\times8$ chessboard, and its representation and repeated transformation are defined directly from chessboard structure.

## 1. Design Objective

The network must preserve three properties of the existing Gadus design. It represents every position as a board-aligned tensor, predicts a legal-move Policy and a side-to-move Value, and supplies both outputs to the existing tree search. The new architecture changes how the shared representation is formed so that a fixed CPU inference budget supports more useful chess computation.

The design follows four complementary architectural choices. Side-to-move canonicalization identifies color-equivalent positions and actions with one representation. A projection-spatial-projection residual path reduces the cost of the trunk, and structural reparameterization gives its local transformation a richer training form that fuses into one inference operation. A learnable board relation supplies efficient spatial communication according to three requirements:

1. Equal local patterns may require different representations on different absolute squares.
2. Squares related by chess movement geometry require short communication paths.
3. The network must remain free to learn relationships outside the supplied geometry.

Together, these requirements define one learnable full-board relation whose initialization reflects chessboard geometry and whose trained form remains unrestricted over square pairs.

## 2. Canonical Chess Representation

Let $x$ be a complete chess state and let $c(x)$ be its side-to-move canonical form. When White is to move, $c(x)$ preserves the board orientation and replaces fixed colors with the roles `friendly` and `opposing`. When Black is to move, $c(x)$ reflects every rank, exchanges the two colors and then assigns the same two roles. In rank-file coordinates, the square transformation is

$$
(r,f)\longmapsto(7-r,f).
$$

The transformation also exchanges the two players' castling rights. Kingside and queenside remain unchanged because the file coordinate is preserved. The en passant square undergoes the same rank reflection, while its file remains unchanged.

The resulting network input contains six friendly piece planes, six opposing piece planes, four role-relative castling planes and one en passant file plane:

$$
s_c=\phi_G(c(x))\in\lbrace0,1\rbrace^{17\times8\times8}.
$$

Every canonical input presents the moving player as `friendly`, so the representation omits the resulting constant side-to-move plane. Canonicalization also applies to Policy labels. For a move played by Black, its source and destination ranks are reflected before its action index is computed. The inverse transformation maps a selected canonical action back to the original state. Consequently, equivalent White-to-move and Black-to-move structures share both their input representation and their Policy parameters.

## 3. Hidden State

Let $C=128$ be the trunk width. A bias-free $3\times3$ convolution with one-square padding, followed by batch normalization and ReLU, maps the canonical input to a board-aligned feature tensor:

$$
z_0=\operatorname{ReLU}\left(
\operatorname{BN}^{\mathrm{stem}}\left(
\operatorname{Conv}^{17\rightarrow C}_{3\times3,\mathrm{stem}}(s_c)
\right)
\right).
$$

A learned square embedding then supplies the absolute identity of each board location:

$$
h_0(q)=z_0(q)+e(q),
\qquad
e(q)\in\mathbb R^C,
$$

where $q$ ranges over the 64 squares. The residual path carries this position-conditioned representation through the trunk. Equal local patterns on different squares can therefore receive different representations immediately, instead of deriving their locations gradually from the board boundary.

Every subsequent hidden tensor has shape

$$
h_j\in\mathbb R^{C\times8\times8}.
$$

Training determines the semantics of the channels, while the network architecture determines which board relationships they can express efficiently.

## 4. Chess-Structured Residual Transformation

The trunk consists of $B$ repetitions of one chess-structured residual transformation. Each repetition receives $h_j$ and produces $h_{j+1}$ through a projection-spatial-projection path whose internal width is $R=128$. This document uses *bottleneck-style* for that topology. The initial equality $R=C$ retains all 128 channels inside the path, while the two $1\times1$ projections replace one of the two full $3\times3$ convolutions in a standard residual block and reduce its arithmetic cost.

For square $q$, the reduction projection produces

$$
b_j(q)=\operatorname{ReLU}\left(
\operatorname{BN}^{\mathrm{down}}_j\left(W^{\mathrm{down}}_j h_j(q)\right)
\right)
\in\mathbb R^R.
$$

The transformation of $b_j$ combines local convolution with a learned relation over all 64 squares before the result is projected back to $C$ channels. The following sections define these two aspects of the same transformation.

### 4.1 Local Geometry

The local term $L_j(b_j)$ is produced by a reparameterized $3\times3$ transformation. During training, it is written as the sum of a $3\times3$ branch, a $1\times1$ branch and an identity branch:

$$
L_j(b)=
\operatorname{BN}_{j,3}\!\left(\operatorname{Conv}_{3\times3,j}(b)\right)
+\operatorname{BN}_{j,1}\!\left(\operatorname{Conv}_{1\times1,j}(b)\right)
+\operatorname{BN}_{j,I}(b).
$$

The three branches provide separate parameterizations for spatial mixing, channel mixing and identity propagation during optimization. Their batch-normalization biases are initialized to zero, and their scale parameters are initialized to $1/\sqrt3$. Under the approximation that the normalized branch outputs have unit variance and weak correlation, their sum therefore has unit variance at initialization. Before inference, each batch-normalization transform is folded into its preceding linear operation, the $1\times1$ kernel is embedded at the center of a $3\times3$ kernel, and the identity transform is represented by a centered identity kernel. Adding the resulting kernels and biases gives one exactly equivalent $3\times3$ convolution:

$$
L_j(b)=\operatorname{Conv}^{\mathrm{fused}}_{3\times3,j}(b).
$$

The three training branches therefore fuse into one inference branch.

### 4.2 Full-Board Relation

The board contains exactly 64 squares, so one $64\times64$ matrix can represent a linear relation between every ordered pair of squares. Gadus uses this fixed board size directly to give every square a one-step path to every other square.

Let $K=8$ be the number of computational relation groups and require $K$ to divide $R$. Their common width is $d=R/K=16$. Reshape the bottleneck tensor as

$$
b_j=
\operatorname{Concat}\left(b_{j,0},\ldots,b_{j,K-1}\right),
\qquad
b_{j,g}\in\mathbb R^{64\times d}.
$$

The geometry dictionary contains $H=8$ normalized basis matrices $\widehat B_0,\ldots,\widehat B_{H-1}$. Each computational group learns a combination of the complete dictionary together with an unrestricted residual matrix:

$$
A_{j,g}=
\sum_{r=0}^{H-1}\alpha_{j,g,r}\widehat B_r
+\Delta A_{j,g}
\in\mathbb R^{64\times64}.
$$

The coefficients $\alpha_{j,g,r}$ and every component of $\Delta A_{j,g}$ are trainable. In the first configuration, where $K=H=8$, initialization assigns a different basis to each group:

$$
\alpha_{j,g,r}=\mathbf 1[g=r],
\qquad
\Delta A_{j,g}=0.
$$

This initialization gives the groups distinct propagation patterns and breaks their permutation symmetry. Training can subsequently mix every geometric basis into every group. The number of computational groups and the number of geometric bases remain conceptually independent and may differ in later configurations.

Write the rank and file coordinates of squares $q$ and $p$ as $(r_q,f_q)$ and $(r_p,f_p)$. The eight binary matrices underlying the geometry dictionary are

$$
\begin{aligned}
B_0(q,p)&=\mathbf 1[q=p],\\
B_1(q,p)&=\mathbf 1[\max(|r_q-r_p|,|f_q-f_p|)=1],\\
B_2(q,p)&=\mathbf 1[(|r_q-r_p|,|f_q-f_p|)\in\{(1,2),(2,1)\}],\\
B_3(q,p)&=\mathbf 1[r_q=r_p\ \land\ q\ne p],\\
B_4(q,p)&=\mathbf 1[f_q=f_p\ \land\ q\ne p],\\
B_5(q,p)&=\mathbf 1[r_q-f_q=r_p-f_p\ \land\ q\ne p],\\
B_6(q,p)&=\mathbf 1[r_q+f_q=r_p+f_p\ \land\ q\ne p],\\
B_7(q,p)&=1.
\end{aligned}
$$

Thus $B_0$ represents identity, $B_1$ king-step adjacency, $B_2$ knight displacement, $B_3$ ranks, $B_4$ files, $B_5$ rising diagonals, $B_6$ falling diagonals and $B_7$ the complete board. Row normalization gives

$$
\widehat B_g(q,p)=
\frac{B_g(q,p)}
{\max\left(1,\sum_{t=0}^{63}B_g(q,t)\right)}.
$$

In particular, every row of $\widehat B_7$ contains 64 entries equal to $1/64$. These matrices describe geometry. Piece values, tactical meanings and move preferences remain learned quantities. The coefficient vector $\alpha_{j,g,:}$ provides coordinated directions for changing geometric relations, while $\Delta A_{j,g}$ can add, remove or reverse individual square-pair connections. Each resulting $A_{j,g}$ can therefore represent any real $64\times64$ linear relation between square positions, and inference stores only the resulting matrix.

For group $g$, the relation transform is

$$
s_{j,g}=A_{j,g}b_{j,g}
\in\mathbb R^{64\times d}.
$$

Concatenating all groups and restoring board layout gives

$$
S_j(b_j)=\operatorname{BoardShape}\left(
\operatorname{Concat}(s_{j,0},\ldots,s_{j,K-1})
\right)
\in\mathbb R^{R\times8\times8}.
$$

The matrices act on absolute square indices, so they can learn position-dependent relationships. Their dense trainable residuals connect arbitrary square pairs, and the shared geometry dictionary provides short optimization paths for common chessboard relations. The hidden features entering the matrices depend on the entire encoded position. Repeated relation transforms, channel projections and nonlinearities therefore make the network output position-dependent even though each trained matrix $A_{j,g}$ remains fixed during inference.

One matrix is shared by the $d$ channels within its computational group. The reduction projection learns how to route trunk channels into the $K$ groups, each group learns its own combination of the geometry dictionary, and the expansion projection recombines their outputs. This factorization limits the cost of one block to $K$ spatial matrices while allowing successive blocks to assign different channel mixtures and geometric combinations to every group.

Before the two paths are combined, non-affine channel normalization standardizes the full-board output:

$$
\widehat S_j(b_j)=\operatorname{Norm}^{\mathrm{relation}}_j(S_j(b_j)).
$$

The normalization maintains running channel means and variances for inference, with unit affine scale and zero affine bias. A learned vector $\lambda_j\in\mathbb R^R$ then controls the full-board contribution to each channel. The two paths are combined with a variance-preserving parameterization:

$$
u_{j,c}=\operatorname{ReLU}\left(
\frac{
L_j(b_j)_c+\lambda_{j,c}\widehat S_j(b_j)_c
}{
\sqrt{1+\lambda_{j,c}^2}
}
\right),
\qquad 0\leq c<R.
$$

Initialization sets $\lambda_{j,c}=1$. Because $L_j(b_j)_c$ and $\widehat S_j(b_j)_c$ both have approximately unit variance at initialization, the denominator keeps their initial combination near unit variance while allowing the supplied geometry to participate in the first forward and backward passes. Training can move each $\lambda_{j,c}$ toward zero, increase its magnitude or reverse its sign according to the role of that channel. The learned coefficients become fixed channel scales during inference.

Taken together, the initialization gives every block balanced local and full-board paths, assigns distinct geometric propagation to its computational groups and sets every unrestricted square-pair correction to zero. Optimization therefore begins from explicit chessboard geometry and can reach arbitrary learned square relations through $\alpha_{j,g,r}$ and $\Delta A_{j,g}$.

### 4.3 Residual Update

The expansion projection and its batch normalization return the bottleneck update to the trunk width. The outer residual path then combines that update with the incoming representation:

$$
h_{j+1}(q)=
\operatorname{ReLU}\left(
h_j(q)+
\operatorname{BN}^{\mathrm{up}}_j\left(W^{\mathrm{up}}_j u_j(q)\right)
\right).
$$

Combining the preceding definitions gives the repeated computation

$$
h_j
\longrightarrow
\text{channel projection}
\longrightarrow
\text{local and full-board relation transform}
\longrightarrow
\text{channel projection and residual addition}
\longrightarrow h_{j+1}.
$$

Every trunk repetition follows this definition. Local convolution remains the spatial base, and the trainable $64\times64$ matrices provide direct communication between arbitrary square pairs from an initialization that reflects chessboard geometry.

### 4.4 Depth-Aware Initialization

The square embedding is initialized to zero, so the first hidden representation begins with the canonical board features and acquires absolute-square corrections through training. Convolutional and linear weights use variance-preserving initialization appropriate to their activation functions. The scale of the final batch normalization in each residual update depends on the length $D$ of its residual sequence:

$$
\gamma^{\mathrm{up}}=\frac1{\sqrt D},
\qquad
\beta^{\mathrm{up}}=0.
$$

The trunk uses $D=B$, the Policy sequence uses $D=2$ and the Value sequence uses $D=1$. If the unscaled residual updates have comparable variance and limited correlation, their accumulated variance is proportional to $D(\gamma^{\mathrm{up}})^2=1$. This initialization keeps the total initial residual contribution bounded as a sequence becomes deeper, while every transformation participates in the first forward and backward passes.

## 5. Policy and Value

After $B$ chess-structured residual transformations, $h_B$ is the shared representation used by the Policy and Value computations:

$$
s_c\longrightarrow h_0\longrightarrow\cdots\longrightarrow h_B
\longrightarrow
\begin{cases}
\ell_\theta(s_c),\\
V_\theta(s_c).
\end{cases}
$$

The Policy first projects $h_B$ to 128 channels and applies two head-specific chess-structured residual transformations. Writing their output as $p_2$, a $1\times1$ projection produces 73 motion-pattern planes, and a learned action bias supplies one scalar for every source-square and motion-pattern pair:

$$
p_0=\operatorname{ReLU}\left(
\operatorname{BN}_P\left(\operatorname{Conv}^{C\rightarrow128}_{1\times1,P}(h_B)\right)
\right),
$$

$$
p_{k+1}=\mathcal F_{P,k}(p_k),
\qquad 0\leq k<2,
$$

$$
L_\theta(s_c)=
\operatorname{Conv}^{128\rightarrow73}_{1\times1,\mathrm{out}}(p_2)+B_P
\in\mathbb R^{73\times8\times8}.
$$

The learned bias $B_P\in\mathbb R^{73\times8\times8}$ assigns an independent offset to every canonical source-square and motion-pattern pair.

Each $\mathcal F_{P,k}$ follows the transformation defined in Section 4 with independent Policy parameters. For canonical source square $q=8r+f$ and motion pattern $m$, the Policy logit is

$$
\ell_\theta(s_c,73q+m)=L_\theta(s_c)_{m,r,f}.
$$

The Policy therefore retains the $64\times73$ action organization. Legal-action inference evaluates only the indices available in the complete state, and the trunk's square embedding supplies the absolute position information used by the Policy computation.

The Value applies one head-specific chess-structured residual transformation to $h_B$, projects the result to 48 channels and flattens the resulting board tensor. A $3072\rightarrow512$ linear map with ReLU produces the hidden Value vector, and a final $512\rightarrow1$ linear map with hyperbolic tangent produces

$$
V_\theta(s_c)\in[-1,1].
$$

The Policy and Value heads have independent parameters after $h_B$. Gradients from both objectives update the shared trunk, while each head separately transforms the shared representation into its required output.

The output layers begin near constant predictions derived from a deterministic stratified sample of the training targets. The sample draws complete HDF5 chunks at evenly spaced locations throughout the dataset so that initialization remains inexpensive even when the complete dataset is large. Let $n_i$ be the number of sampled records whose canonical action index is $i$, let $N$ be the sample size and let $\varepsilon_P>0$ be a smoothing count. The empirical action prior is

$$
\overline\pi_i=
\frac{n_i+\varepsilon_P}
{N+4672\varepsilon_P},
$$

and the Policy action bias is initialized by

$$
B_P(i)=\log\overline\pi_i.
$$

SoftMax applied to these biases produces $\overline\pi$. Let $W^{\mathrm{vp}}$ denote a random output weight matrix drawn with the variance-preserving scale appropriate to its input. The Policy output projection is initialized as

$$
W_P^{(0)}=\eta W_P^{\mathrm{vp}},
\qquad 0<\eta<1.
$$

The factor $\eta$ keeps the initial position-dependent correction smaller than the empirical action bias, while its positive value allows gradients to reach the preceding Policy representation in the first update.

For the sampled Value targets $z_1,\ldots,z_N$, define

$$
\overline z=\frac1N\sum_{n=1}^{N}z_n.
$$

Because $\overline z$ minimizes the mean-squared error among constant predictions, the bias of the final Value map is initialized to

$$
b_V=\operatorname{arctanh}\left(
\operatorname{clip}(\overline z,-1+\varepsilon_V,1-\varepsilon_V)
\right),
$$

where $\varepsilon_V>0$ keeps the inverse hyperbolic tangent finite. The final Value weights use the corresponding initialization

$$
W_V^{(0)}=\eta W_V^{\mathrm{vp}}.
$$

This scale keeps the initial output near $\overline z$ and gives the preceding Value representation a nonzero gradient from the first update. The data-derived biases supply starting predictions rather than fixed chess rules, and training may move them freely.

## 6. Computational Budget

A standard $C$-channel Gadus residual block contains two full $3\times3$ convolutions and therefore approximately

$$
18C^2
$$

convolution weights. A bottleneck path with $R=C=128$ contains two $1\times1$ projections and one $3\times3$ spatial transformation:

$$
CR+9R^2+RC=11C^2.
$$

Its local convolutional path therefore uses approximately $61\%$ of the weights and multiply-accumulate operations of a two-convolution standard block. During training, the full-board relation adds $K$ residual matrices with $64^2$ parameters each, $KH$ geometry coefficients and $R$ path-balance coefficients. Applying one relation matrix to every channel in its group requires

$$
K\times64^2\times d=64^2R
$$

multiply-accumulate operations, because $Kd=R$.

The first concrete configuration uses

$$
C=128,
\qquad
R=128,
\qquad
B=12.
$$

The shared trunk therefore evaluates 12 chess-structured transformations. The Policy head evaluates two additional transformations, and the Value head evaluates one, giving 15 in total for a complete Policy-Value evaluation.

For $C=R=128$ and $K=H=8$, model loading absorbs the $KH$ geometry coefficients into the $K$ relation matrices. One deployed transformation therefore contains

$$
11C^2+K\times64^2+R=213{,}120
$$

principal parameters before normalization terms and biases. Its corresponding board-level arithmetic cost is approximately

$$
64\times11C^2+64^2R=12{,}058{,}624
$$

multiply-accumulate operations. A two-convolution 128-channel standard block requires $18{,}874{,}368$ multiply-accumulate operations on the same board. Fifteen proposed transformations therefore require about $64\%$ of the principal arithmetic and fewer principal parameters than fifteen standard blocks. The training form additionally evaluates the $1\times1$ and identity reparameterization branches.

Including the stem, square embedding, Policy projections, Value projections and Value linear layers gives approximately $4.84$ million principal parameters in the fused inference form. Their raw FP32 storage occupies about $19.3$ MB, excluding the small normalization metadata. For each transformation, the training form additionally contains $R^2$ local-branch weights and $KH$ geometry coefficients, and both are absorbed into fused operators before deployment.

The configuration fixes the dimensions used by the design that follows. Other values of $B$ and $R$ preserve the same architecture when their tensor dimensions remain compatible.

## 7. Execution Definition

The following pseudocode specifies the canonical state and action transformations:

```text
function canonical_state(x):
    if x.side_to_move is White:
        canonical_x = x
    else:
        canonical_x = exchange_piece_colors(reflect_ranks(x))
    return role_planes(canonical_x)

function canonical_move(a, side_to_move):
    return a if side_to_move is White else reflect_move_ranks(a)

function canonical_action_index(a, side_to_move):
    return action_index(canonical_move(a, side_to_move))

function restore_move(canonical_a, side_to_move):
    return canonical_a if side_to_move is White else reflect_move_ranks(canonical_a)
```

`reflect_ranks` transforms piece squares, castling rights and the en passant square according to Section 2. The move transformation reflects both endpoints and preserves the promotion piece.

The fixed relation bases are constructed once from square coordinates:

```text
function construct_relation_bases():
    basis[0] = self_relation()
    basis[1] = king_step_relation()
    basis[2] = knight_relation()
    basis[3] = same_rank_relation()
    basis[4] = same_file_relation()
    basis[5] = rising_diagonal_relation()
    basis[6] = falling_diagonal_relation()
    basis[7] = complete_board_relation()
    return row_normalize(basis)

function initialize_chess_transform(parameters, sequence_depth):
    parameters.alpha = zeros([K, H])
    for g in 0 .. K - 1:
        parameters.alpha[g, g mod H] = 1
    parameters.delta_A = zeros([K, 64, 64])
    parameters.lambda = ones([R])
    parameters.local_bn_scale = fill([3, R], 1 / sqrt(3))
    parameters.local_bn_bias = zeros([3, R])
    parameters.up_bn_scale = fill([C], 1 / sqrt(sequence_depth))
    parameters.up_bn_bias = zeros([C])

function initialize_network(dataset_statistics, eta):
    square_embedding = zeros([64, C])
    for transform in trunk:
        initialize_chess_transform(transform, sequence_depth = B)
    for transform in policy_blocks:
        initialize_chess_transform(transform, sequence_depth = 2)
    initialize_chess_transform(value_block, sequence_depth = 1)
    policy_action_bias = log(smoothed_action_frequencies(dataset_statistics))
    policy_output_weights = eta * variance_preserving_weights()
    value_output_bias = atanh(clipped_mean_value_target(dataset_statistics))
    value_output_weights = eta * variance_preserving_weights()
```

One training-form chess-structured residual transformation is:

```text
function chess_transform(h, parameters, relation_basis):
    b = relu(batch_norm(down_1x1(h)))

    local = batch_norm_3x3(conv_3x3(b))
          + batch_norm_1x1(conv_1x1(b))
          + batch_norm_identity(b)

    grouped = reshape(b, [batch, K, 64, d])
    for g in 0 .. K - 1:
        A[g] = delta_A[g]
        for r in 0 .. H - 1:
            A[g] += alpha[g, r] * relation_basis[r]
        related[g] = A[g] @ grouped[g]

    relation = normalize_without_affine(restore_board(concat(related)))
    local_scale = reciprocal_sqrt(1 + lambda * lambda)
    relation_scale = lambda * local_scale
    u = relu(local_scale * local + relation_scale * relation)
    update = batch_norm_up(up_1x1(u))
    return relu(h + update)
```

The multiplication over $g$ denotes one grouped batched matrix multiplication rather than eight separate tensor launches. Its input has shape $[N,K,64,d]$, its relation tensor has shape $[K,64,64]$, and its output has shape $[N,K,64,d]$. The geometry combinations are shared across the minibatch and are formed once per transformation in each training forward pass.

The complete network evaluation is:

```text
function evaluate(x):
    s = canonical_state(x)
    h = relu(batch_norm(stem_3x3(s))) + square_embedding

    for j in 0 .. B - 1:
        h = chess_transform(h, trunk[j], relation_basis)

    p = relu(batch_norm(policy_1x1(h)))
    for j in 0 .. 1:
        p = chess_transform(p, policy_block[j], relation_basis)
    policy_planes = policy_out_1x1(p) + policy_action_bias

    v = chess_transform(h, value_block, relation_basis)
    v = relu(batch_norm(value_1x1(v)))
    value = tanh(value_out(relu(value_hidden(flatten(v)))))

    return policy_planes, value
```

For a legal move $a$, inference reads `flatten(policy_planes)[canonical_action_index(a, x.side_to_move)]`. SoftMax normalization is then applied across the legal moves of $x$.

## 8. Training and Inference Forms

For one supervised record, let $i^*$ be the canonical action index and let $z\in[-1,1]$ be the Value target from the canonical side-to-move perspective. SoftMax over all 4672 Policy logits gives

$$
\rho_\theta(i\mid s_c)=
\frac{\exp\ell_\theta(s_c,i)}
{\displaystyle\sum_{k=0}^{4671}\exp\ell_\theta(s_c,k)}.
$$

The Policy loss, Value loss and combined supervised objective are

$$
L_P=-\log\rho_\theta(i^*\mid s_c),
\qquad
L_V=\left(V_\theta(s_c)-z\right)^2,
\qquad
L_{\mathrm{sup}}=L_P+w_VL_V,
$$

where $w_V\geq0$ controls the Value contribution. Gradients of $L_{\mathrm{sup}}$ update the shared trunk and both output heads. The canonical state and action transformations apply identically during preprocessing and live inference, so both contexts use the same state and action coordinates.

Inference folds the stem, reduction, expansion and head-projection batch-normalization transforms into their preceding convolutions. It also fuses the three local branches in every chess-structured transformation into one $3\times3$ convolution. For every block and computational relation group, model loading precomputes

$$
A_{j,g}=
\sum_{r=0}^{H-1}\alpha_{j,g,r}\widehat B_r
+\Delta A_{j,g}
$$

It also precomputes the two path coefficients

$$
a_{j,c}=\frac1{\sqrt{1+\lambda_{j,c}^2}},
\qquad
b_{j,c}=\frac{\lambda_{j,c}}{\sqrt{1+\lambda_{j,c}^2}}.
$$

The fixed inference statistics of the non-affine relation normalization turn it into a channelwise affine transform. Its scale is combined with $b_{j,c}$, its additive term is merged into the scaled local-convolution bias, and $a_{j,c}$ is folded into the fused local kernel and bias. A forward pass therefore evaluates one fused local convolution, one fixed grouped board-relation multiplication and one channelwise relation scaling inside each bottleneck-style path.

## 9. Feasibility Review

### 9.1 Representational Feasibility

The full matrix $\Delta A_{j,g}$ makes each $A_{j,g}$ unrestricted. The geometry coefficients give every computational group direct access to the complete basis dictionary, while training can construct any alternative square relation. The local convolution and outer residual path remain available in every block, so the relation transform adds spatial capacity while preserving the convolutional representation.

Each relation matrix is static after training, and the tensor it transforms is produced from the current encoded position. Channel projections before and after the relation multiplication, nonlinear activations and repeated blocks allow the effect of a square relation to depend on that position. Fixed relation matrices and repeated nonlinear transformations provide global connectivity at a predictable cost.

The grouped relation branch does not represent an arbitrary linear map over all $64R$ spatial-channel components in one block. It represents $K$ unrestricted spatial maps, each shared by $d$ routed channels. The learned reduction and expansion projections change that routing in every block, so depth composes multiple spatial and channel factorizations. This is the principal capacity-cost tradeoff of the design.

The path-balance vector allows each block to retain, suppress or emphasize its full-board correction independently in every channel. Consequently, increasing $B$ does not force every additional block to perform a fixed-strength global mixture. The geometric initialization participates from the first training pass because $\lambda_{j,c}=1$, while the learned balance can reduce redundant relation paths in deeper representations.

### 9.2 Initialization and Optimization

The geometry dictionary changes the starting point of optimization without restricting the trained relation matrices. At initialization, each group already propagates information through one defined board relation, so rank, file, diagonal, knight and full-board communication each require one relation multiplication. A network containing only local $3\times3$ convolutions must compose several blocks before information can travel between distant squares.

The initialization also provides coordinated parameter directions. Differentiating a relation matrix with respect to one geometry coefficient gives

$$
\frac{\partial A_{j,g}}{\partial\alpha_{j,g,r}}=\widehat B_r.
$$

One update to $\alpha_{j,g,r}$ therefore changes every square pair represented by $\widehat B_r$ together, while $\Delta A_{j,g}$ remains available for individual square-pair corrections. The model can first adjust a low-dimensional geometric relation and then refine the parts that require different behavior.

This starting point has a conditional optimization advantage. Let $r_g$ be the basis assigned to group $g$ and let $A^*_{j,g}$ denote a useful relation for that group. If the useful relation is close to the assigned basis, it can be written as

$$
A^*_{j,g}=
\widehat B_{r_g}+E_{j,g},
\qquad
\|E_{j,g}\|\ \text{is small}.
$$

The proposed initialization then has distance $\|E_{j,g}\|$ from $A^*_{j,g}$. It begins closer than another initialization $A'_{j,g}$ precisely when $\|E_{j,g}\|<\|A'_{j,g}-A^*_{j,g}\|$. Under a local quadratic approximation to the training objective, an iterative update with contraction factor $0<\rho<1$ satisfies a bound of the form

$$
\|A^{(t)}-A^*\|
\leq
\rho^t\|A^{(0)}-A^*\|.
$$

Reducing the initial distance $\|A^{(0)}-A^*\|$ consequently reduces the iterations required to reach a fixed error under that approximation. The architecture establishes the short propagation paths, coordinated gradient directions and unrestricted final relation exactly. The data and objective determine whether the relation useful to each group is close to its assigned initial basis.

### 9.3 Implementation Feasibility

The canonical codec requires rank reflection, role-relative piece planes and the corresponding reflection of Policy labels. Existing 18-plane HDF5 records contain the required side-to-move and color information, so a data loader can derive the 17 canonical planes and transformed labels during batch construction. The changed input shape and network operations require newly initialized model parameters.

The bottleneck-style and reparameterization operations map directly to existing LibTorch convolution and batch-normalization modules. The relation transform maps to one grouped batched matrix multiplication over contiguous tensors. The complete set of relation residuals occupies about $0.49$ million parameters across 15 transformations, which is small relative to the convolutional parameters. Geometry coefficients and path-balance coefficients add only $KH+R=192$ trainable parameters to each transformation, and fusion absorbs the geometry coefficients into the deployed relation matrices. Relation normalization and path balancing share one channelwise scale on the output of the matrix operation during inference.

### 9.4 Implementation Sequence

The implementation can proceed through four independently testable stages:

1. Implement state and action canonicalization, then verify the equivalences in tests 1 and 7 of Section 10.
2. Implement the bottleneck block and local reparameterization, then verify exact fusion before adding the relation branch.
3. Add the grouped relation multiplication and compare it with a direct reference implementation.
4. Integrate the Policy and Value heads and train the complete network from the initialization defined above.

Each stage preserves a runnable model and isolates one class of implementation error. The architecture is therefore implementable with the current LibTorch training stack and admits a fused inference form.

### 9.5 Feasibility Verdict

The design is mathematically well-defined and can be trained with the existing state, action and Value targets after canonical conversion. Its local branch has an exact fused inference form, and its relation branch has bounded dimensions independent of network depth or game state. Its initialization supplies one-step geometric propagation and coordinated optimization directions while preserving unrestricted trained square relations. These properties establish representational and implementation feasibility.

## 10. Required Verification

The architecture requires tests at eight boundaries:

1. Canonicalization must map a state, its legal actions and its targets to the same representation obtained from the corresponding color-reflected state.
2. Reparameterization fusion must preserve Policy logits and Value outputs within floating-point tolerance.
3. Every fixed relation basis must match its coordinate definition and row normalization.
4. A relation matrix reconstructed from $\alpha_{j,g,r}$, $\widehat B_r$ and $\Delta A_{j,g}$ must match the matrix used by inference.
5. Initialization must reproduce the one-hot geometry coefficients, zero relation residuals, local branch scales $1/\sqrt3$, depth-aware residual scales, path-balance coefficients and data-derived output biases defined above.
6. The training-form path combination and its inference coefficients $a_{j,c}$ and $b_{j,c}$ must produce equal outputs within floating-point tolerance.
7. Training and inference must agree on canonical action indices for ordinary moves, castling, en passant and every promotion type.
8. Forward and backward passes must preserve tensor shapes and finite values for every supported batch size.
