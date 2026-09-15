# Eleginus

Eleginus is a formula-based handcrafted evaluation engine that maps explicit chess signals to a White-perspective score and searches legal continuations with iterative-deepening principal variation search.

## 1. Notation

- $\mathcal X$ is the set of complete chess states maintained by the rules engine.
- $x\in\mathcal X$ is a complete chess state containing the board, side to move, castling rights, en passant state, move counters and repetition history.
- $\mathcal A(x)$ is the set of legal actions in state $x$.
- $T(x,a)$ is the complete state reached by applying legal action $a\in\mathcal A(x)$ to state $x$.
- $B(x)$ is the evaluation projection containing twelve piece bitboards, the side to move and four castling-right bits.
- $\mathcal I_E=\lbrace0,\ldots,693\rbrace$ is the ordered formula-index set.
- $\phi_i(B(x))\in\mathbb Z$ is the scalar signal emitted by formula $i\in\mathcal I_E$.
- $n_t(x)$ is the total number of pieces of type $t$ for $t\in\lbrace P,N,B,R,Q\rbrace$.
- $n_t^0$ is the initial total count of piece type $t$, with $(n_P^0,n_N^0,n_B^0,n_R^0,n_Q^0)=(16,4,4,4,2)$.
- $m_t(x)$ is the normalized material coordinate of piece type $t$.
- $b_i$ is the base coefficient of formula $i$.
- $r_{i,t}$ is the material-response coefficient of formula $i$ for piece type $t$.
- $w_i(x)$ is the effective coefficient of formula $i$ in state $x$.
- $H(x)$ is the White-perspective formula score before conversion and draw-oriented adjustments.
- $E(x)$ is the final White-perspective Eleginus evaluation.
- $C(x)$ is the bounded integer score used by negamax search from the perspective of the side to move.
- $V(x,d,\alpha,\beta)$ is the depth-$d$ search value of state $x$ inside window $[\alpha,\beta)$.
- $\mathrm{clip}_{[l,u]}(y)=\min(u,\max(l,y))$ restricts scalar $y$ to the closed interval $[l,u]$.

## 2. Formula Evaluation

### 2.1 Board Projection

The evaluation projection is

$$
B(x)=
\left(
P_W,N_W,B_W,R_W,Q_W,K_W,
P_B,N_B,B_B,R_B,Q_B,K_B,
\tau,c
\right).
$$

Each piece component is a 64-bit occupancy set. The scalar $\tau\in\lbrace0,1\rbrace$ identifies the side to move. The four low bits of $c$ represent White kingside, White queenside, Black kingside and Black queenside castling rights.

Formula evaluation uses White and Black as two role values. A role-relative vertical reflection maps Black-oriented ranks to White-oriented ranks while preserving files. Piece-square tables, pawn ranks, king regions and directional shifts use these normalized coordinates.

### 2.2 Formula Algebra

Every formula is an expression over square sets, signed integers and Boolean relations. The mathematical notation maps directly to the formula primitives:

| Mathematical notation | Meaning | Source primitive |
| --- | --- | --- |
| $k\in\mathbb Z$ and $X\subseteq\mathcal Q$ | Signed integer signal and square-set signal over the 64-square board $\mathcal Q$ | `NUM`, `BB` |
| $B_a(x)$ | Atom $a$ from the board projection | `INP` |
| $u+v$, $u-v$, $uv$, $|u|$ | Integer arithmetic and absolute value | `ADD`, `SUB`, `MUL`, `ABS` |
| $p\land q$, $p\lor q$, $\neg p$ | Boolean conjunction, disjunction and complement | `LAND`, `LOR`, `LNOT` |
| $\mathbf 1[p]$ | Indicator of a comparison or occupancy relation | `EQ`, `GT`, `LT`, `LE`, `GE`, `ANY` |
| $X\cap Y$, $X\cup Y$, $\overline X$ | Square-set intersection, union and complement | `AND`, `OR`, `NOT` |
| $X\setminus Y$ | $X\cap\overline Y$ | `AND`, `NOT` |
| $\lvert X\rvert$ | Number of squares in $X$ | `POP` |
| $P_{c,t}$ | Pieces of role $c$ and type $t$ | `PCS` |
| $\rho_c(X)$ and $\rho_c(q)$ | Role-relative square-set and square transformations | `REL`, `SQ` |
| $C_{c,w}$ | Castling right of role $c$ on wing $w$ | `CR` |
| $\sigma_{c,d}(X)$ | One role-relative step in direction $d$ | `SH` |

For example, $k=2$ is an integer signal, $X=\lbrace e4\rbrace$ is a one-square signal and $X=R_4$ is the eight-square set representing rank 4. Arithmetic consumes integer signals, while set operations consume square-set signals.

The direction set is

$$
\mathcal D=
\lbrace
\uparrow,\downarrow,\rightarrow,\leftarrow,
\nearrow,\nwarrow,\searrow,\swarrow
\rbrace,
$$

corresponding to forward, backward, east, west and the four diagonals. Boolean values use $0$ and $1$, repeated unions and sums use fixed finite reductions, and every displayed formula expands into the primitives in the table.

The evaluator derives reusable chess quantities through the same algebra. These quantities include pawn attacks, pawn spans, passed pawns, attack unions, double attacks, piece attack maps, sliding rays, mobility histograms, king rings, pawn shelter and pawn storms. Formula evaluation emits one signed scalar for each formula index.

Most signals have the antisymmetric form

$$
\phi_i(B(x))=g_i(W,x)-g_i(B,x),
$$

which gives positive values to White-favored instances and negative values to Black-favored instances. Tempo incorporates $\tau$, while castling and role-relative geometry preserve the same White-perspective convention.

### 2.3 Principal Formula Expressions

Let $c\in\lbrace W,B\rbrace$ denote one role and let $\bar c$ denote the opposing role. For any role-indexed scalar expression $g$, define the role difference

$$
\Delta g=g(W)-g(B).
$$

Let $P_{c,t}$ denote the pieces of role $c$ and type $t$, and let $\mathcal T=\lbrace P,N,B,R,Q,K\rbrace$. The role occupancy and complete occupancy are

$$
O_c=\bigcup_{t\in\mathcal T}P_{c,t},
\qquad
O=O_W\cup O_B.
$$

Repeated role-relative shifts define vertical fill:

$$
\mathcal V_c(X)=
\bigcup_{k=0}^{7}\sigma_{c,\uparrow}^{k}(X),
$$

where $\sigma_{c,\uparrow}^{0}(X)=X$. Filling in both vertical directions gives the occupied-file set

$$
\mathcal L_c(X)=\mathcal V_c(X)\cup\mathcal V_{\bar c}(X).
$$

Pawn attacks are the union of the two forward diagonals:

$$
A_{c,P}=\sigma_{c,\nearrow}(P_{c,P})\cup\sigma_{c,\nwarrow}(P_{c,P}).
$$

King attacks from square set $X$ are

$$
\mathcal K_c(X)=\bigcup_{d\in\mathcal D}\sigma_{c,d}(X).
$$

Knight attacks use the eight two-plus-one shift compositions. With

$$
\mathcal N=\lbrace
(\uparrow,\rightarrow),(\uparrow,\leftarrow),
(\downarrow,\rightarrow),(\downarrow,\leftarrow),
(\rightarrow,\uparrow),(\rightarrow,\downarrow),
(\leftarrow,\uparrow),(\leftarrow,\downarrow)
\rbrace,
$$

their primitive expansion is

$$
\mathcal N_c(X)=
\bigcup_{(a,b)\in\mathcal N}
\sigma_{c,b}\left(\sigma_{c,a}^{2}(X)\right).
$$

A sliding ray begins one step from its source and propagates through empty squares. For direction $d$, define

$$
U_1=\sigma_{c,d}(X),
$$

$$
U_{k+1}=\sigma_{c,d}(U_k\setminus O),
\qquad 1\leq k<7,
$$

and

$$
\mathcal R_{c,d}(X)=\bigcup_{k=1}^{7}U_k.
$$

Bishop attacks combine the four diagonal rays. Rook attacks combine the four orthogonal rays. Queen attacks combine all eight rays. Attack unions combine the corresponding per-piece attack sets, and double-attack sets collect squares reached by at least two friendly pieces. The evaluator computes the same sets through cached bitboard attack generators.

The tempo formula is

$$
\phi_{\mathrm{tempo}}=
\mathbf 1[\tau=W]-\mathbf 1[\tau=B].
$$

The five material formulas are

$$
\phi_{\mathrm{material},t}=
\Delta\lvert P_{c,t}\rvert,
\qquad t\in\lbrace P,N,B,R,Q\rbrace.
$$

For normalized square $q$ and piece type $t$, a piece-square formula is

$$
\phi_{\mathrm{pst},t,q}=
\mathbf 1[q\in\rho_W(P_{W,t})]-
\mathbf 1[q\in\rho_B(P_{B,t})].
$$

The bishop-pair formula is

$$
\phi_{\mathrm{bishopPair}}=
\Delta\mathbf 1[\lvert P_{c,B}\rvert\geq2].
$$

The opposing pawn span and adjacent spans define passed pawns:

$$
S_c=\mathcal V_{\bar c}(P_{\bar c,P}),
$$

$$
P_{c,\mathrm{pass}}=
P_{c,P}\setminus
\left(
S_c\cup\sigma_{c,\rightarrow}(S_c)\cup\sigma_{c,\leftarrow}(S_c)
\right).
$$

For normalized rank mask $R_r$, the passed-pawn rank formula is

$$
\phi_{\mathrm{pass},r}=
\Delta\left\lvert\rho_c(P_{c,\mathrm{pass}})\cap R_r\right\rvert.
$$

Vertically adjacent pawns and isolated pawns use

$$
\phi_{\mathrm{doubled}}=
\Delta\left\lvert
P_{c,P}\cap\sigma_{c,\uparrow}(P_{c,P})
\right\rvert,
$$

$$
J_c=
\sigma_{c,\rightarrow}(\mathcal L_c(P_{c,P}))
\cup
\sigma_{c,\leftarrow}(\mathcal L_c(P_{c,P})),
$$

$$
\phi_{\mathrm{isolated}}=
\Delta\left\lvert P_{c,P}\setminus J_c\right\rvert.
$$

Let $F_c=\mathcal L_c(P_{c,P})$. Rooks on open and semi-open files use

$$
\phi_{\mathrm{rookOpen}}=
\Delta\left\lvert
P_{c,R}\setminus(F_c\cup F_{\bar c})
\right\rvert,
$$

$$
\phi_{\mathrm{rookSemi}}=
\Delta\left\lvert
P_{c,R}\cap(F_{\bar c}\setminus F_c)
\right\rvert.
$$

The mobility area removes blocked pawns, early pawns and opposing pawn control:

$$
M_c=
\overline{
\left(P_{c,P}\cap\sigma_{c,\downarrow}(O)\right)
\cup
\left(P_{c,P}\cap\rho_c(R_2\cup R_3)\right)
\cup A_{\bar c,P}
}.
$$

For piece type $t$, per-piece attack set $A_{c,t}(q)$ and mobility bucket $k$, the primary mobility formula is

$$
\phi_{\mathrm{mobility},t,k}=
\Delta
\sum_{q\in P_{c,t}}
\mathbf 1\left[
\left\lvert A_{c,t}(q)\cap M_c\right\rvert=k
\right].
$$

The per-piece attack set expands through the pawn, knight, king or sliding attack expressions above.

Advanced pawn-supported squares define knight and bishop outposts. With $R_{4:6}$ denoting normalized ranks 4 through 6,

$$
X_c=
\rho_c(R_{4:6})
\cap A_{c,P}
\cap\overline{\mathcal V_{\bar c}(A_{\bar c,P})},
$$

$$
\phi_{\mathrm{outpost},t}=
\Delta\left\lvert P_{c,t}\cap X_c\right\rvert,
\qquad t\in\lbrace N,B\rbrace.
$$

Let $A_c$ denote the complete attack union and $D_c$ the double-attack set. Squares strongly guarded against role $c$ are

$$
G_c=
A_{\bar c,P}\cup(D_{\bar c}\setminus D_c).
$$

For opposing pawns $Y_P=P_{\bar c,P}$ and opposing pieces $Y_M=O_{\bar c}\setminus Y_P$, hanging targets are

$$
H_P(c)=
(Y_P\setminus G_c)\cap A_c\cap\overline{A_{\bar c}},
$$

$$
H_M(c)=
(Y_M\setminus G_c)
\cap A_c
\cap(\overline{A_{\bar c}}\cup D_c).
$$

Each victim-type signal intersects $H_P(c)$ or $H_M(c)$ with $P_{\bar c,t}$, counts the resulting squares and then takes the White-Black role difference.

King escape squares for defending role $c$ are

$$
e_c=\left\lvert
\mathcal K_c(P_{c,K})
\setminus(O_c\cup A_{\bar c})
\right\rvert.
$$

The escape formula is $\phi_{\mathrm{escape}}=\Delta e_c$. For attacking role $c$, inner king-zone pressure is

$$
q_c=
\sum_{t\in\lbrace P,N,B,R,Q\rbrace}
\left\lvert
A_{c,t}\cap
\left(P_{\bar c,K}\cup\mathcal K_{\bar c}(P_{\bar c,K})\right)
\right\rvert,
$$

where each union, intersection, count and sum follows the symbol table in Section 2.2. Section 2.6 gives the smooth response applied to $q_W$ and $q_B$.

The remaining square, rank, mobility, shelter, storm, threat and king-zone instances repeat these same primitive constructions over fixed coordinates and buckets.

### 2.4 Formula Set

The ordered formula set contains 694 scalar signals:

| Group | Signals | Chess structure |
| --- | ---: | --- |
| Tempo | 1 | Side to move |
| Material | 5 | Pawn, knight, bishop, rook and queen counts |
| Piece-square tables | 384 | Six piece types across 64 normalized squares |
| Bishop pair | 1 | Retention of at least two bishops |
| Pawns | 78 | Passed-pawn ranks, path safety, defended advances, blockades, clear files, connected and supported passers, phalanxes, defended pawns, doubled pawns, isolated pawns and king distances |
| Mobility | 64 | Primary mobility histograms for knights, bishops, rooks and queens, plus secondary rook and queen mobility |
| Piece placement | 23 | Minor pieces behind pawns, bishop color complexes, blocked centers, bishop-pawn buckets, bishop x-rays, rook files, outposts, restriction and space |
| Threats | 42 | Hanging targets, pawn attacks, minor-piece attacks, rook attacks, pawn-push attacks, active-side threats, king threats and queen pressure |
| King safety | 96 | Inner and outer king-zone attacks, potential checks, smooth pressure, escape squares, shelter, storms, open files, flank control, castling rights and queen presence |

The group sizes satisfy

$$
1+5+384+1+78+64+23+42+96=694.
$$

Each indexed formula emits one integer signal. A piece-square formula contributes its signed role-relative indicator when its normalized square contains the selected piece type and contributes zero otherwise.

### 2.5 Material-Responsive Coefficients

The five material coordinates are

$$
m_t(x)=\frac{n_t(x)}{n_t^0}-1,
\qquad
t\in\lbrace P,N,B,R,Q\rbrace.
$$

The starting position has $m_t(x)=0$ for every piece type. Piece removal moves the corresponding coordinate in fixed steps toward $-1$.

Each formula owns one base coefficient and five material-response coefficients. Its effective coefficient is

$$
w_i(x)=b_i+
\sum_{t\in\lbrace P,N,B,R,Q\rbrace}
r_{i,t}m_t(x).
$$

The raw HCE score is

$$
H(x)=\sum_{i\in\mathcal I_E}w_i(x)\phi_i(B(x)).
$$

This construction gives every formula its own affine material response, so material, king safety, mobility, pawn structure and endgame signals can vary along distinct material directions.

Each formula uses one base coefficient and five material-response coefficients, so the 694 formulas require $694(1+5)=4164$ scalar coefficients. The king-pressure response uses a center and width. The winnability adjustment uses seven coefficients, and endgame scaling uses five coefficients. Together, these fourteen adjustment parameters bring the complete parameter set to $4164+14=4178$ scalar values.

### 2.6 King-Pressure Response

King pressure aggregates the number of attacked squares in the defending king's inner region. Let $q_W(x)$ and $q_B(x)$ denote White and Black pressure counts. A fixed-point sigmoid converts each count through

$$
S(q)=
\mathrm{round}
\left(
\frac{4096}{1+\exp(-(q-c_P)/s_P)}
\right),
$$

where $c_P$ is the pressure center and $s_P>0$ is the pressure width. The lookup domain contains the integer counts from $0$ through $64$, with endpoint clamping. The king-pressure formula emits

$$
\phi_P(B(x))=S(q_W(x))-S(q_B(x)).
$$

The bounded response gives additional attackers their largest marginal effect near $c_P$ and progressively smaller effects near both ends of the pressure range.

### 2.7 Winnability Adjustment

The sign of $H(x)$ selects the favored side. Let

- $p$ be the total pawn count;
- $f_s$ be the number of files occupied by pawns of both colors;
- $f_a$ be the number of files occupied by exactly one color's pawns;
- $e_P$ indicate a pawn ending;
- $p_+$ be the favored side's pawn count;
- $q_+$ be the favored side's passed-pawn count;
- $o_B$ indicate opposite-colored bishops.

The winnability value is

$$
u(x)=
a_6+a_0p+a_1f_s+a_2f_a+a_3e_P+a_4p_++a_5o_Bq_+.
$$

Its sign-preserving application is

$$
H_W(x)=
\mathrm{sgn}(H(x))
\max\left(0,|H(x)|+u(x)\right).
$$

Pawn availability, pawn-file asymmetry, pure pawn play and passed pawns thereby modulate the conversion potential of the existing advantage.

### 2.8 Endgame Scaling

Minor, bishop, rook and queen material uses piece values $3$, $3$, $5$ and $9$. A thin advantage has an absolute material difference at most $1$. The pawnless condition refers to the favored side.

A pure opposite-colored-bishop ending contains exactly one bishop per side, places the bishops on opposite square colors and gives each side total piece material $3$. A mixed opposite-colored-bishop ending retains the one-bishop-per-side color relation together with additional piece material.

The endgame scale begins at $1$. A thin pawnless advantage contributes scale $s_0$. A pure opposite-colored-bishop ending contributes

$$
s_1+s_4p_++s_3q_+,
$$

and a mixed opposite-colored-bishop ending contributes

$$
s_2+s_3q_+.
$$

Let $\mathcal S(x)$ contain the scale values whose structural conditions hold in state $x$. The thin pawnless condition contributes $s_0$. A pure opposite-colored-bishop condition contributes $s_1+s_4p_++s_3q_+$, and a mixed opposite-colored-bishop condition contributes $s_2+s_3q_+$. The scale is

$$
\lambda(x)=
\mathrm{clip}_{[0,1]}
\left(
\min\left(\{1\}\cup\mathcal S(x)\right)
\right).
$$

When no structural condition holds, $\mathcal S(x)$ is empty and $\lambda(x)=1$.

The final evaluation is

$$
E(x)=\lambda(x)H_W(x).
$$

This final contraction maps drawish pawnless and opposite-colored-bishop structures toward zero while preserving the favored side.

## 3. Evaluation Reuse

### 3.1 Shared Work Within One Position

One evaluation builds each broadly reused board quantity once. Pawn attacks feed passed-pawn detection, mobility areas, outposts, threats, space and king safety. Piece attack maps feed mobility, contested-square, threat and king-zone formulas. King regions feed pressure, escape and flank-control formulas.

Formula execution follows the fixed coefficient order and accumulates each active signal directly with fused multiply-add:

$$
H\leftarrow\mathrm{fma}(w_i,\phi_i,H).
$$

Zero-valued formula roots advance the index and leave the accumulator unchanged. Piece-square tables visit occupied normalized squares and advance across the remaining fixed coordinates.

### 3.2 Reuse Across Positions

A thread-local 1024-entry pawn table uses both pawn bitboards as its key. Each matching entry supplies pawn attacks, spans, files, passed pawns, rank buckets and king-pawn structures.

Attack reuse tracks the twelve piece sets and occupied squares. Knight and king maps follow their source squares. Bishop, rook and queen maps refresh when their piece set changes or changed occupancy intersects a previous ray. Per-square sliding rays retain their occupancy dependency and attack map.

Mobility reuse tracks piece locations, target areas, guard areas, occupied squares and prior reach. A matching dependency state supplies the primary and secondary mobility histograms.

A thread-local 16-entry coefficient table keys the five total piece counts packed into one material signature. Each entry begins with the 694 base coefficients and refreshes the formulas carrying material responses when its material signature changes.

## 4. Search

### 4.1 Search Score

Evaluation enters negamax from the perspective of the side to move. Let

$$
\eta(x)=
\begin{cases}
+1,&\text{White to move},\\
-1,&\text{Black to move}.
\end{cases}
$$

The search score is

$$
C(x)=
\mathrm{round}
\left(
\mathrm{clip}_{[-25000,25000]}
\left(150\,\eta(x)E(x)\right)
\right).
$$

Ordinary static scores occupy $[-25000,25000]$. Checkmate uses magnitude $30000$, mate-distance scores include the current ply, and search infinity uses magnitude $32000$. Drawn terminal states receive score $0$.

The terminal-state layer covers the fifty-move rule, insufficient material, threefold repetition, checkmate and stalemate. Mate-distance bounds tighten every node window to the scores reachable from its ply.

### 4.2 Iterative Deepening

Search completes depths in the sequence

$$
1,2,\ldots,d_{\max}.
$$

Each completed depth publishes its principal move, root scores, searched depth, selective depth, node count and elapsed time. A node budget, time budget or cancellation request returns the deepest completed iteration.

From depth 4 onward, single-line search begins around the previous score with a $\pm32$ window. A failed bound expands by doubling the margin and repeats the same depth. MultiPV uses the full score range for every reported line.

### 4.3 Transposition and Evaluation Tables

The configured hash budget is rounded downward to a power of two. Budgets of at least 2 MiB divide equally between the transposition table and static-evaluation table. A 1 MiB budget belongs to the transposition table, and a zero budget creates empty tables.

Each transposition cluster occupies 64 bytes and contains four 16-byte entries. An entry stores the complete key, a signed score, best move, depth, generation and exact, lower or upper bound. Replacement quality combines depth, exact-bound preference and generation age.

The transposition key mixes the board hash with the halfmove clock and repetition state. Mate scores receive a ply adjustment during storage and the inverse adjustment during retrieval, which preserves mate distance across transpositions.

Each static-evaluation entry occupies 8 bytes and stores a validity bit, 47 key bits and a signed 16-bit score. Direct power-of-two indexing selects one entry from the board hash.

### 4.4 Move Ordering

Move ordering combines five sources in descending priority:

1. the transposition move;
2. tactical moves with favorable static exchange evaluation;
3. killer moves;
4. quiet-move history;
5. tactical moves with unfavorable static exchange evaluation.

Static exchange evaluation follows the target-square capture sequence with the least valuable legal recapture at each step. It accounts for promotions, en passant captures, newly opened sliding rays and king safety. Tactical ordering also includes captured-piece value, moving-piece value and promotion value.

Two killer moves are retained for each ply through ply 63. The history table uses color and source-destination pair as its index. A quiet beta cutoff receives a depth-squared bonus, and earlier failed quiet moves receive a proportional penalty. The bounded gravity update keeps each history value inside its stable range.

### 4.5 Principal Variation Search

The first ordered move receives a full-window search. Later moves receive a null-window search around $\alpha$. A result above $\alpha$ repeats at full depth, and a principal-variation result inside $(\alpha,\beta)$ receives a full-window search.

The recurrence has the negamax form

$$
V(x,d,\alpha,\beta)=
\max_{a\in\mathcal A(x)}
\left[-V(T(x,a),d-1,-\beta,-\alpha)\right],
$$

with narrower scout windows and selective depth reductions applied by principal variation search.

A transposition hit supplies its move for ordering. An entry of sufficient depth tightens the current window according to its bound, and an exact entry supplies the node value. A depth of at least 5 with an empty transposition move receives a one-ply internal iterative reduction.

### 4.6 Forward Pruning and Reductions

Forward pruning applies at null-window nodes with a halfmove clock below $90$, an ordinary score window, at least one retained knight, bishop, rook or queen, and a king outside check. The current static score and the score from two plies earlier define the improving flag.

Reverse futility pruning covers depths through 6 with margin

$$
\mu(d)=
\begin{cases}
70d,&\text{improving},\\
100d,&\text{otherwise}.
\end{cases}
$$

A static score satisfying $C(x)-\mu(d)\geq\beta$ returns the margin-adjusted bound after legal-move confirmation.

Null-move pruning applies from depth 3 in positions with side-to-move minor, rook or queen material and more than 10 pieces. Its reduction is

$$
R=\min\left(d,3+\left\lfloor\frac d4\right\rfloor+
\min\left(3,\left\lfloor\frac{C(x)-\beta}{200}\right\rfloor\right)\right).
$$

Depths from 10 upward verify a null-move cutoff by searching the reduced legal subtree with further null moves suspended across the verification span.

Razoring covers depths through 3 when

$$
C(x)+220d\leq\alpha.
$$

The quiescence result supplies the cutoff bound when it remains at or below $\alpha$.

ProbCut begins at depth 5 when $C(x)\geq\beta-300$. Tactical moves with a favorable static exchange score probe the threshold $\beta+180$ through quiescence and a reduced search at depth $d-4$.

Late quiet pruning at depths through 4 keeps the first

$$
3+d^2+\mathbf 1_{\mathrm{improving}}d^2
$$

quiet candidates. Checking moves, killer moves, advanced pawn moves, castling and the transposition move receive their full ordering role. Shallow quiet futility pruning uses the margin $100+100d$ together with quiet history.

Late-move reduction uses a precomputed table based on

$$
\left\lfloor\frac{\log d\,\log(k+1)}{2}\right\rfloor,
$$

where $k$ is the move-order index. Principal-variation status, improving status and history adjust the reduction, followed by clamping to $[1,d-2]$. Late tactical moves receive a one-ply reduction, while unfavorable tactical exchanges use the table with neutral history.

### 4.7 Quiescence Search

Quiescence search extends unstable leaves through check evasions, captures and promotions. A quiet position begins from the stand-pat evaluation. Favorable stand pat raises $\alpha$ and can reach $\beta$.

Capture pruning compares the stand-pat score, captured-piece value and a 120-point margin with $\alpha$. Static exchange evaluation selects tactically sustainable captures. Quiet promotions join the tactical move list.

Check evasions extend through four plies beyond the ordinary quiescence boundary. The final boundary returns static evaluation for a position with a legal evasion and a mate-distance score for an empty legal set. A pure pawn ending searches one complete legal-move ply at the quiescence root, which exposes immediate pawn breakthroughs, opposition and promotion races before tactical continuation.

### 4.8 Root Lines and Result Selection

Single-PV root search uses the transposition move, full-window first move and null-window later moves. Root scores are sorted by value, with the principal move receiving priority when two bounds share the same score.

MultiPV requests up to eight lines through sequential exclusion when the requested line count is smaller than the legal-move count. Each pass selects the strongest remaining root move. Larger or complete-root requests search every root move over the full score range and sort the resulting lines.

The reported best move is the first move of the highest-scoring completed root line.
