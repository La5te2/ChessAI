inline static constexpr std::array<FormulaParam, 1> tempoWeights{{
	0.16F
}};

FORMULA(tempo) {
	const auto stm = b.INP(Atom::STM);
	// Tempo: +1 when White moves and -1 when Black moves.
	F(diff(b.EQ(stm, us), b.EQ(stm, them)));
}

inline static constexpr std::array<FormulaParam, 5> materialWeights{{
	0.22F, 0.773F, 0.828F, 1.2365F, 2.4515F
}};

FORMULA(material) {
	// Material: friendly minus enemy piece count, once for pawn, knight, bishop, rook and queen.
	for (int type = 0; type < 5; ++type) {
		F(diff(b.POP(b.PCS(us, type)), b.POP(b.PCS(them, type))));
	}
}

inline static constexpr std::array<FormulaParam, 384> pstWeights{{
	-0.013125F, -0.009375F, -0.005625F, -0.001875F, -0.001875F, -0.005625F, -0.009375F, -0.013125F, 0.006875F, 0.010625F,
	0.014375F, 0.018125F, 0.018125F, 0.014375F, 0.010625F, 0.006875F, 0.026875F, 0.030625F, 0.034375F, 0.038125F,
	0.038125F, 0.034375F, 0.030625F, 0.026875F, 0.046875F, 0.050625F, 0.054375F, 0.058125F, 0.058125F, 0.054375F,
	0.050625F, 0.046875F, 0.063125F, 0.066875F, 0.070625F, 0.074375F, 0.074375F, 0.070625F, 0.066875F, 0.063125F,
	0.075625F, 0.079375F, 0.083125F, 0.086875F, 0.086875F, 0.083125F, 0.079375F, 0.075625F, 0.088125F, 0.091875F,
	0.095625F, 0.099375F, 0.099375F, 0.095625F, 0.091875F, 0.088125F, 0.100625F, 0.104375F, 0.108125F, 0.111875F,
	0.111875F, 0.108125F, 0.104375F, 0.100625F, -0.07875F, -0.05625F, -0.03375F, -0.01125F, -0.01125F, -0.03375F,
	-0.05625F, -0.07875F, -0.05625F, -0.03375F, -0.01125F, 0.01125F, 0.01125F, -0.01125F, -0.03375F, -0.05625F,
	-0.03375F, -0.01125F, 0.01125F, 0.03375F, 0.03375F, 0.01125F, -0.01125F, -0.03375F, -0.01125F, 0.01125F,
	0.03375F, 0.05625F, 0.05625F, 0.03375F, 0.01125F, -0.01125F, -0.01125F, 0.01125F, 0.03375F, 0.05625F,
	0.05625F, 0.03375F, 0.01125F, -0.01125F, -0.03375F, -0.01125F, 0.01125F, 0.03375F, 0.03375F, 0.01125F,
	-0.01125F, -0.03375F, -0.05625F, -0.03375F, -0.01125F, 0.01125F, 0.01125F, -0.01125F, -0.03375F, -0.05625F,
	-0.07875F, -0.05625F, -0.03375F, -0.01125F, -0.01125F, -0.03375F, -0.05625F, -0.07875F, -0.0525F, -0.0375F,
	-0.0225F, -0.0075F, -0.0075F, -0.0225F, -0.0375F, -0.0525F, -0.0375F, -0.0225F, -0.0075F, 0.0075F,
	0.0075F, -0.0075F, -0.0225F, -0.0375F, -0.0225F, -0.0075F, 0.0075F, 0.0225F, 0.0225F, 0.0075F,
	-0.0075F, -0.0225F, -0.0075F, 0.0075F, 0.0225F, 0.0375F, 0.0375F, 0.0225F, 0.0075F, -0.0075F,
	-0.0075F, 0.0075F, 0.0225F, 0.0375F, 0.0375F, 0.0225F, 0.0075F, -0.0075F, -0.0225F, -0.0075F,
	0.0075F, 0.0225F, 0.0225F, 0.0075F, -0.0075F, -0.0225F, -0.0375F, -0.0225F, -0.0075F, 0.0075F,
	0.0075F, -0.0075F, -0.0225F, -0.0375F, -0.0525F, -0.0375F, -0.0225F, -0.0075F, -0.0075F, -0.0225F,
	-0.0375F, -0.0525F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
	0.003125F, 0.003125F, 0.003125F, 0.003125F, 0.003125F, 0.003125F, 0.003125F, 0.003125F, 0.00625F, 0.00625F,
	0.00625F, 0.00625F, 0.00625F, 0.00625F, 0.00625F, 0.00625F, 0.009375F, 0.009375F, 0.009375F, 0.009375F,
	0.009375F, 0.009375F, 0.009375F, 0.009375F, 0.0125F, 0.0125F, 0.0125F, 0.0125F, 0.0125F, 0.0125F,
	0.0125F, 0.0125F, 0.015625F, 0.015625F, 0.015625F, 0.015625F, 0.015625F, 0.015625F, 0.015625F, 0.015625F,
	0.035F, 0.035F, 0.035F, 0.035F, 0.035F, 0.035F, 0.035F, 0.035F, 0.021875F, 0.021875F,
	0.021875F, 0.021875F, 0.021875F, 0.021875F, 0.021875F, 0.021875F, -0.02625F, -0.01875F, -0.01125F, -0.00375F,
	-0.00375F, -0.01125F, -0.01875F, -0.02625F, -0.01875F, -0.01125F, -0.00375F, 0.00375F, 0.00375F, -0.00375F,
	-0.01125F, -0.01875F, -0.01125F, -0.00375F, 0.00375F, 0.01125F, 0.01125F, 0.00375F, -0.00375F, -0.01125F,
	-0.00375F, 0.00375F, 0.01125F, 0.01875F, 0.01875F, 0.01125F, 0.00375F, -0.00375F, -0.00375F, 0.00375F,
	0.01125F, 0.01875F, 0.01875F, 0.01125F, 0.00375F, -0.00375F, -0.01125F, -0.00375F, 0.00375F, 0.01125F,
	0.01125F, 0.00375F, -0.00375F, -0.01125F, -0.01875F, -0.01125F, -0.00375F, 0.00375F, 0.00375F, -0.00375F,
	-0.01125F, -0.01875F, -0.02625F, -0.01875F, -0.01125F, -0.00375F, -0.00375F, -0.01125F, -0.01875F, -0.02625F,
	-0.02F, -0.01F, 0.0F, 0.01F, 0.01F, 0.0F, -0.01F, -0.02F, -0.03F, -0.02F,
	-0.01F, 0.0F, 0.0F, -0.01F, -0.02F, -0.03F, -0.025F, -0.015F, -0.005F, 0.005F,
	0.005F, -0.005F, -0.015F, -0.025F, -0.02F, -0.01F, 0.0F, 0.01F, 0.01F, 0.0F,
	-0.01F, -0.02F, -0.025F, -0.015F, -0.005F, 0.005F, 0.005F, -0.005F, -0.015F, -0.025F,
	-0.04F, -0.03F, -0.02F, -0.01F, -0.01F, -0.02F, -0.03F, -0.04F, -0.055F, -0.045F,
	-0.035F, -0.025F, -0.025F, -0.035F, -0.045F, -0.055F, -0.07F, -0.06F, -0.05F, -0.04F,
	-0.04F, -0.05F, -0.06F, -0.07F
}};

FORMULA(pst) {
	// Piece-square tables: friendly minus enemy occupancy at each normalized square, ordered by piece type then square.
	for (int type = 0; type < 6; ++type) {
		const auto friendly = b.REL(us, b.PCS(us, type).bits).bits;
		const auto enemy = b.REL(them, b.PCS(them, type).bits).bits;
		unsigned next = 0;
		// Visit occupied normalized squares in index order while preserving each fixed table coordinate.
		for (int square : b.locations(b.BB(friendly | enemy))) {
			b.skip(square - next);
			const auto mask = 1ULL << square;
			F(diff(b.NUM((friendly & mask) != 0), b.NUM((enemy & mask) != 0)));
			next = square + 1;
		}
		b.skip(64 - next);
	}
}

inline static constexpr std::array<FormulaParam, 1> bishopPairWeights{{
	0.15F
}};

FORMULA(bishopPair) {
	// Bishop pair: one signal when a side retains at least two bishops.
	const auto friendly = b.GE(b.POP(b.PCS(us, 2)), b.NUM(2));
	const auto enemy = b.GE(b.POP(b.PCS(them, 2)), b.NUM(2));
	F(diff(friendly, enemy));
}

inline static constexpr std::array<FormulaParam, 78> pawnsWeights{{
	0.0375F, 0.01375F, 0.01375F, 0.01F, 0.0125F, 0.0175F, -0.01375F, 0.00875F, 0.0175F, 0.0175F,
	0.065F, 0.0225F, 0.02375F, 0.016875F, 0.02F, 0.02875F, -0.025F, 0.015F, 0.02625F, 0.02375F,
	0.1475F, 0.03125F, 0.03375F, 0.02375F, 0.0275F, 0.04F, -0.03625F, 0.02125F, 0.035F, 0.03F,
	0.285F, 0.04F, 0.04375F, 0.030625F, 0.035F, 0.05125F, -0.0475F, 0.0275F, 0.04375F, 0.03625F,
	0.4775F, 0.04875F, 0.05375F, 0.0375F, 0.0425F, 0.0625F, -0.05875F, 0.03375F, 0.0525F, 0.0425F,
	0.725F, 0.0575F, 0.06375F, 0.044375F, 0.05F, 0.07375F, -0.07F, 0.04F, 0.06125F, 0.04875F,
	-0.035F, -0.0275F, 0.006F, -0.006F, 0.004F, -0.004F, 0.002F, -0.002F, 0.0F, 0.0F,
	-0.002F, 0.002F, -0.004F, 0.004F, -0.006F, 0.006F, -0.008F, 0.008F
}};

FORMULA(pawns) {
	// Pawn ranks: ten signals for each normalized rank 2 through 7.
	const auto fp = b.PCS(us, 0);
	const auto ep = b.PCS(them, 0);
	const auto fpass = passedPawns(us, them);
	const auto epass = passedPawns(them, us);
	const auto fatt = pawnAttacks(us);
	const auto eatt = pawnAttacks(them);
	const auto ffiles = files(us);
	const auto efiles = files(them);
	const auto fneighbours = b.OR(b.SH(ffiles, us, 2), b.SH(ffiles, us, 3));
	const auto eneighbours = b.OR(b.SH(efiles, them, 2), b.SH(efiles, them, 3));
	struct PawnSignals {
		std::array<int, 6> passed{};
		std::array<int, 6> safeSpan{};
		std::array<int, 6> safeAdvance{};
		std::array<int, 6> defendedAdvance{};
		std::array<int, 6> blockedAdvance{};
		std::array<int, 6> clearFile{};
		std::array<int, 6> supportedPasser{};
		std::array<int, 6> connectedPasser{};
		std::array<int, 6> phalanx{};
		std::array<int, 6> defended{};
		std::array<int, 8> ownKingDistance{};
		std::array<int, 8> enemyKingDistance{};
		int doubled = 0;
		int isolated = 0;
	};
	const auto collect = [&](InterSignal role, InterSignal opponent, InterSignal pawns, InterSignal passed, InterSignal att, InterSignal neighbours) {
		PawnSignals signals;
		(void)pawns;
		(void)att;
		(void)neighbours;
		const auto &structure = b.pawnStructure(role);
		signals.passed = structure.passed;
		signals.supportedPasser = structure.supported;
		signals.connectedPasser = structure.connected;
		signals.phalanx = structure.phalanx;
		signals.defended = structure.defended;
		signals.doubled = structure.doubled;
		signals.isolated = structure.isolated;
		const auto enemyAttacks = b.attacks(opponent);
		const bool white = number(role.bits) == 0;
		// Dynamic passer safety and king distances depend on the complete position.
		for (int square : b.locations(passed)) {
			const Word bit = 1ULL << square;
			const int rank = white ? square / 8 : 7 - square / 8;
			if (rank >= 1 && rank <= 6) {
				const int index = rank - 1;
				const auto forward = b.SH(b.fill(b.BB(bit), role), role, 0);
				const auto span = b.OR(forward, b.OR(b.SH(forward, role, 2), b.SH(forward, role, 3)));
				const bool clearStoppers = (span.bits & enemyAttacks.bits) == 0;
				const bool clearForward = !clearStoppers && (forward.bits & enemyAttacks.bits) == 0;
				const int push = square + (white ? 8 : -8);
				const bool defendedPush = !clearStoppers && !clearForward
					&& number(b.attackCount(role, b.NUM(push)).bits) > number(b.attackCount(opponent, b.NUM(push)).bits);
				signals.safeSpan[index] += clearStoppers;
				signals.safeAdvance[index] += clearForward;
				signals.defendedAdvance[index] += defendedPush;
				signals.blockedAdvance[index] += (occ.bits & (1ULL << push)) != 0;
				signals.clearFile[index] += (forward.bits & occ.bits) == 0;
			}
			for (int king : b.locations(b.PCS(role, 5))) {
				const int distance = std::max(std::abs(square / 8 - king / 8), std::abs(square % 8 - king % 8));
				++signals.ownKingDistance[distance];
			}
			for (int king : b.locations(b.PCS(opponent, 5))) {
				const int distance = std::max(std::abs(square / 8 - king / 8), std::abs(square % 8 - king % 8));
				++signals.enemyKingDistance[distance];
			}
		}
		return signals;
	};
	const auto friendly = collect(us, them, fp, fpass, fatt, fneighbours);
	const auto enemy = collect(them, us, ep, epass, eatt, eneighbours);
	for (int rank = 0; rank < 6; ++rank) {
		// Passed pawns and their path safety on each normalized rank from 2 through 7.
		F(diff(b.NUM(friendly.passed[rank]), b.NUM(enemy.passed[rank])));
		F(diff(b.NUM(friendly.safeSpan[rank]), b.NUM(enemy.safeSpan[rank])));
		F(diff(b.NUM(friendly.safeAdvance[rank]), b.NUM(enemy.safeAdvance[rank])));
		F(diff(b.NUM(friendly.defendedAdvance[rank]), b.NUM(enemy.defendedAdvance[rank])));
		F(diff(b.NUM(friendly.blockedAdvance[rank]), b.NUM(enemy.blockedAdvance[rank])));
		F(diff(b.NUM(friendly.clearFile[rank]), b.NUM(enemy.clearFile[rank])));
		F(diff(b.NUM(friendly.supportedPasser[rank]), b.NUM(enemy.supportedPasser[rank])));
		F(diff(b.NUM(friendly.connectedPasser[rank]), b.NUM(enemy.connectedPasser[rank])));
		// Adjacent and defended pawns on the same normalized rank.
		F(diff(b.NUM(friendly.phalanx[rank]), b.NUM(enemy.phalanx[rank])));
		F(diff(b.NUM(friendly.defended[rank]), b.NUM(enemy.defended[rank])));
	}
	// Doubled and isolated pawns.
	F(diff(b.NUM(friendly.doubled), b.NUM(enemy.doubled)));
	F(diff(b.NUM(friendly.isolated), b.NUM(enemy.isolated)));
	for (int distance = 0; distance < 8; ++distance) {
		// Passed pawns grouped by Chebyshev distance from their own and opposing king.
		F(diff(b.NUM(friendly.ownKingDistance[distance]), b.NUM(enemy.ownKingDistance[distance])));
		F(diff(b.NUM(friendly.enemyKingDistance[distance]), b.NUM(enemy.enemyKingDistance[distance])));
	}
}

inline static constexpr std::array<FormulaParam, 64> mobilityWeights{{
	-0.108F, -0.081F, -0.054F, -0.027F, 0.027F, 0.054F, 0.081F, 0.108F, -0.1125F, -0.09F,
	-0.0675F, -0.045F, -0.0225F, 0.0225F, 0.045F, 0.0675F, 0.09F, 0.1125F, 0.135F, 0.1575F,
	0.18F, -0.1134F, -0.0972F, -0.081F, -0.0648F, -0.0486F, -0.0324F, -0.0162F, 0.0162F, 0.0324F,
	0.0486F, 0.0648F, 0.081F, 0.0972F, 0.1134F, 0.0F, -0.0972F, -0.0891F, -0.081F, -0.0729F,
	-0.0648F, -0.0567F, -0.0486F, -0.0405F, -0.0324F, -0.0243F, -0.0162F, -0.0081F, 0.0081F, 0.0162F,
	0.0243F, 0.0324F, 0.0405F, 0.0486F, 0.0567F, 0.0648F, 0.0729F, 0.081F, 0.0891F, 0.0972F,
	0.1053F, 0.1134F, 0.1215F, 0.0F
}};

FORMULA(mobility) {
	// Mobility histograms use Clockwork's primary area for knights and bishops and the sum of its
	// primary and secondary histograms for rooks and queens; one recoverable reference bucket is omitted.
	constexpr std::array<int, 4> max{{8, 13, 14, 27}};
	constexpr std::array<int, 4> reference{{4, 5, 7, 12}};
	for (int type = 1; type <= 4; ++type) {
		const auto &f = b.mobility(us, type, mobilityArea(us, them), secondaryArea(us, them, type));
		const auto &e = b.mobility(them, type, mobilityArea(them, us), secondaryArea(them, us, type));
		for (int bucket = 0; bucket <= max[type - 1]; ++bucket) {
			if (bucket == reference[type - 1]) continue;
			const int friendly = f.counts[bucket] + (type >= 3 ? f.secondary[bucket] : 0);
			const int enemy = e.counts[bucket] + (type >= 3 ? e.secondary[bucket] : 0);
			F(diff(b.NUM(friendly), b.NUM(enemy)));
		}
		if (type >= 3) {
			int friendly = 0;
			int enemy = 0;
			for (int bucket = 0; bucket <= max[type - 1]; ++bucket) {
				friendly += bucket * f.secondary[bucket];
				enemy += bucket * e.secondary[bucket];
			}
			F(diff(b.NUM(friendly), b.NUM(enemy)));
		}
	}
}

inline static constexpr std::array<FormulaParam, 23> piecesWeights{{
	0.03F, 0.03F, -0.007F, -0.014F, -0.007F, -0.0065F, -0.0055F, -0.011F, -0.0175F, -0.0245F,
	-0.03F, -0.037F, -0.041F, -0.0525F, 0.015F, 0.095F, 0.065F, 0.1F, 0.065F, 0.11F,
	0.11F, 0.01F, 0.008F
}};
FORMULA(pieces) {
	// Knights and bishops placed immediately behind a pawn of either color.
	const auto allPawns = b.OR(b.PCS(us, 0), b.PCS(them, 0));
	for (int type = 1; type <= 2; ++type) {
		const auto friendly = b.POP(b.AND(b.PCS(us, type), b.SH(allPawns, them, 0)));
		const auto enemy = b.POP(b.AND(b.PCS(them, type), b.SH(allPawns, us, 0)));
		F(diff(friendly, enemy));
	}
	constexpr std::uint64_t light = 0x55AA55AA55AA55AAULL;
	const auto bishoppawns = [&](InterSignal role) {
		const auto l = b.REL(role, light);
		const auto d = b.NOT(l);
		const auto lightValue = b.MUL(b.POP(b.AND(b.PCS(role, 2), l)), b.POP(b.AND(b.PCS(role, 0), l)));
		const auto darkValue = b.MUL(b.POP(b.AND(b.PCS(role, 2), d)), b.POP(b.AND(b.PCS(role, 0), d)));
		return b.ADD(lightValue, darkValue);
	};
	// Bishop and friendly-pawn occupancy on the bishop's square color.
	F(diff(bishoppawns(us), bishoppawns(them)));
	// Bishops without protection from a friendly pawn.
	F(diff(b.POP(b.AND(b.PCS(us, 2), b.NOT(pawnAttacks(us)))), b.POP(b.AND(b.PCS(them, 2), b.NOT(pawnAttacks(them))))));
	const auto central = fileMask(2) | fileMask(3) | fileMask(4) | fileMask(5);
	const auto fblocked = b.AND(b.PCS(us, 0), b.AND(b.SH(occ, us, 1), b.REL(us, central)));
	const auto eblocked = b.AND(b.PCS(them, 0), b.AND(b.SH(occ, them, 1), b.REL(them, central)));
	// Bishops restricted by blocked central friendly pawns.
	F(diff(b.MUL(b.POP(b.PCS(us, 2)), b.POP(fblocked)), b.MUL(b.POP(b.PCS(them, 2)), b.POP(eblocked))));
	const auto bishopPawnBuckets = [&](InterSignal role) {
		std::array<int, 9> result{};
		const Word pawns = b.PCS(role, 0).bits;
		const Word defended = pawnAttacks(role).bits;
		const int blocked = std::popcount(pawns & b.SH(occ, role, 1).bits & central);
		for (int square : b.locations(b.PCS(role, 2))) {
			const Word color = (light & (1ULL << square)) ? light : ~light;
			const int bucket = std::min(8, std::popcount(pawns & color));
			result[static_cast<std::size_t>(bucket)] += ((defended & (1ULL << square)) == 0) + blocked;
		}
		return result;
	};
	const auto friendlyBishopPawns = bishopPawnBuckets(us);
	const auto enemyBishopPawns = bishopPawnBuckets(them);
	for (int bucket = 0; bucket < 9; ++bucket) {
		// Same-color pawn buckets retain Clockwork's unprotected-bishop and blocked-center multiplier.
		F(diff(b.NUM(friendlyBishopPawns[static_cast<std::size_t>(bucket)]), b.NUM(enemyBishopPawns[static_cast<std::size_t>(bucket)])));
	}
	// Enemy pawns lying on a bishop diagonal, independent of current blockers.
	F(diff(bishopXray(us, them), bishopXray(them, us)));
	// Rooks on the normalized seventh rank.
	F(diff(b.POP(b.AND(b.PCS(us, 3), b.REL(us, rankMask(6)))), b.POP(b.AND(b.PCS(them, 3), b.REL(them, rankMask(6))))));
	// Rooks sharing a file with either queen.
	F(diff(rookLine(us, them), rookLine(them, us)));

	const auto fpawnfiles = files(us);
	const auto epawnfiles = files(them);
	const auto openfiles = b.NOT(b.OR(fpawnfiles, epawnfiles));
	const auto fsemifiles = b.AND(b.NOT(fpawnfiles), epawnfiles);
	const auto esemifiles = b.AND(b.NOT(epawnfiles), fpawnfiles);
	// Rooks on files containing no pawn, then files containing only an enemy pawn.
	F(diff(b.POP(b.AND(b.PCS(us, 3), openfiles)), b.POP(b.AND(b.PCS(them, 3), openfiles))));
	F(diff(b.POP(b.AND(b.PCS(us, 3), fsemifiles)), b.POP(b.AND(b.PCS(them, 3), esemifiles))));

	const auto outposts = [&](InterSignal role, InterSignal opponent, int type) {
		const auto ranks = b.REL(role, rankMask(3) | rankMask(4) | rankMask(5));
		const auto future = pawnAttacks(opponent);
		const auto viable = b.AND(ranks, b.AND(pawnAttacks(role), b.NOT(b.fill(future, opponent))));
		return b.POP(b.AND(b.PCS(role, type), viable));
	};
	// Knights and bishops on pawn-supported advanced squares that enemy pawns cannot challenge.
	for (int type = 1; type <= 2; ++type) {
		F(diff(outposts(us, them, type), outposts(them, us, type)));
	}

	const auto restricted = [&](InterSignal role, InterSignal opponent) {
		const auto guarded = strongSquares(role, opponent);
		return b.POP(b.AND(b.AND(b.attacks(role), b.attacks(opponent)), b.NOT(guarded)));
	};
	// Contested attack squares whose protection is unfavorable.
	F(diff(restricted(us, them), restricted(them, us)));
	constexpr std::uint64_t center = 0x00003C3C3C000000ULL;
	const auto space = [&](InterSignal role, InterSignal opponent) { return b.POP(b.AND(b.AND(b.attacks(role), b.REL(role, center)), b.NOT(pawnAttacks(opponent)))); };
	// Safely controlled squares in the normalized central space region.
	F(diff(space(us, them), space(them, us)));
}

inline static constexpr std::array<FormulaParam, 42> threatsWeights{{
	0.07F, 0.063F, 0.028F, 0.0245F, 0.021F, 0.16F, 0.144F, 0.064F, 0.056F, 0.048F,
	0.16F, 0.144F, 0.064F, 0.056F, 0.048F, 0.22F, 0.198F, 0.088F, 0.077F, 0.066F,
	0.3F, 0.27F, 0.12F, 0.105F, 0.09F, -0.0005F, -0.0105F, -0.0065F, -0.0035F, -0.001F,
	-0.002F, 0.2805F, 0.0055F, 0.012F, 0.12F, 0.065F, 0.0375F, 0.055F, 0.045F, 0.066F,
	0.03F, 0.044F
}};
FORMULA(threats) {
	// Build shared threat sets once per attacking side, then count each victim type.
	const auto evalside = [&](InterSignal role, InterSignal opponent) {
		std::array<std::array<InterSignal, 5>, 5> result{};
		const auto guarded = strongSquares(role, opponent);
		const auto targets = own(opponent);
		const auto pawns = b.PCS(opponent, 0);
		const auto nonpawns = b.AND(targets, b.NOT(pawns));
		const auto weak = b.AND(targets, b.NOT(guarded));
		const auto defended = b.AND(nonpawns, guarded);
		const auto attacked = b.attacks(role);
		const auto undefended = b.NOT(b.attacks(opponent));
		const auto hangingPawns = b.AND(b.AND(weak, pawns), b.AND(attacked, undefended));
		const auto hangingPieces = b.AND(b.AND(weak, nonpawns), b.AND(attacked, b.OR(undefended, b.doubleAttacks(role))));
		const auto minor = b.AND(b.OR(weak, defended), b.OR(b.attacks(role, 1), b.attacks(role, 2)));
		const auto rook = b.AND(weak, b.attacks(role, 3));
		const auto pushatt = shared(pushAttack, role, [&] {
			const auto pushed = b.AND(b.SH(b.PCS(role, 0), role, 0), b.NOT(occ));
			return b.OR(b.SH(pushed, role, 4), b.SH(pushed, role, 5));
		});
		for (int victim = 0; victim < 5; ++victim) {
			const auto pieces = b.PCS(opponent, victim);
			const auto hanging = victim == 0 ? hangingPawns : hangingPieces;
			result[static_cast<std::size_t>(victim)] = {b.POP(b.AND(pieces, hanging)), b.POP(b.AND(pieces, pawnAttacks(role))),
				b.POP(b.AND(pieces, minor)), b.POP(b.AND(pieces, rook)), b.POP(b.AND(pieces, pushatt))};
		}
		return result;
	};
	const auto kingthreat = [&](InterSignal role, InterSignal opponent) {
		const auto targets = own(opponent);
		const auto guarded = strongSquares(role, opponent);
		return b.ANY(b.AND(b.AND(targets, b.NOT(guarded)), b.attacks(role, 5)));
	};
	const auto queenpressure = [&](InterSignal role, InterSignal opponent, int type) {
		if (number(b.POP(b.PCS(opponent, 4)).bits) != 1) return z;
		Sum terms;
		auto safe = b.AND(mobilityArea(role, opponent), b.AND(b.NOT(b.PCS(role, 0)), b.NOT(strongSquares(role, opponent))));
		if (type >= 2) safe = b.AND(safe, b.doubleAttacks(role));
		for (int square : b.squares(opponent, 4)) {
			const auto queen = b.ANY(b.AND(b.PCS(opponent, 4), b.REL(opponent, 1ULL << square)));
			InterSignal destinations;
			if (type == 1) {
				destinations = b.attackFrom(role, 1, b.SQ(opponent, square));
			} else {
				const auto queenAttacks = b.attacks(opponent, 4);
				const auto diagonals = b.AND(queenAttacks, b.REL(opponent, diagonalMask(square)));
				destinations = type == 2 ? diagonals : b.AND(queenAttacks, b.NOT(diagonals));
			}
			destinations = b.AND(destinations, b.AND(b.attacks(role, type), safe));
			terms.add(b.MUL(queen, b.POP(destinations)));
		}
		return b.sum(terms);
	};
	const auto friendlyThreats = evalside(us, them);
	const auto enemyThreats = evalside(them, us);
	for (int victim = 0; victim < 5; ++victim) {
		const auto &friendly = friendlyThreats[static_cast<std::size_t>(victim)];
		const auto &enemy = enemyThreats[static_cast<std::size_t>(victim)];
		F(diff(friendly[0], enemy[0]));
		F(diff(friendly[1], enemy[1]));
		F(diff(friendly[2], enemy[2]));
		F(diff(friendly[3], enemy[3]));
		F(diff(friendly[4], enemy[4]));
	}
	const auto stm = b.INP(Atom::STM);
	const auto white = b.EQ(stm, us);
	const auto black = b.EQ(stm, them);
	for (int victim = 0; victim < 5; ++victim) {
		const auto &friendly = friendlyThreats[static_cast<std::size_t>(victim)];
		const auto &enemy = enemyThreats[static_cast<std::size_t>(victim)];
		// Aggregate and active-attacker signals recover both side-to-move threat buckets.
		F(diff(b.MUL(friendly[2], white), b.MUL(enemy[2], black)));
		F(diff(b.MUL(friendly[3], white), b.MUL(enemy[3], black)));
	}
	// A king attacking an insufficiently guarded enemy piece.
	F(diff(kingthreat(us, them), kingthreat(them, us)));
	// Knight, bishop and rook pressure on the enemy queen, split by friendly-queen presence.
	for (int type = 1; type <= 3; ++type) {
		const auto friendly = queenpressure(us, them, type);
		const auto enemy = queenpressure(them, us, type);
		const auto fqueenless = b.EQ(b.POP(b.PCS(us, 4)), z);
		const auto equeenless = b.EQ(b.POP(b.PCS(them, 4)), z);
		F(diff(b.MUL(friendly, b.LNOT(fqueenless)), b.MUL(enemy, b.LNOT(equeenless))));
		F(diff(b.MUL(friendly, fqueenless), b.MUL(enemy, equeenless)));
	}
}

inline static constexpr std::array<FormulaParam, 96> kingsWeights{{
	0.0125F, 0.0045F, 0.006F, 0.025F, 0.009F, 0.012F, 0.025F, 0.009F, 0.012F, 0.03125F,
	0.01125F, 0.015F, 0.04375F, 0.01575F, 0.021F, 0.000006103515625F, 0.015F,
	0.005625F, 0.0045F, 0.00375F, 0.003F, 0.001875F, 0.0015F, -0.0275F, 0.0015F, 0.002F, 0.003F,
	0.004F, 0.01F, -0.005F, 0.015F, 0.01F, -0.05F, -0.0015F, -0.001F, -0.0065F, -0.005F,
	-0.0085F, -0.024F, -0.025F, 0.0025F, -0.0095F, -0.0035F, 0.0035F, 0.0005F, -0.0115F, -0.02F,
	-0.002F, -0.0105F, 0.0025F, 0.005F, 0.001F, -0.0115F, -0.03F, 0.007F, 0.0035F, 0.0155F,
	0.0205F, 0.018F, 0.011F, -0.0025F, 0.0F, 0.0F, 0.0165F, -0.0015F, 0.001F, 0.0115F,
	0.025F, 0.0085F, -0.0815F, -0.024F, 0.002F, 0.0005F, 0.0025F, 0.002F, 0.0035F, -0.0745F,
	-0.018F, -0.002F, -0.002F, -0.002F, 0.002F, 0.006F, -0.0385F, -0.0045F, 0.0045F, 0.004F,
	0.0045F, 0.008F, 0.0005F, -0.03F, 0.012F, 0.0115F, 0.0015F, -0.0005F, 0.005F
}};
FORMULA(kings) {
	static const SigmoidCurve pressureCurve{5.0F, 1.5F};
	b.prepareKingPawns(us);
	b.prepareKingPawns(them);
	// Each attacking piece type emits inner-ring attacks, outer-ring attacks and potential checking moves.
	Sum friendly;
	Sum enemy;
	const auto friendlyInner = b.OR(b.PCS(them, 5), b.attacks(them, 5));
	const auto enemyInner = b.OR(b.PCS(us, 5), b.attacks(us, 5));
	const auto friendlyOuter = kingRegion(them, false);
	const auto enemyOuter = kingRegion(us, false);
	for (int type = 0; type < 5; ++type) {
		const auto fi = b.POP(b.AND(b.attacks(us, type), friendlyInner));
		const auto ei = b.POP(b.AND(b.attacks(them, type), enemyInner));
		friendly.add(fi);
		enemy.add(ei);
		F(diff(fi, ei));
		F(diff(b.POP(b.AND(b.attacks(us, type), friendlyOuter)), b.POP(b.AND(b.attacks(them, type), enemyOuter))));
		F(diff(potentialChecks(us, them, type), potentialChecks(them, us, type)));
	}
	const auto fp = b.sum(friendly);
	const auto ep = b.sum(enemy);
	// Total king pressure follows a smooth bounded response instead of discrete thresholds.
	F(diff(b.SIG(fp, pressureCurve), b.SIG(ep, pressureCurve)));
	// King escape squares not occupied by friendly pieces or controlled by the opponent.
	F(diff(escapes(us, them), escapes(them, us)));
	// Pawn shelter and enemy pawn storm at distances one through three.
	for (int distance = 1; distance <= 3; ++distance) {
		F(diff(kingPawns(us, them, distance, true), kingPawns(them, us, distance, true)));
		F(diff(kingPawns(them, us, distance, false), kingPawns(us, them, distance, false)));
	}
	// Open files crossing the king's three-file neighborhood.
	F(diff(kingOpenFiles(us), kingOpenFiles(them)));
	// Friendly and enemy single/double attacks in each king flank.
	F(diff(flank(b.attacks(us), us), flank(b.attacks(them), them)));
	F(diff(flank(b.attacks(us), them), flank(b.attacks(them), us)));
	F(diff(flank(b.doubleAttacks(us), us), flank(b.doubleAttacks(them), them)));
	F(diff(flank(b.doubleAttacks(us), them), flank(b.doubleAttacks(them), us)));
	const auto fblockedstorm = b.AND(b.PCS(us, 0), b.SH(b.PCS(them, 0), them, 0));
	const auto eblockedstorm = b.AND(b.PCS(them, 0), b.SH(b.PCS(us, 0), us, 0));
	// Locked opposing pawns in the king flank.
	F(diff(flank(fblockedstorm, us), flank(eblockedstorm, them)));
	const auto fnoqueen = b.EQ(b.POP(b.PCS(us, 4)), z);
	const auto enoqueen = b.EQ(b.POP(b.PCS(them, 4)), z);
	// King pressure after the defending side has lost its queen.
	F(diff(b.MUL(fp, enoqueen), b.MUL(ep, fnoqueen)));
	// Remaining king-side and queen-side castling rights.
	for (int wing = 0; wing < 2; ++wing) {
		const auto friendlyright = b.CR(us, wing);
		const auto enemyright = b.CR(them, wing);
		F(diff(friendlyright, enemyright));
	}
	// Absence of the attacking side's queen, independently of the current king-pressure count.
	F(diff(fnoqueen, enoqueen));
	for (int edge = 0; edge < 4; ++edge) {
		for (int rank = 0; rank < 7; ++rank) {
			// Friendly king shelter grouped by edge class and relative pawn rank.
			const unsigned slot = static_cast<unsigned>(7 * edge + rank);
			F(diff(b.shelter(us, slot), b.shelter(them, slot)));
		}
	}
	for (int rank = 0; rank < 7; ++rank) {
		// Enemy pawn storms blocked immediately by a shelter pawn, grouped by relative rank.
		F(diff(b.blockedStorm(us, static_cast<unsigned>(rank)), b.blockedStorm(them, static_cast<unsigned>(rank))));
	}
	for (int edge = 0; edge < 4; ++edge) {
		for (int rank = 0; rank < 7; ++rank) {
			// Enemy pawn storms grouped by edge class and relative pawn rank.
			const unsigned slot = static_cast<unsigned>(7 * edge + rank);
			F(diff(b.storm(us, slot), b.storm(them, slot)));
		}
	}
}

FORMULA(endgames) {
	static constexpr WinnableParams winnable{0.002F, -0.005F, 0.007F, 0.02F, 0.006F, 0.0125F, 0.0F};
	static constexpr EndgameScaleParams scale{0.20F, 0.35F, 0.75F, 0.08F, 0.025F};
	// Conversion and draw scaling follow the favored side selected by the complete preceding HCE score.
	constexpr std::uint64_t light = 0x55AA55AA55AA55AAULL;
	const auto onefb = b.EQ(b.POP(b.PCS(us, 2)), o);
	const auto oneeb = b.EQ(b.POP(b.PCS(them, 2)), o);
	const auto friendlyLight = b.ANY(b.AND(b.PCS(us, 2), b.BB(light)));
	const auto friendlyDark = b.ANY(b.AND(b.PCS(us, 2), b.NOT(b.BB(light))));
	const auto enemyLight = b.ANY(b.AND(b.PCS(them, 2), b.BB(light)));
	const auto enemyDark = b.ANY(b.AND(b.PCS(them, 2), b.NOT(b.BB(light))));
	const auto opposite = b.LOR(b.LAND(friendlyLight, enemyDark), b.LAND(friendlyDark, enemyLight));
	const auto fpawns = b.POP(b.PCS(us, 0));
	const auto epawns = b.POP(b.PCS(them, 0));
	const auto direction = b.direction();
	const auto positive = b.GT(direction, z);
	const auto negative = b.LT(direction, z);
	const auto fpawnless = b.EQ(fpawns, z);
	const auto epawnless = b.EQ(epawns, z);
	const auto thin = b.LE(b.ABS(diff(nonPawnMaterial(us), nonPawnMaterial(them))), o);
	const auto ffiles = files(us);
	const auto efiles = files(them);
	const auto symmetric = b.NUM(std::popcount(ffiles.bits & efiles.bits) >> 3);
	const auto asymmetric = b.NUM(std::popcount(ffiles.bits ^ efiles.bits) >> 3);
	const auto pawnEnding = b.LAND(b.EQ(nonPawnMaterial(us), z), b.EQ(nonPawnMaterial(them), z));
	const auto fpassers = b.POP(passedPawns(us, them));
	const auto epassers = b.POP(passedPawns(them, us));
	const auto strongPawns = b.ADD(b.MUL(positive, fpawns), b.MUL(negative, epawns));
	const auto strongPassers = b.ADD(b.MUL(positive, fpassers), b.MUL(negative, epassers));
	const auto favoredPawnless = b.LOR(b.LAND(positive, fpawnless), b.LAND(negative, epawnless));
	const auto thinPawnless = b.LAND(favoredPawnless, thin);
	const auto pure = b.LAND(b.LAND(b.LAND(onefb, oneeb), opposite),
		b.LAND(b.EQ(nonPawnMaterial(us), b.NUM(3)), b.EQ(nonPawnMaterial(them), b.NUM(3))));
	const auto mixed = b.LAND(b.LAND(b.LAND(onefb, oneeb), opposite), b.LNOT(pure));
	b.END({b.ADD(fpawns, epawns), symmetric, asymmetric, pawnEnding, fpawns, epawns, fpassers, epassers, opposite, fpawnless, epawnless, thin, pure, mixed});

	// Pawn count and pawn-file geometry adjust how readily the current advantage converts.
	b.WIN(b.ADD(fpawns, epawns), symmetric, asymmetric, pawnEnding, strongPawns, b.MUL(opposite, strongPassers), winnable);
	// Pawnless and opposite-colored-bishop structures contract the complete score toward a draw.
	b.SCALE(thinPawnless, pure, mixed, strongPawns, strongPassers, scale);
}

inline static constexpr auto formulaWeights = [] {
	constexpr std::size_t count = tempoWeights.size() + materialWeights.size() + pstWeights.size() + bishopPairWeights.size() + pawnsWeights.size() +
		mobilityWeights.size() + piecesWeights.size() + threatsWeights.size() + kingsWeights.size();
	constexpr std::array<float, 5> fixed{{0.0F, 0.0F, 0.0F, 0.0F, 0.0F}};
	constexpr std::array<float, 5> bishopPairResponse{{-0.05F, 0.0F, 0.0F, 0.0F, 0.0F}};
	constexpr std::array<float, 5> pawnResponse{{0.0F, -0.02F, -0.02F, -0.04F, -0.08F}};
	constexpr std::array<float, 5> threatResponse{{0.0F, 0.02F, 0.02F, 0.05F, 0.12F}};
	constexpr std::array<float, 5> kingResponse{{0.0F, 0.03F, 0.03F, 0.08F, 0.25F}};
	std::array<FormulaParam, count> values{};
	std::size_t index = 0;
	const auto append = [&](const auto &part, const std::array<float, 5> &response) {
		for (FormulaParam value : part) {
			for (std::size_t type = 0; type < response.size(); ++type) {
				value.material[type] += value.base * response[type];
			}
			values[index++] = value;
		}
	};
	append(tempoWeights, fixed);
	append(materialWeights, fixed);
	append(pstWeights, fixed);
	append(bishopPairWeights, bishopPairResponse);
	append(pawnsWeights, pawnResponse);
	append(mobilityWeights, fixed);
	append(piecesWeights, fixed);
	append(threatsWeights, threatResponse);
	append(kingsWeights, kingResponse);
	return values;
}();
