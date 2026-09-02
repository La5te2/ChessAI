#ifdef ELEGINUS_FORMULAS

inline static constexpr std::array<float, 1> tempoInitial{{
	0.16F
}};

FORMULA(tempo) {
	const auto stm = b.ATOM(Atom::STM);
	// Tempo: +1 when White moves and -1 when Black moves.
	F(diff(b.EQ(stm, us), b.EQ(stm, them)));
}

inline static constexpr std::array<float, 5> materialInitial{{
	0.22F, 0.773F, 0.828F, 1.2365F, 2.4515F
}};

FORMULA(material) {
	// Material: friendly minus enemy piece count, once for pawn, knight, bishop, rook and queen.
	for (int type = 0; type < 5; ++type)
		F(diff(b.POP(b.PCS(us, type)), b.POP(b.PCS(them, type))));
}

inline static constexpr std::array<float, 384> pstInitial{{
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
		// Visit occupied normalized squares in index order; opposing activations may cancel in the score but remain part of the condition.
		for (int square : b.locations(b.BB(friendly | enemy))) {
			b.skip(square - next);
			const auto mask = 1ULL << square;
			F(diff(b.NUM((friendly & mask) != 0), b.NUM((enemy & mask) != 0)));
			next = square + 1;
		}
		b.skip(64 - next);
	}
}

inline static constexpr std::array<float, 1> bishopPairInitial{{
	0.15F
}};

FORMULA(bishopPair) {
	// Bishop pair: one signal when a side retains at least two bishops.
	const auto friendly = b.GE(b.POP(b.PCS(us, 2)), b.NUM(2));
	const auto enemy = b.GE(b.POP(b.PCS(them, 2)), b.NUM(2));
	F(diff(friendly, enemy));
}

inline static constexpr std::array<float, 78> pawnsInitial{{
	0.0375F, 0.01375F, 0.01375F, 0.01F, 0.0125F, 0.0175F, -0.01375F, 0.00875F, 0.0175F, 0.0175F,
	0.065F, 0.0225F, 0.02375F, 0.016875F, 0.02F, 0.02875F, -0.025F, 0.015F, 0.02625F, 0.02375F,
	0.1475F, 0.03125F, 0.03375F, 0.02375F, 0.0275F, 0.04F, -0.03625F, 0.02125F, 0.035F, 0.03F,
	0.285F, 0.04F, 0.04375F, 0.030625F, 0.035F, 0.05125F, -0.0475F, 0.0275F, 0.04375F, 0.03625F,
	0.4775F, 0.04875F, 0.05375F, 0.0375F, 0.0425F, 0.0625F, -0.05875F, 0.03375F, 0.0525F, 0.0425F,
	0.725F, 0.0575F, 0.06375F, 0.044375F, 0.05F, 0.07375F, -0.07F, 0.04F, 0.06125F, 0.04875F,
	-0.035F, -0.0275F, -0.0325F, -0.02F, 0.006F, -0.006F, 0.004F, -0.004F, 0.002F, -0.002F,
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
	const auto ffiles = shared(pawnFile, us, [&] { return files(fp, us, them); });
	const auto efiles = shared(pawnFile, them, [&] { return files(ep, them, us); });
	const auto fneighbours = b.OR(b.SH(ffiles, us, 2), b.SH(ffiles, us, 3));
	const auto eneighbours = b.OR(b.SH(efiles, them, 2), b.SH(efiles, them, 3));
	std::array<std::array<int, 78>, 2> counts{};
	const auto collect = [&](unsigned side, IS role, IS opponent, IS pawns, IS passed, IS att, IS oppatt, IS neighbours) {
		auto &values = counts[side];
		const auto shape = b.pawnShape(role);
		values[60] += shape[0];
		values[61] += std::popcount(pawns.bits & ~neighbours.bits);
		values[62] += std::popcount(pawns.bits & ~att.bits & b.SH(oppatt, opponent, 0).bits);
		values[63] += shape[1];
		// Per-passer signals: passed, clear path, safe path, wide-safe path, controlled push, supported, blocked and connected.
		std::array<Word, 8> sets{};
		if (passed.bits) {
			const auto a = b.attacks(role), e = b.attacks(opponent);
			const auto wide = b.BB(e.bits | b.SH(e, role, 2).bits | b.SH(e, role, 3).bits);
			const auto control = b.BB((a.bits & ~e.bits) | (b.doubleAttacks(role).bits & ~b.doubleAttacks(opponent).bits));
			sets = {passed.bits, ~b.SH(b.fill(occ, opponent), opponent, 0).bits, ~b.SH(b.fill(e, opponent), opponent, 0).bits, ~b.SH(b.fill(wide, opponent), opponent, 0).bits,
			    b.SH(control, role, 1).bits, att.bits, b.SH(occ, opponent, 0).bits, b.SH(passed, role, 2).bits | b.SH(passed, role, 3).bits};
		}
		// The remaining two rank signals are phalanx membership and support by a friendly pawn.
		const Word phalanx = b.SH(pawns, role, 2).bits;
		const bool white = number(role.bits) == 0;
		// Visit each pawn once for rank signals and both king-distance histograms.
		for (int square : b.locations(pawns)) {
			const Word bit = 1ULL << square;
			const int rank = white ? square / 8 : 7 - square / 8;
			if (rank >= 1 && rank <= 6) {
				const int start = 10 * (rank - 1);
				values[start + 8] += (phalanx & bit) != 0;
				values[start + 9] += (att.bits & bit) != 0;
				if (passed.bits & bit)
					for (int i = 0; i < 8; ++i)
						values[start + i] += (sets[i] & bit) != 0;
			}
			if (!(passed.bits & bit))
				continue;
			for (int k = 0; k < 2; ++k)
				for (int king : b.locations(b.PCS(k == 0 ? role : opponent, 5))) {
					const int d = std::max(std::abs(square / 8 - king / 8), std::abs(square % 8 - king % 8));
					if (d != 3)
						values[64 + 2 * (d < 3 ? d : d - 1) + k] += 1;
				}
		}
	};
	collect(0, us, them, fp, fpass, fatt, eatt, fneighbours);
	collect(1, them, us, ep, epass, eatt, fatt, eneighbours);
	// Global pawn signals follow the rank coordinates: doubled, isolated, backward and island counts,
	// then friendly/enemy king-distance histograms for passed pawns.
	for (std::size_t i = 0; i < counts[0].size(); ++i)
		F(diff(b.NUM(counts[0][i]), b.NUM(counts[1][i])));
}

inline static constexpr std::array<float, 64> mobilityInitial{{
	-0.108F, -0.081F, -0.054F, -0.027F, 0.027F, 0.054F, 0.081F, 0.108F, -0.1125F, -0.09F,
	-0.0675F, -0.045F, -0.0225F, 0.0225F, 0.045F, 0.0675F, 0.09F, 0.1125F, 0.135F, 0.1575F,
	0.18F, -0.1134F, -0.0972F, -0.081F, -0.0648F, -0.0486F, -0.0324F, -0.0162F, 0.0162F, 0.0324F,
	0.0486F, 0.0648F, 0.081F, 0.0972F, 0.1134F, 0.01F, -0.0972F, -0.0891F, -0.081F, -0.0729F,
	-0.0648F, -0.0567F, -0.0486F, -0.0405F, -0.0324F, -0.0243F, -0.0162F, -0.0081F, 0.0081F, 0.0162F,
	0.0243F, 0.0324F, 0.0405F, 0.0486F, 0.0567F, 0.0648F, 0.0729F, 0.081F, 0.0891F, 0.0972F,
	0.1053F, 0.1134F, 0.1215F, 0.005F
}};

FORMULA(mobility) {
	// Mobility histograms count pieces by legal attack-set size; one reference bucket per piece is omitted.
	// Rooks and queens also emit their total mobility outside the secondary unsafe area.
	constexpr std::array<int, 4> max{{8, 13, 14, 27}};
	constexpr std::array<int, 4> reference{{4, 5, 7, 12}};
	for (int type = 1; type <= 4; ++type) {
		const auto &f = b.mobility(us, type, mobilityArea(us, them), secondaryArea(us, them, type));
		const auto &e = b.mobility(them, type, mobilityArea(them, us), secondaryArea(them, us, type));
		for (int bucket = 0; bucket <= max[type - 1]; ++bucket) {
			if (bucket == reference[type - 1])
				continue;
			F(diff(b.NUM(f.counts[bucket]), b.NUM(e.counts[bucket])));
		}
		if (type >= 3)
			F(diff(b.NUM(f.secondary), b.NUM(e.secondary)));
	}
}

inline static constexpr std::array<float, 34> piecesInitial{{
	0.03F, 0.03F, -0.007F, -0.014F, -0.007F, 0.015F, 0.095F, 0.065F, 0.1F, 0.065F,
	0.11F, 0.11F, 0.01F, 0.008F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
	0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
	0.0F, 0.0F, 0.0F, 0.0F
}};

FORMULA(pieces) {
	// Knights and bishops placed immediately behind a friendly pawn.
	for (int type = 1; type <= 2; ++type) {
		const auto friendly = b.POP(b.AND(b.PCS(us, type), b.SH(b.PCS(us, 0), them, 0)));
		const auto enemy = b.POP(b.AND(b.PCS(them, type), b.SH(b.PCS(them, 0), us, 0)));
		F(diff(friendly, enemy));
	}
	constexpr std::uint64_t light = 0x55AA55AA55AA55AAULL;
	const auto bishoppawns = [&](IS role) {
		const auto l = b.REL(role, light);
		const auto d = b.NOT(l);
		return b.ADD(b.MUL(b.POP(b.AND(b.PCS(role, 2), l)), b.POP(b.AND(b.PCS(role, 0), l))), b.MUL(b.POP(b.AND(b.PCS(role, 2), d)), b.POP(b.AND(b.PCS(role, 0), d))));
	};
	// Bishop and friendly-pawn occupancy on the bishop's square color.
	F(diff(bishoppawns(us), bishoppawns(them)));
	// Bishops without protection from a friendly pawn.
	F(diff(b.POP(b.AND(b.PCS(us, 2), b.NOT(pawnAttacks(us)))), b.POP(b.AND(b.PCS(them, 2), b.NOT(pawnAttacks(them))))));
	const auto central = fileMask(2) | fileMask(3) | fileMask(4) | fileMask(5);
	const auto fblocked = b.AND(b.PCS(us, 0), b.AND(b.SH(occ, us, 1), b.REL(us, central & (rankMask(1) | rankMask(2)))));
	const auto eblocked = b.AND(b.PCS(them, 0), b.AND(b.SH(occ, them, 1), b.REL(them, central & (rankMask(1) | rankMask(2)))));
	// Bishops restricted by blocked central friendly pawns.
	F(diff(b.MUL(b.POP(b.PCS(us, 2)), b.POP(fblocked)), b.MUL(b.POP(b.PCS(them, 2)), b.POP(eblocked))));
	// Enemy pawns lying on a bishop diagonal, independent of current blockers.
	F(diff(bishopXray(us, them), bishopXray(them, us)));
	// Rooks on the normalized seventh rank.
	F(diff(b.POP(b.AND(b.PCS(us, 3), b.REL(us, rankMask(6)))), b.POP(b.AND(b.PCS(them, 3), b.REL(them, rankMask(6))))));
	// Rooks sharing a file with either queen.
	F(diff(rookLine(us, them), rookLine(them, us)));

	{
		IS fopen = z;
		IS eopen = z;
		IS fsemi = z;
		IS esemi = z;
		for (int file = 0; file < 8; ++file) {
			const auto mask = b.BB(fileMask(file));
			const auto frooks = b.POP(b.AND(b.PCS(us, 3), mask));
			const auto erooks = b.POP(b.AND(b.PCS(them, 3), mask));
			const auto fp = b.ANY(b.AND(b.PCS(us, 0), mask));
			const auto ep = b.ANY(b.AND(b.PCS(them, 0), mask));
			fopen = b.ADD(fopen, b.MUL(frooks, b.LNOT(b.LOR(fp, ep))));
			eopen = b.ADD(eopen, b.MUL(erooks, b.LNOT(b.LOR(fp, ep))));
			fsemi = b.ADD(fsemi, b.MUL(frooks, b.LAND(b.LNOT(fp), ep)));
			esemi = b.ADD(esemi, b.MUL(erooks, b.LAND(b.LNOT(ep), fp)));
		}
		// Rooks on files containing no pawn, then files containing only an enemy pawn.
		F(diff(fopen, eopen));
		F(diff(fsemi, esemi));
	}

	const auto outposts = [&](IS role, IS opponent, int type) {
		const auto ranks = b.REL(role, rankMask(3) | rankMask(4) | rankMask(5));
		const auto future = pawnAttacks(opponent);
		const auto viable = b.AND(ranks, b.AND(pawnAttacks(role), b.NOT(b.fill(future, opponent))));
		return b.POP(b.AND(b.PCS(role, type), viable));
	};
	// Knights and bishops on pawn-supported advanced squares that enemy pawns cannot challenge.
	for (int type = 1; type <= 2; ++type) {
		F(diff(outposts(us, them, type), outposts(them, us, type)));
	}

	const auto restricted = [&](IS role, IS opponent) {
		const auto guarded = strongSquares(role, opponent);
		return b.POP(b.AND(b.AND(b.attacks(role), b.attacks(opponent)), b.NOT(guarded)));
	};
	// Contested attack squares whose protection is unfavorable.
	F(diff(restricted(us, them), restricted(them, us)));
	constexpr std::uint64_t center = 0x00003C3C3C000000ULL;
	const auto space = [&](IS role, IS opponent) { return b.POP(b.AND(b.AND(b.attacks(role), b.REL(role, center)), b.NOT(pawnAttacks(opponent)))); };
	// Safely controlled squares in the normalized central space region.
	F(diff(space(us, them), space(them, us)));

	// Knights and bishops placed immediately behind an enemy pawn.
	for (int type = 1; type <= 2; ++type) {
		const auto friendly = b.POP(b.AND(b.PCS(us, type), b.SH(b.PCS(them, 0), them, 0)));
		const auto enemy = b.POP(b.AND(b.PCS(them, type), b.SH(b.PCS(us, 0), us, 0)));
		F(diff(friendly, enemy));
	}

	const auto bishopContext = [&](IS role) {
		std::array<std::array<int, 9>, 2> values{};
		const Word pawns = b.PCS(role, 0).bits;
		const Word defended = pawnAttacks(role).bits;
		const Word earlyCenter = b.REL(role, central & (rankMask(1) | rankMask(2))).bits;
		const int blocked = std::popcount(pawns & b.SH(occ, role, 1).bits & earlyCenter);
		for (int square : b.locations(b.PCS(role, 2))) {
			const Word colorMask = (light & (1ULL << square)) != 0 ? light : ~light;
			const int sameColorPawns = std::popcount(pawns & colorMask);
			values[0][sameColorPawns] += (defended & (1ULL << square)) == 0;
			values[1][sameColorPawns] += blocked;
		}
		return values;
	};
	const auto friendlyBishops = bishopContext(us);
	const auto enemyBishops = bishopContext(them);
	// Each same-color-pawn bucket retains its joint relation with pawn protection and blocked central pawns.
	for (unsigned state = 0; state < friendlyBishops.size(); ++state)
		for (unsigned bucket = 0; bucket < friendlyBishops[state].size(); ++bucket)
			F(diff(b.NUM(friendlyBishops[state][bucket]), b.NUM(enemyBishops[state][bucket])));
}

inline static constexpr std::array<float, 32> threatsInitial{{
	0.07F, 0.063F, 0.028F, 0.0245F, 0.021F, 0.16F, 0.144F, 0.064F, 0.056F, 0.048F,
	0.16F, 0.144F, 0.064F, 0.056F, 0.048F, 0.22F, 0.198F, 0.088F, 0.077F, 0.066F,
	0.3F, 0.27F, 0.12F, 0.105F, 0.09F, 0.065F, 0.0375F, 0.055F, 0.045F, 0.066F,
	0.03F, 0.044F
}};

FORMULA(threats) {
	// For each victim type, emit hanging targets and targets attacked by a pawn, minor, rook or pawn push.
	const auto evalside = [&](IS role, IS opponent, int victim) {
		const auto targets = b.PCS(opponent, victim);
		const auto guarded = strongSquares(role, opponent);
		const auto weak = b.AND(targets, b.NOT(guarded));
		const auto hanging = b.AND(targets, b.AND(b.attacks(role), b.OR(b.NOT(b.attacks(opponent)), b.doubleAttacks(role))));
		const auto minor = b.AND(weak, b.OR(b.attacks(role, 1), b.attacks(role, 2)));
		const auto rook = b.AND(weak, b.attacks(role, 3));
		const auto pushatt = shared(pushAttack, role, [&] {
			const auto pushed = b.AND(b.SH(b.PCS(role, 0), role, 0), b.NOT(occ));
			return b.OR(b.SH(pushed, role, 4), b.SH(pushed, role, 5));
		});
		return std::array<IS, 5>{b.POP(hanging), b.POP(b.AND(targets, pawnAttacks(role))), b.POP(minor), b.POP(rook), b.POP(b.AND(targets, pushatt))};
	};
	const auto kingthreat = [&](IS role, IS opponent) {
		const auto targets = own(opponent);
		const auto guarded = strongSquares(role, opponent);
		return b.ANY(b.AND(b.AND(targets, b.NOT(guarded)), b.attacks(role, 5)));
	};
	const auto queenpressure = [&](IS role, IS opponent, int type) {
		Sum terms;
		const auto guarded = strongSquares(role, opponent);
		for (int square : b.squares(opponent, 4)) {
			const auto queen = b.ANY(b.AND(b.PCS(opponent, 4), b.REL(opponent, 1ULL << square)));
			const auto sources = b.AND(b.PCS(role, type), b.attackFrom(role, type, b.SQ(opponent, square)));
			terms.add(b.MUL(queen, b.POP(b.AND(sources, b.NOT(guarded)))));
		}
		return b.sum(terms);
	};

	for (int victim = 0; victim < 5; ++victim) {
		{
			const auto friendly = evalside(us, them, victim);
			const auto enemy = evalside(them, us, victim);
			F(diff(friendly[0], enemy[0]));
			F(diff(friendly[1], enemy[1]));
			F(diff(friendly[2], enemy[2]));
			F(diff(friendly[3], enemy[3]));
			F(diff(friendly[4], enemy[4]));
		}
	}
	// A king attacking an insufficiently guarded enemy piece.
	F(diff(kingthreat(us, them), kingthreat(them, us)));
	{
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
}

inline static constexpr std::array<float, 108> kingsInitial{{
	0.0125F, 0.0045F, 0.006F, 0.025F, 0.009F, 0.012F, 0.025F, 0.009F, 0.012F, 0.03125F,
	0.01125F, 0.015F, 0.04375F, 0.01575F, 0.021F, 0.0025F, 0.005F, 0.0075F, 0.01F, 0.015F,
	0.005625F, 0.0045F, 0.00375F, 0.003F, 0.001875F, 0.0015F, -0.0275F, 0.0015F, 0.002F, 0.003F,
	0.004F, 0.01F, -0.005F, 0.015F, 0.01F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
	0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
	0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
	0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
	0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
	0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
	0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
	0.0F, 0.0F, 0.0F
}};

FORMULA(kings) {
	// Each attacking piece type emits inner-ring attacks, outer-ring attacks and potential checking moves.
	Sum friendly;
	Sum enemy;
	for (int type = 0; type < 5; ++type) {
		const auto fi = ringAttacks(us, them, type, 1);
		const auto ei = ringAttacks(them, us, type, 1);
		friendly.add(fi);
		enemy.add(ei);
		F(diff(fi, ei));
		F(diff(ringAttacks(us, them, type, 2), ringAttacks(them, us, type, 2)));
		F(diff(potentialChecks(us, them, type), potentialChecks(them, us, type)));
	}
	const auto fp = b.sum(friendly);
	const auto ep = b.sum(enemy);
	// Total king pressure crossing four fixed thresholds.
	for (const int threshold : {2, 4, 6, 8}) {
		F(diff(b.GE(fp, b.NUM(threshold)), b.GE(ep, b.NUM(threshold))));
	}
	// King escape squares not occupied by friendly pieces or controlled by the opponent.
	F(diff(escapes(us, them), escapes(them, us)));
	// Pawn shelter and enemy pawn storm at distances one through three.
	{
		for (int distance = 1; distance <= 3; ++distance) {
			F(diff(kingPawns(us, them, distance, true), kingPawns(them, us, distance, true)));
			F(diff(kingPawns(them, us, distance, false), kingPawns(us, them, distance, false)));
		}
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

	struct ShelterSignals {
		std::array<int, 32> own{};
		std::array<int, 32> storm{};
		std::array<int, 8> blocked{};
	};
	const auto shelter = [&](IS role, IS opponent) {
		ShelterSignals values;
		const Word ownPawns = b.REL(role, b.PCS(role, 0).bits & ~pawnAttacks(opponent).bits).bits;
		const Word enemyPawns = b.REL(role, b.PCS(opponent, 0).bits).bits;
		for (int square : b.squares(role, 5)) {
			const int rank = square / 8;
			const int center = std::clamp(square % 8, 1, 6);
			const Word fromRank = ~((1ULL << (8 * rank)) - 1);
			for (int file = center - 1; file <= center + 1; ++file) {
				const Word mask = fileMask(file);
				const Word own = ownPawns & mask & fromRank;
				const Word enemy = enemyPawns & mask & fromRank;
				const int ownRank = own ? std::countr_zero(own) / 8 : 0;
				const int enemyRank = enemy ? std::countr_zero(enemy) / 8 : 0;
				const int edge = std::min(file, 7 - file);
				++values.own[8 * edge + ownRank];
				if (ownRank != 0 && ownRank == enemyRank - 1)
					++values.blocked[enemyRank];
				else
					++values.storm[8 * edge + enemyRank];
			}
		}
		return values;
	};
	const auto friendlyShelter = shelter(us, them);
	const auto enemyShelter = shelter(them, us);
	// Complete shelter and storm geometry: king-edge group crossed with nearest pawn rank.
	for (unsigned i = 0; i < friendlyShelter.own.size(); ++i)
		F(diff(b.NUM(friendlyShelter.own[i]), b.NUM(enemyShelter.own[i])));
	for (unsigned i = 0; i < friendlyShelter.storm.size(); ++i)
		F(diff(b.NUM(friendlyShelter.storm[i]), b.NUM(enemyShelter.storm[i])));
	// Blocked pawn storms retain the enemy pawn's normalized rank.
	for (unsigned i = 0; i < friendlyShelter.blocked.size(); ++i)
		F(diff(b.NUM(friendlyShelter.blocked[i]), b.NUM(enemyShelter.blocked[i])));
	// Queen absence remains available as an independent condition for learned king-safety interactions.
	F(diff(fnoqueen, enoqueen));
}

inline static constexpr std::array<float, 22> endgamesInitial{{
	-0.05F, -0.15F, 0.002F, -0.005F, 0.007F, 0.02F, 0.006F, 0.0125F, 0.0F, 0.0F,
	0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
	0.0F, 0.0F
}};

FORMULA(endgames) {
	// Opposite-colored single bishops, signed by the side with the material advantage.
	constexpr std::uint64_t light = 0x55AA55AA55AA55AAULL;
	const auto onefb = b.EQ(b.POP(b.PCS(us, 2)), o);
	const auto oneeb = b.EQ(b.POP(b.PCS(them, 2)), o);
	const auto opposite = b.LOR(b.LAND(b.ANY(b.AND(b.PCS(us, 2), b.BB(light))), b.ANY(b.AND(b.PCS(them, 2), b.NOT(b.BB(light))))),
	    b.LAND(b.ANY(b.AND(b.PCS(us, 2), b.NOT(b.BB(light)))), b.ANY(b.AND(b.PCS(them, 2), b.BB(light)))));
	const auto fpawns = b.POP(b.PCS(us, 0));
	const auto epawns = b.POP(b.PCS(them, 0));
	const auto material = b.ADD(diff(nonPawnMaterial(us), nonPawnMaterial(them)), diff(fpawns, epawns));
	const auto positive = b.GT(material, z);
	const auto negative = b.LT(material, z);
	const auto direction = diff(positive, negative);
	F(b.MUL(b.LAND(b.LAND(onefb, oneeb), opposite), direction));
	const auto fpawnless = b.EQ(fpawns, z);
	const auto epawnless = b.EQ(epawns, z);
	const auto thin = b.LE(b.ABS(diff(nonPawnMaterial(us), nonPawnMaterial(them))), o);
	// A thin non-pawn material advantage whose stronger side has no pawns.
	F(diff(b.LAND(b.LAND(positive, fpawnless), thin), b.LAND(b.LAND(negative, epawnless), thin)));

	IS symmetric = z;
	IS asymmetric = z;
	for (int file = 0; file < 8; ++file) {
		const auto fp = b.ANY(b.AND(b.PCS(us, 0), b.BB(fileMask(file))));
		const auto ep = b.ANY(b.AND(b.PCS(them, 0), b.BB(fileMask(file))));
		symmetric = b.ADD(symmetric, b.LAND(fp, ep));
		asymmetric = b.ADD(asymmetric, b.LOR(b.LAND(fp, b.LNOT(ep)), b.LAND(ep, b.LNOT(fp))));
	}
	// Total pawns, symmetric pawn files, asymmetric pawn files and bare kings, each signed by material advantage.
	F(b.MUL(direction, b.ADD(fpawns, epawns)));
	F(b.MUL(direction, symmetric));
	F(b.MUL(direction, asymmetric));
	F(b.MUL(direction, b.EQ(phase, z)));
	const auto strongpawns = diff(b.MUL(positive, fpawns), b.MUL(negative, epawns));
	// Pawn count belonging to the materially stronger side.
	F(strongpawns);
	const auto strongpassers = diff(b.MUL(positive, b.POP(passedPawns(us, them))), b.MUL(negative, b.POP(passedPawns(them, us))));
	// Passed pawns of the stronger side in opposite-colored-bishop endings.
	F(b.MUL(b.LAND(b.LAND(onefb, oneeb), opposite), strongpassers));

	const auto sidePhase = [&](IS role) {
		return b.ADD(b.ADD(b.POP(b.PCS(role, 1)), b.POP(b.PCS(role, 2))),
		    b.ADD(b.MUL(b.NUM(2), b.POP(b.PCS(role, 3))), b.MUL(b.NUM(4), b.POP(b.PCS(role, 4)))));
	};
	const auto fphase = sidePhase(us);
	const auto ephase = sidePhase(them);
	const auto otherPieces = b.ADD(b.ADD(b.POP(b.PCS(us, 1)), b.POP(b.PCS(them, 1))),
	    b.ADD(b.ADD(b.POP(b.PCS(us, 3)), b.POP(b.PCS(them, 3))), b.ADD(b.POP(b.PCS(us, 4)), b.POP(b.PCS(them, 4)))));
	const auto pureOpposite = b.LAND(b.LAND(b.LAND(onefb, oneeb), opposite), b.EQ(otherPieces, z));
	const auto mixedOpposite = b.LAND(b.LAND(b.LAND(onefb, oneeb), opposite), b.GT(otherPieces, z));
	const auto strongPhase = b.ADD(b.MUL(positive, fphase), b.MUL(negative, ephase));
	const auto weakPhase = b.ADD(b.MUL(positive, ephase), b.MUL(negative, fphase));
	const auto strongPawnCount = b.ADD(b.MUL(positive, fpawns), b.MUL(negative, epawns));
	const auto weakPawnCount = b.ADD(b.MUL(positive, epawns), b.MUL(negative, fpawns));
	const auto strongPawnLead = b.GE(strongPawnCount, b.ADD(weakPawnCount, b.NUM(2)));
	IS friendlyPieces = z;
	IS enemyPieces = z;
	for (int type = 0; type < 5; ++type) {
		friendlyPieces = b.ADD(friendlyPieces, b.POP(b.PCS(us, type)));
		enemyPieces = b.ADD(enemyPieces, b.POP(b.PCS(them, type)));
	}
	const auto strongPieceCount = b.ADD(b.MUL(positive, friendlyPieces), b.MUL(negative, enemyPieces));
	const auto missingStrongPawns = b.SUB(b.NUM(8), strongPawnCount);
	// Symmetric state signals serve as graybox conditions for the public endgame scaling knowledge.
	for (const auto signal : {pureOpposite, mixedOpposite, strongPhase, weakPhase, weakPawnCount, strongPawnLead, strongPieceCount,
	         b.MUL(missingStrongPawns, missingStrongPawns)})
		F(diff(signal, signal));

	const auto pawnlessStrong = b.EQ(strongPawnCount, z);
	const auto thinPhase = b.LE(b.SUB(strongPhase, weakPhase), o);
	const auto strongBelowTwo = b.LT(strongPhase, b.NUM(2));
	const auto weakAtMostOne = b.LE(weakPhase, o);
	const auto pawnlessBare = b.LAND(b.LAND(pawnlessStrong, thinPhase), strongBelowTwo);
	const auto pawnlessWeak = b.LAND(b.LAND(b.LAND(pawnlessStrong, thinPhase), b.LNOT(strongBelowTwo)), weakAtMostOne);
	const auto pawnlessOther = b.LAND(b.LAND(b.LAND(pawnlessStrong, thinPhase), b.LNOT(strongBelowTwo)), b.LNOT(weakAtMostOne));
	const auto strongPasserCount = b.ADD(b.MUL(positive, b.POP(passedPawns(us, them))), b.MUL(negative, b.POP(passedPawns(them, us))));
	for (const auto signal : {pawnlessBare, pawnlessWeak, pawnlessOther, b.MUL(pureOpposite, strongPasserCount),
	         b.MUL(pureOpposite, strongPawnLead), b.MUL(mixedOpposite, strongPieceCount)})
		F(diff(signal, signal));
}

inline static constexpr auto initialWeights = [] {
	std::array<float, kFormulaCount> values{};
	std::size_t index = 0;
	const auto append = [&](const auto &part) {
		for (float value : part)
			values[index++] = value;
	};
	append(tempoInitial);
	append(materialInitial);
	append(pstInitial);
	append(bishopPairInitial);
	append(pawnsInitial);
	append(mobilityInitial);
	append(piecesInitial);
	append(threatsInitial);
	append(kingsInitial);
	append(endgamesInitial);
	return values;
}();
static_assert(initialWeights.size() == kFormulaCount);

#endif
