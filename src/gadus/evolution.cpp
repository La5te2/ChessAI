#include "gadus/fcpi.hpp"

#include <hdf5.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "gadus/args.hpp"
#include "gadus/checkpoint.hpp"

namespace gadus {

namespace {

inline constexpr const char *kFcpiFormula = "gadus_adaptive_value_expansion_td_kl";

std::string zero_pad(int value, int width) {
	std::ostringstream output;
	output << std::setfill('0') << std::setw(width) << value;
	return output.str();
}

struct Position {
	int game_id = 0;
	PackedState state{};
	std::string fen;
	float root_value = 0.0F;
	std::vector<int> legal_indices;
	std::vector<float> legal_prior;
	int played_index = 0;
	std::vector<int> candidate_indices;
	float value_target = 0.0F;
	std::vector<float> policy_target;
	std::vector<float> candidate_q;
	int aggregate_count = 1;
};

struct Trajectory {
	int game_id = 0;
	chess::Board board;
	std::vector<Position> positions;
};

struct Branch {
	std::size_t record = 0;
	std::size_t candidate = 0;
	chess::Board board;
	int depth = 1;
	std::vector<float> estimates;
	std::vector<float> policy;
	float current_value = 0.0F;
	float last_residual = 0.0F;
	float last_change = 0.0F;
	bool terminal = false;
};

struct TargetSummary {
	std::int64_t branches = 0;
	std::int64_t branch_plies = 0;
	std::int64_t terminal_branches = 0;
	double residual_sum = 0.0;
	std::int64_t residual_count = 0;
};

struct SamplingSpec {
	std::string fen;
};

void require_h5(herr_t status, const std::string &operation) {
	if (status < 0) {
		throw std::runtime_error("HDF5 operation failed: " + operation);
	}
}

hid_t require_id(hid_t id, const std::string &operation) {
	if (id < 0) {
		throw std::runtime_error("HDF5 operation failed: " + operation);
	}
	return id;
}

void write_string_attribute(hid_t object, const char *name, const std::string &value) {
	const hid_t space = require_id(H5Screate(H5S_SCALAR), name);
	const hid_t type = require_id(H5Tcopy(H5T_C_S1), name);
	require_h5(H5Tset_size(type, value.size() + 1), name);
	const hid_t attribute =
		require_id(H5Acreate2(object, name, type, space, H5P_DEFAULT, H5P_DEFAULT), name);
	require_h5(H5Awrite(attribute, type, value.c_str()), name);
	H5Aclose(attribute);
	H5Tclose(type);
	H5Sclose(space);
}

void write_dataset(hid_t file, const char *name, hid_t file_type, hid_t memory_type,
				   const std::vector<hsize_t> &shape, const void *data) {
	const hid_t space =
		require_id(H5Screate_simple(static_cast<int>(shape.size()), shape.data(), nullptr), name);
	const hid_t properties = require_id(H5Pcreate(H5P_DATASET_CREATE), name);
	if (shape[0] > 0) {
		auto chunk = shape;
		chunk[0] = std::min<hsize_t>(shape[0], 4096);
		require_h5(H5Pset_chunk(properties, static_cast<int>(chunk.size()), chunk.data()), name);
		require_h5(H5Pset_deflate(properties, 1), name);
	}
	const hid_t dataset = require_id(
		H5Dcreate2(file, name, file_type, space, H5P_DEFAULT, properties, H5P_DEFAULT), name);
	require_h5(H5Dwrite(dataset, memory_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data), name);
	H5Dclose(dataset);
	H5Pclose(properties);
	H5Sclose(space);
}

std::vector<float> normalize(std::vector<float> values) {
	double total = 0.0;
	for (float &value : values) {
		value = std::max(0.0F, value);
		total += value;
	}
	if (!std::isfinite(total) || total <= 0.0) {
		const float uniform = 1.0F / std::max<std::size_t>(1, values.size());
		std::fill(values.begin(), values.end(), uniform);
		return values;
	}
	for (float &value : values) {
		value = static_cast<float>(value / total);
	}
	return values;
}

std::vector<float> stable_softmax(const std::vector<double> &logits) {
	if (logits.empty()) {
		return {};
	}
	const double maximum = *std::max_element(logits.begin(), logits.end());
	std::vector<float> values(logits.size());
	for (std::size_t index = 0; index < logits.size(); ++index) {
		values[index] =
			static_cast<float>(std::exp(std::clamp(logits[index] - maximum, -80.0, 0.0)));
	}
	return normalize(std::move(values));
}

std::size_t sample_index(const std::vector<float> &probabilities, std::mt19937_64 &rng) {
	std::discrete_distribution<std::size_t> distribution(probabilities.begin(),
														 probabilities.end());
	return distribution(rng);
}

float mixed_depth_q(const std::vector<float> &estimates, double lambda) {
	if (estimates.empty()) {
		throw std::runtime_error("counterfactual branch has no estimates");
	}
	if (estimates.size() == 1) {
		return estimates[0];
	}
	const double bounded = std::clamp(lambda, 0.0, 1.0);
	double total = 0.0;
	for (std::size_t depth = 0; depth + 1 < estimates.size(); ++depth) {
		total += (1.0 - bounded) * std::pow(bounded, static_cast<double>(depth)) * estimates[depth];
	}
	total += std::pow(bounded, static_cast<double>(estimates.size() - 1)) * estimates.back();
	return static_cast<float>(std::clamp(total, -1.0, 1.0));
}

std::string packed_key(const PackedState &state) {
	return std::string(reinterpret_cast<const char *>(state.data()), state.size());
}

std::vector<SamplingSpec> make_sampling_specs(const FcpiOptions &options, int iteration,
											  nlohmann::json &summary) {
	std::mt19937_64 rng(options.seed + iteration);
	const int games = std::max(1, options.games_per_iter);
	const double fraction = std::clamp(options.startpos_fraction, 0.0, 1.0);
	const int startpos_games =
		options.opening_book.empty()
			? games
			: std::clamp(static_cast<int>(std::llround(games * fraction)), 0, games);
	const int book_games = games - startpos_games;
	std::vector<SamplingSpec> specs;
	specs.reserve(games);
	for (int index = 0; index < startpos_games; ++index) {
		specs.push_back({std::string(chess::constants::STARTPOS)});
	}
	int cycles = 0;
	std::vector<std::string> positions;
	if (book_games > 0) {
		positions = load_opening_positions(options.opening_book, options.book_plies,
										   options.max_book_positions, options.seed + iteration);
		if (positions.empty()) {
			throw std::runtime_error("opening book contains no FCPI positions");
		}
		while (static_cast<int>(specs.size()) < games) {
			auto cycle = positions;
			std::shuffle(cycle.begin(), cycle.end(), rng);
			for (const auto &fen : cycle) {
				if (static_cast<int>(specs.size()) >= games) {
					break;
				}
				specs.push_back({fen});
			}
			++cycles;
		}
	}
	std::shuffle(specs.begin(), specs.end(), rng);
	summary = {
		{"games", games},
		{"startpos_games", startpos_games},
		{"book_games", book_games},
		{"book_positions", positions.size()},
		{"book_cycles", cycles},
		{"reused_book_starts", std::max(0, book_games - static_cast<int>(positions.size()))},
	};
	return specs;
}

std::vector<SearchResult> evaluate_chunks(Searcher &searcher,
										  const std::vector<chess::Board> &boards, int batch_size) {
	std::vector<SearchResult> output;
	output.reserve(boards.size());
	for (std::size_t begin = 0; begin < boards.size(); begin += std::max(1, batch_size)) {
		const auto end = std::min(boards.size(), begin + std::max(1, batch_size));
		std::vector<chess::Board> chunk(boards.begin() + begin, boards.begin() + end);
		auto results = searcher.search_many(chunk);
		output.insert(output.end(), std::make_move_iterator(results.begin()),
					  std::make_move_iterator(results.end()));
	}
	return output;
}

std::vector<int> choose_candidates(const std::vector<int> &legal, const std::vector<float> &scores,
								   int played, int topk) {
	std::vector<std::size_t> order(legal.size());
	std::iota(order.begin(), order.end(), 0);
	std::stable_sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
		return scores[left] > scores[right];
	});
	const std::size_t count = std::min<std::size_t>(std::max(1, topk), order.size());
	std::vector<int> selected;
	for (std::size_t index = 0; index < count; ++index) {
		selected.push_back(legal[order[index]]);
	}
	if (std::find(selected.begin(), selected.end(), played) == selected.end()) {
		selected.back() = played;
	}
	selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
	return selected;
}

std::vector<Position> collect_selfplay(Model model, const torch::Device &device,
									   const FcpiOptions &options, int iteration,
									   nlohmann::json &sampling_summary) {
	SearchOptions closed;
	closed.type = SearchType::Closed;
	closed.mcts_sims = 0;
	closed.mcts_batch_size = options.inference_batch_size;
	Searcher evaluator(model, device, closed);
	nlohmann::json starts;
	const auto specs = make_sampling_specs(options, iteration, starts);
	std::mt19937_64 rng(options.seed + iteration);
	std::vector<Trajectory> trajectories;
	trajectories.reserve(specs.size());
	int completed = 0;
	std::cout << "fcpi self-play start: iteration=" << iteration << " arch_type=" << kArchType
			  << " games=" << specs.size() << " max_plies=" << options.max_plies
			  << " device=" << device.str() << std::endl;
	std::cout << "fcpi starts: " << starts.dump() << std::endl;

	for (std::size_t group_start = 0; group_start < specs.size();
		 group_start += std::max(1, options.games_in_flight)) {
		const auto group_end =
			std::min(specs.size(), group_start + std::max(1, options.games_in_flight));
		std::vector<Trajectory> group;
		for (std::size_t index = group_start; index < group_end; ++index) {
			Trajectory trajectory;
			trajectory.game_id = static_cast<int>(index) + 1;
			trajectory.board = chess::Board(specs[index].fen);
			group.push_back(std::move(trajectory));
		}
		std::vector<bool> done(group.size(), false);
		while (std::find(done.begin(), done.end(), false) != done.end()) {
			std::vector<std::size_t> active_indices;
			std::vector<chess::Board> boards;
			for (std::size_t index = 0; index < group.size(); ++index) {
				if (!done[index]) {
					active_indices.push_back(index);
					boards.push_back(group[index].board);
				}
			}
			const auto results = evaluate_chunks(evaluator, boards, options.inference_batch_size);
			for (std::size_t row = 0; row < active_indices.size(); ++row) {
				auto &trajectory = group[active_indices[row]];
				auto &board = trajectory.board;
				const auto moves = legal_moves(board);
				std::vector<int> legal;
				std::vector<float> prior;
				for (const auto &move : moves) {
					const int action = move_to_index(move);
					legal.push_back(action);
					prior.push_back(results[row].policy[action]);
				}
				prior = normalize(std::move(prior));
				const double temperature = std::max(1e-4, options.behavior_temperature);
				std::vector<float> behavior(prior.size());
				for (std::size_t index = 0; index < prior.size(); ++index) {
					behavior[index] = static_cast<float>(
						std::pow(std::clamp(static_cast<double>(prior[index]), 1e-12, 1.0),
								 1.0 / temperature));
				}
				behavior = normalize(std::move(behavior));
				const double mix = std::clamp(options.uniform_mix, 0.0, 1.0);
				for (float &probability : behavior) {
					probability =
						static_cast<float>((1.0 - mix) * probability + mix / behavior.size());
				}
				const std::size_t choice = sample_index(behavior, rng);
				const int played = legal[choice];
				Position position;
				position.game_id = trajectory.game_id;
				position.state = encode_state(board);
				position.fen = board.getFen();
				position.root_value = results[row].value;
				position.legal_indices = legal;
				position.legal_prior = prior;
				position.played_index = played;
				position.candidate_indices =
					choose_candidates(legal, prior, played, options.counterfactual_topk);
				trajectory.positions.push_back(std::move(position));
				board.makeMove(moves[choice]);
				const bool terminal = game_is_over(board);
				const bool truncated =
					static_cast<int>(trajectory.positions.size()) >= options.max_plies;
				if (terminal || truncated) {
					done[active_indices[row]] = true;
					++completed;
					std::cout << "fcpi game: completed=" << completed << '/' << specs.size()
							  << " game_id=" << trajectory.game_id
							  << " plies=" << trajectory.positions.size()
							  << " result=" << (terminal ? game_result(board) : "bootstrap")
							  << " truncated=" << (truncated && !terminal ? "true" : "false")
							  << std::endl;
				}
			}
		}
		for (auto &trajectory : group) {
			trajectories.push_back(std::move(trajectory));
		}
	}

	std::vector<chess::Board> truncated_boards;
	std::vector<std::size_t> truncated_indices;
	for (std::size_t index = 0; index < trajectories.size(); ++index) {
		if (!game_is_over(trajectories[index].board)) {
			truncated_indices.push_back(index);
			truncated_boards.push_back(trajectories[index].board);
		}
	}
	const auto truncated_results =
		evaluate_chunks(evaluator, truncated_boards, options.inference_batch_size);
	std::unordered_map<std::size_t, float> bootstrap;
	for (std::size_t index = 0; index < truncated_indices.size(); ++index) {
		bootstrap[truncated_indices[index]] = truncated_results[index].value;
	}
	const double lambda = std::clamp(options.td_lambda, 0.0, 1.0);
	for (std::size_t trajectory_index = 0; trajectory_index < trajectories.size();
		 ++trajectory_index) {
		auto &trajectory = trajectories[trajectory_index];
		float next_return = game_is_over(trajectory.board)
								? terminal_value_side_to_move(trajectory.board)
								: bootstrap.at(trajectory_index);
		const float final_value = next_return;
		for (int index = static_cast<int>(trajectory.positions.size()) - 1; index >= 0; --index) {
			const float next_value = index + 1 == static_cast<int>(trajectory.positions.size())
										 ? final_value
										 : trajectory.positions[index + 1].root_value;
			const float current_return =
				-static_cast<float>((1.0 - lambda) * next_value + lambda * next_return);
			trajectory.positions[index].value_target = std::clamp(current_return, -1.0F, 1.0F);
			next_return = current_return;
		}
	}

	std::mt19937_64 sample_rng(options.seed + iteration + 1'000'003);
	std::vector<Position> records;
	std::int64_t source_positions = 0;
	std::int64_t unique_positions = 0;
	int capped_games = 0;
	for (auto &trajectory : trajectories) {
		source_positions += trajectory.positions.size();
		std::unordered_set<std::string> seen;
		std::vector<std::size_t> indices;
		for (std::size_t index = 0; index < trajectory.positions.size(); ++index) {
			if (seen.insert(packed_key(trajectory.positions[index].state)).second) {
				indices.push_back(index);
			}
		}
		unique_positions += indices.size();
		if (static_cast<int>(indices.size()) > options.positions_per_game) {
			std::shuffle(indices.begin(), indices.end(), sample_rng);
			indices.resize(options.positions_per_game);
			std::sort(indices.begin(), indices.end());
			++capped_games;
		}
		for (const auto index : indices) {
			records.push_back(std::move(trajectory.positions[index]));
		}
	}
	sampling_summary = {
		{"games", trajectories.size()},
		{"source_positions", source_positions},
		{"unique_positions", unique_positions},
		{"selected_positions", records.size()},
		{"positions_per_game", options.positions_per_game},
		{"capped_games", capped_games},
		{"starts", starts},
	};
	std::cout << "fcpi position sampling: " << sampling_summary.dump() << std::endl;
	return records;
}

void evaluate_frontier(std::vector<Branch *> branches, Searcher &evaluator,
					   const FcpiOptions &options) {
	std::vector<chess::Board> boards;
	std::vector<Branch *> pending;
	for (auto *branch : branches) {
		if (!branch->terminal) {
			pending.push_back(branch);
			boards.push_back(branch->board);
		}
	}
	const auto results = evaluate_chunks(evaluator, boards, options.inference_batch_size);
	for (std::size_t index = 0; index < pending.size(); ++index) {
		auto *branch = pending[index];
		branch->current_value = results[index].value;
		branch->policy = results[index].policy;
		const float sign = branch->depth % 2 == 0 ? 1.0F : -1.0F;
		branch->estimates.push_back(sign * results[index].value);
	}
}

void construct_targets(std::vector<Position> &records, Model model, const torch::Device &device,
					   const FcpiOptions &options, TargetSummary &summary) {
	SearchOptions closed;
	closed.type = SearchType::Closed;
	closed.mcts_sims = 0;
	closed.mcts_batch_size = options.inference_batch_size;
	Searcher evaluator(model, device, closed);
	const int min_plies = std::max(1, options.counterfactual_min_plies);
	const int max_plies = std::max(min_plies, options.counterfactual_max_plies);
	const double target_average =
		std::clamp(options.counterfactual_target_average_plies, static_cast<double>(min_plies),
				   static_cast<double>(max_plies));
	std::cout << "fcpi counterfactual start: positions=" << records.size()
			  << " target_average_plies=" << target_average << std::endl;

	for (std::size_t subset_begin = 0; subset_begin < records.size();
		 subset_begin += std::max(1, options.target_records_per_batch)) {
		const auto subset_end =
			std::min(records.size(), subset_begin + std::max(1, options.target_records_per_batch));
		std::vector<Branch> branches;
		for (std::size_t record_index = subset_begin; record_index < subset_end; ++record_index) {
			auto &record = records[record_index];
			chess::Board board(record.fen);
			for (std::size_t candidate = 0; candidate < record.candidate_indices.size();
				 ++candidate) {
				const auto move = index_to_move(record.candidate_indices[candidate], board);
				if (move.move() == chess::Move::NO_MOVE) {
					throw std::runtime_error("FCPI candidate action is illegal");
				}
				Branch branch;
				branch.record = record_index;
				branch.candidate = candidate;
				branch.board = board;
				branch.board.makeMove(move);
				if (game_is_over(branch.board)) {
					branch.estimates.push_back(-terminal_value_side_to_move(branch.board));
					branch.terminal = true;
					++summary.terminal_branches;
				}
				branches.push_back(std::move(branch));
			}
		}
		std::vector<Branch *> all;
		for (auto &branch : branches) {
			all.push_back(&branch);
		}
		evaluate_frontier(all, evaluator, options);
		for (auto &branch : branches) {
			if (!branch.terminal) {
				branch.last_residual =
					std::abs(records[branch.record].root_value + branch.current_value);
				branch.last_change =
					std::abs(branch.estimates.back() - records[branch.record].root_value);
				summary.residual_sum += branch.last_residual;
				++summary.residual_count;
			}
		}

		const std::int64_t target_depth =
			static_cast<std::int64_t>(std::llround(branches.size() * target_average));
		while (true) {
			std::vector<Branch *> expandable;
			for (auto &branch : branches) {
				if (!branch.terminal && branch.depth < max_plies) {
					expandable.push_back(&branch);
				}
			}
			if (expandable.empty()) {
				break;
			}
			std::int64_t depth_sum = 0;
			for (const auto &branch : branches) {
				depth_sum += branch.depth;
			}
			std::vector<Branch *> active;
			for (auto *branch : expandable) {
				if (branch->depth < min_plies) {
					active.push_back(branch);
				}
			}
			if (active.empty()) {
				const std::int64_t remaining = target_depth - depth_sum;
				if (remaining <= 0) {
					break;
				}
				std::unordered_map<std::size_t, float> best;
				for (const auto &branch : branches) {
					best[branch.record] = std::max(branch.estimates.back(),
												   best.contains(branch.record)
													   ? best[branch.record]
													   : -std::numeric_limits<float>::infinity());
				}
				std::sort(expandable.begin(), expandable.end(),
						  [&](const Branch *left, const Branch *right) {
							  auto priority = [&](const Branch *branch) {
								  const double competitiveness =
									  1.0 -
									  std::min(1.0, std::max(0.0, static_cast<double>(
																	  best[branch->record] -
																	  branch->estimates.back()) /
																	  2.0));
								  return std::max(branch->last_residual, branch->last_change) +
										 0.05 * competitiveness;
							  };
							  return priority(left) > priority(right);
						  });
				active.assign(expandable.begin(),
							  expandable.begin() +
								  std::min<std::int64_t>(remaining, expandable.size()));
			}

			std::unordered_map<Branch *, std::pair<float, float>> previous;
			for (auto *branch : active) {
				const auto moves = legal_moves(branch->board);
				auto selected = std::max_element(moves.begin(), moves.end(),
												 [&](const auto &left, const auto &right) {
													 return branch->policy[move_to_index(left)] <
															branch->policy[move_to_index(right)];
												 });
				previous.emplace(branch,
								 std::pair(branch->current_value, branch->estimates.back()));
				branch->board.makeMove(*selected);
				branch->depth += 1;
				branch->policy.clear();
				if (game_is_over(branch->board)) {
					const float sign = branch->depth % 2 == 0 ? 1.0F : -1.0F;
					branch->current_value = terminal_value_side_to_move(branch->board);
					branch->estimates.push_back(sign * branch->current_value);
					branch->terminal = true;
					++summary.terminal_branches;
				}
			}
			evaluate_frontier(active, evaluator, options);
			for (auto *branch : active) {
				branch->last_residual = std::abs(previous.at(branch).first + branch->current_value);
				branch->last_change =
					std::abs(branch->estimates.back() - previous.at(branch).second);
				summary.residual_sum += branch->last_residual;
				++summary.residual_count;
			}
		}

		for (auto &branch : branches) {
			auto &record = records[branch.record];
			if (record.candidate_q.empty()) {
				record.candidate_q.resize(record.candidate_indices.size());
			}
			record.candidate_q[branch.candidate] =
				mixed_depth_q(branch.estimates, options.counterfactual_lambda);
			summary.branch_plies += branch.depth;
			++summary.branches;
		}
		if (subset_end == records.size() || subset_end % std::max(1, options.log_every) == 0) {
			std::cout << "fcpi counterfactual: positions=" << subset_end << '/' << records.size()
					  << " branches=" << summary.branches
					  << " branch_plies=" << summary.branch_plies << std::endl;
		}
	}

	const double played_weight = std::clamp(options.played_return_weight, 0.0, 1.0);
	const double temperature = std::max(1e-4, options.policy_temperature);
	for (auto &record : records) {
		for (std::size_t index = 0; index < record.candidate_indices.size(); ++index) {
			if (record.candidate_indices[index] == record.played_index) {
				record.candidate_q[index] =
					static_cast<float>((1.0 - played_weight) * record.candidate_q[index] +
									   played_weight * record.value_target);
			}
		}
		std::vector<float> q_all(record.legal_indices.size(), record.root_value);
		for (std::size_t candidate = 0; candidate < record.candidate_indices.size(); ++candidate) {
			const auto found = std::find(record.legal_indices.begin(), record.legal_indices.end(),
										 record.candidate_indices[candidate]);
			q_all[found - record.legal_indices.begin()] = record.candidate_q[candidate];
		}
		std::vector<double> logits(record.legal_indices.size());
		for (std::size_t index = 0; index < logits.size(); ++index) {
			logits[index] = options.prior_power *
								std::log(std::clamp(static_cast<double>(record.legal_prior[index]),
													1e-12, 1.0)) +
							(q_all[index] - record.root_value) / temperature;
		}
		record.policy_target = stable_softmax(logits);
	}
	std::cout << "fcpi counterfactual summary: branches=" << summary.branches << " average_depth="
			  << (summary.branches > 0
					  ? static_cast<double>(summary.branch_plies) / summary.branches
					  : 0.0)
			  << " terminal_branches=" << summary.terminal_branches << " mean_residual="
			  << (summary.residual_count > 0 ? summary.residual_sum / summary.residual_count : 0.0)
			  << std::endl;
}

std::vector<Position> aggregate_records(std::vector<Position> records, nlohmann::json &summary) {
	const std::size_t source_count = records.size();
	std::unordered_map<std::string, std::size_t> groups;
	std::vector<Position> output;
	for (auto &record : records) {
		const auto key = packed_key(record.state);
		const auto found = groups.find(key);
		if (found == groups.end()) {
			groups.emplace(key, output.size());
			output.push_back(std::move(record));
			continue;
		}
		auto &merged = output[found->second];
		if (merged.legal_indices != record.legal_indices) {
			throw std::runtime_error("identical encoded states produced different legal actions");
		}
		const float old_count = static_cast<float>(merged.aggregate_count);
		const float new_count = old_count + 1.0F;
		for (std::size_t index = 0; index < merged.legal_prior.size(); ++index) {
			merged.legal_prior[index] =
				(merged.legal_prior[index] * old_count + record.legal_prior[index]) / new_count;
			merged.policy_target[index] =
				(merged.policy_target[index] * old_count + record.policy_target[index]) / new_count;
		}
		merged.value_target = (merged.value_target * old_count + record.value_target) / new_count;
		for (std::size_t candidate = 0; candidate < record.candidate_indices.size(); ++candidate) {
			const int action = record.candidate_indices[candidate];
			const auto existing =
				std::find(merged.candidate_indices.begin(), merged.candidate_indices.end(), action);
			if (existing == merged.candidate_indices.end()) {
				merged.candidate_indices.push_back(action);
				merged.candidate_q.push_back(record.candidate_q[candidate]);
			} else {
				const auto index = existing - merged.candidate_indices.begin();
				merged.candidate_q[index] =
					(merged.candidate_q[index] * old_count + record.candidate_q[candidate]) /
					new_count;
			}
		}
		merged.aggregate_count += 1;
	}
	for (auto &record : output) {
		record.legal_prior = normalize(std::move(record.legal_prior));
		record.policy_target = normalize(std::move(record.policy_target));
	}
	summary = {
		{"source_positions", source_count},
		{"aggregated_positions", output.size()},
		{"merged_positions", source_count - output.size()},
	};
	return output;
}

nlohmann::json write_fcpi_h5(const std::filesystem::path &path, std::vector<Position> &records) {
	nlohmann::json aggregation;
	records = aggregate_records(std::move(records), aggregation);
	std::cout << "fcpi position aggregation: " << aggregation.dump() << std::endl;
	if (records.empty()) {
		throw std::runtime_error("FCPI generated no training positions");
	}
	const std::size_t legal_width =
		std::max_element(records.begin(), records.end(), [](const auto &left, const auto &right) {
			return left.legal_indices.size() < right.legal_indices.size();
		})->legal_indices.size();
	const std::size_t candidate_width =
		std::max_element(records.begin(), records.end(), [](const auto &left, const auto &right) {
			return left.candidate_indices.size() < right.candidate_indices.size();
		})->candidate_indices.size();
	const std::size_t count = records.size();
	std::vector<std::uint8_t> states(count * kStatePlanes * 8);
	std::vector<std::int32_t> legal(count * legal_width, 0);
	std::vector<float> priors(count * legal_width, 0.0F);
	std::vector<float> policy(count * legal_width, 0.0F);
	std::vector<std::uint8_t> legal_counts(count);
	std::vector<float> values(count);
	std::vector<std::int32_t> candidates(count * candidate_width, 0);
	std::vector<float> candidate_q(count * candidate_width, 0.0F);
	std::vector<std::uint8_t> candidate_counts(count);
	for (std::size_t row = 0; row < count; ++row) {
		std::copy(records[row].state.begin(), records[row].state.end(),
				  states.begin() + row * kStatePlanes * 8);
		legal_counts[row] = static_cast<std::uint8_t>(records[row].legal_indices.size());
		candidate_counts[row] = static_cast<std::uint8_t>(records[row].candidate_indices.size());
		values[row] = records[row].value_target;
		for (std::size_t column = 0; column < records[row].legal_indices.size(); ++column) {
			legal[row * legal_width + column] = records[row].legal_indices[column];
			priors[row * legal_width + column] = records[row].legal_prior[column];
			policy[row * legal_width + column] = records[row].policy_target[column];
		}
		for (std::size_t column = 0; column < records[row].candidate_indices.size(); ++column) {
			candidates[row * candidate_width + column] = records[row].candidate_indices[column];
			candidate_q[row * candidate_width + column] = records[row].candidate_q[column];
		}
	}
	if (!path.parent_path().empty()) {
		std::filesystem::create_directories(path.parent_path());
	}
	const hid_t file = require_id(
		H5Fcreate(path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), path.string());
	write_string_attribute(file, "arch_type", kArchType);
	write_string_attribute(file, "fcpi_formula", kFcpiFormula);
	write_dataset(file, "states", H5T_STD_U8LE, H5T_NATIVE_UINT8, {count, kStatePlanes, 8},
				  states.data());
	write_dataset(file, "legal_indices", H5T_STD_I32LE, H5T_NATIVE_INT32, {count, legal_width},
				  legal.data());
	write_dataset(file, "legal_priors", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT, {count, legal_width},
				  priors.data());
	write_dataset(file, "policy_targets", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT, {count, legal_width},
				  policy.data());
	write_dataset(file, "legal_counts", H5T_STD_U8LE, H5T_NATIVE_UINT8, {count},
				  legal_counts.data());
	write_dataset(file, "value_targets", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT, {count}, values.data());
	write_dataset(file, "candidate_indices", H5T_STD_I32LE, H5T_NATIVE_INT32,
				  {count, candidate_width}, candidates.data());
	write_dataset(file, "candidate_q", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT, {count, candidate_width},
				  candidate_q.data());
	write_dataset(file, "candidate_counts", H5T_STD_U8LE, H5T_NATIVE_UINT8, {count},
				  candidate_counts.data());
	H5Fclose(file);
	return {
		{"path", path.string()},	  {"positions", count},
		{"legal_width", legal_width}, {"counterfactual_width", candidate_width},
		{"formula", kFcpiFormula},	  {"aggregation", aggregation},
	};
}

nlohmann::json train_candidate(const std::filesystem::path &source,
							   const std::filesystem::path &candidate, Model model,
							   const torch::Device &device, std::vector<Position> &records,
							   const FcpiOptions &options) {
	model->to(device);
	model->train();
	torch::optim::AdamW optimizer(
		model->parameters(),
		torch::optim::AdamWOptions(options.learning_rate).weight_decay(options.weight_decay));
	std::vector<std::size_t> order(records.size());
	std::iota(order.begin(), order.end(), 0);
	std::mt19937_64 rng(options.seed);
	std::int64_t steps = 0;
	double total_loss = 0.0;
	double total_policy = 0.0;
	double total_value = 0.0;
	double total_kl = 0.0;
	double total_entropy = 0.0;
	for (int epoch = 0; epoch < std::max(0, options.epochs); ++epoch) {
		std::shuffle(order.begin(), order.end(), rng);
		for (std::size_t begin = 0; begin < order.size();
			 begin += std::max(1, options.batch_size)) {
			const auto end = std::min(order.size(), begin + std::max(1, options.batch_size));
			const std::int64_t batch = static_cast<std::int64_t>(end - begin);
			std::size_t width = 1;
			std::vector<std::uint8_t> packed(batch * kStatePlanes * 8);
			for (std::size_t index = begin; index < end; ++index) {
				width = std::max(width, records[order[index]].legal_indices.size());
				std::copy(records[order[index]].state.begin(), records[order[index]].state.end(),
						  packed.begin() + (index - begin) * kStatePlanes * 8);
			}
			auto states = decode_states(packed.data(), batch).to(device, true);
			auto legal = torch::zeros({batch, static_cast<std::int64_t>(width)}, torch::kInt64);
			auto priors = torch::zeros({batch, static_cast<std::int64_t>(width)}, torch::kFloat32);
			auto targets = torch::zeros({batch, static_cast<std::int64_t>(width)}, torch::kFloat32);
			auto counts = torch::zeros({batch}, torch::kInt64);
			auto values = torch::zeros({batch}, torch::kFloat32);
			auto legal_access = legal.accessor<std::int64_t, 2>();
			auto prior_access = priors.accessor<float, 2>();
			auto target_access = targets.accessor<float, 2>();
			auto count_access = counts.accessor<std::int64_t, 1>();
			auto value_access = values.accessor<float, 1>();
			for (std::size_t index = begin; index < end; ++index) {
				const auto &record = records[order[index]];
				const auto row = static_cast<std::int64_t>(index - begin);
				count_access[row] = record.legal_indices.size();
				value_access[row] = record.value_target;
				for (std::size_t column = 0; column < record.legal_indices.size(); ++column) {
					legal_access[row][column] = record.legal_indices[column];
					prior_access[row][column] = record.legal_prior[column];
					target_access[row][column] = record.policy_target[column];
				}
			}
			legal = legal.to(device, true);
			priors = priors.to(device, true);
			targets = targets.to(device, true);
			counts = counts.to(device, true);
			values = values.to(device, true);

			optimizer.zero_grad();
			auto [logits, predicted] = model->forward(states);
			auto selected = logits.gather(1, legal).to(torch::kFloat32);
			auto columns = torch::arange(static_cast<std::int64_t>(width), counts.options());
			auto mask = columns.unsqueeze(0) < counts.unsqueeze(1);
			selected = selected.masked_fill(~mask, -1e9);
			auto log_probability = torch::log_softmax(selected, 1);
			auto masked_targets = targets * mask;
			masked_targets = masked_targets / masked_targets.sum(1, true).clamp_min(1e-8);
			auto masked_priors = priors.clamp_min(1e-8) * mask;
			masked_priors = masked_priors / masked_priors.sum(1, true).clamp_min(1e-8);
			auto probabilities = torch::exp(log_probability);
			auto policy_loss = -(masked_targets * log_probability).sum(1).mean();
			auto kl =
				(probabilities * (log_probability - torch::log(masked_priors.clamp_min(1e-8))))
					.sum(1)
					.mean();
			auto entropy = -(probabilities * log_probability).sum(1).mean();
			auto value_loss = torch::nn::functional::smooth_l1_loss(predicted.squeeze(1), values);
			auto loss = options.policy_weight * policy_loss + options.value_weight * value_loss +
						options.kl_weight * kl - options.entropy_weight * entropy;
			loss.backward();
			if (options.grad_clip > 0.0) {
				torch::nn::utils::clip_grad_norm_(model->parameters(), options.grad_clip);
			}
			optimizer.step();
			++steps;
			total_loss += loss.item<double>();
			total_policy += policy_loss.item<double>();
			total_value += value_loss.item<double>();
			total_kl += kl.item<double>();
			total_entropy += entropy.item<double>();
			if (options.log_every > 0 && (steps == 1 || steps % options.log_every == 0)) {
				std::cout << "fcpi train: step=" << steps
						  << " policy=" << policy_loss.item<double>()
						  << " value=" << value_loss.item<double>() << " kl=" << kl.item<double>()
						  << " entropy=" << entropy.item<double>()
						  << " loss=" << loss.item<double>() << std::endl;
			}
			if (options.train_max_steps > 0 && steps >= options.train_max_steps) {
				break;
			}
		}
		if (options.train_max_steps > 0 && steps >= options.train_max_steps) {
			break;
		}
	}
	CheckpointInfo source_info;
	load_checkpoint(source, torch::Device(torch::kCPU), &source_info);
	save_checkpoint_atomic(candidate, model,
						   {source_info.channels, source_info.blocks,
							source_info.epoch + std::max(0, options.epochs),
							source_info.global_step + steps, "fcpi"});
	const double divisor = static_cast<double>(std::max<std::int64_t>(1, steps));
	return {
		{"steps", steps},
		{"epochs_requested", options.epochs},
		{"candidate", candidate.string()},
		{"metrics",
		 {
			 {"loss", total_loss / divisor},
			 {"policy", total_policy / divisor},
			 {"value", total_value / divisor},
			 {"kl", total_kl / divisor},
			 {"entropy", total_entropy / divisor},
		 }},
	};
}

} // namespace

void run_fcpi(const FcpiOptions &options) {
	if (options.iterations <= 0 || options.games_per_iter <= 0 || options.games_in_flight <= 0) {
		throw std::invalid_argument("FCPI iteration and game counts must be positive");
	}
	if (!std::filesystem::is_regular_file(options.model)) {
		throw std::runtime_error("FCPI model not found: " + options.model.string());
	}
	const auto run_id = create_run_id("fcpi");
	const auto data_dir = std::filesystem::path("data/runs") / run_id;
	const auto model_dir = std::filesystem::path("models/runs") / run_id;
	if (std::filesystem::exists(data_dir) || std::filesystem::exists(model_dir)) {
		throw std::runtime_error("FCPI run directory already exists: " + run_id);
	}
	std::filesystem::create_directories(data_dir);
	std::filesystem::create_directories(model_dir);
	const auto initial = model_dir / "initial.pth";
	const auto current = model_dir / "current.pth";
	atomic_copy(options.model, initial);
	atomic_copy(initial, current);
	const auto device = resolve_device(options.device);
	std::cout << "fcpi run id: " << run_id << std::endl;
	std::cout << "fcpi architecture: " << kArchType << std::endl;
	std::cout << "fcpi formula: " << kFcpiFormula << std::endl;
	std::cout << "fcpi current model: " << current.string() << std::endl;
	nlohmann::json summaries = nlohmann::json::array();

	for (int iteration = 1; iteration <= options.iterations; ++iteration) {
		std::cout << "fcpi iteration " << iteration << std::endl;
		CheckpointInfo current_info;
		auto model = load_checkpoint(current, device, &current_info);
		nlohmann::json sampling;
		auto records = collect_selfplay(model, device, options, iteration, sampling);
		TargetSummary target_summary;
		construct_targets(records, model, device, options, target_summary);
		const auto data_path = data_dir / ("fcpi_iter_" + zero_pad(iteration, 3) + ".h5");
		auto data_summary = write_fcpi_h5(data_path, records);
		data_summary["sampling"] = sampling;
		data_summary["counterfactual"] = {
			{"branches", target_summary.branches},
			{"average_depth",
			 target_summary.branches > 0
				 ? static_cast<double>(target_summary.branch_plies) / target_summary.branches
				 : 0.0},
			{"terminal_branches", target_summary.terminal_branches},
		};
		const auto candidate = model_dir / ("candidate_iter_" + zero_pad(iteration, 3) + ".pth");
		auto train_summary = train_candidate(current, candidate, model, device, records, options);
		auto arena_options = options.arena;
		arena_options.candidate = candidate;
		arena_options.baseline = current;
		arena_options.device = options.device;
		arena_options.seed = options.seed + iteration;
		auto arena_summary = evaluate_models(arena_options);
		const bool accepted = arena_summary["accepted"].get<bool>();
		if (accepted) {
			atomic_copy(candidate, current);
			std::cout << "fcpi promoted: " << current.string() << std::endl;
		} else {
			std::cout << "fcpi candidate rejected: " << candidate.string() << std::endl;
		}
		summaries.push_back({
			{"iteration", iteration},
			{"architecture", kArchType},
			{"formula", kFcpiFormula},
			{"data", data_summary},
			{"train", train_summary},
			{"arena", arena_summary},
			{"accepted", accepted},
		});
		std::ofstream summary_file(data_dir / "summary.json");
		summary_file << nlohmann::json({
										   {"run_id", run_id},
										   {"initial_model", initial.string()},
										   {"current_model", current.string()},
										   {"summaries", summaries},
									   })
							.dump(2);
	}
}

} // namespace gadus
