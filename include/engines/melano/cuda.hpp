#pragma once

// Optional fixed-shape CUDA Graph execution for Melano training and inference.

#include "melano/model.hpp"
#include "melano/precision.hpp"
#include <memory>
#include <torch/optim.h>

namespace melano {

struct TrainingStep {
	torch::Tensor policy_loss;
	torch::Tensor value_bce;
	torch::Tensor loss;
};

class TrainingGraph {
public:
	TrainingGraph();
	~TrainingGraph();
	TrainingGraph(const TrainingGraph &) = delete;
	TrainingGraph &operator=(const TrainingGraph &) = delete;

	/// Runs one backward pass and captures repeated full-size CUDA batches after warmup.
	TrainingStep run(Model &model, torch::optim::AdamW &optimizer, torch::Tensor states, torch::Tensor moves, torch::Tensor values,
	                 const torch::Device &device, ComputePrecision precision, double value_weight, bool fixed_shape);

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

class InferenceGraphs {
public:
	InferenceGraphs();
	~InferenceGraphs();
	InferenceGraphs(const InferenceGraphs &) = delete;
	InferenceGraphs &operator=(const InferenceGraphs &) = delete;

	/// Reuses one CUDA Graph for each encountered state/legal-index tensor shape.
	std::tuple<torch::Tensor, torch::Tensor> run(
	    Model &model, torch::Tensor states, torch::Tensor legal_indices, const torch::Device &device, ComputePrecision precision);

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

} // namespace melano
