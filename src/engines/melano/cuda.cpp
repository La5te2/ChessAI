// Implements fixed-shape CUDA Graph replay while preserving eager CPU behavior.

#include "melano/cuda.hpp"
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

#ifdef GADIDAE_MELANO_CUDA
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#endif

namespace melano {
namespace {

// Numerically stable binary cross-entropy evaluated directly from logits.
torch::Tensor bce_logits(const torch::Tensor &logits, const torch::Tensor &targets) {
	const auto score_logits = logits.squeeze(1).to(torch::kFloat32) * 2.0;
	const auto probabilities = ((targets.to(torch::kFloat32) + 1.0) * 0.5).clamp(0.0, 1.0);
	return torch::nn::functional::binary_cross_entropy_with_logits(score_logits, probabilities);
}

TrainingStep eager_training_step(Model &model, torch::optim::AdamW &optimizer, const torch::Tensor &states, const torch::Tensor &moves,
                                 const torch::Tensor &values, const torch::Device &device, ComputePrecision precision, double value_weight) {
	optimizer.zero_grad();
	const auto device_states = states.to(device, true);
	const auto device_moves = moves.to(device, true).to(torch::kInt64);
	const auto device_values = values.to(device, true);
	torch::Tensor policy;
	torch::Tensor predicted_value;
	{
		AutocastGuard autocast(precision, device);
		std::tie(policy, predicted_value) = model->forward_training(device_states);
	}
	auto policy_loss = torch::nn::functional::cross_entropy(policy.to(torch::kFloat32), device_moves);
	auto value_bce = bce_logits(predicted_value, device_values);
	auto loss = policy_loss + value_weight * value_bce;
	loss.backward();
	return {policy_loss, value_bce, loss};
}

std::tuple<torch::Tensor, torch::Tensor> eager_inference(Model &model, const torch::Tensor &states, const torch::Tensor &legal_indices,
                                                        const torch::Device &device, ComputePrecision precision) {
	const auto device_states = states.to(device, true);
	const auto device_indices = legal_indices.to(device, true);
	AutocastGuard autocast(precision, device);
	auto [logits, value] = model->forward_legal(device_states, device_indices.clamp_min(0));
	auto probabilities = torch::softmax(logits.to(torch::kFloat32).masked_fill(device_indices < 0, -std::numeric_limits<float>::infinity()), 1);
	return {probabilities, value};
}

#ifdef GADIDAE_MELANO_CUDA

int cuda_device_index(const torch::Device &device) {
	return device.index() >= 0 ? device.index() : c10::cuda::current_device();
}

struct TrainingEntry {
	at::cuda::CUDAGraph graph;
	torch::Tensor states;
	torch::Tensor moves;
	torch::Tensor values;
	TrainingStep result;
};

struct InferenceEntry {
	at::cuda::CUDAGraph graph;
	torch::Tensor states;
	torch::Tensor legal_indices;
	torch::Tensor policy;
	torch::Tensor value;
};

void warmup_training(Model &model, torch::optim::AdamW &optimizer, TrainingEntry &entry, ComputePrecision precision, double value_weight,
                     const c10::cuda::CUDAStream &stream) {
	c10::cuda::CUDAStreamGuard stream_guard(stream);
	for (int iteration = 0; iteration < 3; ++iteration) {
		optimizer.zero_grad(false);
		torch::Tensor policy;
		torch::Tensor predicted_value;
		{
			AutocastGuard autocast(precision, entry.states.device());
			std::tie(policy, predicted_value) = model->forward_training(entry.states);
		}
		auto policy_loss = torch::nn::functional::cross_entropy(policy.to(torch::kFloat32), entry.moves.to(torch::kInt64));
		auto value_bce = bce_logits(predicted_value, entry.values);
		(policy_loss + value_weight * value_bce).backward();
	}
}

void capture_training(Model &model, torch::optim::AdamW &optimizer, TrainingEntry &entry, ComputePrecision precision, double value_weight,
                      const c10::cuda::CUDAStream &stream) {
	c10::cuda::CUDAStreamGuard stream_guard(stream);
	entry.graph.capture_begin();
	optimizer.zero_grad(false);
	torch::Tensor policy;
	torch::Tensor predicted_value;
	{
		AutocastGuard autocast(precision, entry.states.device());
		std::tie(policy, predicted_value) = model->forward_training(entry.states);
	}
	entry.result.policy_loss = torch::nn::functional::cross_entropy(policy.to(torch::kFloat32), entry.moves.to(torch::kInt64));
	entry.result.value_bce = bce_logits(predicted_value, entry.values);
	entry.result.loss = entry.result.policy_loss + value_weight * entry.result.value_bce;
	entry.result.loss.backward();
	entry.graph.capture_end();
}

void warmup_inference(Model &model, InferenceEntry &entry, ComputePrecision precision, const c10::cuda::CUDAStream &stream) {
	c10::cuda::CUDAStreamGuard stream_guard(stream);
	torch::InferenceMode guard;
	for (int iteration = 0; iteration < 3; ++iteration) {
		AutocastGuard autocast(precision, entry.states.device());
		auto [logits, value] = model->forward_legal(entry.states, entry.legal_indices.clamp_min(0));
		entry.policy = torch::softmax(logits.to(torch::kFloat32).masked_fill(entry.legal_indices < 0, -std::numeric_limits<float>::infinity()), 1);
		entry.value = value;
	}
}

void capture_inference(Model &model, InferenceEntry &entry, ComputePrecision precision, const c10::cuda::CUDAStream &stream) {
	c10::cuda::CUDAStreamGuard stream_guard(stream);
	torch::InferenceMode guard;
	entry.graph.capture_begin();
	{
		AutocastGuard autocast(precision, entry.states.device());
		auto [logits, value] = model->forward_legal(entry.states, entry.legal_indices.clamp_min(0));
		entry.policy = torch::softmax(logits.to(torch::kFloat32).masked_fill(entry.legal_indices < 0, -std::numeric_limits<float>::infinity()), 1);
		entry.value = value;
	}
	entry.graph.capture_end();
}

#endif

} // namespace

struct TrainingGraph::Impl {
#ifdef GADIDAE_MELANO_CUDA
	std::unique_ptr<TrainingEntry> entry;
#endif
};

TrainingGraph::TrainingGraph() : impl_(std::make_unique<Impl>()) {}
TrainingGraph::~TrainingGraph() = default;

TrainingStep TrainingGraph::run(Model &model, torch::optim::AdamW &optimizer, torch::Tensor states, torch::Tensor moves, torch::Tensor values,
                                const torch::Device &device, ComputePrecision precision, double value_weight, bool fixed_shape) {
#ifdef GADIDAE_MELANO_CUDA
	if (fixed_shape && device.is_cuda()) {
		if (!impl_->entry) {
			auto entry = std::make_unique<TrainingEntry>();
			entry->states = torch::empty(states.sizes(), torch::TensorOptions().dtype(states.scalar_type()).device(device));
			entry->moves = torch::empty(moves.sizes(), torch::TensorOptions().dtype(moves.scalar_type()).device(device));
			entry->values = torch::empty(values.sizes(), torch::TensorOptions().dtype(values.scalar_type()).device(device));
			entry->states.copy_(states, true);
			entry->moves.copy_(moves, true);
			entry->values.copy_(values, true);
			c10::cuda::device_synchronize();
			const auto stream = c10::cuda::getStreamFromPool(false, cuda_device_index(device));
			warmup_training(model, optimizer, *entry, precision, value_weight, stream);
			stream.synchronize();
			capture_training(model, optimizer, *entry, precision, value_weight, stream);
			stream.synchronize();
			entry->graph.replay();
			impl_->entry = std::move(entry);
		} else {
			if (impl_->entry->states.sizes() != states.sizes() || impl_->entry->moves.sizes() != moves.sizes() || impl_->entry->values.sizes() != values.sizes()) {
				throw std::runtime_error("Melano training CUDA Graph received a different fixed batch shape");
			}
			impl_->entry->states.copy_(states, true);
			impl_->entry->moves.copy_(moves, true);
			impl_->entry->values.copy_(values, true);
			impl_->entry->graph.replay();
		}
		return impl_->entry->result;
	}
#endif
	(void)fixed_shape;
	return eager_training_step(model, optimizer, states, moves, values, device, precision, value_weight);
}

struct InferenceGraphs::Impl {
#ifdef GADIDAE_MELANO_CUDA
	using Shape = std::pair<std::pair<std::int64_t, std::int64_t>, ComputePrecision>;
	std::map<Shape, std::unique_ptr<InferenceEntry>> entries;
	std::map<Shape, int> observations;
#endif
};

InferenceGraphs::InferenceGraphs() : impl_(std::make_unique<Impl>()) {}
InferenceGraphs::~InferenceGraphs() = default;

std::tuple<torch::Tensor, torch::Tensor> InferenceGraphs::run(
    Model &model, torch::Tensor states, torch::Tensor legal_indices, const torch::Device &device, ComputePrecision precision) {
#ifdef GADIDAE_MELANO_CUDA
	if (device.is_cuda()) {
		const Impl::Shape shape{{states.size(0), legal_indices.size(1)}, precision};
		auto found = impl_->entries.find(shape);
		if (found == impl_->entries.end()) {
			if (++impl_->observations[shape] < 2) {
				return eager_inference(model, states, legal_indices, device, precision);
			}
			auto entry = std::make_unique<InferenceEntry>();
			entry->states = torch::empty(states.sizes(), torch::TensorOptions().dtype(states.scalar_type()).device(device));
			entry->legal_indices = torch::empty(legal_indices.sizes(), torch::TensorOptions().dtype(legal_indices.scalar_type()).device(device));
			entry->states.copy_(states, true);
			entry->legal_indices.copy_(legal_indices, true);
			c10::cuda::device_synchronize();
			const auto stream = c10::cuda::getStreamFromPool(false, cuda_device_index(device));
			warmup_inference(model, *entry, precision, stream);
			stream.synchronize();
			capture_inference(model, *entry, precision, stream);
			stream.synchronize();
			found = impl_->entries.emplace(shape, std::move(entry)).first;
		} else {
			found->second->states.copy_(states, true);
			found->second->legal_indices.copy_(legal_indices, true);
			found->second->graph.replay();
		}
		return {found->second->policy, found->second->value};
	}
#endif
	return eager_inference(model, states, legal_indices, device, precision);
}

} // namespace melano
