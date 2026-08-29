// Implements geometry-biased attention with batched matrix operations.

#include "melano/attention.hpp"
#include <ATen/ops/scaled_dot_product_attention.h>
#include <cmath>
#include <stdexcept>

namespace melano {

void center_gab_rows(torch::Tensor templates) {
	if (templates.dim() != 3) {
		throw std::invalid_argument("GAB templates must have [relations, queries, keys] shape");
	}
	torch::NoGradGuard guard;
	templates.sub_(templates.mean(-1, true));
}

torch::Tensor geometry_attention(
    torch::Tensor query, torch::Tensor key, torch::Tensor value, torch::Tensor coefficients, torch::Tensor templates) {
	if (query.dim() != 4 || key.sizes() != query.sizes() || value.sizes() != query.sizes()) {
		throw std::invalid_argument("query, key, and value must have equal [batch, heads, tokens, channels] shapes");
	}
	if (coefficients.dim() != 3 || coefficients.size(0) != query.size(0) || coefficients.size(1) != query.size(1) ||
	    templates.dim() != 3 || templates.size(0) != coefficients.size(2) || templates.size(1) != query.size(2) ||
	    templates.size(2) != query.size(2)) {
		throw std::invalid_argument("geometry coefficients and templates do not match the attention shape");
	}

	const auto tokens = query.size(2);
	const auto channels = query.size(3);
	auto bias = torch::matmul(coefficients, templates.reshape({templates.size(0), tokens * tokens}));
	bias = bias.view({query.size(0), query.size(1), tokens, tokens});
	return at::scaled_dot_product_attention(query, key, value, bias, 0.0, false, 1.0 / std::sqrt(static_cast<double>(channels)), false);
}

} // namespace melano
