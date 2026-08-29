#pragma once

// Geometry-biased attention expressed as batched matrix operations.

#include <torch/types.h>

namespace melano {

/// Computes softmax(QK^T / sqrt(d) + sum_r coefficients_r templates_r)V.
torch::Tensor geometry_attention(
    torch::Tensor query, torch::Tensor key, torch::Tensor value, torch::Tensor coefficients, torch::Tensor templates);

/// Removes the key-axis mean from every GAB template row in place.
void center_gab_rows(torch::Tensor templates);

} // namespace melano
