import numpy as np
import torch

from decision import inference_profile_for_model


class EvaluationBatch:
    def __init__(self, policies, values, expansion_payload=None):
        self.policies = policies
        self.values = values
        self.expansion_payload = expansion_payload


class BatchedEvaluator:
    """Batch neural policy/value inference for MCTS leaf positions."""

    def __init__(self, model, device="cuda", batch_size=32):
        self.model = model
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.inference = inference_profile_for_model(model)
        self.state_codec = self.inference.state_codec

    @torch.no_grad()
    def evaluate_boards_full(self, boards):
        if not boards:
            action_size = self.inference.move_codec.action_size
            return EvaluationBatch(
                np.zeros((0, action_size), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                None,
            )

        self.model.eval()
        policies = []
        values = []
        payloads = []
        for start in range(0, len(boards), self.batch_size):
            batch = boards[start:start + self.batch_size]
            x = np.stack([self.state_codec.tensor_from_board(board) for board in batch])
            tensor = torch.from_numpy(x).to(self.device, non_blocking=True)
            output = self.inference.evaluate_tensor(self.model, tensor)
            policies.append(torch.softmax(output.policy_logits, dim=1).cpu().numpy())
            values.append(output.value.squeeze(1).cpu().numpy())
            payload = self.inference.output_payload_to_numpy(output)
            if payload is not None:
                payloads.append(payload)

        expansion_payload = self.inference.merge_payloads(payloads)
        return EvaluationBatch(
            np.concatenate(policies, axis=0).astype(np.float32, copy=False),
            np.concatenate(values, axis=0).astype(np.float32, copy=False),
            expansion_payload,
        )

    def evaluate_boards(self, boards):
        evaluation = self.evaluate_boards_full(boards)
        return evaluation.policies, evaluation.values

    def evaluate_one_full(self, board):
        evaluation = self.evaluate_boards_full([board])
        payload = self.inference.payload_for_index(evaluation.expansion_payload, 0)
        return evaluation.policies[0], float(evaluation.values[0]), payload

    def evaluate_one(self, board):
        policy, value, _ = self.evaluate_one_full(board)
        return policy, value

    def legal_priors(self, board):
        policy, value = self.evaluate_one(board)
        return self.inference.move_codec.policy_to_legal_distribution(
            policy,
            board,
            normalize=True,
        ), value
