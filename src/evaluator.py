import numpy as np
import torch

from chess_env import board_to_tensor
from move_encoder import policy_to_legal_distribution


class BatchedEvaluator:
    """Batch neural policy/value inference for MCTS leaf positions."""

    def __init__(self, model, device="cuda", batch_size=32):
        self.model = model
        self.device = device
        self.batch_size = max(1, int(batch_size))

    @torch.no_grad()
    def evaluate_boards(self, boards):
        if not boards:
            action_size = getattr(self.model, "action_size", 4672)
            return (
                np.zeros((0, action_size), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        self.model.eval()
        policies = []
        values = []
        for start in range(0, len(boards), self.batch_size):
            batch = boards[start:start + self.batch_size]
            x = np.stack([board_to_tensor(board) for board in batch]).astype(np.float32)
            tensor = torch.from_numpy(x).to(self.device, non_blocking=True)
            logits, value = self.model(tensor)
            policies.append(torch.softmax(logits, dim=1).cpu().numpy())
            values.append(value.squeeze(1).cpu().numpy())

        return (
            np.concatenate(policies, axis=0).astype(np.float32, copy=False),
            np.concatenate(values, axis=0).astype(np.float32, copy=False),
        )

    def evaluate_one(self, board):
        policies, values = self.evaluate_boards([board])
        return policies[0], float(values[0])

    def legal_priors(self, board):
        policy, value = self.evaluate_one(board)
        return policy_to_legal_distribution(policy, board, normalize=True), value
