import bisect

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from chess_env import packed_to_tensor


class H5ChessDataset(Dataset):
    """Supervised HDF5: states / moves / values."""

    def __init__(self, path):
        self.path = path
        self._file = None
        with h5py.File(path, "r") as h5:
            self.length = int(h5["states"].shape[0])
            self.has_policy = "policy" in h5
            self.has_moves = "moves" in h5

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_file"] = None
        return state

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        h5 = self._open()
        state = torch.from_numpy(packed_to_tensor(h5["states"][idx]))
        value = torch.tensor(float(h5["values"][idx]), dtype=torch.float32)
        if "policy" in h5:
            policy = torch.from_numpy(np.asarray(h5["policy"][idx], dtype=np.float32))
            return state, policy, value
        move = torch.tensor(int(h5["moves"][idx]), dtype=torch.long)
        return state, move, value


class MultiSelfLearnDataset(Dataset):
    """Concatenate teacher-constrained self-learning HDF5 files."""

    REQUIRED = (
        "states",
        "target_policy",
        "terminal_values",
        "terminal_valid",
        "teacher_values",
        "teacher_weights",
        "regret_cp",
    )

    def __init__(self, paths):
        self.paths = [str(path) for path in paths]
        if not self.paths:
            raise ValueError("empty self-learning replay list")
        self._files = [None] * len(self.paths)
        self.lengths = []
        for path in self.paths:
            with h5py.File(path, "r") as h5:
                missing = [key for key in self.REQUIRED if key not in h5]
                if missing:
                    raise ValueError(f"{path} missing datasets: {missing}")
                self.lengths.append(int(h5["states"].shape[0]))

        self.cumulative = []
        total = 0
        for length in self.lengths:
            total += length
            self.cumulative.append(total)
        self.length = total

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_files"] = [None] * len(self.paths)
        return state

    def _open(self, index):
        if self._files[index] is None:
            self._files[index] = h5py.File(self.paths[index], "r")
        return self._files[index]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        file_index = bisect.bisect_right(self.cumulative, idx)
        previous = 0 if file_index == 0 else self.cumulative[file_index - 1]
        local_index = idx - previous
        h5 = self._open(file_index)
        return (
            torch.from_numpy(packed_to_tensor(h5["states"][local_index])),
            torch.from_numpy(np.asarray(h5["target_policy"][local_index], dtype=np.float32)),
            torch.tensor(float(h5["terminal_values"][local_index]), dtype=torch.float32),
            torch.tensor(float(h5["terminal_valid"][local_index]), dtype=torch.float32),
            torch.tensor(float(h5["teacher_values"][local_index]), dtype=torch.float32),
            torch.tensor(float(h5["teacher_weights"][local_index]), dtype=torch.float32),
            torch.tensor(float(h5["regret_cp"][local_index]), dtype=torch.float32),
        )
