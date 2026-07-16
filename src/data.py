import h5py
import torch
from torch.utils.data import Dataset

from architectures import (
    RESNET_PVA_GAD,
    RESNET_PV_LINEAR,
    architecture_spec,
    normalize_arch_type,
)
from state_codecs import get_state_codec


def _decode_attr(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if hasattr(value, "decode"):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def supervised_h5_schema(arch_type: str):
    spec = architecture_spec(arch_type)
    return {
        "datasets": spec.supervised_datasets,
        "state_encoding": spec.state_encoding,
        "target_schema": spec.target_schema,
        "move_encoding": spec.move_encoding,
    }


def validate_supervised_h5(path: str, arch_type: str):
    arch_type = normalize_arch_type(arch_type)
    schema = supervised_h5_schema(arch_type)
    expected_datasets = set(schema["datasets"])
    expected_state_encoding = str(schema["state_encoding"])
    expected_move_encoding = str(schema["move_encoding"])
    expected_target_schema = str(schema["target_schema"])

    with h5py.File(path, "r") as h5:
        for attr in ("arch_type", "state_encoding", "move_encoding", "target_schema"):
            if attr not in h5.attrs:
                raise ValueError(f"{path} missing required attr {attr!r}")

        actual_arch_type = normalize_arch_type(_decode_attr(h5.attrs["arch_type"]))
        actual_state_encoding = _decode_attr(h5.attrs["state_encoding"])
        actual_move_encoding = _decode_attr(h5.attrs["move_encoding"])
        actual_target_schema = _decode_attr(h5.attrs["target_schema"])
        if actual_arch_type != arch_type:
            raise ValueError(
                f"{path} arch_type={actual_arch_type!r}, expected {arch_type!r}"
            )
        if actual_state_encoding != expected_state_encoding:
            raise ValueError(
                f"{path} state_encoding={actual_state_encoding!r}, "
                f"expected {expected_state_encoding!r}"
            )
        if actual_move_encoding != expected_move_encoding:
            raise ValueError(
                f"{path} move_encoding={actual_move_encoding!r}, "
                f"expected {expected_move_encoding!r}"
            )
        if actual_target_schema != expected_target_schema:
            raise ValueError(
                f"{path} target_schema={actual_target_schema!r}, "
                f"expected {expected_target_schema!r}"
            )

        actual_datasets = set(str(name) for name in h5.keys())
        missing = sorted(expected_datasets - actual_datasets)
        unexpected = sorted(actual_datasets - expected_datasets)
        if missing or unexpected:
            raise ValueError(
                f"{path} datasets mismatch for {arch_type}: "
                f"missing={missing}, unexpected={unexpected}"
            )

        lengths = {name: int(h5[name].shape[0]) for name in schema["datasets"]}
        length = next(iter(lengths.values()))
        mismatched = {
            name: value
            for name, value in lengths.items()
            if value != length
        }
        if mismatched:
            raise ValueError(
                f"{path} dataset lengths mismatch: expected {length}, got {mismatched}"
            )
        if length <= 0:
            raise ValueError(f"{path} is empty")

    return {
        "arch_type": arch_type,
        "state_encoding": expected_state_encoding,
        "move_encoding": expected_move_encoding,
        "target_schema": expected_target_schema,
        "length": length,
        "datasets": tuple(schema["datasets"]),
    }


def read_resnet_pv_linear_row(h5, idx, state_codec):
    return (
        torch.from_numpy(state_codec.decode_state(h5["states"][idx])),
        torch.tensor(int(h5["moves"][idx]), dtype=torch.long),
        torch.tensor(float(h5["values"][idx]), dtype=torch.float32),
    )


def read_resnet_pva_gad_row(h5, idx, state_codec):
    return (
        torch.from_numpy(state_codec.decode_state(h5["states"][idx])),
        torch.tensor(int(h5["moves"][idx]), dtype=torch.long),
        torch.tensor(float(h5["values"][idx]), dtype=torch.float32),
        torch.tensor(int(h5["adv_moves"][idx]), dtype=torch.long),
        torch.tensor(float(h5["adv_values"][idx]), dtype=torch.float32),
        torch.tensor(float(h5["adv_weights"][idx]), dtype=torch.float32),
    )


SUPERVISED_ROW_READERS = {
    RESNET_PV_LINEAR: read_resnet_pv_linear_row,
    RESNET_PVA_GAD: read_resnet_pva_gad_row,
}


class H5ChessDataset(Dataset):
    """Architecture-specific supervised HDF5 dataset."""

    def __init__(self, path, arch_type):
        self.path = path
        metadata = validate_supervised_h5(path, arch_type)
        self.arch_type = metadata["arch_type"]
        self.state_encoding = metadata["state_encoding"]
        self.target_schema = metadata["target_schema"]
        self.move_encoding = metadata["move_encoding"]
        self.datasets = metadata["datasets"]
        self.length = int(metadata["length"])
        self.state_codec = get_state_codec(self.state_encoding)
        self.row_reader = SUPERVISED_ROW_READERS[self.arch_type]
        self._file = None

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
        return self.row_reader(h5, idx, self.state_codec)
