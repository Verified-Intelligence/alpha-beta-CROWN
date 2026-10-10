#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2026 The α,β-CROWN Team                        ##
##   Team leaders:                                                     ##
##          Faculty:   Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##          Student:   Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##   See CONTRIBUTORS for all current and past developers in the team. ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENSE file in the repository root.       ##
##                                                                     ##
#########################################################################
"""
Record domains
"""

from dataclasses import dataclass
from typing import List
from hashlib import sha256
import pathlib
import pickle


@dataclass(slots=True, frozen=True, order=True)
class ActivationSplit:
    """A single ReLU activation split used in branch-and-bound.

    Attributes:
        layer_name: Name of the layer where the split occurs.
        idx: Index into the unstable activations of that layer.
        status: Split direction; -1 for negative (inactive), 1 for positive (active).
    """

    layer_name: str
    idx: int
    status: int

    @staticmethod
    def from_tuple_of_tensor(layer_name, tup: tuple) -> "List[ActivationSplit]":

        idx_arr = tup[0]
        status_arr = tup[1]
        return [
            ActivationSplit(layer_name, int(idx_arr[i]), int(status_arr[i]))
            for i in range(len(idx_arr))
        ]

    def to_tuple(self) -> tuple:
        return (str(self.layer_name), int(self.idx), int(self.status))


@dataclass(slots=True, frozen=True, order=True)
class DomainRecord:
    """A single BaB domain defined by a sequence of ReLU activation splits.

    Attributes:
        depth: Depth of this domain in the search tree.
        activation_splits: Tuple of ActivationSplit decisions that define this domain.
    """
    depth: int
    activation_splits: tuple[ActivationSplit, ...]

    @staticmethod
    def from_d(d_hist: dict, d_depths: list) -> "List[DomainRecord]":

        domain_records = []

        for i, hist in enumerate(d_hist):
            act_splits = []
            for layer_name, hist_tuple in hist.items():
                act_splits.extend(
                    ActivationSplit.from_tuple_of_tensor(layer_name, hist_tuple)
                )
            record = DomainRecord(d_depths[i], tuple(act_splits))
            domain_records.append(record)

        return domain_records

    def to_tuple(self) -> tuple:
        t = (
            int(self.depth),
            tuple(act_split.to_tuple() for act_split in self.activation_splits),
        )
        return t

    def sha256_hash(self) -> str:
        hasher = sha256()
        self_tuple = self.to_tuple()
        hasher.update(pickle.dumps(self_tuple))
        return hasher.hexdigest()


@dataclass
class DomainDB:
    """A simple database for storing, serializing, and comparing BaB domains.

    Attributes:
        records: List of DomainRecord entries.
        hash_values: List of SHA-256 hashes corresponding to each record.
    """
    records: List[DomainRecord]
    hash_values: List[str]

    @staticmethod
    def new_db() -> "DomainDB":
        return DomainDB([], [])

    def add_records(self, records: List[DomainRecord]) -> None:
        self.records.extend(records)
        for record in records:
            self.hash_values.append(record.sha256_hash())

    def add_records_from_d(self, d_hist: dict, d_depths: list) -> None:
        records = DomainRecord.from_d(d_hist, d_depths)
        self.add_records(records)

    def to_dict(self) -> dict:
        return {
            "records": [record.to_tuple() for record in self.records],
            "hash_values": self.hash_values,
        }

    @staticmethod
    def from_dict(d: dict) -> "DomainDB":
        records: list[DomainRecord] = []
        for record_tuple in d["records"]:
            depth = record_tuple[0]
            act_splits = tuple(
                ActivationSplit(
                    layer_name=act_split_tuple[0],
                    idx=act_split_tuple[1],
                    status=act_split_tuple[2],
                )
                for act_split_tuple in record_tuple[1]
            )
            record = DomainRecord(depth=depth, activation_splits=act_splits)
            records.append(record)
        hash_values: list[str] = d["hash_values"]
        return DomainDB(records=records, hash_values=hash_values)

    def dump(self, file_path: str, batch_idx: int | None = None) -> None:
        if batch_idx is not None:
            p = pathlib.Path(file_path)
            basename = p.stem
            suffix = p.suffix
            dirpath = p.parent
            file_path_path = dirpath / f"{basename}_batch{batch_idx}{suffix}"
            file_path_path.parent.mkdir(parents=True, exist_ok=True)
            file_path = str(file_path_path)

        with open(file_path, "wb") as f:
            pickle.dump(self.to_dict(), f)

    def load(self, file_path: str) -> None:
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
            if isinstance(obj, DomainDB):
                self.records = obj.records
                self.hash_values = obj.hash_values
            elif isinstance(obj, dict):
                db = DomainDB.from_dict(obj)
                self.records = db.records
                self.hash_values = db.hash_values
            else:
                raise ValueError("Invalid data format in the file.")

    def merge(self, other: "DomainDB") -> None:
        if self is other:
            return
        self.records.extend(other.records)
        self.hash_values.extend(other.hash_values)

    def compare_with(self, other: "DomainDB"):
        hash_values1 = set(self.hash_values)
        hash_values2 = set(other.hash_values)
        
        print(f"DB1 size: {len(hash_values1)}")
        print(f"DB2 size: {len(hash_values2)}")

        common_hash_values = hash_values1.intersection(hash_values2)
        unique_hash_values1 = hash_values1 - common_hash_values
        unique_hash_values2 = hash_values2 - common_hash_values

        print(f"Common hash values: {len(common_hash_values)}")
        print(f"Unique hash values in db1: {len(unique_hash_values1)}")
        print(f"Unique hash values in db2: {len(unique_hash_values2)}")
        print(
            f"diff rate: {max(len(unique_hash_values1), len(unique_hash_values2)) / len(hash_values1):%}"
        )

    def diff(self, other: "DomainDB") -> List[DomainRecord]:
        hash_values_other = set(other.hash_values)
        diff_records = [
            record
            for record in self.records
            if record.sha256_hash() not in hash_values_other
        ]
        return diff_records