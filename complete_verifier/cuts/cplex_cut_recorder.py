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
"""Utilities for recording and replaying CPLEX cuts."""
import os
import time
import pickle
import csv


class CplexCutRecorder:
    """Manages recording and replaying of CPLEX cuts."""

    def __init__(self, batch_id, record_csv_basepath=None, replay_csv_basepath=None):
        """
        :param batch_id: Batch identifier for organizing cut records.
        :param record_csv_basepath: Path to CSV file for recording cuts. The batch ID will be automatically appended to the filename. If None, recording is disabled.
        :param replay_csv_basepath: Path to CSV file for replaying cuts. The batch ID will be automatically appended to the filename. If None, replay is disabled.
        """
        # Assert that record and replay modes cannot both be enabled
        assert not (
            record_csv_basepath is not None and replay_csv_basepath is not None
        ), "record_cplex_cut_to and read_cplex_cut_from cannot both be enabled at the same time"

        self.batch_id = batch_id
        self.record_call_count = 0
        self.replay_call_count = 0
        self.replay_data = None
        self.saved_var_names = False
        if record_csv_basepath is not None:
            base, ext = os.path.splitext(record_csv_basepath)
            self.record_csv_path = f"{base}_batch_{batch_id}{ext}"

            csv_dir = os.path.dirname(self.record_csv_path)
            if csv_dir:
                os.makedirs(csv_dir, exist_ok=True)

            with open(record_csv_basepath, "w") as f:
                f.write(f"This is a dummy file. Please look at other files {base}_batch_*.csv for the actual data.")

            csv_basename = os.path.splitext(os.path.basename(self.record_csv_path))[0]
            self.pickle_dir = os.path.join(csv_dir, csv_basename)
            os.makedirs(self.pickle_dir, exist_ok=True)

            self.csv_path = self.record_csv_path

            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "number_of_call",
                        "recorded_timestamp",
                        "path_to_cuts",
                    ]
                )
        else:
            self.record_csv_path = None

        if replay_csv_basepath is not None:

            base, ext = os.path.splitext(replay_csv_basepath)
            self.replay_csv_path = f"{base}_batch_{batch_id}{ext}"

            csv_dir = os.path.dirname(self.replay_csv_path)
            csv_basename = os.path.splitext(os.path.basename(self.replay_csv_path))[0]
            self.pickle_dir = os.path.join(csv_dir, csv_basename)
            if self.pickle_dir:
                os.makedirs(self.pickle_dir, exist_ok=True)
            self._load_replay_data()
        else:
            self.replay_csv_path = None

    def _load_replay_data(self):
        """Load replay data from CSV file."""
        assert self.replay_csv_path is not None, "Replay CSV path is not set"
        csv_path = self.replay_csv_path
        if not os.path.exists(csv_path):
            print(
                f"Warning: Replay CSV file not found at {csv_path}. Replay will return None, -1 for all calls."
            )
            self.replay_data = []
            return

        self.replay_data = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.replay_data.append(
                    {
                        "number_of_call": int(row["number_of_call"]),
                        "recorded_timestamp": float(row["recorded_timestamp"]),
                        "path_to_cuts": row["path_to_cuts"],
                    }
                )

        print(f"Loaded {len(self.replay_data)} cut records for replay from {csv_path}")

    def record_cplex_cut_to(self, cuts, cut_timestamp):
        """
        Record cuts and metadata to files.

        :param cuts: The cuts data structure (can be None)
        :param cut_timestamp: The cut timestamp (can be -1)
        :return: None
        """
        if self.record_csv_path is None:
            return

        recorded_timestamp = time.time()

        # Save cuts as pickle in the pickle directory
        cuts_filename = f"cuts_batch_{self.batch_id}_{self.record_call_count:06d}.pkl"
        cuts_path = os.path.join(self.pickle_dir, cuts_filename)

        with open(cuts_path, "wb") as f:
            pickle.dump((cuts, cut_timestamp), f)

        # Save relative path in CSV (relative to pickle directory)
        cuts_path_for_csv = os.path.join(
            os.path.basename(self.pickle_dir), cuts_filename
        )

        # Append to CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    self.record_call_count,
                    recorded_timestamp,
                    cuts_path_for_csv,
                ]
            )

        self.record_call_count += 1
        print(f"Recorded cut call {self.record_call_count - 1} to {cuts_path}")

    def read_cplex_cut_from(self):
        """
        Read cuts from recorded data for replay.

        :return: (cuts, cut_timestamp) tuple, or (None, -1) if no more records.
                 Requires replay to be enabled (replay_csv_path must be set).
        """
        assert self.replay_csv_path is not None, "Replay CSV path is not set"
        if self.replay_data is None:
            self._load_replay_data()

        assert self.replay_data is not None, "Replay data is not loaded"

        # Check if we have more records
        if self.replay_call_count >= len(self.replay_data):
            print(
                f"Replay: No more recorded cuts (call {self.replay_call_count} >= {len(self.replay_data)} records)"
            )
            return None, -1

        record = self.replay_data[self.replay_call_count]

        # Path in CSV is relative to pickle directory, resolve it relative to CSV directory
        cuts_path = record["path_to_cuts"]
        csv_dir = os.path.dirname(self.replay_csv_path)
        cuts_path = os.path.join(csv_dir, cuts_path)

        with open(cuts_path, "rb") as f:
            cuts, cut_timestamp = pickle.load(f)

        print(f"Replay: Loaded cut call {self.replay_call_count} from {cuts_path}")
        self.replay_call_count += 1
        return cuts, cut_timestamp

    def save_var_names(self, var_names):
        """
        Save variable names to a file for cplex cuts.
        """
        assert self.record_csv_path is not None, "Record CSV path is not set"
        if self.saved_var_names:
            return
        self.saved_var_names = True
        var_names_path = os.path.join(
            self.pickle_dir, f"var_names_batch_{self.batch_id}.pkl"
        )
        with open(var_names_path, "wb") as f:
            pickle.dump(var_names, f)

    def load_var_names(self):
        """
        Load variable names from a file for cplex cuts.
        """
        assert self.replay_csv_path is not None, "Replay CSV path is not set"
        var_names_path = os.path.join(
            self.pickle_dir, f"var_names_batch_{self.batch_id}.pkl"
        )
        with open(var_names_path, "rb") as f:
            var_names = pickle.load(f)
            return var_names

    @staticmethod
    def create_from_csv(csv_path, batch_id):
        """
        Create a recorder instance for replay from a CSV file path.

        :param csv_path: Base path to the CSV file (the batch ID will be automatically appended)
        :param batch_id: Batch identifier for the CSV file
        :return: CplexCutRecorder instance configured for replay
        """
        return CplexCutRecorder(batch_id=batch_id, replay_csv_basepath=csv_path)

    def report(self):
        """report stats"""
        print(f"Cut recording completed for batch {self.batch_id}. Total calls recorded: {self.record_call_count}")
