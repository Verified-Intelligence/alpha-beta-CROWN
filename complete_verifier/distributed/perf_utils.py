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
Performance profiling tool frequently used
in the development of distributed version.
"""
import os
import torch
from functools import wraps


class torch_profile_decorator:

    static_global_counter = {}

    def __init__(self, file_prefix: str, function_name: str = ""):
        self.file_prefix = file_prefix
        self.function_name = function_name
        if function_name not in torch_profile_decorator.static_global_counter:
            torch_profile_decorator.static_global_counter[function_name] = 0

    def __call__(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if os.getenv("TORCH_PROFILER_ENABLE", "0") != "1":
                print(
                    "TORCH_PROFILER_ENABLE environment variable is not set to '1'."
                    " Skipping profiling."
                )
                return fn(*args, **kwargs)
            torch.cuda.memory._record_memory_history(max_entries=int(1e5))
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
            ):
                result = fn(*args, **kwargs)
            # print("[1/2] Dumping memory profile...")
            # prof.export_memory_timeline(f"{self.file_prefix}.html", device="cuda:0")
            # print("Skipeed")
            print("[2/2] Dumping memory snapshot...")
            mem_gb = torch.cuda.memory_reserved() / (1024**3)
            print(f"Current reserved memory: {mem_gb:.2f} GB")
            if mem_gb < 7:
                # no dump
                torch_profile_decorator.static_global_counter[self.function_name] += 1
                return result

            counter = torch_profile_decorator.static_global_counter[self.function_name]
            torch.cuda.memory._dump_snapshot(
                f"{self.file_prefix}.{self.function_name}.{counter}.pickle"
            )
            torch_profile_decorator.static_global_counter[self.function_name] += 1
            return result

        return wrapper
