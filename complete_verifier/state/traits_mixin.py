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
Helper mix-in classes to make dataclasses dict-like or list-like, 
without inheriting from dict or list.
"""


from typing import TYPE_CHECKING, Any, Iterator


class DictLikeMixIn:
    """
    A mixin class to make a dataclass dict-like without inheriting from dict.

    Subclasses must make `self._data: dict[str, Any]` field.
    """

    if TYPE_CHECKING:
        _data: dict[str, Any]

    def __getitem__(self, key: str):
        return self._data[key]

    def __setitem__(self, key: str, value) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def get(self, key: str, default):
        return self._data.get(key, default)

class ListLikeMixIn:
    """
    A mixin class to make a dataclass list-like without inheriting from list.

    Subclasses must make `self._data: list[Any]` field.
    """

    if TYPE_CHECKING:
        _data: list[Any]

    def __getitem__(self, index: int):
        return self._data[index]

    def __setitem__(self, index: int, value) -> None:
        self._data[index] = value

    def __delitem__(self, index: int) -> None:
        del self._data[index]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)