"""Tests for backend/api/utils.py."""
from __future__ import annotations

import pytest

from api.utils import normalize_state


class TestNormalizeState:
    def test_none_returns_none(self) -> None:
        assert normalize_state(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert normalize_state("") is None

    def test_whitespace_only_returns_none(self) -> None:
        assert normalize_state("   ") is None

    def test_tab_only_returns_none(self) -> None:
        assert normalize_state("\t") is None

    def test_newline_only_returns_none(self) -> None:
        assert normalize_state("\n") is None

    def test_lowercase_is_uppercased(self) -> None:
        assert normalize_state("pa") == "PA"

    def test_already_uppercase_unchanged(self) -> None:
        assert normalize_state("PA") == "PA"

    def test_mixed_case_is_uppercased(self) -> None:
        assert normalize_state("Pa") == "PA"

    def test_leading_whitespace_stripped(self) -> None:
        assert normalize_state("  PA") == "PA"

    def test_trailing_whitespace_stripped(self) -> None:
        assert normalize_state("PA  ") == "PA"

    def test_both_sides_whitespace_stripped(self) -> None:
        assert normalize_state("  pa  ") == "PA"

    def test_single_char_state(self) -> None:
        assert normalize_state("c") == "C"

    def test_full_state_name_lowercased(self) -> None:
        assert normalize_state("california") == "CALIFORNIA"

    def test_full_state_name_mixed_case(self) -> None:
        assert normalize_state("California") == "CALIFORNIA"

    def test_numeric_string_passthrough(self) -> None:
        assert normalize_state("42") == "42"

    def test_common_two_letter_abbreviations(self) -> None:
        for state in ["NY", "CA", "TX", "FL", "IL"]:
            assert normalize_state(state) == state
            assert normalize_state(state.lower()) == state

    def test_returns_string_type(self) -> None:
        result = normalize_state("pa")
        assert isinstance(result, str)

    def test_return_type_is_none_for_blank(self) -> None:
        result = normalize_state("")
        assert result is None

    def test_whitespace_around_valid_state(self) -> None:
        assert normalize_state("\t NY \n") == "NY"
