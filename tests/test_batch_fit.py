from __future__ import annotations

import multiprocessing as mp

import pytest

from inftools.batch import fit_many


def square_item(x):
    return x * x


def maybe_fail_item(x):
    if x == 2:
        raise ValueError("bad object")
    return 10 * x


def test_fit_many_serial_preserves_catalog_order():
    result = fit_many([1, 2, 3], square_item, executor="serial")

    assert result.results == [1, 4, 9]
    assert result.failures == []
    assert result.successful_indices == [0, 1, 2]


def test_fit_many_can_record_failures_without_stopping_catalog():
    result = fit_many([1, 2, 3], maybe_fail_item, executor="serial", fail_fast=False)

    assert result.results == [10, None, 30]
    assert len(result.failures) == 1
    assert result.failures[0].index == 1
    assert result.failures[0].error_type == "ValueError"
    assert result.successful_indices == [0, 2]


def test_fit_many_process_executor_runs_picklable_top_level_function():
    try:
        mp.get_context("fork")
    except ValueError:
        pytest.skip("fork multiprocessing context is unavailable on this platform")

    try:
        result = fit_many([1, 2, 3, 4], square_item, n_workers=2, executor="process", mp_context="fork")
    except (NotImplementedError, PermissionError) as exc:
        pytest.skip(f"process pools are unavailable in this runtime: {exc}")

    assert result.results == [1, 4, 9, 16]
    assert result.failures == []
