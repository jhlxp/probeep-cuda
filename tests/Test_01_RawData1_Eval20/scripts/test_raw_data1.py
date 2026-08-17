from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("raw_data1.py")
SPEC = importlib.util.spec_from_file_location("probeep_tests_raw_data1", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
raw_data1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(raw_data1)


def test_selectors_are_explicit() -> None:
    assert raw_data1.selector_layers("raw_data1_eval20") == list(range(20))
    assert raw_data1.selector_layers("raw_data1_all") == list(range(58))
    assert raw_data1.selector_layers("raw_data1_layer_57") == [57]


def test_dataset_is_complete_dsv3() -> None:
    summary, scaled = raw_data1.describe(
        raw_data1.DEFAULT_DATA_DIR, "raw_data1_all", 16, 4096, 8
    )
    assert summary["selected_layer_count"] == 58
    assert summary["num_experts"] == 256
    assert summary["routes_per_layer"] == 524288
    assert summary["paper_eligible"] is True
    assert scaled.shape == (58, 256)
    assert np.all(scaled.sum(axis=1) == 524288)


def test_exact_topk_realization_preserves_histogram() -> None:
    counts = np.full(256, 2, dtype=np.int64)
    routes = raw_data1.realize_exact_topk(counts, num_tokens=64, topk=8)
    assert routes.shape == (64, 8)
    assert np.all(np.diff(np.sort(routes.astype(np.int32), axis=1), axis=1) != 0)
    np.testing.assert_array_equal(np.bincount(routes.ravel(), minlength=256), counts)
