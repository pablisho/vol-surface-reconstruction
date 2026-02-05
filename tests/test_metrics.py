# tests/test_metrics.py
"""Tests for evaluation/metrics.py — reconstruction quality metrics."""

from __future__ import annotations

import pytest
import torch

from evaluation.metrics import compute_metrics


class TestComputeMetrics:
    def test_perfect_prediction(self) -> None:
        target = torch.rand(2, 1, 4, 5)
        mask = torch.ones(2, 4, 5)
        m = compute_metrics(target, target, mask)
        assert m.rmse == pytest.approx(0.0, abs=1e-12)
        assert m.mae == pytest.approx(0.0, abs=1e-12)
        assert m.max_error == pytest.approx(0.0, abs=1e-12)

    def test_known_error(self) -> None:
        target = torch.zeros(1, 1, 2, 2)
        pred = torch.ones(1, 1, 2, 2)
        mask = torch.ones(1, 2, 2)
        m = compute_metrics(pred, target, mask)
        assert m.rmse == pytest.approx(1.0, abs=1e-7)
        assert m.mae == pytest.approx(1.0, abs=1e-7)
        assert m.max_error == pytest.approx(1.0, abs=1e-7)

    def test_observed_vs_missing_split(self) -> None:
        target = torch.zeros(1, 1, 1, 4)
        pred = torch.tensor([[[[0.0, 0.0, 1.0, 1.0]]]])
        # Mask: first 2 observed, last 2 missing
        mask = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])
        m = compute_metrics(pred, target, mask)
        assert m.rmse_observed == pytest.approx(0.0, abs=1e-12)
        assert m.rmse_missing == pytest.approx(1.0, abs=1e-7)

    def test_all_metrics_non_negative(self) -> None:
        pred = torch.randn(3, 1, 4, 5)
        target = torch.randn(3, 1, 4, 5)
        mask = (torch.rand(3, 4, 5) > 0.3).float()
        m = compute_metrics(pred, target, mask)
        assert m.rmse >= 0.0
        assert m.mae >= 0.0
        assert m.rmse_observed >= 0.0
        assert m.rmse_missing >= 0.0
        assert m.max_error >= 0.0

    def test_2d_input(self) -> None:
        """compute_metrics should also work with unbatched 2-D tensors."""
        target = torch.rand(4, 5)
        pred = target + 0.01
        mask = torch.ones(4, 5)
        m = compute_metrics(pred, target, mask)
        assert m.rmse > 0
        assert m.rmse == pytest.approx(0.01, abs=1e-6)
