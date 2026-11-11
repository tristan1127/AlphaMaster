"""
单元测试：_build_walk_forward_folds 的 gap 机制

验证以下需求：
  T2.1 – 正常情况下每折 val_start == train_end + gap
  T2.2 – 越界折不被添加（折叠不越界）
  T2.3 – 数据量不足时显式退化为非 OOS 单折，绝不静默缩小 gap
"""
import pytest
from model_core.engine import _build_walk_forward_folds


class TestNormalCase:
    """T2.1 – 正常情况：val_start == train_end + gap（使用折内记录的实际 gap）"""

    def test_val_start_equals_train_end_plus_gap(self):
        """T=500, n_folds=5, gap=20：每折的 val_start 严格等于 train_end + gap。"""
        folds = _build_walk_forward_folds(T=500, n_folds=5, gap=20)
        assert len(folds) > 0, "应至少返回一折"
        for i, fold in enumerate(folds):
            actual_gap = fold["gap"]
            assert fold["val_start"] == fold["train_end"] + actual_gap, (
                f"折 {i}: val_start={fold['val_start']} != "
                f"train_end={fold['train_end']} + gap={actual_gap}"
            )

    def test_gap_preserved_when_data_sufficient(self):
        """当数据充足时，即使 T mod n_folds 较小也必须保留指定 gap。
        T=53, n_folds=3, gap=1：53 mod 3 = 2 >= 1*(3-1)=2，gap 恰好被保留为 1。
        """
        folds = _build_walk_forward_folds(T=53, n_folds=3, gap=1)
        assert len(folds) > 0, "应返回至少一折"
        for fold in folds:
            assert "gap" in fold, "折字典缺少 'gap' 键"
            assert fold["gap"] == 1, f"gap 应保留为 1，实际 gap={fold['gap']}"

    def test_expected_fold_count(self):
        """T=500, n_folds=5, gap=20：应有 4 折（k=1~4）。"""
        folds = _build_walk_forward_folds(T=500, n_folds=5, gap=20)
        assert len(folds) == 4

    def test_train_start_moves_forward(self):
        """rolling-window 每折训练起点必须向前推进，而非全部从 0 开始。"""
        folds = _build_walk_forward_folds(T=500, n_folds=5, gap=20)
        assert [f["train_start"] for f in folds] == [0, 104, 208, 312]
        assert all(f["gap"] == 20 for f in folds)
        assert all(f["val_end"] - f["val_start"] == 84 for f in folds)

    def test_val_end_within_bounds(self):
        """所有折的 val_end <= T。"""
        T = 500
        folds = _build_walk_forward_folds(T=T, n_folds=5, gap=20)
        for i, fold in enumerate(folds):
            assert fold["val_end"] <= T, (
                f"折 {i}: val_end={fold['val_end']} 超出 T={T}"
            )


class TestGapZero:
    """gap=0 时 val_start 应等于 train_end。"""

    def test_val_start_equals_train_end_when_gap_zero(self):
        folds = _build_walk_forward_folds(T=500, n_folds=5, gap=0)
        assert len(folds) > 0
        for i, fold in enumerate(folds):
            assert fold["val_start"] == fold["train_end"], (
                f"折 {i}: gap=0 时 val_start 应等于 train_end，"
                f"实际 val_start={fold['val_start']}, train_end={fold['train_end']}"
            )

    def test_gap_stored_as_zero(self):
        folds = _build_walk_forward_folds(T=500, n_folds=5, gap=0)
        for fold in folds:
            assert fold["gap"] == 0


class TestInsufficientData:
    """T2.3 – 无法满足 purge gap 时不制造虚假的 Walk-Forward 验证。"""

    def test_infeasible_gap_falls_back_without_pretending_oos(self):
        T = 50
        folds = _build_walk_forward_folds(T=T, n_folds=5, gap=20)
        assert folds == [{
            "train_start": 0, "train_end": T,
            "val_start": 0, "val_end": T, "gap": 0,
        }]

    def test_feasible_gap_is_never_silently_reduced(self):
        folds = _build_walk_forward_folds(T=120, n_folds=5, gap=5)
        assert len(folds) == 4
        assert all(f["gap"] == 5 for f in folds)

    def test_each_validation_block_is_disjoint_and_equal_length(self):
        for T, n_folds, gap in ((500, 5, 20), (53, 3, 1), (120, 5, 5)):
            folds = _build_walk_forward_folds(T=T, n_folds=n_folds, gap=gap)
            assert len(folds) == n_folds - 1
            for f in folds:
                assert 0 <= f["train_start"] < f["train_end"]
                assert f["train_end"] + gap == f["val_start"]
                assert f["val_start"] < f["val_end"] <= T
                assert f["train_end"] - f["train_start"] == f["val_end"] - f["val_start"]

    @pytest.mark.parametrize("T,n_folds,gap", [
        (-1, 5, 20), (500, 1, 20), (500, 5, -1),
    ])
    def test_invalid_args(self, T, n_folds, gap):
        with pytest.raises(ValueError):
            _build_walk_forward_folds(T=T, n_folds=n_folds, gap=gap)


class TestDegenerateCase:
    """T2.3 – T < n_folds×2 时退化为单折（全量训练，无验证）。"""

    def test_returns_single_fold_when_T_too_small(self):
        """T=8, n_folds=5：fold_size=8//5=1 < 2，应退化为单折。"""
        folds = _build_walk_forward_folds(T=8, n_folds=5, gap=20)
        assert len(folds) == 1, (
            f"T=8, n_folds=5 时应退化为单折，实际返回 {len(folds)} 折"
        )

    def test_degenerate_fold_covers_full_range(self):
        """退化单折的 train 和 val 均覆盖全量 [0, T)。"""
        T = 8
        folds = _build_walk_forward_folds(T=T, n_folds=5, gap=20)
        fold = folds[0]
        assert fold["train_start"] == 0
        assert fold["train_end"] == T
        assert fold["val_start"] == 0
        assert fold["val_end"] == T

    def test_degenerate_fold_gap_is_zero(self):
        """退化单折的 gap 为 0（无意义的 gap）。"""
        folds = _build_walk_forward_folds(T=8, n_folds=5, gap=20)
        assert folds[0]["gap"] == 0

    def test_T_equals_n_folds_times_2_minus_1(self):
        """fold_size = (n_folds*2 - 1) // n_folds = 1 < 2，仍退化为单折。"""
        T = 5 * 2 - 1   # = 9, fold_size = 9//5 = 1
        folds = _build_walk_forward_folds(T=T, n_folds=5, gap=0)
        assert len(folds) == 1

    def test_T_equals_n_folds_times_2_gives_multiple_folds(self):
        """fold_size = (n_folds*2) // n_folds = 2 >= 2，应产生多折。"""
        T = 5 * 2   # = 10, fold_size = 10//5 = 2
        folds = _build_walk_forward_folds(T=T, n_folds=5, gap=0)
        assert len(folds) > 1, (
            f"fold_size=2 时应产生多折，实际返回 {len(folds)} 折"
        )


class TestReturnStructure:
    """验证返回的折字典包含所有必要键。"""

    REQUIRED_KEYS = {"train_start", "train_end", "val_start", "val_end", "gap"}

    def test_all_required_keys_present(self):
        folds = _build_walk_forward_folds(T=500, n_folds=5, gap=20)
        for i, fold in enumerate(folds):
            missing = self.REQUIRED_KEYS - set(fold.keys())
            assert not missing, f"折 {i} 缺少键：{missing}"

    def test_degenerate_fold_has_required_keys(self):
        folds = _build_walk_forward_folds(T=8, n_folds=5, gap=20)
        for i, fold in enumerate(folds):
            missing = self.REQUIRED_KEYS - set(fold.keys())
            assert not missing, f"退化折 {i} 缺少键：{missing}"
