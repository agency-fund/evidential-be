"""Tests for permuted block randomization module."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from xngin.stats.block_randomize import BlockRandomize


class TestReturnsValidIndex:
    def test_single_draw_in_range(self):
        br = BlockRandomize(block_multiple=5)
        for length in (2, 3, 5, 10):
            idx = br.random_index_for("exp-1", length)
            assert 0 <= idx < length

    def test_returns_int(self):
        br = BlockRandomize(block_multiple=3)
        idx = br.random_index_for("exp-1", 4)
        assert isinstance(idx, int)


class TestPerfectBalanceOverOneBlock:
    @pytest.mark.parametrize("length", [2, 3, 5, 7])
    @pytest.mark.parametrize("block_multiple", [1, 5, 10])
    def test_exact_counts(self, length: int, block_multiple: int):
        br = BlockRandomize(block_multiple=block_multiple)
        block_size = length * block_multiple
        results = [br.random_index_for("exp-balance", length) for _ in range(block_size)]
        counts = Counter(results)
        assert set(counts.keys()) == set(range(length))
        for arm_idx in range(length):
            assert counts[arm_idx] == block_multiple


class TestAutoRefillOnExhaustion:
    def test_balance_across_two_blocks(self):
        length, block_multiple = 3, 4
        br = BlockRandomize(block_multiple=block_multiple)
        block_size = length * block_multiple

        for block_num in range(2):
            results = [br.random_index_for("exp-refill", length) for _ in range(block_size)]
            counts = Counter(results)
            for arm_idx in range(length):
                assert counts[arm_idx] == block_multiple, (
                    f"Block {block_num}: arm {arm_idx} count {counts[arm_idx]} != {block_multiple}"
                )


class TestIndependentExperimentIds:
    def test_separate_blocks_per_id(self):
        br = BlockRandomize(block_multiple=4)
        length = 3
        block_size = length * 4

        for exp_id in ("exp-A", "exp-B"):
            results = [br.random_index_for(exp_id, length) for _ in range(block_size)]
            counts = Counter(results)
            for arm_idx in range(length):
                assert counts[arm_idx] == 4


class TestIndependentLengths:
    def test_same_id_different_lengths(self):
        br = BlockRandomize(block_multiple=6)

        results_2 = [br.random_index_for("exp-shared", 2) for _ in range(12)]
        results_3 = [br.random_index_for("exp-shared", 3) for _ in range(18)]

        counts_2 = Counter(results_2)
        counts_3 = Counter(results_3)

        assert set(counts_2.keys()) == {0, 1}
        assert all(c == 6 for c in counts_2.values())

        assert set(counts_3.keys()) == {0, 1, 2}
        assert all(c == 6 for c in counts_3.values())


class TestSingleArm:
    def test_always_returns_zero(self):
        br = BlockRandomize(block_multiple=10)
        results = [br.random_index_for("exp-single", 1) for _ in range(25)]
        assert all(idx == 0 for idx in results)


class TestInvalidLengthRaises:
    @pytest.mark.parametrize("length", [0, -1, -100])
    def test_raises_value_error(self, length: int):
        br = BlockRandomize(block_multiple=5)
        with pytest.raises(ValueError, match="length must be >= 1"):
            br.random_index_for("exp-bad", length)

    @pytest.mark.parametrize("block_multiple", [0, -1])
    def test_invalid_block_multiple_raises(self, block_multiple: int):
        with pytest.raises(ValueError, match="block_multiple must be >= 1"):
            BlockRandomize(block_multiple=block_multiple)


class TestShuffleIsRandom:
    def test_two_blocks_differ(self):
        """Two consecutive blocks for the same key are (almost certainly) in different order."""
        length, block_multiple = 5, 20
        br = BlockRandomize(block_multiple=block_multiple)
        block_size = length * block_multiple

        block_1 = [br.random_index_for("exp-shuffle", length) for _ in range(block_size)]
        block_2 = [br.random_index_for("exp-shuffle", length) for _ in range(block_size)]

        # Both blocks have perfect balance
        assert Counter(block_1) == Counter(block_2)
        # But the order should differ (probability of identical order is vanishingly small)
        assert block_1 != block_2


class TestConcurrentAccess:
    def test_threaded_draws_are_valid_and_complete(self):
        length, block_multiple = 4, 25
        br = BlockRandomize(block_multiple=block_multiple)
        n_threads = 8
        # Total draws = exact multiple of block_size so balance is checkable
        block_size = length * block_multiple
        total_draws = block_size * 4  # 4 full blocks
        draws_per_thread = total_draws // n_threads

        all_results: list[int] = []

        def draw_many():
            return [br.random_index_for("exp-concurrent", length) for _ in range(draws_per_thread)]

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futures = [pool.submit(draw_many) for _ in range(n_threads)]
            for future in as_completed(futures):
                all_results.extend(future.result())

        assert len(all_results) == total_draws
        assert all(0 <= idx < length for idx in all_results)

        counts = Counter(all_results)
        assert set(counts.keys()) == set(range(length))
        # Under the lock, blocks are consumed atomically so total balance
        # across complete blocks is maintained.
        for arm_idx in range(length):
            assert counts[arm_idx] == total_draws // length


class TestBlockMultipleRespected:
    @pytest.mark.parametrize("block_multiple", [1, 5, 50])
    def test_block_size_matches_multiple(self, block_multiple: int):
        length = 3
        br = BlockRandomize(block_multiple=block_multiple)
        block_size = length * block_multiple

        results = [br.random_index_for("exp-mult", length) for _ in range(block_size)]
        counts = Counter(results)
        for arm_idx in range(length):
            assert counts[arm_idx] == block_multiple
