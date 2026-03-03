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


class TestWeightedProportions:
    def test_weighted_proportions_over_one_block(self):
        """weights [30, 70] → base block [3, 7], tiled by block_multiple."""
        br = BlockRandomize(block_multiple=5)
        # base_size = 10, block = 10 * 5 = 50
        block_size = 50
        results = [br.random_index_for("exp-w1", 2, weights=[30, 70]) for _ in range(block_size)]
        counts = Counter(results)
        assert counts[0] == 15  # 3 * 5
        assert counts[1] == 35  # 7 * 5

    def test_weighted_three_arms(self):
        """weights [1, 2, 7] → probs [0.1, 0.2, 0.7], base [1, 2, 7], base_size=10."""
        br = BlockRandomize(block_multiple=3)
        block_size = 30  # 10 * 3
        results = [br.random_index_for("exp-w3", 3, weights=[1, 2, 7]) for _ in range(block_size)]
        counts = Counter(results)
        assert counts[0] == 3  # 1 * 3
        assert counts[1] == 6  # 2 * 3
        assert counts[2] == 21  # 7 * 3

    def test_weighted_equal_matches_uniform(self):
        """Equal weights should produce the same block size as uniform."""
        block_multiple = 4
        br = BlockRandomize(block_multiple=block_multiple)
        length = 3
        # uniform base_size = 3, block = 12
        # equal-weights base: probs [1/3, 1/3, 1/3], lcm_denom = 3, base [1,1,1], base_size = 3
        block_size = length * block_multiple

        results = [br.random_index_for("exp-weq", length, weights=[1, 1, 1]) for _ in range(block_size)]
        counts = Counter(results)
        for arm_idx in range(length):
            assert counts[arm_idx] == block_multiple

    def test_weighted_auto_refill(self):
        """Proportions hold across two consecutive blocks."""
        br = BlockRandomize(block_multiple=4)
        # weights [25, 75] → probs [0.25, 0.75], base [1, 3], base_size=4, block=16
        block_size = 16
        for block_num in range(2):
            results = [br.random_index_for("exp-wrf", 2, weights=[25, 75]) for _ in range(block_size)]
            counts = Counter(results)
            assert counts[0] == 4, f"Block {block_num}: expected 4, got {counts[0]}"
            assert counts[1] == 12, f"Block {block_num}: expected 12, got {counts[1]}"


class TestMaxBlockSize:
    def test_caps_multiple_uniform(self):
        """max_block_size=20, block_multiple=100, length=3 → effective_multiple=6, block=18."""
        br = BlockRandomize(block_multiple=100, max_block_size=20)
        # base_size=3, effective = min(100, 20//3) = min(100, 6) = 6
        block_size = 18
        results = [br.random_index_for("exp-cap", 3) for _ in range(block_size)]
        counts = Counter(results)
        for arm_idx in range(3):
            assert counts[arm_idx] == 6
        # Next draw triggers a new block (not from the same one)
        extra = br.random_index_for("exp-cap", 3)
        assert 0 <= extra < 3

    def test_caps_weighted(self):
        """weights [1, 99] → base_size=100, max_block_size=100 → effective_multiple=1."""
        br = BlockRandomize(block_multiple=10, max_block_size=100)
        block_size = 100
        results = [br.random_index_for("exp-wcap", 2, weights=[1, 99]) for _ in range(block_size)]
        counts = Counter(results)
        assert counts[0] == 1
        assert counts[1] == 99


class TestWeightedValidation:
    def test_weights_length_mismatch_raises(self):
        br = BlockRandomize()
        with pytest.raises(ValueError, match="len\\(weights\\) must equal length"):
            br.random_index_for("exp-bad", 3, weights=[1, 2])

    def test_zero_weight_raises(self):
        br = BlockRandomize()
        with pytest.raises(ValueError, match="all weights must be > 0"):
            br.random_index_for("exp-bad", 2, weights=[0.5, 0])

    def test_negative_weight_raises(self):
        br = BlockRandomize()
        with pytest.raises(ValueError, match="all weights must be > 0"):
            br.random_index_for("exp-bad", 2, weights=[1, -1])


class TestWeightedConcurrentAccess:
    def test_threaded_draws_with_weights(self):
        br = BlockRandomize(block_multiple=5)
        # weights [30, 70] → base [3,7], base_size=10, block=50
        n_threads = 8
        block_size = 50
        total_draws = block_size * 4  # 4 full blocks = 200
        draws_per_thread = total_draws // n_threads

        all_results: list[int] = []

        def draw_many():
            return [br.random_index_for("exp-wcon", 2, weights=[30, 70]) for _ in range(draws_per_thread)]

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futures = [pool.submit(draw_many) for _ in range(n_threads)]
            for future in as_completed(futures):
                all_results.extend(future.result())

        assert len(all_results) == total_draws
        assert all(0 <= idx < 2 for idx in all_results)

        counts = Counter(all_results)
        # 4 blocks * 15 = 60 for arm 0, 4 * 35 = 140 for arm 1
        assert counts[0] == 60
        assert counts[1] == 140


class TestRandomState:
    def test_seeded_block_is_reproducible(self):
        """Same random_state produces the same block order on a fresh instance."""
        length, block_multiple = 4, 5
        block_size = length * block_multiple

        br1 = BlockRandomize(block_multiple=block_multiple)
        br2 = BlockRandomize(block_multiple=block_multiple)
        br3 = BlockRandomize(block_multiple=block_multiple)

        results1 = [br1.random_index_for("exp-seed", length, random_state=42) for _ in range(block_size)]
        results2 = [br2.random_index_for("exp-seed", length, random_state=42) for _ in range(block_size)]
        # But different seeds produce different blocks
        results3 = [br3.random_index_for("exp-seed-diff", length, random_state=41) for _ in range(block_size)]

        assert results1 == results2
        assert results1 != results3

    def test_seeded_block_is_balanced(self):
        """random_state seeds the shuffle but does not break balance guarantees."""
        length, block_multiple = 3, 4
        block_size = length * block_multiple
        br = BlockRandomize(block_multiple=block_multiple)

        results = [br.random_index_for("exp-seed-bal", length, random_state=7) for _ in range(block_size)]
        counts = Counter(results)
        for arm_idx in range(length):
            assert counts[arm_idx] == block_multiple

    def test_seed_only_affects_new_block_generation(self):
        """Once a block exists, subsequent random_state values are ignored for that block."""
        length, block_multiple = 2, 4
        block_size = length * block_multiple
        # br = BlockRandomize(block_multiple=block_multiple)
        br1 = BlockRandomize(block_multiple=block_multiple)
        br2 = BlockRandomize(block_multiple=block_multiple)

        reference = [br1.random_index_for("exp-seed-firstcall", length, random_state=42) for _ in range(block_size)]
        # Seed is used on first call (creates block)
        first = br2.random_index_for("exp-seed-firstcall2", length, random_state=42)
        # Subsequent calls drain the same block regardless of random_state
        rest = [br2.random_index_for("exp-seed-firstcall2", length, random_state=i) for i in range(block_size - 1)]
        all_results = [first, *rest]

        assert reference == all_results

    def test_seeded_weighted_block_is_reproducible(self):
        """random_state also seeds weighted block shuffles."""
        length, block_multiple = 2, 5
        block_size = 50  # weights [30,70] → base 10, * 5 = 50

        br1 = BlockRandomize(block_multiple=block_multiple)
        br2 = BlockRandomize(block_multiple=block_multiple)

        r1 = [br1.random_index_for("exp-w-seed", length, weights=[30, 70], random_state=13) for _ in range(block_size)]
        r2 = [br2.random_index_for("exp-w-seed", length, weights=[30, 70], random_state=13) for _ in range(block_size)]

        assert r1 == r2
        # Still balanced
        counts = Counter(r1)
        assert counts[0] == 15
        assert counts[1] == 35
