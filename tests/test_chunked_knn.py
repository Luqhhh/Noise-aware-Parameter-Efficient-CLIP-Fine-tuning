"""Tests for chunked_topk_cosine — verify equality with brute-force."""
import torch
import torch.nn.functional as F
import pytest
from common.diagnostic_metrics import chunked_topk_cosine


@pytest.fixture
def small_data():
    """Small query and bank for exhaustive comparison."""
    rng = torch.Generator().manual_seed(99)
    query = torch.randn(30, 16, generator=rng)
    bank = torch.randn(50, 16, generator=rng)
    query = F.normalize(query, dim=-1)
    bank = F.normalize(bank, dim=-1)
    return query, bank


def brute_force_topk(query, bank, k):
    """Reference: full similarity matrix, exact top-k."""
    sim = torch.mm(query, bank.T)
    return sim.topk(k, dim=1, largest=True).values


class TestChunkedTopK:
    def test_vs_brute_force_default_chunks(self, small_data):
        query, bank = small_data
        k = 5
        idx_c, sim_c = chunked_topk_cosine(query, bank, k, query_chunk_size=8, bank_chunk_size=10)
        sim_bf = brute_force_topk(query, bank, k)
        assert torch.allclose(sim_c, sim_bf, atol=1e-5), f"max diff: {(sim_c - sim_bf).abs().max()}"

    def test_vs_brute_force_single_query_chunk(self, small_data):
        query, bank = small_data
        k = 3
        idx_c, sim_c = chunked_topk_cosine(query, bank, k, query_chunk_size=30, bank_chunk_size=50)
        sim_bf = brute_force_topk(query, bank, k)
        assert torch.allclose(sim_c, sim_bf, atol=1e-5)

    def test_vs_brute_force_single_bank_chunk(self, small_data):
        query, bank = small_data
        k = 3
        idx_c, sim_c = chunked_topk_cosine(query, bank, k, query_chunk_size=5, bank_chunk_size=50)
        sim_bf = brute_force_topk(query, bank, k)
        assert torch.allclose(sim_c, sim_bf, atol=1e-5)

    def test_k_is_one(self, small_data):
        query, bank = small_data
        k = 1
        idx_c, sim_c = chunked_topk_cosine(query, bank, k)
        sim_bf = brute_force_topk(query, bank, k)
        assert torch.allclose(sim_c, sim_bf, atol=1e-5)

    def test_k_equals_bank_size(self, small_data):
        query, bank = small_data
        k = bank.size(0)
        idx_c, sim_c = chunked_topk_cosine(query, bank, k)
        sim_bf = brute_force_topk(query, bank, k)
        # Sort each row of sim_bf for comparison
        sim_bf_sorted, _ = sim_bf.sort(dim=1, descending=True)
        sim_c_sorted, _ = sim_c.sort(dim=1, descending=True)
        assert torch.allclose(sim_c_sorted, sim_bf_sorted, atol=1e-5)

    def test_output_shapes(self, small_data):
        query, bank = small_data
        k = 7
        idx, sim = chunked_topk_cosine(query, bank, k)
        assert idx.shape == (query.size(0), k)
        assert sim.shape == (query.size(0), k)

    def test_indices_in_range(self, small_data):
        query, bank = small_data
        k = 5
        idx, sim = chunked_topk_cosine(query, bank, k)
        assert (idx >= 0).all() and (idx < bank.size(0)).all()

    def test_cpu_device(self, small_data):
        query, bank = small_data
        k = 5
        idx, sim = chunked_topk_cosine(query, bank, k, device="cpu")
        assert idx.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self, small_data):
        query, bank = small_data
        k = 5
        idx, sim = chunked_topk_cosine(query, bank, k, device="cuda")
        assert idx.device.type == "cpu"  # returned to CPU
        # Just check it runs without error
        assert idx.shape == (query.size(0), k)
