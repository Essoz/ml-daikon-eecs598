import torch
import unittest

class TestQuantileBug(unittest.TestCase):
    def test_quantile_dim_none(self):
        A = torch.randn(1000, 10)
        device = torch.device("mps")

        cpu_result = torch.quantile(A, 0.5, dim=None, interpolation="nearest")
        mps_result = torch.quantile(A.to(device), 0.5, dim=None, interpolation="nearest").to("cpu")

        self.assertTrue(torch.equal(cpu_result, mps_result), "MPS and CPU give different result when dim=None")

    def test_quantile_dim_zero(self):
        A = torch.randn(1000, 10)
        device = torch.device("mps")

        cpu_result = torch.quantile(A, 0.5, dim=0, interpolation="nearest")
        mps_result = torch.quantile(A.to(device), 0.5, dim=0, interpolation="nearest").to("cpu")

        self.assertTrue(torch.equal(cpu_result, mps_result), "MPS and CPU give different result when dim=0")

    def test_quantile_sorted_tensor(self):
        A = torch.randn(1000, 10)
        device = torch.device("mps")

        cpu_result = torch.quantile(torch.sort(A, dim=0).values, 0.5, dim=0, interpolation="nearest")
        mps_result = torch.quantile(torch.sort(A.to(device), dim=0).values, 0.5, dim=0, interpolation="nearest").to("cpu")

        self.assertTrue(torch.equal(cpu_result, mps_result), "MPS and CPU give different result with sorted tensor")

if __name__ == '__main__':
    unittest.main()