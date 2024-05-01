import unittest
import torch

class TestMatmulBug(unittest.TestCase):
    def test_matmul_bug(self):
        zeros = torch.zeros(911, 9, 1, device=torch.device("cpu"))
        ones = torch.ones(1, 32769, device=torch.device("cpu"))
        result_cpu = zeros @ ones

        zeros_mps = torch.zeros(911, 9, 1, device=torch.device("mps"))
        ones_mps = torch.ones(1, 32769, device=torch.device("mps"))
        result_mps = zeros_mps @ ones_mps

        self.assertTrue(torch.allclose(result_cpu, result_mps.to('cpu')), "Results are different between CPU and MPS devices")

if __name__ == '__main__':
    unittest.main()
