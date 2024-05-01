import torch
import unittest

class TestGradBug(unittest.TestCase):
    def test_gradients(self):
        A = torch.randn(1000, 10, requires_grad=True)
        device = torch.device("mps")

        # CPU: correctly returns non-zero gradients for 1 element in each column of A
        out_cpu = torch.sum(torch.quantile(A, 0.5, 0, interpolation="nearest"))
        out_cpu.backward()
        cpu_grad = torch.where(A.grad != 0)

        A.grad = None

        # MPS: usually returns no non-zero elements
        # some seeds give non-zero elements but in the wrong places (always in the first row)
        out_mps = torch.sum(torch.quantile(A.to(device), 0.5, 0, interpolation="nearest"))
        out_mps.backward()
        device_grad = torch.where(A.grad != 0)

        for i in range(len(cpu_grad)):
            if not torch.equal(cpu_grad[i], device_grad[i]):
                self.fail("MPS and CPU gradients are different")

if __name__ == '__main__':
    unittest.main()