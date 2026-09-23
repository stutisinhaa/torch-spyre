import torch

torch.manual_seed(42)

def test(x):
    return torch.transpose(x, -2, -1).clone()

x = torch.randn(2, 64, dtype=torch.float16)

compiled = torch.compile(test, backend="inductor")
output = compiled(x.to("spyre")).cpu()
expected = test(x)

torch.testing.assert_close(output, expected, atol=0.1, rtol=0.1)
