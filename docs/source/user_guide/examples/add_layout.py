import torch, torch_spyre

a = torch.randn(4, 128, dtype=torch.float16, device="spyre:0")
b = torch.randn(4, 128, dtype=torch.float16, device="spyre:0")
print("a layout:", a.device_tensor_layout())
print("b layout:", b.device_tensor_layout())
torch.compile(lambda x, y: x + y)(a, b)
