import torch
import torch_spyre

a = torch.randn(4, 128, dtype=torch.float16, device="spyre")
b = torch.randn(4, 128, dtype=torch.float16, device="spyre")

compiled = torch.compile(lambda x, y: x.T + y.T)
result = compiled(a, b)

print("result:", result)
print("shape:", result.shape)
print("stride:", result.stride())
print("data_ptr:", result.data_ptr())
print("input a data_ptr:", a.data_ptr())
print("input b data_ptr:", b.data_ptr())
print("result layout:", result.device_tensor_layout())
