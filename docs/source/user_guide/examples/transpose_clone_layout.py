import torch
import torch_spyre

a = torch.randn(4, 128, dtype=torch.float16, device="spyre")

compiled = torch.compile(lambda x: x.T.clone())
result = compiled(a)

print("result:", result)
print("shape:", result.shape)
print("stride:", result.stride())
print("data_ptr:", result.data_ptr())
print("input data_ptr:", a.data_ptr())
print("result layout:", result.device_tensor_layout())
