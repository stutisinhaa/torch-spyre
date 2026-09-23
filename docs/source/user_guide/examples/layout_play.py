import torch

# Ensure the Spyre PyTorch C++ extension library is loaded
try:
    import torch_spyre
except ImportError:
    raise RuntimeError(
        "torch_spyre backend library is missing. "
        "Install via `pip install torch-spyre` on a machine with Spyre hardware drivers."
    )

def inspect_spyre_tensor(tensor: torch.Tensor):
    """
    Extracts layout metadata directly from a live tensor residing on a Spyre device.
    """
    if tensor.device.type != "spyre":
        raise ValueError(f"Expected a tensor on 'spyre' device, but got '{tensor.device}'")
        
    print("=" * 65)
    print(f"LIVE SPYRE TENSOR METADATA (Device: {tensor.device})")
    print("=" * 65)
    
    # 1. Standard PyTorch Metadata
    print(f"Logical Shape      : {tuple(tensor.shape)}")
    print(f"Logical Strides    : {tuple(tensor.stride())}")
    print(f"Data Type          : {tensor.dtype}")
    print(f"Is Contiguous      : {tensor.is_contiguous()}")
    print("-" * 65)

    # 2. Spyre Physical Hardware Layout Properties
    # torch_spyre exposes hardware metadata on Spyre tensors
    try:
        # Access hardware-specific layout object
        layout = torch_spyre.get_spyre_layout(tensor)
        
        device_size = tuple(layout.device_size)
        stride_map = tuple(layout.stride_map)
        stick_size = layout.stick_size if hasattr(layout, 'stick_size') else 64
        
        print(f"Device Size        : {device_size}")
        print(f"Stride Map         : {stride_map}")
        print(f"Physical Stick Size: {stick_size}")
        
    except AttributeError:
        # Fallback to direct tensor attributes if C++ bindings expose them directly
        device_size = getattr(tensor, "device_size", "N/A")
        stride_map = getattr(tensor, "stride_map", "N/A")
        print(f"Device Size        : {device_size}")
        print(f"Stride Map         : {stride_map}")
        
    print("-" * 65)
    
    # 3. Element Access Mapping
    print("SAMPLE LOGICAL-TO-PHYSICAL ELEMENT ADDRESSING:")
    flat_tensor = tensor.flatten()
    test_indices = [0, 63, 64, tensor.numel() - 1]
    
    for idx in test_indices:
        if idx < tensor.numel():
            # Query Spyre physical offset/coordinates if API allows
            try:
                phys_coords, phys_offset = torch_spyre.get_element_location(tensor, idx)
                print(f"  Logical Index [{idx:3d}] -> Physical Coords: {phys_coords} | Memory Offset: {phys_offset}")
            except (AttributeError, NotImplementedError):
                # Manual coordinate extraction based on reported stride_map & device_size
                if isinstance(stride_map, tuple) and len(stride_map) >= 2:
                    s_idx = idx // stride_map[0]
                    e_idx = idx % stride_map[0]
                    p_offset = (s_idx * stride_map[0]) + (e_idx * stride_map[-1])
                    print(f"  Logical Index [{idx:3d}] -> Stick: {s_idx}, Offset In Stick: {e_idx} | Memory Offset: {p_offset}")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    # Check if Spyre accelerator hardware is available
    device = torch.device("spyre:0" if torch.spyre.is_available() else "cpu")
    print(f"Running on target device: {device}\n")
    
    if device.type == "spyre":
        # Example 1: 1D Tensor spanning across hardware sticks
        print("--> Test Case 1: 1D Tensor with 65 elements (Float16)")
        t1_spyre = torch.arange(65, dtype=torch.float16, device=device)
        inspect_spyre_tensor(t1_spyre)

        # Example 2: 2D Matrix (32x100)
        print("--> Test Case 2: 2D Matrix (32x100) on Spyre")
        t2_spyre = torch.randn(32, 100, dtype=torch.float16, device=device)
        inspect_spyre_tensor(t2_spyre)

        # Example 3: Non-contiguous Slice
        print("--> Test Case 3: Non-Contiguous Tensor View")
        t3_spyre = torch.randn(64, 128, dtype=torch.float16, device=device)
        t3_sliced = t3_spyre[::2, :] # Strided slice
        inspect_spyre_tensor(t3_sliced)
    else:
        print("Spyre device not detected. Ensure drivers and hardware are available.")
