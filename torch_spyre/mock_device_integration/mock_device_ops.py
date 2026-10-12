# Copyright 2025 The Torch-Spyre Authors.
# Mock device operations for SDSC JSON generation

"""
Mock device operations that intercept tensor.to("spyre") calls
and enable SDSC JSON generation without actual hardware.
"""

import torch
import os
import json
import sys
from typing import Any, Dict, List

try:
    from backends.spyre.torch_patches import install_mock_torch_patches
except Exception:
    install_mock_torch_patches = None

MOCK_DEVICE_ENABLED = os.environ.get("TORCH_SPYRE_MOCK_DEVICE", "0") == "1"
MOCK_VERBOSE = os.environ.get("TORCH_SPYRE_MOCK_VERBOSE", "0") == "1"


def _mock_print(*args, **kwargs):
    """Print to stderr to avoid polluting subprocess stdout captures."""
    print(*args, file=sys.stderr, **kwargs)

# Global storage for tracking operations
_operation_log = []
_sdsc_graph = {
    "version": "1.0",
    "operations": [],
    "tensors": {}
}


def log_operation(op_name: str, inputs: List[Any], output: Any, metadata: Dict[str, Any] | None = None):
    """Log an operation for SDSC JSON generation"""
    if not MOCK_DEVICE_ENABLED:
        return
    
    op_entry = {
        "name": op_name,
        "inputs": [_tensor_info(inp) for inp in inputs if isinstance(inp, torch.Tensor)],
        "output": _tensor_info(output) if isinstance(output, torch.Tensor) else None,
        "metadata": metadata if metadata is not None else {}
    }
    _operation_log.append(op_entry)
    _sdsc_graph["operations"].append(op_entry)
    
    if MOCK_DEVICE_ENABLED:
        _mock_print(f"[MOCK_DEVICE] Logged operation: {op_name}")


def _tensor_info(tensor):
    """Extract tensor information for SDSC"""
    if tensor is None:
        return None
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "requires_grad": tensor.requires_grad
    }


def get_sdsc_json():
    """Get the accumulated SDSC JSON"""
    return json.dumps(_sdsc_graph, indent=2)


def clear_sdsc_log():
    """Clear the operation log"""
    global _operation_log, _sdsc_graph
    _operation_log = []
    _sdsc_graph = {
        "version": "1.0",
        "operations": [],
        "tensors": {}
    }


def print_sdsc_json():
    """Print the SDSC JSON"""
    _mock_print("\n" + "="*60)
    _mock_print("SDSC JSON OUTPUT")
    _mock_print("="*60)
    _mock_print(get_sdsc_json())
    _mock_print("="*60 + "\n")


# Mock tensor.to() implementation
class MockSpyreTensor:
    """Wrapper that keeps tensor on CPU but tracks as Spyre device"""
    
    def __init__(self, cpu_tensor):
        self._cpu_tensor = cpu_tensor
        self._device_str = "spyre:0"
        if MOCK_DEVICE_ENABLED:
            _mock_print(f"[MOCK_DEVICE] Created mock Spyre tensor: shape={cpu_tensor.shape}, dtype={cpu_tensor.dtype}")
    
    @property
    def device(self):
        return torch.device(self._device_str)
    
    def cpu(self):
        """Return the underlying CPU tensor"""
        if MOCK_DEVICE_ENABLED:
            _mock_print(f"[MOCK_DEVICE] Moving tensor back to CPU")
        return self._cpu_tensor
    
    def __getattr__(self, name):
        """Forward all other attributes to CPU tensor"""
        return getattr(self._cpu_tensor, name)


def mock_tensor_to(tensor, device, *args, **kwargs):
    """Mock implementation of tensor.to() for Spyre device"""
    if isinstance(device, str) and 'spyre' in device.lower():
        if MOCK_DEVICE_ENABLED:
            _mock_print(f"[MOCK_DEVICE] Intercepted tensor.to('spyre')")
        # Keep tensor on CPU but mark as Spyre device
        return MockSpyreTensor(tensor)
    # For other devices, use original implementation
    return tensor.to(device, *args, **kwargs)


def install_mock_device_hooks():
    """Install hooks to intercept device operations"""
    if not MOCK_DEVICE_ENABLED:
        return
    
    _mock_print("[MOCK_DEVICE] Installing device operation hooks")
    
    if install_mock_torch_patches is not None:
        install_mock_torch_patches()
    
    # We'll patch torch.Tensor.to to intercept device transfers
    # This is a simplified approach - full implementation would need more hooks
    original_to = torch.Tensor.to
    
    def patched_to(self, *args, **kwargs):
        # Check if first arg is a device string containing 'spyre'
        if args and isinstance(args[0], (str, torch.device)):
            device_str = str(args[0])
            if 'spyre' in device_str.lower():
                return mock_tensor_to(self, args[0], *args[1:], **kwargs)  # type: ignore
        return original_to(self, *args, **kwargs)
    
    torch.Tensor.to = patched_to  # type: ignore
    _mock_print("[MOCK_DEVICE] Hooks installed successfully")


def register_spyre_backend_ops():
    """Register missing ATen operators for Spyre backend"""
    if not MOCK_DEVICE_ENABLED:
        return
    
    if MOCK_VERBOSE:
        _mock_print("[MOCK_DEVICE] Registering Spyre backend operators")
    
    try:
        from torch.library import impl
        
        # Register aten::empty.memory_format for spyre backend
        @impl("aten::empty.memory_format", "PrivateUse1")
        def empty_memory_format(size, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None):
            """
            Implementation of aten::empty.memory_format for Spyre backend.
            
            In mock mode, we create the tensor on CPU but wrap it in MockSpyreTensor
            to make it appear as a Spyre device tensor.
            """
            if MOCK_VERBOSE:
                _mock_print(f"[MOCK_DEVICE] aten::empty.memory_format called: size={size}, dtype={dtype}, device={device}")
            
            # Create tensor on CPU
            cpu_tensor = torch.empty(size, dtype=dtype, layout=layout, device='cpu', pin_memory=False, memory_format=memory_format)
            
            # Import here to avoid circular dependency
            try:
                from torch_spyre.mock_device_integration.mock_spyre_tensor import MockSpyreTensor
                result = MockSpyreTensor(cpu_tensor)
                if MOCK_VERBOSE:
                    _mock_print(f"[MOCK_DEVICE] Created MockSpyreTensor with shape {size}")
                return result
            except ImportError:
                # Fallback: just return CPU tensor
                if MOCK_VERBOSE:
                    _mock_print("[MOCK_DEVICE] Warning: Could not import MockSpyreTensor, returning CPU tensor")
                return cpu_tensor
        
        if MOCK_VERBOSE:
            _mock_print("[MOCK_DEVICE] Registered aten::empty.memory_format for PrivateUse1")
        
    except Exception as e:
        if MOCK_VERBOSE:
            _mock_print(f"[MOCK_DEVICE] Warning: Could not register aten::empty.memory_format: {e}")
            import traceback
            traceback.print_exc()
    
    try:
        from torch.library import impl
        
        # Also register aten::empty.out variant if needed
        @impl("aten::empty.out", "PrivateUse1")
        def empty_out(size, out=None):
            """Implementation of aten::empty.out for Spyre backend"""
            if MOCK_VERBOSE:
                _mock_print(f"[MOCK_DEVICE] aten::empty.out called: size={size}")
            
            if out is None:
                return torch.empty(size, device='cpu')
            
            # Resize the output tensor if needed
            if out.shape != torch.Size(size):
                out.resize_(size)
            
            return out
        
        if MOCK_VERBOSE:
            _mock_print("[MOCK_DEVICE] Registered aten::empty.out for PrivateUse1")
        
    except Exception as e:
        if MOCK_VERBOSE:
            _mock_print(f"[MOCK_DEVICE] Note: Could not register aten::empty.out: {e}")

    # spyre::to_dtype_cpu is defined in torch_spyre with device_types="spyre"
    # only, so it has no CPU kernel. In mock mode the compiled graph runs with
    # CPU-backed tensors, so the generated `spyre.to_dtype_cpu(x, dtype)` call
    # dispatches to the CPU backend and raises NotImplementedError. Register a
    # CPU kernel that performs the dtype conversion. This adds a backend impl for
    # the mock; it does not change torch_spyre's op definition or SDSC.
    try:
        # Ensure the op schema exists before registering an additional kernel.
        import torch_spyre._inductor.customops  # noqa: F401
        from torch.library import impl

        @impl("spyre::to_dtype_cpu", "CPU")
        def _to_dtype_cpu_cpu(input, dtype):
            return input.to(dtype=dtype)

        if MOCK_VERBOSE:
            _mock_print("[MOCK_DEVICE] Registered spyre::to_dtype_cpu for CPU backend")
    except Exception as e:
        if MOCK_VERBOSE:
            _mock_print(f"[MOCK_DEVICE] Note: Could not register spyre::to_dtype_cpu CPU kernel: {e}")


# Initialize hooks on import if mock device is enabled
if MOCK_DEVICE_ENABLED:
    install_mock_device_hooks()
    register_spyre_backend_ops()
