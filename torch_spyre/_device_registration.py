"""
Minimal device registration for mock mode
"""
import os
import sys
import torch


def _stub_copy_impl(self, src, non_blocking=False):
    """Stub implementation for copy_ operation"""
    # For mock device, just copy the data on CPU
    if hasattr(src, '_mock_cpu_tensor'):
        self.copy_(src._mock_cpu_tensor, non_blocking=non_blocking)
    else:
        self.copy_(src, non_blocking=non_blocking)
    return self


def _register_privateuse1_ops():
    """Register device operations for PrivateUse1 backend (spyre)"""
    try:
        # For the mock device, patch tensor.to() to redirect spyre allocations to CPU
        # This prevents "PyTorch is not linked with support for spyre devices" errors
        
        original_tensor_to = torch.Tensor.to
        
        def patched_tensor_to(self, *args, **kwargs):
            """Patch tensor.to() to redirect spyre device to CPU"""
            # Check if target device is spyre (from args or kwargs)
            device_arg = None
            is_spyre = False
            
            # Check positional argument
            if len(args) > 0:
                device_arg = args[0]
            # Check keyword argument
            elif 'device' in kwargs:
                device_arg = kwargs['device']
            
            if device_arg is not None:
                if isinstance(device_arg, torch.device):
                    is_spyre = (device_arg.type == 'spyre')
                elif isinstance(device_arg, str):
                    is_spyre = (device_arg == 'spyre' or device_arg.startswith('spyre:'))
            
            if is_spyre:
                # Redirect spyre device to CPU  
                # The tensor data stays on CPU, which is fine for mock mode
                if len(args) > 0:
                    # Positional argument: replace it
                    new_args = ('cpu',) + args[1:]
                    return original_tensor_to(self, *new_args, **kwargs)
                else:
                    # Keyword argument: replace in kwargs
                    new_kwargs = kwargs.copy()
                    new_kwargs['device'] = 'cpu'
                    return original_tensor_to(self, *args, **new_kwargs)
            
            # Default behavior for non-spyre devices
            return original_tensor_to(self, *args, **kwargs)
        
        # Apply the patch
        torch.Tensor.to = patched_tensor_to
        sys.stderr.write("[MOCK_DEVICE] Patched torch.Tensor.to() to redirect spyre to CPU\n")
        
        # Patch torch.accelerator functions to handle spyre device
        _patch_accelerator_functions()
        
    except Exception as e:
        sys.stderr.write(f"[MOCK_DEVICE] Note: Could not patch device operations: {e}\n")


def _patch_accelerator_functions():
    """Patch torch.accelerator functions to handle mock spyre device"""
    try:
        import torch.accelerator
        
        # Patch current_device_index to return 0 for spyre device
        original_current_device_index = torch.accelerator.current_device_index
        
        def patched_current_device_index(device_type=None):
            """Patch current_device_index to handle spyre device"""
            try:
                # If device_type is spyre, return 0 (no device index needed for mock)
                if device_type == 'spyre':
                    return 0
                # Otherwise use original implementation (which takes no args)
                return original_current_device_index()
            except RuntimeError as e:
                # If we get "not linked with support for spyre devices", return 0
                if 'spyre' in str(e):
                    return 0
                raise
        
        torch.accelerator.current_device_index = patched_current_device_index
        sys.stderr.write("[MOCK_DEVICE] Patched torch.accelerator functions for spyre device\n")
        
        # Patch current_stream to handle spyre device
        original_current_stream = torch.accelerator.current_stream
        
        def patched_current_stream(device=None):
            """Patch current_stream to handle spyre device"""
            try:
                # For spyre device, return a dummy CPU stream
                if device == 'spyre' or (isinstance(device, torch.device) and device.type == 'spyre'):
                    return torch.cuda.current_stream() if torch.cuda.is_available() else torch.Stream()
                # Otherwise use original implementation
                return original_current_stream(device)
            except RuntimeError as e:
                # If we get "not linked with support for spyre devices", return dummy stream
                if 'spyre' in str(e):
                    return torch.Stream()
                raise
        
        torch.accelerator.current_stream = patched_current_stream
        sys.stderr.write("[MOCK_DEVICE] Patched torch.accelerator.current_stream for spyre device\n")
        
    except Exception as e:
        sys.stderr.write(f"[MOCK_DEVICE] Note: Could not patch accelerator functions: {e}\n")


def register_mock_spyre_device():
    """Register spyre as a PrivateUse1 device when in mock mode"""
    mock_enabled = os.environ.get('TORCH_SPYRE_MOCK_DEVICE', '0') == '1'
    if not mock_enabled:
        return False
    
    try:
        # Check if the function exists (PyTorch 1.13+)
        if not hasattr(torch.utils, 'rename_privateuse1_backend'):
            sys.stderr.write("[MOCK_DEVICE] Warning: rename_privateuse1_backend not available (requires PyTorch 1.13+)\n")
            return False
        
        # Try to rename PrivateUse1 to 'spyre'
        # This allows torch.device("spyre") to work
        torch.utils.rename_privateuse1_backend("spyre")  # type: ignore
        sys.stderr.write("[MOCK_DEVICE] Registered 'spyre' as PrivateUse1 backend\n")
        
        # Generate standard methods for the backend
        if hasattr(torch.utils, 'generate_methods_for_privateuse1_backend'):
            try:
                torch.utils.generate_methods_for_privateuse1_backend()  # type: ignore
                sys.stderr.write("[MOCK_DEVICE] Generated methods for spyre backend\n")
            except Exception as e:
                sys.stderr.write(f"[MOCK_DEVICE] Note: Could not generate methods: {e}\n")
        
        # Register device-specific operations (including tensor.to() patching)
        _register_privateuse1_ops()
        
        return True
        
    except Exception as e:
        sys.stderr.write(f"[MOCK_DEVICE] Could not register device: {e}\n")
        return False

