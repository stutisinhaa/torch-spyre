# Copyright 2025 The Torch-Spyre Authors.
import functools
# Stub implementation of _C module for when C++ extensions are not available

"""
Stub implementation of torch_spyre._C module.
This provides minimal dummy implementations to allow imports when
the wheel is built with stub dependencies.

MVP Mock Device: This stub now includes basic device registration
to enable SDSC JSON generation without actual hardware.
"""

import warnings
import os
import types
import math
import sympy
import torch

# Check if mock device is enabled
MOCK_DEVICE_ENABLED = os.environ.get('TORCH_SPYRE_MOCK_DEVICE', '0') == '1'
MOCK_VERBOSE = os.environ.get("TORCH_SPYRE_MOCK_VERBOSE", "0") == "1"


def _mock_print(*args, **kwargs):
    if MOCK_DEVICE_ENABLED and MOCK_VERBOSE:
        print(*args, **kwargs)


# Install import patch early to suppress codegen_ops errors
if MOCK_DEVICE_ENABLED:
    try:
        import builtins
        original_import = builtins.__import__
        
        def patched_import(name, *args, **kwargs):
            try:
                return original_import(name, *args, **kwargs)
            except RuntimeError as e:
                if "codegen_ops" in name and "already a kernel registered" in str(e):
                    _mock_print(f"[MOCK_DEVICE] Suppressed codegen_ops import error")
                    import types
                    return types.ModuleType(name)
                raise
        
        builtins.__import__ = patched_import
        _mock_print("[MOCK_DEVICE] Installed import patch for codegen_ops")
    except Exception as e:
        _mock_print(f"[MOCK_DEVICE] Failed to install import patch: {e}")


# Show warning once when this stub is imported
if not MOCK_DEVICE_ENABLED:
    warnings.warn(
        "Using stub implementation of torch_spyre._C module. "
        "Full functionality requires actual IBM Spyre libraries. "
        "Set TORCH_SPYRE_MOCK_DEVICE=1 to enable mock device.",
        RuntimeWarning,
        stacklevel=2
    )
else:
    _mock_print("[MOCK_DEVICE] Spyre mock device enabled - SDSC JSON generation active")


class DataFormats:
    """Stub DataFormats enum-like class with elems_per_stick() method"""
    
    def __init__(self, value, name, elems=1):
        self._value = value
        self._name = name
        self._elems = elems
    
    def __int__(self):
        return self._value
    
    def __repr__(self):
        return f"DataFormats.{self._name}"
    
    def __eq__(self, other):
        if isinstance(other, DataFormats):
            return self._value == other._value
        return self._value == other
    
    def __hash__(self):
        return hash(self._value)
    
    @property
    def value(self):
        return self._value
    
    @property
    def name(self):
        return self._name
    
    def elems_per_stick(self):
        """Return number of elements per stick for this data format"""
        return self._elems

# Create enum-like instances
DataFormats.SEN169_FP16 = DataFormats(0, "SEN169_FP16", elems=64)
DataFormats.IEEE_FP32 = DataFormats(1, "IEEE_FP32", elems=32)
DataFormats.SEN169_FP8 = DataFormats(2, "SEN169_FP8", elems=64)
DataFormats.IEEE_INT32 = DataFormats(3, "IEEE_INT32", elems=32)
DataFormats.SENINT8 = DataFormats(4, "SENINT8", elems=128)  # NEW torch-spyre format - 128 elements per stick (1 byte each)

# Remaining members of the real torch_spyre._C.DataFormats enum.  These are
# needed so mock mode mirrors the native enum's full surface -- e.g. the SDSC
# codegen relabels 32-bit index tensors as DataFormats.SENUINT32
# (see torch_spyre/_inductor/codegen/superdsc.py).  Values must stay distinct
# from the ones above (0-4) so __eq__/__hash__ never collide; elems is the
# 128-byte stick capacity (128 / bytes-per-element) where meaningful.
DataFormats.INVALID = DataFormats(5, "INVALID", elems=1)
DataFormats.SEN143_FP8 = DataFormats(6, "SEN143_FP8", elems=128)
DataFormats.SEN152_FP8 = DataFormats(7, "SEN152_FP8", elems=128)
DataFormats.SEN153_FP9 = DataFormats(8, "SEN153_FP9", elems=64)
DataFormats.SENINT2 = DataFormats(9, "SENINT2", elems=512)
DataFormats.SENINT4 = DataFormats(10, "SENINT4", elems=256)
DataFormats.SENINT16 = DataFormats(11, "SENINT16", elems=64)
DataFormats.SENINT24 = DataFormats(12, "SENINT24", elems=32)
DataFormats.IEEE_INT64 = DataFormats(13, "IEEE_INT64", elems=16)
DataFormats.SENUINT32 = DataFormats(14, "SENUINT32", elems=32)
DataFormats.SENUINT2 = DataFormats(15, "SENUINT2", elems=512)
DataFormats.IEEE_FP16 = DataFormats(16, "IEEE_FP16", elems=64)
DataFormats.BOOL = DataFormats(17, "BOOL", elems=128)
DataFormats.BFLOAT16 = DataFormats(18, "BFLOAT16", elems=64)
DataFormats.SEN18F_FP24 = DataFormats(19, "SEN18F_FP24", elems=32)

class ElementArrangement:
    """Stub ElementArrangement enum for NEW torch-spyre (May 27+)"""
    
    def __init__(self, value, name):
        self._value = value
        self._name = name
    
    def __int__(self):
        return self._value
    
    def __repr__(self):
        return f"ElementArrangement.{self._name}"
    
    def __eq__(self, other):
        if isinstance(other, ElementArrangement):
            return self._value == other._value
        return self._value == other
    
    def __hash__(self):
        return hash(self._value)
    
    @property
    def value(self):
        return self._value
    
    @property
    def name(self):
        return self._name

# Create enum-like instances for ElementArrangement.
# Values must match the real C++ enum (see torch_spyre/csrc/spyre_tensor_impl.h
# and module.cpp): STANDARD=0, DL16_TO_FP32=1, QFP8CH=2, EXX2=3.
ElementArrangement.STANDARD = ElementArrangement(0, "STANDARD")
ElementArrangement.DL16_TO_FP32 = ElementArrangement(1, "DL16_TO_FP32")
ElementArrangement.QFP8CH = ElementArrangement(2, "QFP8CH")
ElementArrangement.EXX2 = ElementArrangement(3, "EXX2")



class SpyreTensorLayout:
    """Stub SpyreTensorLayout class with required attributes for inductor"""
    element_arrangement = 0  # STANDARD (class-level fallback)
    
    def __setattr__(self, name, value):
        """Intercept device_size assignments to fix stick dimension"""
        if name == 'device_size' and value is not None:
            # Fix stick dimension if it's too small
            if hasattr(value, '__iter__') and len(value) > 0:
                if hasattr(self, 'device_dtype') and hasattr(self.device_dtype, 'elems_per_stick'):
                    expected_stick_size = self.device_dtype.elems_per_stick()
                    if value[-1] < expected_stick_size:
                        # Fix the stick size
                        was_tuple = isinstance(value, tuple)
                        size_list = list(value)
                        size_list[-1] = expected_stick_size
                        value = tuple(size_list) if was_tuple else size_list
                        if os.environ.get("TORCH_SPYRE_MOCK_DEVICE", "0") == "1":
                            _mock_print(f"[MOCK_DEVICE][SETATTR] Fixed device_size stick dimension from {list(value)[:-1] + [value[-1] if isinstance(value, (list, tuple)) else value]} to {value}")
        # Use object.__setattr__ to avoid recursion
        object.__setattr__(self, name, value)
    
    def __init__(self, *args, **kwargs):
        # Handle 3 calling patterns based on number of args
        if MOCK_DEVICE_ENABLED:
            _mock_print(f"[MOCK_DEVICE] SpyreTensorLayout.__init__ called with {len(args)} args: {args}")
        
        # Initialize dim_order to None by default
        self.dim_order = None
        
        if len(args) == 2:
            # Pattern 1: SpyreTensorLayout(size, dtype)
            host_size = list(args[0]) if hasattr(args[0], '__iter__') else [args[0]]

            device_dtype = get_device_dtype(args[1])
            self.device_dtype = device_dtype
            self.data_format = device_dtype

            stick_size = device_dtype.elems_per_stick()

            if len(host_size) == 0:
                # Scalar: ensure at least one valid stride for the stick dimension
                self.device_size = [1, stick_size]
                self.dim_map = [-1, -1]
                self.stride_map = [stick_size, 1]  # Valid strides instead of all -1
            elif len(host_size) == 1:
                self.device_size = [math.ceil(host_size[0] / stick_size), stick_size]
                self.dim_map = [0, 0]
                self.stride_map = [stick_size, 1]
            else:
                host_stride = []
                stride = 1
                for i in range(len(host_size) - 1, -1, -1):
                    host_stride.insert(0, stride)
                    stride *= host_size[i]

                stick_dim = len(host_size) - 1
                outer = math.ceil(host_size[stick_dim] / stick_size)
                prefix_dims = [host_size[i] for i in range(len(host_size) - 1)]

                if len(host_size) == 2:
                    self.device_size = [outer] + prefix_dims + [stick_size]
                    self.dim_map = [stick_dim, 0, stick_dim]
                    self.stride_map = [stick_size, host_stride[0], 1]
                else:
                    leading_dims = host_size[:-1]
                    self.device_size = [leading_dims[-1], outer] + list(reversed(leading_dims[:-1])) + [stick_size]
                    self.dim_map = [len(host_size) - 2, stick_dim] + list(range(len(host_size) - 3, -1, -1)) + [stick_dim]
                    self.stride_map = [host_stride[len(host_size) - 2], stick_size, *[host_stride[idx] for idx in range(len(host_size) - 3, -1, -1)], 1]
        
        elif len(args) == 3:
            # Pattern C: SpyreTensorLayout(device_size, stride_map, device_dtype)
            # This is used when creating layouts from existing device layouts
            if MOCK_DEVICE_ENABLED:
                _mock_print(f"[MOCK_DEVICE] Pattern C (3-arg): device_size={args[0]}, stride_map={args[1]}, device_dtype={args[2]}")
            
            # CRITICAL: Ensure device_size is NEVER empty - this causes IndexError crashes
            raw_device_size = args[0]
            if not hasattr(raw_device_size, '__iter__') or len(raw_device_size) == 0:
                # Empty device_size - create minimal valid layout with stick dimension
                stick_size = args[2].elems_per_stick() if hasattr(args[2], 'elems_per_stick') else 64
                self.device_size = (1, stick_size)  # Minimal 2D layout
                self.stride_map = [stick_size, 1]
                self.dim_map = [0, 0]
                if MOCK_DEVICE_ENABLED:
                    _mock_print(f"[MOCK_DEVICE] Pattern C: Fixed empty device_size to {self.device_size}")
            else:
                self.device_size = tuple(raw_device_size)
                self.stride_map = list(args[1]) if hasattr(args[1], '__iter__') and not isinstance(args[1], str) else []
                self.dim_map = list(range(len(self.device_size)))
            
            self.device_dtype = args[2]
            self.data_format = args[2]
            
            if MOCK_DEVICE_ENABLED:
                _mock_print(f"[MOCK_DEVICE] Pattern C computed: device_size={self.device_size}, dim_map={self.dim_map}, stride_map={self.stride_map}")
            
            
        elif len(args) == 4:
            # Two possible patterns - detect by checking if arg[3] is already a DataFormats
            if isinstance(args[3], DataFormats):
                # Pattern B: SpyreTensorLayout(device_size, dim_map, stride_map, device_dtype)
                if MOCK_DEVICE_ENABLED:
                    _mock_print(f"[MOCK_DEVICE] Pattern B: device_size={args[0]}, dim_map={args[1]}, stride_map={args[2]}, device_dtype={args[3]}")
                self.device_size = tuple(args[0]) if hasattr(args[0], '__iter__') else (args[0],)
                # dim_map should always be a list (matches C++ std::vector<int32_t>)
                self.dim_map = list(args[1]) if hasattr(args[1], '__iter__') else []
                # stride_map should always be a list (matches C++ std::vector<int32_t>)
                self.stride_map = list(args[2]) if hasattr(args[2], '__iter__') and not isinstance(args[2], str) else []
                self.device_dtype = args[3]
                self.data_format = args[3]
            else:
                # Pattern A: SpyreTensorLayout(size, stride, dtype, dim_order)
                # This matches C++ init(host_size, host_strides, dtype, dim_order) at line 141
                if MOCK_DEVICE_ENABLED:
                    _mock_print(f"[MOCK_DEVICE] Pattern A: size={args[0]}, stride={args[1]}, dtype={args[2]}, dim_order={args[3]}")
                
                host_size = list(args[0]) if hasattr(args[0], '__iter__') else [args[0]]
                host_stride = list(args[1]) if hasattr(args[1], '__iter__') else [args[1]]
                dim_order = args[3] if isinstance(args[3], list) else list(range(len(host_size)))
                
                # Store dim_order for compatibility checking
                self.dim_order = dim_order
                
                # Convert torch dtype to device dtype
                device_dtype = get_device_dtype(args[2])
                self.device_dtype = device_dtype
                self.data_format = device_dtype
                
                stick_size = device_dtype.elems_per_stick()

                if len(host_size) == 0 or len(dim_order) == 0:
                    # Scalar tensor (e.g. a full reduction to a scalar). There is
                    # no host/stick dimension to lay out; produce the minimal
                    # valid scalar layout matching the 2-arg scalar path so a
                    # within-stick coordinate always exists.
                    self.device_size = [1, stick_size]
                    self.dim_map = [-1, -1]
                    self.stride_map = [stick_size, 1]
                    if MOCK_DEVICE_ENABLED:
                        _mock_print(f"[MOCK_DEVICE] Pattern A scalar: device_size={self.device_size}")
                    return

                sparse = dim_order[-1] == -1

                if sparse:
                    logical_order = list(dim_order[:-1])
                    if len(host_size) == 0 or len(logical_order) == 0:
                        self.device_size = [1, stick_size]
                        self.dim_map = [-1, -1]
                        self.stride_map = [-1, 1]
                    else:
                        self.device_size = [host_size[logical_order[-1]], 1] + [host_size[d] for d in logical_order[:-1]] + [stick_size]
                        self.dim_map = [logical_order[-1], -1] + logical_order[:-1] + [-1]
                        self.stride_map = [host_stride[logical_order[-1]], 1] + [host_stride[d] for d in logical_order[:-1]] + [1]
                else:
                    stick_dim = dim_order[-1]
                    outer = math.ceil(host_size[stick_dim] / stick_size)
                    nonstick_dims = list(dim_order[:-1])

                    if len(host_size) == 1:
                        self.device_size = [outer, stick_size]
                        self.dim_map = [stick_dim, stick_dim]
                        self.stride_map = [stick_size, 1]
                    elif len(host_size) == 2:
                        self.device_size = [outer, host_size[nonstick_dims[0]], stick_size]
                        self.dim_map = [stick_dim, nonstick_dims[0], stick_dim]
                        self.stride_map = [stick_size, host_stride[nonstick_dims[0]], 1]
                    else:
                        # Lay out device dims in the same canonical order as the
                        # 2-arg constructor (leading-host-dim major):
                        #   [host[nonstick[-1]], stick_blocks, *reversed(nonstick[:-1]),
                        #    stick_size]
                        # The previous form put `outer` first and only handled exactly
                        # two non-stick dims, producing (a) a dim order inconsistent with
                        # the 2-arg constructor and (b) a stick-block stride of
                        # stick_size*host_size[stick_dim] instead of
                        # stick_size*host_stride[stick_dim] -- the latter yields degenerate
                        # device coordinates (e.g. floor(M*c/64)) for a batched matmul
                        # whose M dimension is not stick-aligned.
                        rest = list(reversed(nonstick_dims[:-1]))
                        self.device_size = (
                            [host_size[nonstick_dims[-1]], outer]
                            + [host_size[d] for d in rest]
                            + [stick_size]
                        )
                        self.dim_map = (
                            [nonstick_dims[-1], stick_dim] + rest + [stick_dim]
                        )
                        self.stride_map = (
                            [host_stride[nonstick_dims[-1]], stick_size * host_stride[stick_dim]]
                            + [host_stride[d] for d in rest]
                            + [1]
                        )

                if MOCK_DEVICE_ENABLED:
                    _mock_print(f"[MOCK_DEVICE] Pattern A computed: dim_map={self.dim_map}, device_size={self.device_size}, stride_map={self.stride_map}")
            
        else:
            # Pattern 3: Named parameters (fallback)
            # torch_spyre passes the dtype as `device_dtype`; older callers use
            # `data_format`. Honor either, defaulting to IEEE_FP32.
            data_format = kwargs.get('device_dtype', kwargs.get('data_format', DataFormats.IEEE_FP32))
            self.device_dtype = data_format
            self.data_format = data_format

            raw_device_size = kwargs.get('device_size', ())
            if not hasattr(raw_device_size, '__iter__') or len(raw_device_size) == 0:
                # Empty device_size (scalar layout, e.g. a full reduction like
                # torch.amax over all dims). Mirror the positional scalar path
                # (Pattern 1, len(host_size) == 0) so device_coordinates() has a
                # valid stick dimension instead of crashing on coords[-1].
                stick_size = data_format.elems_per_stick() if hasattr(data_format, 'elems_per_stick') else 64
                self.device_size = [1, stick_size]
                self.dim_map = [-1, -1]
                self.stride_map = [stick_size, 1]
                if MOCK_DEVICE_ENABLED:
                    _mock_print(f"[MOCK_DEVICE] Pattern 3: Fixed empty device_size to {self.device_size} (scalar layout)")
            else:
                self.device_size = raw_device_size
                self.dim_map = kwargs.get('dim_map', {})
                self.stride_map = kwargs.get('stride_map', [])
        
        # UNIVERSAL EMPTY-LAYOUT GUARD: a scalar tensor (e.g. the result of a
        # full reduction like torch.amax over all dims) produces an empty
        # device_size on every construction path (Pattern B/C positional and the
        # kwargs fallback). torch_spyre then indexes coords[-1] / in_layout.size[-1]
        # and crashes with IndexError. Normalize any empty device_size to a
        # minimal valid scalar layout so a within-stick coordinate always exists.
        if not hasattr(self, 'device_size') or self.device_size is None \
                or not hasattr(self.device_size, '__iter__') or len(self.device_size) == 0:
            dtype = getattr(self, 'device_dtype', DataFormats.IEEE_FP32)
            stick_size = dtype.elems_per_stick() if hasattr(dtype, 'elems_per_stick') else 64
            self.device_size = [1, stick_size]
            self.dim_map = [-1, -1]
            self.stride_map = [stick_size, 1]
            if MOCK_DEVICE_ENABLED:
                _mock_print(f"[MOCK_DEVICE] Normalized empty device_size to {self.device_size} (scalar layout)")

        # CRITICAL FIX: Ensure at least one dimension has BOTH stride > 0 AND size > 1
        # compute_coordinates() ignores dimensions where stride <= 0 OR size == 1
        # We need at least one valid dimension to prevent empty coordinate lists
        if hasattr(self, 'stride_map') and hasattr(self, 'device_size'):
            if isinstance(self.stride_map, list) and len(self.stride_map) > 0:
                # Check if any dimension has both stride > 0 and size > 1
                has_valid_dim = False
                for i, (stride, size) in enumerate(zip(self.stride_map, self.device_size)):
                    if stride > 0 and size > 1:
                        has_valid_dim = True
                        break
                
                if not has_valid_dim:
                    # No valid dimension found - fix the last dimension (stick dim)
                    # Ensure it has stride > 0 and size > 1
                    if self.stride_map[-1] <= 0:
                        self.stride_map[-1] = 1
                    if self.device_size[-1] <= 1:
                        # Convert to list if tuple, modify, keep as tuple if it was tuple
                        was_tuple = isinstance(self.device_size, tuple)
                        size_list = list(self.device_size)
                        # Use the actual stick size from device_dtype, not hardcoded 2
                        stick_size = self.device_dtype.elems_per_stick() if hasattr(self.device_dtype, 'elems_per_stick') else 64
                        size_list[-1] = stick_size
                        # CRITICAL: Actually assign the fixed size back!
                        self.device_size = tuple(size_list) if was_tuple else size_list
                    
                    if MOCK_DEVICE_ENABLED:
                        _mock_print(f"[MOCK_DEVICE] Fixed layout to ensure valid coordinates: stride_map[-1]={self.stride_map[-1]}, device_size[-1]={self.device_size[-1]}")
        
        if MOCK_DEVICE_ENABLED:
            _mock_print(f"[MOCK_DEVICE] SpyreTensorLayout created: device_size={self.device_size}, stride_map={getattr(self, 'stride_map', 'N/A')}")
    
    def _dim_map_to_stride_map(self, dim_map, host_size, host_stride, device_size):
        """
        Python port of C++ dim_map_to_stride_map from spyre_tensor_impl.cpp lines 111-131.
        
        Args:
            dim_map: List of dimension mappings from device to host
            host_size: Host tensor dimensions
            host_stride: Host tensor strides
            device_size: Device tensor dimensions (with stick tiling)
        
        Returns:
            List of stride values (INCLUDING -1 values, matching C++ behavior)
        """
        if len(dim_map) == 0:
            return []
        
        n = len(dim_map)
        stride_map = [-1] * n
        last_stride = [-1] * n
        
        for j in range(n - 1, -1, -1):
            d = dim_map[j]
            if d == -1 or host_size[d] == 1:
                stride_map[j] = -1
            elif last_stride[d] == -1:
                stride_map[j] = host_stride[d]
                last_stride[d] = stride_map[j] * device_size[j]
            else:
                stride_map[j] = last_stride[d]
                last_stride[d] = stride_map[j] * device_size[j]
        
        # Return the full stride_map including -1 values (matches C++ line 130)
        # IMPORTANT: Ensure at least the last element (stick dimension) has a valid stride
        # to prevent empty coordinate lists in compute_coordinates()
        if all(s == -1 for s in stride_map) and n > 0:
            # All strides are -1, set the last one (stick dim) to 1
            stride_map[-1] = 1
        return stride_map
    
    def elems_per_stick(self):
        return self.device_dtype.elems_per_stick()

    def host_stick_dim(self):
        if not self.dim_map:
            return None
        if self.dim_map[-1] == -1:
            return None
        return self.dim_map[-1]

    def __eq__(self, other):
        return (
            isinstance(other, SpyreTensorLayout)
            and list(self.device_size) == list(other.device_size)
            and list(self.dim_map) == list(other.dim_map)
            and list(self.stride_map) == list(other.stride_map)
            and self.device_dtype == other.device_dtype
        )
    def __hash__(self):
        """Make SpyreTensorLayout hashable so it can be used as dict key"""
        return hash((
            tuple(self.device_size),
            tuple(self.dim_map),
            tuple(self.stride_map),
            self.device_dtype
        ))


    def __str__(self):
        return (
            f"SpyreTensorLayout(device_size={list(self.device_size)}, "
            f"dim_map ={list(self.dim_map)}, "
            f"stride_map ={list(self.stride_map)}, "
            f"device_dtype={self.device_dtype})"
        )

    def __repr__(self):
        return self.__str__()


def get_spyre_tensor_layout(tensor):
    """Return a mock SpyreTensorLayout for mock spyre tensors."""
    layout = getattr(tensor, "_mock_device_layout", None)
    if layout is not None:
        return layout
    try:
        return SpyreTensorLayout(list(tensor.shape), tensor.dtype)
    except Exception:
        return None


def to_with_layout(tensor, layout):
    """Create a mock spyre tensor and preserve the requested layout."""
    from torch_spyre.mock_device_integration.mock_spyre_tensor import MockSpyreTensor

    result = MockSpyreTensor(tensor)
    try:
        object.__setattr__(result, "_mock_device_layout", layout)
    except Exception:
        pass
    return result


def empty_with_layout(*args, **kwargs):
    """Create a mock spyre tensor and preserve the requested layout."""
    import torch
    from torch_spyre.mock_device_integration.mock_spyre_tensor import MockSpyreTensor

    size = args[0] if args else kwargs.get("size")
    layout = args[1] if len(args) > 1 else kwargs.get("device_layout")
    dtype = args[2] if len(args) > 2 else kwargs.get("dtype", torch.float32)

    result = MockSpyreTensor(torch.empty(size, dtype=dtype))
    try:
        object.__setattr__(result, "_mock_device_layout", layout)
    except Exception:
        pass
    return result


def spyre_empty_with_layout(*args, **kwargs):
    """
    Stub for spyre_empty_with_layout.
    Real signature: spyre_empty_with_layout(size, stride, dtype, layout)
    """
    import torch
    if len(args) >= 3:
        # args[0] = size, args[1] = stride, args[2] = dtype, args[3] = layout
        size, stride, dtype = args[0], args[1], args[2]
        layout = args[3] if len(args) > 3 else None
        if MOCK_DEVICE_ENABLED:
            _mock_print(
                f"[MOCK_DEVICE] spyre_empty_with_layout: size={size}, stride={stride}, dtype={dtype}"
            )
        from torch_spyre.mock_device_integration.mock_spyre_tensor import MockSpyreTensor

        result = MockSpyreTensor(torch.empty(size, dtype=dtype))
        # Preserve requested logical view metadata from empty_strided allocation.
        try:
            result._mock_logical_shape = tuple(size)
            result._mock_logical_stride = tuple(stride)
            result._mock_storage_offset = 0
            if layout is not None:
                result._mock_device_layout = layout
        except Exception:
            pass
        return result
    else:
        # Fallback for unexpected call pattern
        return torch.empty(*args[:1])


def reinterpret_tensor(tensor, size, stride, storage_offset):
    """
    Reinterpret a tensor with new size and stride (view operation).
    This is used by the wrapper to create views like squeeze/unsqueeze/permute.
    """
    if MOCK_DEVICE_ENABLED:
        _mock_print(f"[MOCK_DEVICE] reinterpret_tensor: input type={type(tensor).__name__}, {tuple(tensor.shape)} -> {tuple(size)}, stride={tuple(stride)}, offset={storage_offset}")
    
    # Use as_strided to create a view with the new shape and stride
    result = torch.as_strided(tensor, size, stride, storage_offset)
    
    if MOCK_DEVICE_ENABLED:
        _mock_print(f"[MOCK_DEVICE] reinterpret_tensor: result type={type(result).__name__}, device={result.device}")
    
    # In mock mode, always wrap results in MockSpyreTensor
    if MOCK_DEVICE_ENABLED:
        from torch_spyre.mock_device_integration.mock_spyre_tensor import MockSpyreTensor
        if isinstance(result, torch.Tensor) and not isinstance(result, MockSpyreTensor):
            # Wrap in MockSpyreTensor to preserve device attribute
            _mock_print(f"[MOCK_DEVICE] reinterpret_tensor: wrapping result in MockSpyreTensor")
            mock_result = MockSpyreTensor.__new__(MockSpyreTensor, result)
            mock_result._mock_device = torch.device("spyre", 0)
            # Preserve logical view metadata from as_strided reinterpretation.
            mock_result._mock_logical_shape = tuple(size)
            mock_result._mock_logical_stride = tuple(stride)
            mock_result._mock_storage_offset = int(storage_offset)
            # Preserve layout if input had one
            if hasattr(tensor, '_mock_device_layout'):
                mock_result._mock_device_layout = tensor._mock_device_layout
            _mock_print(f"[MOCK_DEVICE] reinterpret_tensor: wrapped result device={mock_result.device}")
            return mock_result
    
    return result


def reinterpret_tensor_with_layout(tensor, size, stride, storage_offset, layout):
    """
    Reinterpret a tensor with new size, stride, and layout (view operation).
    This is used by the wrapper to create views with specific device layouts.
    """
    if MOCK_DEVICE_ENABLED:
        _mock_print(f"[MOCK_DEVICE] reinterpret_tensor_with_layout: {tuple(tensor.shape)} -> {tuple(size)}, stride={tuple(stride)}, offset={storage_offset}, layout={layout}")
    
    # Use as_strided to create a view with the new shape and stride
    result = torch.as_strided(tensor, size, stride, storage_offset)
    
    # Preserve mock device attributes if present
    if hasattr(tensor, '_mock_device'):
        from torch_spyre.mock_device_integration.mock_spyre_tensor import MockSpyreTensor
        if isinstance(result, torch.Tensor) and not isinstance(result, MockSpyreTensor):
            # Wrap in MockSpyreTensor to preserve attributes
            mock_result = MockSpyreTensor.__new__(MockSpyreTensor, result)
            mock_result._mock_device = tensor._mock_device
            mock_result._mock_logical_shape = tuple(size)
            mock_result._mock_logical_stride = tuple(stride)
            mock_result._mock_storage_offset = int(storage_offset)
            if hasattr(tensor, '_mock_device_layout'):
                mock_result._mock_device_layout = layout
            return mock_result
    
    return result


def encode_constant(value, format):
    """Stub function"""
    return value


def convert_artifacts(*args, **kwargs):
    """Stub function"""
    pass


def prepare_kernel(*args, **kwargs):
    """Stub for prepare_kernel (NEW torch-spyre May 27+)"""
    _mock_print("[MOCK_DEVICE] prepare_kernel called (stub)")
    return None

def launch_jobplan(*args, **kwargs):
    """Stub for launch_jobplan (NEW torch-spyre May 27+)"""
    _mock_print("[MOCK_DEVICE] launch_jobplan called (stub)")
    return None

def launch_kernel(*args, **kwargs):
    """Stub function"""
    raise NotImplementedError(
        "launch_kernel requires actual torch_spyre._C module. "
        "Rebuild with IBM Spyre libraries for full functionality."
    )


_mock_current_device_index = 0
_mock_stream_state = {
    "next_id": 1,
    "default_streams": {},
    "current_streams": {},
}


class _MockStreamHandle:
    def __init__(self, device, stream_id, priority=0, is_default=False):
        self._device = torch.device(device)
        self._id = int(stream_id)
        self._priority = int(priority)
        self._is_default = bool(is_default)
        self._complete = True

    def synchronize(self):
        self._complete = True
        return None

    def query(self):
        return True

    @property
    def device(self):
        return self._device

    @property
    def id(self):
        return self._id

    @property
    def priority(self):
        return self._priority

    def __repr__(self):
        return (
            f"torch_spyre.Stream(device='{self._device}', "
            f"id={self._id}, priority={self._priority})"
        )


def _normalize_mock_device(device=None):
    if device is None:
        return torch.device("spyre", _mock_current_device_index)
    if isinstance(device, int):
        return torch.device("spyre", device)
    if isinstance(device, str):
        return torch.device(device)
    return torch.device(device)


def _device_key(device):
    dev = _normalize_mock_device(device)
    return (dev.type, 0 if dev.index is None else dev.index)


def default_stream(device=None):
    dev = _normalize_mock_device(device)
    key = _device_key(dev)
    stream = _mock_stream_state["default_streams"].get(key)
    if stream is None:
        stream = _MockStreamHandle(dev, stream_id=0, priority=0, is_default=True)
        _mock_stream_state["default_streams"][key] = stream
    _mock_stream_state["current_streams"].setdefault(key, stream)
    return stream


def current_stream(device=None):
    dev = _normalize_mock_device(device)
    key = _device_key(dev)
    stream = _mock_stream_state["current_streams"].get(key)
    if stream is None:
        stream = default_stream(dev)
    return stream


def get_stream_from_pool(device=None, priority=0):
    dev = _normalize_mock_device(device)
    stream_id = _mock_stream_state["next_id"]
    _mock_stream_state["next_id"] += 1
    return _MockStreamHandle(dev, stream_id=stream_id, priority=priority)


def set_current_stream(stream):
    if stream is None:
        return None
    key = _device_key(stream.device())
    _mock_stream_state["current_streams"][key] = stream
    return None


def synchronize(device=None):
    if device is None:
        for stream in list(_mock_stream_state["current_streams"].values()):
            stream.synchronize()
        for stream in list(_mock_stream_state["default_streams"].values()):
            stream.synchronize()
        return None
    current_stream(device).synchronize()
    return None


def current_device():
    return _mock_current_device_index


def set_device(index):
    global _mock_current_device_index
    _mock_current_device_index = int(index)
    default_stream(torch.device("spyre", _mock_current_device_index))
    return None


def _patch_torch_stream_api_for_mock():
    if not MOCK_DEVICE_ENABLED:
        return
    if getattr(torch, "_spyre_mock_stream_patched", False):
        return

    from torch_spyre.streams import Stream as _SpyreStream

    class _MockTorchStream(_SpyreStream):
        pass

    torch.Stream = _MockTorchStream
    torch._streambase = getattr(torch, "_streambase", None)
    torch._spyre_mock_stream_patched = True
    _mock_print("[MOCK_DEVICE] Patched torch.Stream for mock spyre backend")


def _patch_torch_accelerator_for_mock():
    """Patch torch.accelerator APIs that otherwise call into missing C++ spyre support."""
    if not MOCK_DEVICE_ENABLED:
        return

    try:
        import torch
    except Exception as e:
        _mock_print(f"[MOCK_DEVICE] Warning: could not import torch for accelerator patching: {e}")
        return

    accelerator = getattr(torch, "accelerator", None)
    if accelerator is None:
        return

    if getattr(accelerator, "_spyre_mock_patched", False):
        return

    def _mock_current_device_index():
        return 0

    def _mock_current_stream(device=None):
        try:
            import torch_spyre
            if hasattr(torch_spyre, "current_stream"):
                return torch_spyre.current_stream(device)
        except Exception:
            pass
        return current_stream(device)

    def _mock_set_device_index(device=None):
        return None

    def _mock_synchronize(device=None):
        return synchronize(device)

    accelerator.current_device_index = _mock_current_device_index
    accelerator.current_stream = _mock_current_stream
    accelerator.set_device_index = _mock_set_device_index
    accelerator.synchronize = _mock_synchronize
    accelerator._spyre_mock_patched = True
    _mock_print("[MOCK_DEVICE] Patched torch.accelerator APIs for mock spyre backend")


def _patch_fake_tensor_ops_for_mock():
    """Patch selected fake-tensor execution paths to use CPU semantics in mock Spyre mode."""
    if not MOCK_DEVICE_ENABLED:
        return

    try:
        import operator
        from torch._dynamo import utils as dynamo_utils
        import torch
        import torch._subclasses.fake_tensor as fake_tensor_mod
        from torch._subclasses.fake_tensor import FakeTensor
        from torch._subclasses.functional_tensor import FunctionalTensor
        from torch.fx import Interpreter
    except Exception as e:
        _mock_print(f"[MOCK_DEVICE] Warning: could not import fake-tensor utilities for patching: {e}")
        return

    if getattr(dynamo_utils, "_spyre_mock_patched", False):
        return

    try:
        full_default = torch.ops.spyre.full.default
    except Exception:
        full_default = None
    else:
        if not getattr(full_default, "_spyre_mock_cpu_wrapped", False):
            original_full_default = full_default

            @functools.wraps(original_full_default)
            def _spyre_full_default_with_mock_cpu_fallback(*args, **kwargs):
                device = None
                if len(args) > 2:
                    device = args[2]
                elif isinstance(kwargs, dict):
                    device = kwargs.get("device")

                if getattr(device, "type", None) == "cpu":
                    size = args[0] if len(args) > 0 else kwargs.get("size")
                    fill_value = args[1] if len(args) > 1 else kwargs.get("fill_value")
                    dtype = args[3] if len(args) > 3 else kwargs.get("dtype")
                    if MOCK_DEVICE_ENABLED:
                        _mock_print("[MOCK_DEVICE] direct CPU fallback for torch.ops.spyre.full.default(..., device=cpu)")
                    return torch.full(size, fill_value, dtype=dtype, device="cpu")

                return original_full_default(*args, **kwargs)

            setattr(_spyre_full_default_with_mock_cpu_fallback, "_spyre_mock_cpu_wrapped", True)
            torch.ops.spyre.full.default = _spyre_full_default_with_mock_cpu_fallback

    original_run_node = dynamo_utils.run_node
    original_find_common_device = FakeTensor._find_common_device
    original_interpreter_call_function = Interpreter.call_function
    original_fake_tensor_torch_function = getattr(FakeTensor, "__torch_function__", None)
    try:
        import torch._functorch._aot_autograd.functional_utils as aot_functional_utils
        import torch._functorch._aot_autograd.runtime_wrappers as aot_runtime_wrappers
    except Exception:
        aot_functional_utils = None
        aot_runtime_wrappers = None
        original_gen_alias_from_base = None
    else:
        original_gen_alias_from_base = aot_functional_utils.gen_alias_from_base

    def _is_mock_spyre_tensor(obj):
        # Primary check: FakeTensor with spyre device (during compilation/tracing).
        if isinstance(obj, FakeTensor) and getattr(getattr(obj, "device", None), "type", None) == "spyre":
            return True
        # Also catch real MockSpyreTensor during execution (has no fake_mode).
        # Use type name to avoid importing MockSpyreTensor (circular import risk).
        if (type(obj).__name__ == "MockSpyreTensor"
                or (isinstance(obj, torch.Tensor)
                    and not isinstance(obj, FakeTensor)
                    and getattr(getattr(obj, "device", None), "type", None) == "spyre")):
            return True
        return False

    def _logical_shape(obj):
        shape = getattr(obj, "_mock_logical_shape", None)
        if shape is not None:
            return tuple(shape)
        try:
            return tuple(obj.shape)
        except Exception:
            return None

    def _contains_mock_spyre_tensor(obj):
        if _is_mock_spyre_tensor(obj):
            return True
        if isinstance(obj, FunctionalTensor):
            try:
                inner = obj.from_functional()
            except Exception:
                inner = None
            return _contains_mock_spyre_tensor(inner)
        if isinstance(obj, (tuple, list)):
            return any(_contains_mock_spyre_tensor(item) for item in obj)
        if isinstance(obj, dict):
            return any(_contains_mock_spyre_tensor(item) for item in obj.values())
        return False

    def _to_cpu_tree(obj):
        if _is_mock_spyre_tensor(obj):
            return obj.to(device="cpu")
        if isinstance(obj, FunctionalTensor):
            inner = obj.from_functional()
            return _to_cpu_tree(inner)
        if isinstance(obj, tuple):
            return tuple(_to_cpu_tree(item) for item in obj)
        if isinstance(obj, list):
            return [_to_cpu_tree(item) for item in obj]
        if isinstance(obj, dict):
            return {key: _to_cpu_tree(value) for key, value in obj.items()}
        return obj

    def _describe_mock_result(obj):
        if isinstance(obj, torch.Tensor):
            return (
                f"type={type(obj).__name__} "
                f"device={getattr(getattr(obj, 'device', None), 'type', None)} "
                f"shape={tuple(obj.shape)} "
                f"logical_shape={getattr(obj, '_mock_logical_shape', None)} "
                f"logical_stride={getattr(obj, '_mock_logical_stride', None)} "
                f"logical_offset={getattr(obj, '_mock_storage_offset', None)}"
            )
        if isinstance(obj, (tuple, list)):
            return type(obj).__name__ + "[" + ", ".join(_describe_mock_result(v) for v in obj) + "]"
        if isinstance(obj, dict):
            return "dict{" + ", ".join(f"{k}: {_describe_mock_result(v)}" for k, v in obj.items()) + "}"
        return repr(obj)

    def _to_spyre_tree(obj):
        if isinstance(obj, tuple):
            return tuple(_to_spyre_tree(item) for item in obj)
        if isinstance(obj, list):
            return [_to_spyre_tree(item) for item in obj]
        if isinstance(obj, dict):
            return {key: _to_spyre_tree(value) for key, value in obj.items()}
        if isinstance(obj, torch.Tensor) and getattr(getattr(obj, "device", None), "type", None) == "cpu":
            if type(obj).__name__ == "FakeTensor":
                if os.environ.get("TORCH_SPYRE_MOCK_DEBUG_VIEW", "0") == "1":
                    _mock_print("[MOCK_DEVICE] _to_spyre_tree leaving FakeTensor on cpu to avoid invalid subclass rewrap")
                return obj
            from torch_spyre.mock_device_integration.mock_spyre_tensor import MockSpyreTensor
            wrapped = MockSpyreTensor(obj)
            if os.environ.get("TORCH_SPYRE_MOCK_DEBUG_VIEW", "0") == "1":
                _mock_print(f"[MOCK_DEVICE] _to_spyre_tree wrapped {_describe_mock_result(wrapped)}")
            return wrapped
        return obj

    def _run_node_with_cpu_fallback(tracer, node, args, kwargs, nnmodule):
        target = getattr(node, "target", None)
        if node.op == "call_method" and target == "copy_" and (
            _contains_mock_spyre_tensor(args) or _contains_mock_spyre_tensor(kwargs)
        ):
            if MOCK_DEVICE_ENABLED:
                _mock_print("[MOCK_DEVICE] Dynamo fake-tensor direct fallback for method copy_ on spyre")
            dst = args[0]
            src = args[1] if len(args) > 1 else None
            if _is_mock_spyre_tensor(dst) and src is not None:
                try:
                    object.__setattr__(dst, "_mock_spyre_copy_src", src)
                except Exception:
                    pass
            return dst

        if node.op == "call_method" and target == "matmul" and (
            _contains_mock_spyre_tensor(args) or _contains_mock_spyre_tensor(kwargs)
        ):
            if MOCK_DEVICE_ENABLED:
                _mock_print("[MOCK_DEVICE] Dynamo fake-tensor direct CPU fallback for method matmul on spyre")
            lhs = args[0]
            rhs = args[1] if len(args) > 1 else None
            cpu_kwargs = _to_cpu_tree(kwargs)

            def _maybe_mock_shape(x):
                return _logical_shape(x)

            if lhs is not None and rhs is not None:
                lhs_shape = _maybe_mock_shape(lhs)
                rhs_shape = _maybe_mock_shape(rhs)

                if lhs_shape is not None and rhs_shape is not None:
                    if (
                        len(lhs_shape) >= 2
                        and len(rhs_shape) >= 2
                        and lhs_shape[-1] != rhs_shape[-2]
                        and lhs_shape[-1] == rhs_shape[-1]
                    ):
                        if MOCK_DEVICE_ENABLED:
                            _mock_print(
                                "[MOCK_DEVICE] Dynamo matmul fake fallback: fixing stale RHS transpose shape before CPU matmul"
                            )
                        rhs = rhs.transpose(-1, -2)

            if hasattr(lhs, "cpu") and rhs is not None and hasattr(rhs, "cpu"):
                result = lhs.cpu().matmul(rhs.cpu(), **cpu_kwargs)
                return _to_spyre_tree(result)

        def _cpu_fallback_from_run_node():
            cpu_args = _to_cpu_tree(args)
            cpu_kwargs = _to_cpu_tree(kwargs)

            if target is operator.getitem:
                if MOCK_DEVICE_ENABLED:
                    _mock_print("[MOCK_DEVICE] Dynamo fake-tensor CPU fallback for builtins.getitem on spyre")
                result = target(*cpu_args, **cpu_kwargs)
                return _to_spyre_tree(result)

            if node.op == "call_method" and isinstance(target, str):
                if target == "copy_":
                    if MOCK_DEVICE_ENABLED:
                        _mock_print("[MOCK_DEVICE] Dynamo fake-tensor CPU fallback for method copy_ on spyre")
                    dst = args[0]
                    src = args[1] if len(args) > 1 else None
                    if _is_mock_spyre_tensor(dst) and src is not None:
                        try:
                            object.__setattr__(dst, "_mock_spyre_copy_src", src)
                        except Exception:
                            pass
                    return dst
                if MOCK_DEVICE_ENABLED:
                    _mock_print(f"[MOCK_DEVICE] Dynamo fake-tensor CPU fallback for method {target} on spyre")
                result = getattr(cpu_args[0], target)(*cpu_args[1:], **cpu_kwargs)
                return _to_spyre_tree(result)

            if callable(target):
                target_name = getattr(target, "__name__", "")
                target_overloadpacket = getattr(target, "overloadpacket", None)
                target_packet_name = getattr(target_overloadpacket, "__name__", "")

                if target_packet_name == "full":
                    if MOCK_DEVICE_ENABLED:
                        _mock_print("[MOCK_DEVICE] Dynamo fake-tensor CPU fallback for spyre.full on spyre")
                    size = cpu_args[0]
                    fill_value = cpu_args[1]
                    device = cpu_args[2]
                    dtype = cpu_args[3] if len(cpu_args) > 3 else cpu_kwargs.get("dtype")
                    return _to_spyre_tree(torch.full(size, fill_value, dtype=dtype, device=device))

                if MOCK_DEVICE_ENABLED:
                    _mock_print(
                        f"[MOCK_DEVICE] Dynamo fake-tensor CPU fallback for "
                        f"{target_name or repr(target)} on spyre"
                    )
                if target_name == "addmm":
                    out_arg = kwargs.get("out")
                    if out_arg is not None and _is_mock_spyre_tensor(out_arg):
                        cpu_input = cpu_args[0] if len(cpu_args) > 0 else None
                        cpu_mat1 = cpu_args[1] if len(cpu_args) > 1 else None
                        cpu_mat2 = cpu_args[2] if len(cpu_args) > 2 else None
                        alpha = kwargs.get("alpha", 1)
                        beta = kwargs.get("beta", 1)
                        computed = torch.addmm(cpu_input, cpu_mat1, cpu_mat2, beta=beta, alpha=alpha)
                        try:
                            object.__setattr__(out_arg, "_mock_spyre_copy_src", computed)
                        except Exception:
                            pass
                        return computed

                result = target(*cpu_args, **cpu_kwargs)
                if target_name == "addmm" and len(args) >= 1 and _contains_mock_spyre_tensor(args[0]):
                    return _to_spyre_tree(result)
                return _to_spyre_tree(result)

            raise

        try:
            result = original_run_node(tracer, node, args, kwargs, nnmodule)
            if os.environ.get("TORCH_SPYRE_MOCK_DEBUG_VIEW", "0") == "1" and (
                _contains_mock_spyre_tensor(args) or _contains_mock_spyre_tensor(kwargs)
            ):
                _mock_print(f"[MOCK_DEVICE] run_node result {_describe_mock_result(result)}")
            return result
        except RuntimeError as e:
            msg = str(e)
            if "PyTorch is not linked with support for spyre devices" not in msg and (
                "does not have a kernel registered for cpu" not in msg or "full" not in msg
            ):
                raise
            if not _contains_mock_spyre_tensor(args) and not _contains_mock_spyre_tensor(kwargs):
                raise
            return _cpu_fallback_from_run_node()

    def _find_common_device_with_mock_spyre_support(func, flat_args):
        try:
            return original_find_common_device(func, flat_args)
        except RuntimeError as e:
            msg = str(e)
            if "Unhandled FakeTensor Device Propagation" not in msg:
                raise

            devices = []
            has_spyre = False
            has_cpu = False
            has_meta = False
            for arg in flat_args:
                # Check both FakeTensors and real MockSpyreTensors (e.g. scalar
                # fill values in aten.where are real tensors, not FakeTensors).
                if isinstance(arg, torch.Tensor):
                    device = getattr(arg, "device", None)
                    if device is not None:
                        devices.append(device)
                        if device.type == "spyre":
                            has_spyre = True
                        elif device.type == "cpu":
                            has_cpu = True
                        elif device.type == "meta":
                            has_meta = True

            # Prefer spyre over cpu or meta — meta tensors are shape-only
            # placeholders (e.g. bool masks in aten.where), so the result
            # device should follow the data tensor (spyre).
            if has_spyre and (has_cpu or has_meta):
                if MOCK_DEVICE_ENABLED:
                    _mock_print("[MOCK_DEVICE] FakeTensor device propagation fallback: preferring spyre over cpu/meta")
                return fake_tensor_mod.torch.device("spyre", 0), False

            raise

    def _interpreter_call_function_with_mock_spyre_fallback(self, target, args, kwargs):
        try:
            result = original_interpreter_call_function(self, target, args, kwargs)
            if os.environ.get("TORCH_SPYRE_MOCK_DEBUG_VIEW", "0") == "1" and (
                _contains_mock_spyre_tensor(args) or _contains_mock_spyre_tensor(kwargs)
            ):
                _mock_print(f"[MOCK_DEVICE] interpreter result {_describe_mock_result(result)}")
            return result
        except (RuntimeError, AttributeError, AssertionError) as e:
            msg = str(e)
            is_mock_spyre = _contains_mock_spyre_tensor(args) or _contains_mock_spyre_tensor(kwargs)
            target_name = getattr(target, "__name__", "")
            is_supported_mock_failure = (
                "PyTorch is not linked with support for spyre devices" in msg
                or "'CppOverrides' object has no attribute 'overwrite'" in msg
                or msg == "matmul"
                or "AssertionError: matmul" in msg
                or "batchmatmul: failed to map stick_dims to host coords" in msg
            )
            if not is_supported_mock_failure or not is_mock_spyre:
                raise

            cpu_args = _to_cpu_tree(args)
            cpu_kwargs = _to_cpu_tree(kwargs)

            if target is operator.getitem:
                if MOCK_DEVICE_ENABLED:
                    _mock_print("[MOCK_DEVICE] FX interpreter CPU fallback for builtins.getitem on spyre")
                return target(*cpu_args, **cpu_kwargs)

            if target_name == "overwrite":
                if MOCK_DEVICE_ENABLED:
                    _mock_print("[MOCK_DEVICE] FX interpreter CPU fallback for overwrite on spyre")
                value, stride, offset, _gap = cpu_args
                del stride  # mock fallback only needs logical offset placement
                return value

            if target_name in {"reduction", "_reduction"}:
                if len(cpu_args) >= 4:
                    dtype, src_dtype, reduction_type, value = cpu_args[:4]
                    if reduction_type in {"matmul", "batchmatmul"} and isinstance(value, tuple) and len(value) == 2:
                        if MOCK_DEVICE_ENABLED:
                            _mock_print(
                                f"[MOCK_DEVICE] FX interpreter CPU fallback for reduction({reduction_type}) on spyre"
                            )
                        lhs, rhs = value
                        return lhs * rhs

            target_overloadpacket = getattr(target, "overloadpacket", None)
            target_packet_name = getattr(target_overloadpacket, "__name__", "")
            if target_packet_name == "full":
                if MOCK_DEVICE_ENABLED:
                    _mock_print("[MOCK_DEVICE] FX interpreter CPU fallback for spyre.full on spyre")
                size, fill_value, device, *rest = cpu_args
                dtype = rest[0] if rest else cpu_kwargs.get("dtype")
                return torch.full(size, fill_value, dtype=dtype, device=device)

            if callable(target):
                if MOCK_DEVICE_ENABLED:
                    _mock_print(
                        f"[MOCK_DEVICE] FX interpreter CPU fallback for "
                        f"{getattr(target, '__name__', repr(target))} on spyre"
                    )
                return target(*cpu_args, **cpu_kwargs)

            raise

    def _fake_tensor_torch_function_with_mock_spyre_fallback(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        func_name = getattr(func, "__name__", "")
        flat_args = list(args) + list(kwargs.values())

        if func_name == "copy_":
            if any(_is_mock_spyre_tensor(arg) for arg in flat_args):
                if MOCK_DEVICE_ENABLED:
                    _mock_print("[MOCK_DEVICE] FakeTensor __torch_function__ fallback for copy_ on spyre")
                dst = args[0]
                src = args[1] if len(args) > 1 else None
                try:
                    if src is not None:
                        object.__setattr__(dst, "_mock_spyre_copy_src", src)
                except Exception:
                    pass
                return dst

        if func_name == "contiguous" and args and _is_mock_spyre_tensor(args[0]):
            if MOCK_DEVICE_ENABLED:
                _mock_print("[MOCK_DEVICE] FakeTensor __torch_function__ handling contiguous on spyre")
            # For FakeTensors, PyTorch already handles contiguous() correctly
            # We just need to preserve mock device attributes
            assert original_fake_tensor_torch_function is not None
            result = original_fake_tensor_torch_function(func, types, args, kwargs)
            base = args[0]
            if isinstance(result, FakeTensor) and _is_mock_spyre_tensor(base):
                try:
                    object.__setattr__(result, "_mock_device", getattr(base, "_mock_device", "spyre"))
                    if hasattr(base, "_mock_device_layout"):
                        object.__setattr__(result, "_mock_device_layout", base._mock_device_layout)
                except Exception:
                    pass
            return result

        if func_name in {"transpose", "permute", "t", "unsqueeze", "squeeze"} and any(
            _is_mock_spyre_tensor(arg) for arg in flat_args
        ):
            # Delegate to the original FakeTensor __torch_function__ so that the
            # result is registered in ProxyTorchDispatchMode's tensor_tracker.
            # Creating a brand-new FakeTensor via fake_mode.from_tensor() produces
            # an untracked tensor; when a subsequent op's functionalization kernel
            # tries to look it up in tensor_tracker it raises
            # "cannot find FakeTensor in tensor_tracker", breaking compilation of
            # any graph that chains view ops (e.g. transpose → contiguous →
            # transpose inside a spyre_hint block).
            assert original_fake_tensor_torch_function is not None
            try:
                result = original_fake_tensor_torch_function(func, types, args, kwargs)
                # Propagate mock device marker so downstream ops keep seeing the
                # tensor as a spyre tensor.
                base = args[0] if args else None
                if isinstance(result, FakeTensor) and base is not None and _is_mock_spyre_tensor(base):
                    try:
                        object.__setattr__(result, "_mock_device", getattr(base, "_mock_device", "spyre"))
                        if hasattr(base, "_mock_device_layout"):
                            object.__setattr__(result, "_mock_device_layout", base._mock_device_layout)
                    except Exception:
                        pass
                return result
            except Exception:
                pass
            # Hard fallback: should rarely be reached
            return original_fake_tensor_torch_function(func, types, args, kwargs)

        if func_name in {"mm", "matmul", "bmm"} and any(_is_mock_spyre_tensor(arg) for arg in flat_args):
            if MOCK_DEVICE_ENABLED:
                _mock_print(
                    f"[MOCK_DEVICE] FakeTensor __torch_function__ CPU fallback for {func_name} on spyre"
                )
            cpu_args = _to_cpu_tree(args)
            cpu_kwargs = _to_cpu_tree(kwargs)
            return func(*cpu_args, **cpu_kwargs)

        if str(func) == "aten.bmm.default" and any(_is_mock_spyre_tensor(arg) for arg in flat_args):
            if MOCK_DEVICE_ENABLED:
                _mock_print("[MOCK_DEVICE] FakeTensor __torch_function__ CPU fallback for aten.bmm.default on spyre")
            cpu_args = _to_cpu_tree(args)
            cpu_kwargs = _to_cpu_tree(kwargs)
            return func(*cpu_args, **cpu_kwargs)

        # Void / inplace custom ops on spyre tensors: FakeTensorMode has no
        # shape rule for ops like spyre::overwrite_f or any other custom op
        # registered only for "spyre" device.  For shape-inference purposes an
        # inplace op doesn't change the output shape — return the first tensor
        # arg that lives on spyre (the output / mutated arg) unchanged.
        # "overwrite_f" specifically: signature is (input, output, dims, offsets)
        # so args[1] is the output tensor.  For other void inplace ops args[0]
        # is typically the mutated tensor, which is safe to return.
        _has_spyre_tensor_arg = any(
            getattr(getattr(arg, "device", None), "type", None) == "spyre"
            for arg in flat_args
            if isinstance(arg, torch.Tensor)
        )
        if _has_spyre_tensor_arg:
            _func_ns = getattr(func, "namespace", None) or ""
            _is_custom_op = _func_ns not in ("", "aten", "prims", "prim")
            if _is_custom_op:
                if MOCK_DEVICE_ENABLED:
                    _mock_print(f"[MOCK_DEVICE] FakeTensor no-op for custom op {func_name} on spyre")
                if "overwrite_f" in func_name:
                    return args[1] if len(args) > 1 else (args[0] if args else None)
                for _a in args:
                    if isinstance(_a, torch.Tensor) and getattr(getattr(_a, "device", None), "type", None) == "spyre":
                        return _a
                return args[0] if args else None

        # General catch-all: for any op involving mock spyre FakeTensors that
        # wasn't explicitly handled above, convert spyre FakeTensors to CPU
        # FakeTensors using fake_mode.from_tensor (same pattern as the
        # transpose handler) so shape/dtype inference runs without hitting
        # "PyTorch is not linked with support for spyre devices".
        # We do NOT use _to_cpu_tree / obj.to("cpu") here because that would
        # re-enter __torch_function__ and cause infinite recursion.
        if any(_is_mock_spyre_tensor(arg) for arg in flat_args):
            fake_mode = None
            for _arg in flat_args:
                if isinstance(_arg, FakeTensor):
                    fake_mode = getattr(_arg, "fake_mode", None)
                    if fake_mode is not None:
                        break

            if fake_mode is not None:
                if MOCK_DEVICE_ENABLED:
                    _mock_print(
                        f"[MOCK_DEVICE] FakeTensor catch-all CPU fallback for {func_name} on spyre"
                    )

                def _spyre_ft_to_cpu(obj):
                    if _is_mock_spyre_tensor(obj):
                        # Access shape/stride/dtype WITHOUT triggering __torch_function__:
                        # FakeTensor.shape calls aten.size() under the hood, which would
                        # re-enter _fake_tensor_torch_function_with_mock_spyre_fallback
                        # and cause infinite recursion.
                        with torch._C.DisableTorchFunctionSubclass():
                            shape = list(obj.shape)
                            stride = list(obj.stride())
                            dtype = obj.dtype
                            fake_mode_ref = obj.fake_mode
                        # Disable ALL dispatch modes to get a raw meta tensor.
                        # unset_fake_temporarily only exits FakeTensorMode, not
                        # FunctionalTensorMode — so when FunctionalTensorMode is also
                        # active (AOT compilation), empty_strided would return a
                        # FunctionalTensor, and FakeTensor(_make_subclass) on a
                        # FunctionalTensor raises "use FunctionalTensor with a
                        # corresponding FunctionalTensorMode()".
                        with torch._C._DisableTorchDispatch():
                            meta = torch.empty_strided(shape, stride, dtype=dtype, device='meta')
                        return FakeTensor(fake_mode_ref, meta, torch.device('cpu'))
                    # Also convert plain real CPU tensors (e.g. _tensor_constant0 in
                    # bitwise/logical ops) into CPU FakeTensors so FakeTensorMode can
                    # handle the mixed FakeTensor+real-tensor dispatch.
                    if (
                        isinstance(obj, torch.Tensor)
                        and not isinstance(obj, FakeTensor)
                        and fake_mode is not None
                    ):
                        shape = list(obj.shape)
                        stride = list(obj.stride())
                        dtype = obj.dtype
                        with torch._C._DisableTorchDispatch():
                            meta = torch.empty_strided(shape, stride, dtype=dtype, device='meta')
                        return FakeTensor(fake_mode, meta, torch.device('cpu'))
                    return obj

                cpu_args = tuple(_spyre_ft_to_cpu(a) for a in args)
                cpu_kwargs = {k: _spyre_ft_to_cpu(v) for k, v in (kwargs or {}).items()}
                try:
                    assert original_fake_tensor_torch_function is not None
                    return original_fake_tensor_torch_function(func, types, cpu_args, cpu_kwargs)
                except Exception as _e:
                    _mock_print(
                        f"[MOCK_DEVICE] FakeTensor catch-all also failed for {func_name}: {_e}"
                    )

        assert original_fake_tensor_torch_function is not None
        return original_fake_tensor_torch_function(func, types, args, kwargs)

    if original_gen_alias_from_base is not None:
        def _gen_alias_from_base_with_mock_spyre_fallback(
            aliased_base_tensor,
            target_meta_tensor,
            target_requires_grad,
            target_view_meta_sequence=None,
            *,
            replay_views,
        ):
            if getattr(getattr(aliased_base_tensor, "device", None), "type", None) == "spyre":
                if replay_views and target_view_meta_sequence is not None:
                    try:
                        replayed = aot_functional_utils._functionalization.apply_view_meta_sequence(
                            aliased_base_tensor,
                            target_view_meta_sequence.sequence,
                        )
                    except Exception:
                        replay_views = False
                    else:
                        if replayed.shape != target_meta_tensor.shape:
                            replay_views = False
            return original_gen_alias_from_base(
                aliased_base_tensor,
                target_meta_tensor,
                target_requires_grad,
                target_view_meta_sequence,
                replay_views=replay_views,
            )

        aot_functional_utils.gen_alias_from_base = _gen_alias_from_base_with_mock_spyre_fallback
        if aot_runtime_wrappers is not None:
            aot_runtime_wrappers.gen_alias_from_base = _gen_alias_from_base_with_mock_spyre_fallback
    dynamo_utils.run_node = _run_node_with_cpu_fallback
    FakeTensor._find_common_device = staticmethod(_find_common_device_with_mock_spyre_support)
    Interpreter.call_function = _interpreter_call_function_with_mock_spyre_fallback
    if original_fake_tensor_torch_function is not None:
        FakeTensor.__torch_function__ = classmethod(_fake_tensor_torch_function_with_mock_spyre_fallback)
    dynamo_utils._spyre_mock_patched = True
    _mock_print("[MOCK_DEVICE] Patched Dynamo fake-tensor run_node for mock spyre backend")


def _patch_cpp_reduction_codegen_for_mock():
    if not MOCK_DEVICE_ENABLED:
        return

    try:
        import torch._inductor.codegen.cpp as cpp_codegen
    except Exception as e:
        _mock_print(f"[MOCK_DEVICE] Warning: could not import cpp codegen for patching: {e}")
        return

    if getattr(cpp_codegen, "_spyre_mock_matmul_codegen_patched", False):
        return

    original_reduction_combine = cpp_codegen.reduction_combine
    original_reduction_init = cpp_codegen.reduction_init

    def _mock_reduction_combine(reduction_type, a, b, index=None, **kwargs):
        if reduction_type in {"matmul", "batchmatmul"}:
            if isinstance(b, tuple) and len(b) == 2:
                lhs, rhs = b
                return f"({lhs}) * ({rhs})"
            return "0"
        return original_reduction_combine(reduction_type, a, b, index=index, **kwargs)

    def _mock_reduction_init(reduction_type, dtype):
        if reduction_type in {"matmul", "batchmatmul"}:
            return "0"
        return original_reduction_init(reduction_type, dtype)

    cpp_codegen.reduction_combine = _mock_reduction_combine
    cpp_codegen.reduction_init = _mock_reduction_init

    original_gen_parallel_reduction_buffers = cpp_codegen.CppKernel._gen_parallel_reduction_buffers
    original_codegen_functions = cpp_codegen.CppKernelProxy.codegen_functions
    original_kernel_reduction = cpp_codegen.CppKernel.reduction
    original_vec_kernel_reduction = None
    if hasattr(cpp_codegen, 'CppVecKernel'):
        original_vec_kernel_reduction = cpp_codegen.CppVecKernel.reduction

    def _mock_gen_parallel_reduction_buffers(
        self,
        acc,
        acc_type,
        reduction_type,
        dtype,
        reduction_combine_fn=_mock_reduction_combine,
        reduction_init_fn=_mock_reduction_init,
    ):
        return original_gen_parallel_reduction_buffers(
            self,
            acc,
            acc_type,
            reduction_type,
            dtype,
            reduction_combine_fn=reduction_combine_fn,
            reduction_init_fn=reduction_init_fn,
        )

    def _mock_codegen_functions(self, fn_list, var_sizes_list):
        return original_codegen_functions(self, fn_list, var_sizes_list)

    def _mock_kernel_reduction(self, dtype, src_dtype, reduction_type, value):
        if reduction_type in {"matmul", "batchmatmul"}:
            # For matmul/batchmatmul, treat them as 'sum' reductions
            # This avoids C++ vectorization issues while still generating correct code
            # The actual matmul logic is in the inner_fn, not the reduction type
            # Use the base CppKernel.reduction (non-vectorized) for matmul
            if isinstance(self, cpp_codegen.CppVecKernel):
                # For CppVecKernel, also use 'sum' to avoid vec matmul issues
                return original_vec_kernel_reduction(self, dtype, src_dtype, "sum", value)
            return original_kernel_reduction(self, dtype, src_dtype, "sum", value)
        # For non-matmul reductions (e.g., sum for rms_norm), dispatch to the
        # correct class method so CppVecKernel gets its vectorized codegen
        if isinstance(self, cpp_codegen.CppVecKernel):
            return original_vec_kernel_reduction(self, dtype, src_dtype, reduction_type, value)
        return original_kernel_reduction(self, dtype, src_dtype, reduction_type, value)

    cpp_codegen.CppKernel._gen_parallel_reduction_buffers = _mock_gen_parallel_reduction_buffers
    cpp_codegen.CppKernelProxy.codegen_functions = _mock_codegen_functions
    cpp_codegen.CppKernel.reduction = _mock_kernel_reduction
    
    # Also patch CppVecKernel.reduction if it exists
    if hasattr(cpp_codegen, 'CppVecKernel'):
        cpp_codegen.CppVecKernel.reduction = _mock_kernel_reduction
        _mock_print("[MOCK_DEVICE] Patched CppVecKernel.reduction for matmul/batchmatmul")
    
    # Patch CppOverrides to add overwrite method for torch.cat support
    def _mock_overwrite(self, input, stride, offset, gap):
        """Mock overwrite operation for torch.cat support in mock device mode."""
        # In mock mode, just return the input value
        # The actual overwrite logic is handled by the Spyre backend
        return input
    
    if hasattr(cpp_codegen, 'CppOverrides'):
        if not hasattr(cpp_codegen.CppOverrides, 'overwrite'):
            cpp_codegen.CppOverrides.overwrite = _mock_overwrite
            _mock_print("[MOCK_DEVICE] Added overwrite method to CppOverrides")
    
    cpp_codegen._spyre_mock_matmul_codegen_patched = True
    _mock_print("[MOCK_DEVICE] Patched C++ reduction codegen for mock matmul/batchmatmul")
    
    # Install CPU kernel for overwrite operation
    _install_mock_overwrite_kernel()


def _patch_spyre_kernel_for_mock():
    """Patch selected Spyre inductor planning logic only in mock mode."""
    if not MOCK_DEVICE_ENABLED:
        return

    try:
        import torch_spyre._inductor.spyre_kernel as spyre_kernel
    except Exception as e:
        _mock_print(f"[MOCK_DEVICE] Warning: could not import spyre_kernel for patching: {e}")
        return

    if getattr(spyre_kernel, "_spyre_mock_patched", False):
        return

    original_simplify_op_spec = spyre_kernel.simplify_op_spec

    def _simplify_op_spec_with_mock_boundary_fix(op_spec):
        # Mock device doesn't have the nonuniform stick indexing limitation
        # that real Spyre hardware has, so we skip that validation check
        
        try:
            return original_simplify_op_spec(op_spec)
        except ValueError as e:
            if "is not in list" not in str(e):
                raise

            try:
                import torch_spyre._inductor.views as spyre_views
            except Exception:
                raise

            var_ranges = {var: val[0] for var, val in op_spec.iteration_space.items()}
            op_it_space_splits = {var: val[1] for var, val in op_spec.iteration_space.items()}

            splits = {var: set() for var in var_ranges.keys()}
            breakdown = []
            stick_dim = []
            stick_size = []
            for arg in op_spec.args:
                tensor = {
                    "size": arg.device_size,
                    "coordinates": arg.device_coordinates,
                }
                intervals = spyre_views.normalize_coordinates(
                    var_ranges, tensor["size"], tensor["coordinates"]
                )
                stick_dim.append(intervals[-1][2])
                stick_size.append(intervals[-1][-1])
                breakdown.append(intervals)
                for num, den, var, mod, dim_size in intervals:
                    if var is not None:
                        if den != stick_size[-1] or var != stick_dim[-1]:
                            splits[var].add(den)
                        if mod != stick_size[-1] or var != stick_dim[-1]:
                            splits[var].add(mod)

            splits = {
                var: sorted({sympy.S.One, var_ranges[var], *val})
                for var, val in splits.items()
            }

            new_var_ranges = {}
            new_op_it_space_splits = {}
            n = 0
            remap = {}
            for var, split in splits.items():
                div = op_it_space_splits[var] if var in op_it_space_splits else 1
                if len(split) > 1:
                    new_var_ranges[var] = split[1] // split[0]
                    remap[var] = [var]
                    for i in range(1, len(split) - 1):
                        new_var = sympy.symbols(f"z{n}")
                        n += 1
                        new_var_ranges[new_var] = split[i + 1] // split[i]
                        remap[var].append(new_var)
                    for v in reversed(remap[var]):
                        new_op_it_space_splits[v] = math.gcd(div, new_var_ranges[v])
                        div //= new_op_it_space_splits[v]
                else:
                    new_var_ranges[var] = var_ranges[var]
                    new_op_it_space_splits[var] = (
                        op_it_space_splits[var] if var in op_it_space_splits else 1
                    )

            new_tensors = []
            for j, intervals in enumerate(breakdown):
                size = []
                coordinates = []
                for num, den, var, mod, dim_size in intervals[:-1]:
                    if var is None:
                        size.append(dim_size)
                        coordinates.append(sympy.S.Zero)
                        continue
                    low = (
                        0
                        if var == stick_dim[j]
                        and den == stick_size[j]
                        and den not in splits[var]
                        else splits[var].index(den)
                    )
                    for i in reversed(range(low, splits[var].index(mod))):
                        if i == splits[var].index(mod) - 1:
                            size.append(dim_size * den // splits[var][i])
                        else:
                            size.append(splits[var][i + 1] // splits[var][i])
                        coordinates.append(remap[var][i])
                    if var == stick_dim[j] and den == stick_size[j] and den not in splits[var]:
                        size[-1] //= den
                        coordinates[-1] //= den
                    if num > 1:
                        size.append(num)
                        coordinates.append(sympy.S.Zero)
                num, den, var, mod, dim_size = intervals[-1]
                size.append(dim_size)
                coordinates.append(var % dim_size if var is not None else sympy.S.Zero)
                new_tensors.append({"size": size, "coordinates": coordinates})

            rank = 0
            for i, t in enumerate(new_tensors):
                if stick_dim[i] is None:
                    not_found = 1
                    for c, s in zip(t["coordinates"][:-1], t["size"][:-1]):
                        if c == 0 and s == 1:
                            not_found = 0
                            break
                    rank = max(rank, len(t["size"]) + not_found)
                    continue
                found = 1
                for c, s in zip(t["coordinates"][:-1], t["size"][:-1]):
                    if stick_dim[i] in c.free_symbols or s == 1:
                        found = 0
                        break
                rank = max(rank, len(t["size"]) + found)

            for t in new_tensors:
                gap = rank - len(t["size"])
                t["size"] = [sympy.S.One] * gap + t["size"]
                t["coordinates"] = [sympy.S.Zero] * gap + t["coordinates"]

            for t in new_tensors:
                vars = t["coordinates"][-1].free_symbols
                if len(vars) == 1:
                    stick_dim_var = next(iter(vars))
                    found = False
                    for i in range(len(t["coordinates"]) - 1):
                        vars = t["coordinates"][i].free_symbols
                        if stick_dim_var in vars:
                            found = True
                            continue
                    if not found:
                        for i in range(len(t["coordinates"]) - 1):
                            if t["size"][i] == 1:
                                t["coordinates"][i] = stick_dim_var // t["size"][-1]
                                t["coordinates"][-1] = stick_dim_var % t["size"][-1]
                                break

            op_spec.iteration_space = {
                k: (v, new_op_it_space_splits[k]) for k, v in new_var_ranges.items()
            }
            for arg, t in zip(op_spec.args, new_tensors):
                arg.device_size = t["size"]
                arg.device_coordinates = t["coordinates"]

            if MOCK_DEVICE_ENABLED:
                _mock_print("[MOCK_DEVICE] Applied mock-only Spyre planner boundary fix")
            return None

    spyre_kernel.simplify_op_spec = _simplify_op_spec_with_mock_boundary_fix
    setattr(spyre_kernel, "_spyre_mock_patched", True)
    _mock_print("[MOCK_DEVICE] Patched spyre_kernel.simplify_op_spec for mock spyre backend")


def start_runtime():
    """Stub function - initializes mock device"""
    if MOCK_DEVICE_ENABLED:
        _mock_print("[MOCK_DEVICE] Runtime initialized")
        _register_privateuse1_backend()
        default_stream(torch.device("spyre", _mock_current_device_index))
        _patch_torch_stream_api_for_mock()
        _patch_torch_accelerator_for_mock()
        _patch_fake_tensor_ops_for_mock()
        _patch_cpp_reduction_codegen_for_mock()
        _patch_spyre_kernel_for_mock()
    pass


def _register_privateuse1_backend():
    """Register Spyre as PrivateUse1 backend with PyTorch"""
    if not MOCK_DEVICE_ENABLED:
        return
    
    try:
        import torch
        # Register the backend name
        torch._register_device_module('spyre', 'torch_spyre')
        _mock_print("[MOCK_DEVICE] Registered 'spyre' as PrivateUse1 backend")
    except Exception as e:
        _mock_print(f"[MOCK_DEVICE] Warning: Could not register backend: {e}")


def get_device_dtype(dtype):
    """Stub function - maps torch dtype to DataFormats"""
    import torch
    if dtype == torch.float16:
        return DataFormats.SEN169_FP16
    elif dtype == torch.float32:
        return DataFormats.IEEE_FP32
    else:
        # Default to FP16 for other types
        return DataFormats.SEN169_FP16


def get_elem_in_stick(dtype):
    """Returns number of elements in a stick (128 bytes) for given dtype"""
    import torch
    
    # Stick size is 128 bytes
    STICK_SIZE_BYTES = 128
    
    # Map dtype to element size in bytes
    dtype_sizes = {
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float32: 4,
        torch.float64: 8,
        torch.int8: 1,
        torch.uint8: 1,
        torch.int16: 2,
        torch.int32: 4,
        torch.int64: 8,
        torch.bool: 1,
    }
    
    element_size = dtype_sizes.get(dtype, 2)  # Default to 2 bytes (float16)
    return STICK_SIZE_BYTES // element_size


# Mock device registration functions
def is_available():
    """Mock device is always available when enabled"""
    return MOCK_DEVICE_ENABLED


def device_count():
    """Return 1 mock device"""
    return 1 if MOCK_DEVICE_ENABLED else 0

def _get_default_generator(device_id=0):
    """Return a stub generator for the mock device"""
    import torch
    if MOCK_DEVICE_ENABLED:
        _mock_print(f"[MOCK_DEVICE] _get_default_generator({device_id}) - returning CPU generator")
    # Return CPU's default generator as fallback for mock device
    return torch.default_generator



def current_device():
    """Return current device (always 0 for mock)"""
    return 0


def set_device(device_id):
    """Set current device (no-op for mock)"""
    if MOCK_DEVICE_ENABLED:
        _mock_print(f"[MOCK_DEVICE] set_device({device_id}) - no-op")
    pass


def _register_device():
    """Register device (no-op for mock - already registered via _register_privateuse1_backend)"""
    if MOCK_DEVICE_ENABLED:
        _mock_print("[MOCK_DEVICE] _register_device() called - no-op")


def _install_mock_overwrite_kernel():
    """Install CPU kernel for torch.ops.spyre.overwrite to support cat in mock mode."""
    if not MOCK_DEVICE_ENABLED:
        return
    
    # In mock mode, we don't need to register the overwrite kernel
    # because we're using the fallback lowering for cat instead
    _mock_print("[MOCK_DEVICE] Skipping overwrite kernel registration (using fallback lowering)")


def _suppress_codegen_ops_import_error():
    """Suppress the duplicate kernel registration error in codegen_ops.py for mock mode."""
    if not MOCK_DEVICE_ENABLED:
        return
    
    try:
        import sys
        import builtins
        
        # Patch the built-in __import__ to catch RuntimeError when importing codegen_ops
        original_import = builtins.__import__
        
        def patched_import(name, *args, **kwargs):
            try:
                return original_import(name, *args, **kwargs)
            except RuntimeError as e:
                if "codegen_ops" in name and "already a kernel registered" in str(e):
                    _mock_print(f"[MOCK_DEVICE] Suppressed codegen_ops import error: {e}")
                    # Return a dummy module
                    import types
                    return types.ModuleType(name)
                raise
        
        builtins.__import__ = patched_import
        _mock_print("[MOCK_DEVICE] Patched __import__ to suppress codegen_ops errors")
    except Exception as e:
        _mock_print(f"[MOCK_DEVICE] Failed to patch __import__: {e}")
    return True


def _is_device_available():
    """Check if device is available"""
    return MOCK_DEVICE_ENABLED


def _get_device_count():
    """Get device count"""
    return 1 if MOCK_DEVICE_ENABLED else 0

def get_downcast_warning():
    """Get downcast warning setting for dtype conversions"""
    return True

def set_downcast_warning(enabled):
    """Set downcast warning setting (no-op in mock mode)"""
    pass


# Mock tensor storage - keeps data on CPU but tracks device metadata
class MockSpyreTensorStorage:
    """Mock storage that keeps tensor data on CPU but tracks Spyre device metadata"""
    def __init__(self, cpu_tensor):
        self.cpu_data = cpu_tensor
        self.device_id = 0
        self.is_mock = True
        if MOCK_DEVICE_ENABLED:
            _mock_print(f"[MOCK_DEVICE] Created mock tensor storage: shape={cpu_tensor.shape}, dtype={cpu_tensor.dtype}")


def copy_tensor(tensor, layout=None):
    """Stub function for copy_tensor - returns a copy of the tensor."""
    if MOCK_DEVICE_ENABLED:
        _mock_print(f"[MOCK_DEVICE] copy_tensor called with tensor shape {tensor.shape}")
    
    # For mock device mode, just return a CPU copy
    result = tensor.cpu().clone()
    
    # If layout is provided, try to preserve it in mock form
    if layout is not None:
        try:
            object.__setattr__(result, "_mock_device_layout", layout)
        except Exception:
            pass
    
    return result


# Add any other functions/classes that might be imported from _C
__all__ = [
    'DataFormats',
    'ElementArrangement',
    'SpyreTensorLayout',
    'get_spyre_tensor_layout',
    'to_with_layout',
    'empty_with_layout',
    'spyre_empty_with_layout',
    'reinterpret_tensor',
    'reinterpret_tensor_with_layout',
    'encode_constant',
    'convert_artifacts',
    'prepare_kernel',
    'launch_jobplan',
    'launch_kernel',
    'start_runtime',
    'current_stream',
    'default_stream',
    'get_stream_from_pool',
    'set_current_stream',
    'synchronize',
    'current_device',
    'set_device',
    'get_device_dtype',
    'get_elem_in_stick',
    '_register_device',
    '_is_device_available',
    '_get_device_count',
    '_get_default_generator',
    'device_count',
    'get_downcast_warning',
    'set_downcast_warning',
    'MockSpyreTensorStorage',
    'copy_tensor',
    'MOCK_DEVICE_ENABLED',
]
