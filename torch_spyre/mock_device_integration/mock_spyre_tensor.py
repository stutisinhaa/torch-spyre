# Copyright 2025 The Torch-Spyre Authors.
# Mock Spyre Tensor - Spoofs device as "spyre" while keeping data on CPU

"""
Mock Spyre Tensor Implementation

This creates a tensor that:
1. Reports device as "spyre" to PyTorch
2. Keeps actual data on CPU
3. Triggers Spyre inductor backend for compilation
4. Generates real SDSC JSON through the normal path

This is the key to making torch.compile() use the Spyre backend
without actual hardware.
"""

import torch
import os
import types

MOCK_DEVICE_ENABLED = os.environ.get('TORCH_SPYRE_MOCK_DEVICE', '0') == '1'
MOCK_VERBOSE = os.environ.get("TORCH_SPYRE_MOCK_VERBOSE", "0") == "1"


def _mock_print(*args, **kwargs):
    if MOCK_DEVICE_ENABLED and MOCK_VERBOSE:
        print(*args, **kwargs)

def _mock_has_debug_enabled():
    return MOCK_DEVICE_ENABLED and (
        MOCK_VERBOSE or os.environ.get("TORCH_SPYRE_MOCK_DEBUG_VIEW", "0") == "1"
    )

def _mock_debug_print(*args, **kwargs):
    if _mock_has_debug_enabled():
        print(*args, **kwargs)


def _build_functional_aten_map():
    """Map torch.nn.functional activation composites to their direct aten ops.

    Used by MockSpyreTensor.__torch_function__ to avoid dynamo lifting the
    subclass input to a frozen (fake) constant. Only includes functionals whose
    aten counterpart has the same call signature for the single-tensor case.
    """
    import torch.nn.functional as _F

    mapping = {}
    for fn_name, aten_name in (
        ("silu", "silu"),
        ("mish", "mish"),
        ("gelu", "gelu"),
        ("softplus", "softplus"),
    ):
        fn = getattr(_F, fn_name, None)
        aten_op = getattr(torch.ops.aten, aten_name, None)
        if fn is not None and aten_op is not None:
            mapping[fn] = aten_op
    return mapping


_MOCK_FUNCTIONAL_ATEN = _build_functional_aten_map()


# Bound C-level tensor accessors captured once at import. Used to bypass the
# Python property overrides below during torch.compile tracing: reading
# ``self.shape``/``self.stride()`` etc. through ``super()`` makes Dynamo
# graph-break on the SuperVariable access, which severs the spyre_hint
# annotation region and strips the coarse-tile hint from in-scope ops. Calling
# these C accessors directly is traceable and avoids the break.
_TENSOR_SIZE = torch.Tensor.size
_TENSOR_STRIDE = torch.Tensor.stride


class MockSpyreTensor(torch.Tensor):
    """
    A tensor subclass that reports device as 'spyre' but stores data on CPU.
    
    This tricks PyTorch's inductor into using the Spyre compilation backend,
    which generates real SDSC JSON through the normal code path.
    """
    
    @staticmethod
    def __new__(cls, data, *args, **kwargs):
        # Idempotent wrapping: re-wrapping an existing MockSpyreTensor must
        # return the SAME object so tensor identity is preserved. On real
        # hardware `.to("spyre")` on an already-spyre tensor is a no-op; the mock
        # must match that, otherwise identity-keyed metadata (e.g. the named-dim
        # annotations in propagate_named_dims, keyed on V.get_real_inputs()) is
        # lost and dims fall back to `_untracked_<size>`.
        if isinstance(data, cls):
            return data
        # Cannot wrap FakeTensors or meta tensors as MockSpyreTensor
        if isinstance(data, torch.Tensor):
            if (type(data).__name__ == 'FakeTensor' or
                    getattr(getattr(data, 'device', None), 'type', None) == 'meta'):
                return data
            cpu_tensor = data.cpu()
        else:
            cpu_tensor = torch.tensor(data, *args, **kwargs)

        # Create our subclass wrapping the CPU tensor. Pass require_grad
        # positionally: Dynamo's _make_subclass polyfill asserts len(kwargs)==0,
        # so a `require_grad=` kwarg graph-breaks during tracing (e.g. inside the
        # view unwrap/rewrap path), severing the spyre_hint annotation region.
        return torch.Tensor._make_subclass(cls, cpu_tensor, cpu_tensor.requires_grad)
    
    def __init__(self, data=None, *args, **kwargs):
        super().__init__()
        self._mock_device = torch.device("spyre", 0)
        # data is None when Dynamo inlines __init__ during the view-unwrap/rewrap
        # path (e.g. after contiguous() on a MockSpyreTensor).  _make_subclass
        # constructs the object without passing data through __init__, so Dynamo
        # sees a zero-argument call.  Making data optional lets argument binding
        # succeed and avoids the graph break that would sever the spyre_hint
        # annotation region (causing LoopSpec to disappear from the generated
        # source).  The debug print below is eager-only, so data=None is fine.
        if MOCK_DEVICE_ENABLED and not torch.compiler.is_compiling() and data is not None:
            try:
                cpu_data = data.cpu() if isinstance(data, torch.Tensor) else torch.tensor(data)
                _mock_print(
                    f"[MOCK_DEVICE] Created MockSpyreTensor: shape={tuple(cpu_data.shape)}, dtype={cpu_data.dtype}"
                )
            except Exception:
                _mock_print("[MOCK_DEVICE] Created MockSpyreTensor")
    
    @property
    def device(self):
        """Report device as 'spyre' to trick inductor"""
        # Handle case where _mock_device might not be set (e.g., after copy_())
        if not hasattr(self, '_mock_device'):
            self._mock_device = torch.device("spyre", 0)
        return self._mock_device

    @property
    def shape(self):
        # During torch.compile tracing, bypass the logical-shape override and
        # the super() access (which Dynamo graph-breaks on, stripping the
        # spyre_hint). The physical C-level shape is what the FX graph needs.
        if torch.compiler.is_compiling():
            return _TENSOR_SIZE(self)
        logical_shape = getattr(self, "_mock_logical_shape", None)
        if logical_shape is not None:
            return torch.Size(logical_shape)
        return super().shape

    def size(self, *args):
        if torch.compiler.is_compiling():
            return _TENSOR_SIZE(self, *args)
        shape = self.shape
        if args:
            return shape[args[0]]
        return shape

    def stride(self, dim=None):
        if torch.compiler.is_compiling():
            return _TENSOR_STRIDE(self) if dim is None else _TENSOR_STRIDE(self, dim)
        logical_stride = getattr(self, "_mock_logical_stride", None)
        if logical_stride is None:
            return super().stride() if dim is None else super().stride(dim)
        if dim is None:
            return logical_stride
        return logical_stride[dim]

    # storage_offset() is intentionally NOT overridden here.
    #
    # A Python override that returns a concrete int breaks torch.compile: Dynamo
    # traces into the Python method, sees `call_method; target: storage_offset`
    # returning type `int` instead of a SymInt, and issues a graph break that
    # severs the spyre_hint annotation region (so LoopSpec disappears from the
    # generated source).  The base C-level implementation is traced natively by
    # Dynamo and returns a SymInt as expected.
    #
    # The _mock_storage_offset attribute (set by _rewrap inside __torch_function__)
    # is read directly by cpu() via getattr — no override needed.
    
    def device_tensor_layout(self):
        """
        Return a SpyreTensorLayout that reflects the tensor's actual memory order.

        For contiguous tensors (or 0-/1-D), use the 2-arg constructor so that
        SpyreTensorLayout computes the canonical stick-stride-map automatically.
        For non-contiguous tensors (transposed, permuted) use the 4-arg constructor
        with a dim_order derived from the physical stride order, so the restickify
        cost optimizer sees the real layout instead of always assuming row-major.
        """
        if MOCK_DEVICE_ENABLED:
            _mock_print(f"[MOCK_DEVICE] device_tensor_layout() called for tensor shape={self.shape}")

        existing_layout = getattr(self, "_mock_device_layout", None)
        if existing_layout is not None:
            if MOCK_DEVICE_ENABLED:
                _mock_print("[MOCK_DEVICE] Returning preserved mock device layout")
            return existing_layout

        try:
            from torch_spyre._C import SpyreTensorLayout

            ndim = self.dim()
            shape = list(self.shape)
            strides = list(self.stride())

            # Determine whether this tensor is contiguous in the default
            # (last-dim-fastest) sense.  Use the 2-arg constructor for the
            # common case so SpyreTensorLayout fills in the stride_map itself.
            is_contiguous = self.is_contiguous()

            if ndim <= 1 or is_contiguous:
                layout = SpyreTensorLayout(shape, self.dtype)
            else:
                # Infer dim_order: sort dimensions from slowest stride to
                # fastest (largest stride first), matching physical memory order.
                # Dimensions with stride 0 (broadcast) go last.
                dim_order = sorted(
                    range(ndim),
                    key=lambda d: (-strides[d] if strides[d] > 0 else -1e18, d),
                )
                layout = SpyreTensorLayout(shape, strides, self.dtype, dim_order)

            if MOCK_DEVICE_ENABLED:
                _mock_print(
                    f"[MOCK_DEVICE] Created SpyreTensorLayout: shape={shape} "
                    f"contiguous={is_contiguous}"
                )
            return layout
        except Exception as e:
            if MOCK_DEVICE_ENABLED:
                _mock_print(f"[MOCK_DEVICE] Warning: Could not create SpyreTensorLayout: {e}")
                import traceback
                traceback.print_exc()
            class MockLayout:
                def __init__(self):
                    self.device_size = ()
                    self.dim_map = {}
                    self.stride_map = {}
            return MockLayout()
    
    def to(self, *args, **kwargs):
        """
        Override to() to handle device transfers and dtype conversions.
        If target is 'spyre', return self (already "on" spyre).
        Dtype-only conversion stays on spyre (returns MockSpyreTensor).
        Device transfer to CPU/other returns a plain tensor.
        """
        # Parse arguments
        device = None
        dtype = None
        for arg in args:
            if isinstance(arg, (str, torch.device)):
                device = torch.device(arg) if isinstance(arg, str) else arg
            elif isinstance(arg, torch.dtype):
                dtype = arg

        device = kwargs.get('device', device)
        dtype = kwargs.get('dtype', dtype)

        # If target is spyre, we're already there
        if device and 'spyre' in str(device):
            if dtype is None or dtype == self.dtype:
                return self
            # dtype change only, stay on spyre
            cpu_result = torch.Tensor._make_subclass(torch.Tensor, self).to(dtype=dtype)
            wrapped = MockSpyreTensor.__new__(MockSpyreTensor, cpu_result)
            wrapped._mock_device = self._mock_device if hasattr(self, '_mock_device') else torch.device("spyre", 0)
            return wrapped

        # Dtype-only conversion (no device specified) — stay on spyre
        if device is None and dtype is not None:
            if MOCK_DEVICE_ENABLED:
                _mock_print(f"[MOCK_DEVICE] dtype-only .to({dtype}) — staying on spyre")
            cpu_result = torch.Tensor._make_subclass(torch.Tensor, self).to(dtype=dtype)
            wrapped = MockSpyreTensor.__new__(MockSpyreTensor, cpu_result)
            wrapped._mock_device = self._mock_device if hasattr(self, '_mock_device') else torch.device("spyre", 0)
            return wrapped

        # Device transfer to CPU / other — return plain tensor
        if MOCK_DEVICE_ENABLED:
            _mock_print(f"[MOCK_DEVICE] Converting MockSpyreTensor to regular tensor for device: {device}")
        regular_tensor = torch.Tensor._make_subclass(torch.Tensor, self)
        if dtype is None and regular_tensor.dtype != self.dtype:
            regular_tensor = regular_tensor.to(dtype=self.dtype)
        return regular_tensor.to(*args, **kwargs)
    
    def cpu(self):
        """Return as regular CPU tensor, preserving dtype and logical view metadata"""
        # During torch.compile the logical-view metadata is not used (Dynamo
        # maintains symbolic shapes itself) and calling storage_offset() on
        # the base tensor would return a concrete int that graph-breaks the
        # trace.  Return the plain subclass-stripped tensor.
        if torch.compiler.is_compiling():
            return torch.Tensor._make_subclass(torch.Tensor, self)

        if MOCK_DEVICE_ENABLED:
            _mock_print(f"[MOCK_DEVICE] Converting MockSpyreTensor to CPU tensor")
        regular_tensor = torch.Tensor._make_subclass(torch.Tensor, self)
        logical_shape = getattr(self, "_mock_logical_shape", None)
        logical_stride = getattr(self, "_mock_logical_stride", None)
        logical_offset = getattr(self, "_mock_storage_offset", None)

        if (
            logical_shape is not None
            or logical_stride is not None
            or logical_offset is not None
        ):
            _mock_debug_print(
                "[MOCK_DEVICE] cpu() logical metadata "
                f"shape={logical_shape} stride={logical_stride} offset={logical_offset} "
                f"base_shape={tuple(regular_tensor.shape)} "
                f"base_stride={tuple(torch.Tensor.stride(regular_tensor))}"
            )

        if (
            logical_shape is not None
            or logical_stride is not None
            or logical_offset is not None
        ):
            base = regular_tensor
            if logical_offset is None:
                logical_offset = base.storage_offset()
            if logical_shape is None:
                logical_shape = tuple(base.shape)
            if logical_stride is None:
                logical_stride = tuple(base.stride())
            regular_tensor = torch.as_strided(
                base, size=logical_shape, stride=logical_stride, storage_offset=logical_offset
            )

        return regular_tensor
    
    def __repr__(self):
        return f"MockSpyreTensor({super().__repr__()}, device='{self.device}')"

    # Reflected arithmetic operators — needed so expressions like
    # ``1.0 - spyre_tensor`` or ``2.0 / spyre_tensor`` work in eager mode.
    # Python tries ``float.__sub__(1.0, tensor)`` first; that returns
    # NotImplemented, then Python calls ``tensor.__rsub__(1.0)``.  Without these
    # the fallback is a TypeError instead of going through __torch_function__.
    def __radd__(self, other):
        return torch.add(other, self)

    def __rsub__(self, other):
        return torch.sub(other, self)

    def __rmul__(self, other):
        return torch.mul(other, self)

    def __rtruediv__(self, other):
        return torch.div(other, self)

    @classmethod
    def __torch_function__(cls, func, types_, args=(), kwargs=None):
        # Check stick dimension compatibility for pointwise operations
        func_name = getattr(func, "__name__", "")
        if func_name in {"add", "mul", "sub", "div"}:
            # Get all MockSpyreTensor inputs
            mock_tensors = [arg for arg in args if isinstance(arg, MockSpyreTensor)]
            
            if len(mock_tensors) >= 2:
                # Extract stick dimensions from each tensor's layout
                stick_dims = []
                for tensor in mock_tensors:
                    layout = getattr(tensor, "_mock_device_layout", None)
                    if layout is not None:
                        dim_map = getattr(layout, "dim_map", None)
                        if dim_map is not None and len(dim_map) > 0:
                            # Stick dimension is the last element of dim_map
                            stick_dim = dim_map[-1]
                            stick_dims.append(stick_dim)
                
                # Check if all stick dimensions are the same
                if len(stick_dims) >= 2:
                    first_stick_dim = stick_dims[0]
                    for stick_dim in stick_dims[1:]:
                        if stick_dim != first_stick_dim:
                            raise RuntimeError(
                                "Spyre limitation: pointwise op with nonuniform stick indexing"
                            )

        kwargs = kwargs or {}

        # Detect CPU tensor × Spyre tensor device mismatch in eager mode.
        # In compiled mode this is handled by the inductor backend.
        if not torch.compiler.is_compiling():
            func_name_check = getattr(func, "__name__", "")
            if func_name_check in {"mul", "add", "sub", "div", "matmul", "__mul__", "__add__"}:
                has_spyre = any(isinstance(a, MockSpyreTensor) for a in args)
                has_cpu_tensor = any(
                    isinstance(a, torch.Tensor)
                    and not isinstance(a, MockSpyreTensor)
                    and getattr(getattr(a, "device", None), "type", "cpu") == "cpu"
                    for a in args
                )
                if has_spyre and has_cpu_tensor:
                    raise RuntimeError(
                        "Device mismatch: cannot mix cpu tensor with spyre tensor"
                    )

        # Redirect torch.nn.functional.* activation composites to their direct
        # aten ops. These Python composites route through __torch_function__ in a
        # way that makes dynamo bake the (subclass) input as a frozen constant
        # instead of keeping it as a graph input. In mock mode that frozen
        # constant is captured as a FakeTensor (no real data), so the compiled
        # Spyre kernel runs on garbage. Builtin ops (e.g. torch.sigmoid) do not
        # have this problem. Rewriting to the aten op makes dynamo trace these the
        # same way as the builtins, keeping the input as a real graph argument.
        import os as _os
        if _os.environ.get("SPYRE_CONST_DBG"):
            print(f"[TF_DBG] func={getattr(func,'__name__','?')} in_map={func in _MOCK_FUNCTIONAL_ATEN}", flush=True)
        _functional_aten = _MOCK_FUNCTIONAL_ATEN.get(func)
        if _functional_aten is not None:
            # F.xxx activations pass `inplace` but their aten counterparts don't accept it.
            filtered_kw = {k: v for k, v in (kwargs or {}).items() if k != "inplace"}
            return _functional_aten(*args, **filtered_kw)

        def _unwrap(x):
            if isinstance(x, MockSpyreTensor):
                return x.cpu()
            if isinstance(x, (tuple, list)):
                return type(x)(_unwrap(v) for v in x)
            if isinstance(x, dict):
                return {k: _unwrap(v) for k, v in x.items()}
            return x

        def _rewrap(x):
            if isinstance(x, MockSpyreTensor):
                return x
            if isinstance(x, torch.Tensor):
                try:
                    wrapped = MockSpyreTensor(x)
                except RuntimeError as e:
                    msg = str(e)
                    if (
                        "already associated to a python object of type FakeTensor" in msg
                        or "Creating a new Tensor subclass" in msg
                    ):
                        return x
                    raise
                wrapped._mock_logical_shape = tuple(x.shape)
                wrapped._mock_logical_stride = tuple(x.stride())
                # x.storage_offset() returns a concrete int.  During
                # torch.compile, Dynamo traces __torch_function__ and inlines
                # _rewrap; the call_method node for storage_offset then returns
                # type `int` instead of a SymInt, causing a graph break that
                # severs the spyre_hint annotation region.  The attribute is
                # only used by cpu() in eager mode, so skip it while compiling.
                if not torch.compiler.is_compiling():
                    wrapped._mock_storage_offset = x.storage_offset()
                return wrapped
            if isinstance(x, (tuple, list)):
                return type(x)(_rewrap(v) for v in x)
            if isinstance(x, dict):
                return {k: _rewrap(v) for k, v in x.items()}
            return x

        def _is_view_like(torch_func):
            name = getattr(torch_func, "__name__", "")
            return name in {
                "unsqueeze",
                "unsqueeze_",
                "squeeze",
                "squeeze_",
                "view",
                "reshape",
                "permute",
                "transpose",
                "t",
                "detach",
                "alias",
                "as_strided",
                # flatten/unflatten are shape-only ops whose Python-level
                # descriptors trigger InternalTorchDynamoError in
                # is_tensor_base_attr_getter when super().__torch_function__
                # calls func(*args, **kwargs) during tracing.  Routing them
                # through DisableTorchFunctionSubclass avoids the breakage.
                "flatten",
                "unflatten",
            }

        def _is_mock_only_cpu_boundary(torch_func, torch_args, torch_kwargs):
            name = getattr(torch_func, "__name__", "")
            has_mock_tensor_arg = any(
                isinstance(arg, MockSpyreTensor) for arg in torch_args
            )
            out_tensor = torch_kwargs.get("out") if isinstance(torch_kwargs, dict) else None
            has_mock_out = isinstance(out_tensor, MockSpyreTensor)

            if name in {"rsqrt", "__getitem__", "copy_", "cat", "sin", "cos", "arange", "where",
                        "to"}:
                return has_mock_tensor_arg or has_mock_out

            if name == "addmm":
                return has_mock_out or has_mock_tensor_arg

            return False

        def _warn_and_run_cpu_fallback(torch_func, torch_args, torch_kwargs):
            from torch_spyre.ops.fallbacks import fallback_ops, warn_fallback

            cpu_args = _unwrap(torch_args)
            cpu_kwargs = _unwrap(torch_kwargs)

            fallback_name = None
            func_name = getattr(torch_func, "__name__", "")
            normalized_func_names = {
                func_name,
                func_name.replace("aten::", ""),
                func_name.replace("aten::", "").split(".")[0],
                func_name.split(".")[0],
            }
            for fallback_op in fallback_ops:
                op_name = getattr(fallback_op, "_name", "")
                if not isinstance(op_name, str):
                    continue
                normalized_op_names = {
                    op_name,
                    op_name.replace("aten::", ""),
                    op_name.replace("aten::", "").split(".")[0],
                    op_name.split(".")[0],
                    op_name.split(".")[-1],
                }
                if normalized_func_names & normalized_op_names:
                    fallback_name = op_name
                    break

            warn_fallback(fallback_name or func_name or str(torch_func))
            result = torch_func(*cpu_args, **cpu_kwargs)

            func_name = getattr(torch_func, "__name__", "")
            if func_name == "copy_" and torch_args:
                dst = torch_args[0]
                if isinstance(dst, MockSpyreTensor):
                    return dst

            if func_name in {"sin", "cos", "arange"}:
                out_tensor = torch_kwargs.get("out") if isinstance(torch_kwargs, dict) else None
                if isinstance(out_tensor, MockSpyreTensor):
                    return out_tensor

            return _rewrap(result)

        # During torch.compile Dynamo handles contiguous() natively as an aten
        # op; returning args[0] here would cause the compiled graph to skip the
        # contiguous call entirely instead of tracing it as an FX node.  Only
        # short-circuit in eager mode.
        if (
            not torch.compiler.is_compiling()
            and getattr(func, "__name__", "") == "contiguous"
            and args
            and isinstance(args[0], MockSpyreTensor)
        ):
            return args[0]

        # Handle w.T: Dynamo calls __torch_function__ with func=torch.Tensor.T.__get__
        # (__name__=="__get__", __self__.__name__=="T").  Falling through to
        # super().__torch_function__ triggers _convert -> as_subclass, which either
        # raises "assert source is None" (graph inputs) or creates an FX-untracked
        # ghost tensor (other tensors), breaking AOT compilation.
        # In Dynamo mode, bypass __torch_function__ entirely via
        # DisableTorchFunctionSubclass so aten.t.default goes straight through
        # __torch_dispatch__ and ProxyTorchDispatchMode registers the result.
        # In eager mode, fall through to the _is_view_like CPU unwrap/rewrap path.
        if (
            getattr(func, "__name__", "") == "__get__"
            and getattr(getattr(func, "__self__", None), "__name__", "") == "T"
            and args
        ):
            if torch._dynamo.is_compiling():
                with torch._C.DisableTorchFunctionSubclass():
                    return torch.ops.aten.t.default(args[0])
            else:
                cpu_args = _unwrap(args)
                result = func(*cpu_args, **{})
                return _rewrap(result)

        if _is_view_like(func):
            # During torch.compile, view ops must be traced as FX nodes so the
            # Spyre inductor can see them (e.g. to insert restickify nodes).
            # The unwrap/rewrap path materialises the op eagerly and Dynamo bakes
            # the result as a frozen constant, breaking coarse-tile hint groups.
            # Bypass __torch_function__ so the op goes straight through
            # __torch_dispatch__ (ProxyTorchDispatchMode registers the result in
            # tensor_tracker and no graph break occurs).
            if torch.compiler.is_compiling():
                with torch._C.DisableTorchFunctionSubclass():
                    return func(*args, **kwargs)
            cpu_args = _unwrap(args)
            cpu_kwargs = _unwrap(kwargs)
            result = func(*cpu_args, **cpu_kwargs)
            return _rewrap(result)

        if _is_mock_only_cpu_boundary(func, args, kwargs):
            if MOCK_DEVICE_ENABLED:
                _mock_print(
                    f"[MOCK_DEVICE] mock-only CPU boundary fallback for {getattr(func, '__name__', func)}"
                )
            # Don't warn for mock-only CPU boundary operations - they're expected to run on CPU
            cpu_args = _unwrap(args)
            cpu_kwargs = _unwrap(kwargs)
            result = func(*cpu_args, **cpu_kwargs)

            # Special handling for copy_ - return the destination tensor
            func_name = getattr(func, "__name__", "")
            if func_name == "copy_" and args:
                dst = args[0]
                if isinstance(dst, MockSpyreTensor):
                    return dst
            
            # Special handling for operations with 'out' parameter
            if func_name in {"sin", "cos", "arange"}:
                out_tensor = kwargs.get("out")
                if isinstance(out_tensor, MockSpyreTensor):
                    return out_tensor
            
            return _rewrap(result)

        # For OpOverload functions called during compilation, avoid
        # super().__torch_function__ which calls _convert -> as_subclass and
        # creates a ghost FakeTensor that is NOT registered in
        # ProxyTorchDispatchMode's tensor_tracker, or triggers functionalization
        # machinery that fails for custom ops with tuple/void returns.
        # Instead, call the op directly with DisableTorchFunctionSubclass so
        # the aten dispatch goes straight through __torch_dispatch__.
        try:
            from torch._ops import OpOverload as _OpOverload
            from torch._subclasses import FakeTensor as _FakeTensor
            try:
                from torch._subclasses.functional_tensor import FunctionalTensor as _FunctionalTensor
            except ImportError:
                _FunctionalTensor = None
            _first_arg = args[0] if args else None
            if isinstance(func, _OpOverload) and _first_arg is not None:
                _is_fake_or_functional = isinstance(_first_arg, _FakeTensor) or (
                    _FunctionalTensor is not None and isinstance(_first_arg, _FunctionalTensor)
                )
                # Check if we're inside tracing with any arg being a FakeTensor/FunctionalTensor
                if not _is_fake_or_functional and torch.compiler.is_compiling():
                    _is_fake_or_functional = any(
                        isinstance(a, _FakeTensor) or (
                            _FunctionalTensor is not None and isinstance(a, _FunctionalTensor)
                        )
                        for a in args
                        if isinstance(a, torch.Tensor)
                    )
                # Also bypass super().__torch_function__ for custom (non-aten) ops
                # during torch.compile: super() triggers FunctionalTensorMode dispatch
                # which fails for ops with tuple returns or in-place void returns.
                _is_custom_op = (
                    torch.compiler.is_compiling()
                    and not _is_fake_or_functional
                    and hasattr(func, 'namespace')
                    and func.namespace not in ("aten", "prims", "prim", "spyre")
                )
                if _is_fake_or_functional or _is_custom_op:
                    with torch._C.DisableTorchFunctionSubclass():
                        return func(*args, **kwargs)
        except (ImportError, AttributeError, TypeError):
            pass

        try:
            result = super().__torch_function__(func, types_, args, kwargs)
        except RuntimeError as e:
            msg = str(e)
            if "PyTorch is not linked with support for spyre devices" not in msg:
                raise
            if MOCK_DEVICE_ENABLED:
                _mock_print(
                    f"[MOCK_DEVICE] __torch_function__ fallback for {getattr(func, '__name__', func)}"
                )
            out_tensor = kwargs.get("out")
            cpu_kwargs = _unwrap(kwargs)
            cpu_out_tensor = cpu_kwargs.get("out") if isinstance(cpu_kwargs, dict) else None
            result = _warn_and_run_cpu_fallback(func, args, kwargs)

            if isinstance(out_tensor, MockSpyreTensor) and isinstance(cpu_out_tensor, torch.Tensor):
                return MockSpyreTensor(cpu_out_tensor)

            return result
        except Exception as e:
            # Dynamo's SuperVariable path may raise InternalTorchDynamoError
            # (e.g. AttributeError: 'function' has no attribute '__objclass__')
            # for ops like unflatten that have Python-level descriptors rather
            # than C TensorBase method-wrappers.  Fall through to
            # DisableTorchFunctionSubclass so the op is traced natively.
            try:
                from torch._dynamo.exc import InternalTorchDynamoError
                if isinstance(e, InternalTorchDynamoError):
                    if torch.compiler.is_compiling():
                        with torch._C.DisableTorchFunctionSubclass():
                            return func(*args, **kwargs)
            except ImportError:
                pass
            raise
        else:
            copy_src = getattr(result, "_mock_spyre_copy_src", None)
            if copy_src is not None:
                func_name = getattr(func, "__name__", "")
                if func_name == "copy_" and args:
                    dst = args[0]
                    dst_copy_src = getattr(dst, "_mock_spyre_copy_src", None)
                    if dst_copy_src is not None:
                        return _rewrap(_unwrap(dst_copy_src))
                    return _rewrap(_unwrap(dst))

                out_tensor = kwargs.get("out")
                if isinstance(out_tensor, MockSpyreTensor):
                    out_copy_src = getattr(out_tensor, "_mock_spyre_copy_src", None)
                    if out_copy_src is not None:
                        return _rewrap(_unwrap(out_copy_src))

                if func_name in {"add_", "mul_"} and args:
                    dst = args[0]
                    if isinstance(dst, MockSpyreTensor):
                        return dst

                return _rewrap(_unwrap(copy_src))

            if getattr(func, "__name__", "") in {"add_", "mul_"} and args:
                dst = args[0]
                if isinstance(dst, MockSpyreTensor):
                    return dst

            if any(isinstance(arg, MockSpyreTensor) for arg in args):
                func_name = getattr(func, "__name__", "")
                if func_name:
                    try:
                        from torch_spyre.ops.fallbacks import fallback_ops, warn_fallback

                        normalized_func_names = {
                            func_name,
                            func_name.replace("aten::", ""),
                            func_name.replace("aten::", "").split(".")[0],
                            func_name.split(".")[0],
                        }
                        for fallback_op in fallback_ops:
                            op_name = getattr(fallback_op, "_name", "")
                            if not isinstance(op_name, str):
                                continue
                            normalized_op_names = {
                                op_name,
                                op_name.replace("aten::", ""),
                                op_name.replace("aten::", "").split(".")[0],
                                op_name.split(".")[0],
                                op_name.split(".")[-1],
                            }
                            if normalized_func_names & normalized_op_names:
                                warn_fallback(op_name)
                                break
                    except Exception:
                        pass

            return result


def to_mock_spyre(tensor):
    """
    Convert a regular tensor to MockSpyreTensor.
    
    This is the key function that enables the mock device workflow:
    1. Takes CPU tensor
    2. Wraps in MockSpyreTensor (reports as spyre device)
    3. torch.compile() sees spyre device
    4. Uses Spyre inductor backend
    5. Generates real SDSC JSON
    """
    if isinstance(tensor, MockSpyreTensor):
        return tensor
    
    if MOCK_DEVICE_ENABLED:
        _mock_print(f"[MOCK_DEVICE] Converting tensor to mock Spyre device")
    
    return MockSpyreTensor(tensor)


# Monkey-patch tensor.to() to intercept "spyre" device requests
_original_tensor_to = torch.Tensor.to

def _patched_tensor_to(self, *args, **kwargs):
    """Patched version of tensor.to() that handles 'spyre' device"""
    # Check if target is spyre device - need to catch before PyTorch validates
    try:
        # Check if target is spyre device
        device = None
        for arg in args:
            if isinstance(arg, (str, torch.device)):
                device_str = str(arg)
                if 'spyre' in device_str.lower():
                    device = arg
                    break
        
        if not device:
            device = kwargs.get('device')
            if device and 'spyre' in str(device).lower():
                pass
            else:
                device = None
        
        # If target is spyre, convert to MockSpyreTensor
        if device:
            if MOCK_DEVICE_ENABLED:
                _mock_print(f"[MOCK_DEVICE] Intercepted .to('spyre') call")
            return to_mock_spyre(self)
        
        # Otherwise use original implementation
        return _original_tensor_to(self, *args, **kwargs)
    except RuntimeError as e:
        # If PyTorch rejects the device, check if it's spyre and handle it
        if 'spyre' in str(e).lower() or 'privateuse1' in str(e).lower():
            if MOCK_DEVICE_ENABLED:
                _mock_print(f"[MOCK_DEVICE] Caught PyTorch device error, converting to MockSpyreTensor")
            return to_mock_spyre(self)
        raise


def install_mock_device_patches():
    """Install patches to enable mock Spyre device"""
    if not MOCK_DEVICE_ENABLED:
        return
    
    _mock_print("[MOCK_DEVICE] Installing tensor.to() patch for Spyre device")
    torch.Tensor.to = _patched_tensor_to  # type: ignore
    _mock_print("[MOCK_DEVICE] Mock device patches installed successfully")


# Auto-install if mock device is enabled
if MOCK_DEVICE_ENABLED:
    install_mock_device_patches()
