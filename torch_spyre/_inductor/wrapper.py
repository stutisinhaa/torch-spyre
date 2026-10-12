# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Optional

import sympy
from torch._inductor.codegen.wrapper import (
    BufferLike,
    PythonWrapperCodegen,
    SubgraphPythonWrapperCodegen,
)
from torch._inductor import config, ir
from torch._dynamo.utils import counters
from torch._inductor.ir import GraphPartitionSignature
from torch._inductor.virtualized import V
from torch._inductor.sizevars import SizeVarAllocator

from .errors import Unsupported
from .ir import FixedTiledLayout


<<<<<<< Updated upstream
class _SpyreWrapperCodegenMixin(PythonWrapperCodegen):
    """Spyre wrapper-codegen behavior shared by the top-level graph wrapper and
    the invoke_subgraph wrapper.

    These overrides are about *how a Spyre buffer/op is emitted* (device-layout
    allocation, HBM-pool reuse/free, constant-tensor fallback). They are
    graph-role-agnostic, so both the parent graph and a nested_compile_region
    subgraph need them. Role-specific behavior (header emission, benchmark
    harness, launcher naming) is NOT here — it stays on the concrete wrapper
    classes, so a subgraph keeps the stock ``write_header = pass`` and does not
    double-emit imports.
    """

    def _patch_sizevars(self) -> None:
        # Spyre device layout is not fully visible to Inductor, so its loop
        # simplification would be wrong (see noop_simplify_loops_impl). Applied
        # per GraphLowering, so the subgraph's own sizevars is patched too.
=======
class SpyreSubgraphPythonWrapperCodegen(SubgraphPythonWrapperCodegen):
    """Custom subgraph wrapper that supports generate_const_tensor_fallback."""

    def generate_const_tensor_fallback(self, node):
        # Check if we're in mock device mode
        if os.environ.get("TORCH_SPYRE_MOCK_DEVICE", "0") == "1":
            # In mock mode, create tensor and move to spyre device (MockSpyreTensor handles this)
            value = node.constant_args[0]
            dtype = node.layout.dtype
            device = node.layout.device
            self.writeline(f'{node.get_name()} = torch.tensor({value}, dtype={dtype}).to("{device}")')
        else:
            # Hardware mode: use spyre_constant_tensor
            value = node.constant_args[0]
            dtype = node.layout.dtype
            device = node.layout.device
            self.writeline(
                f'{node.get_name()} = spyre_constant_tensor({value}, torch.device("{device}"), {dtype})'
            )


class SpyrePythonWrapperCodegen(PythonWrapperCodegen):
    def __init__(self):
        super().__init__()
>>>>>>> Stashed changes
        V.graph.sizevars._simplify_loops_impl = noop_simplify_loops_impl.__get__(
            V.graph.sizevars, SizeVarAllocator
        )

<<<<<<< Updated upstream
=======
    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: Optional[str],
        parent_wrapper: Optional[PythonWrapperCodegen],
        partition_signatures: Optional[GraphPartitionSignature] = None,
    ):
        if is_subgraph:
            assert subgraph_name is not None
            assert parent_wrapper is not None
            return SpyreSubgraphPythonWrapperCodegen(
                subgraph_name, parent_wrapper, partition_signatures
            )
        return SpyrePythonWrapperCodegen()

    def write_header(self) -> None:
        super().write_header()
        self.imports.splice(
            """
                from sympy import sympify
                from torch_spyre._inductor.op_spec import TensorArg, OpSpec, UnimplementedOp, LoopSpec, spyre_constant_tensor, IndirectAccess
                from torch_spyre.execution.async_compile import SpyreAsyncCompile
                from torch_spyre._C import DataFormats, SpyreTensorLayout, spyre_empty_with_layout
                import subprocess
            """,
            strip=True,
        )
        self.header.writeline(
            "from torch_spyre._C import reinterpret_tensor as reinterpret_tensor"
        )
        self.header.writeline(
            "from torch_spyre._C import reinterpret_tensor_with_layout"
        )
        self.header.writeline("del async_compile")
        self.header.writeline("async_compile = SpyreAsyncCompile()")

    def generate(self, is_inference):
        """Override to add pool allocation/deallocation around kernel calls."""
        result_tuple = super().generate(is_inference)
        wrapper_value_with_linemap, kernel_decls = result_tuple

        # Build a map of {real_buf: alias_buf} for MutationLayoutSHOULDREMOVE
        # buffers that were removed.  When the scheduler stores a reduction into
        # a pre-allocated buffer (e.g. buf1 = empty_default, buf0 = matmul result
        # mutating buf1), buf0 is added to removed_buffers and the kernel writes
        # to buf1.  The generated return statement still references buf0, so we
        # must emit "buf0 = buf1" after the buf1 allocation to define the alias.
        scheduler = getattr(V.graph, "scheduler", None)
        mutation_real_name = (
            getattr(scheduler, "mutation_real_name", {}) if scheduler else {}
        )
        removed = V.graph.removed_buffers
        alias_map = {
            real: alias
            for alias, real in mutation_real_name.items()
            if alias in removed
        }

        needs_str_edit = alias_map or getattr(V.graph, "pool_size", 0) > 0
        if needs_str_edit:
            wrapper_str = str(wrapper_value_with_linemap.value)
            lines = wrapper_str.split("\n")

            # Insert "alias = real" after each "real = spyre_empty_with_layout(...)"
            # line so that removed mutation buffers are defined in the generated code.
            if alias_map:
                i = 0
                while i < len(lines):
                    stripped = lines[i].strip()
                    for real_name, alias_name in alias_map.items():
                        if stripped.startswith(f"{real_name} = spyre_empty_with_layout("):
                            indent = len(lines[i]) - len(lines[i].lstrip())
                            lines.insert(i + 1, " " * indent + f"{alias_name} = {real_name}")
                            break
                    i += 1

            pool_size = getattr(V.graph, "pool_size", 0)
            if pool_size > 0:
                # Add `del _pool` before `return (` statement.
                for i in range(len(lines) - 1, -1, -1):
                    line = lines[i].strip()
                    if line.startswith("return ("):
                        indent = len(lines[i]) - len(lines[i].lstrip())
                        lines.insert(i, " " * indent + "del _pool")
                        break

                # Add pool allocation before first kernel call (`.run(`).
                pool_alloc_code = self.allocate_pool()
                for i, line in enumerate(lines):
                    if ".run(" in line:
                        indent = len(line) - len(line.lstrip())
                        lines.insert(i, " " * indent + pool_alloc_code)
                        break

            wrapper_str = "\n".join(lines)
            wrapper_value_with_linemap = ValueWithLineMap(
                value=wrapper_str, line_map=wrapper_value_with_linemap.line_map
            )

        return (wrapper_value_with_linemap, kernel_decls)

>>>>>>> Stashed changes
    def make_buffer_allocation(self, buffer: BufferLike):
        layout = buffer.get_layout()
        if not isinstance(layout, FixedTiledLayout):
            return super().make_buffer_allocation(buffer)

        name = buffer.get_name()
        codegen_shape_tuple = self.codegen_python_shape_tuple(tuple(layout.size))
        codegen_stride_tuple = self.codegen_python_shape_tuple(tuple(layout.stride))

        out = (
            f"{name} = spyre_empty_with_layout("
            f"{codegen_shape_tuple}, "
            f"{codegen_stride_tuple}, "
            f"{layout.dtype}, "
            f"{layout.device_layout!r}, "
            f"device=torch.device('{layout.device}'))"
        )

        return out

    def generate_const_tensor_fallback(self, node):
        # Check if we're in mock device mode
        if os.environ.get("TORCH_SPYRE_MOCK_DEVICE", "0") == "1":
            # In mock mode, create tensor and move to spyre device (MockSpyreTensor handles this)
            value = node.constant_args[0]
            dtype = node.layout.dtype
            device = node.layout.device
            self.writeline(f'{node.get_name()} = torch.tensor({value}, dtype={dtype}).to("{device}")')
        else:
            # Hardware mode: use spyre_constant_tensor
            value = node.constant_args[0]
            dtype = node.layout.dtype
            device = node.layout.device
            self.writeline(
                f'{node.get_name()} = spyre_constant_tensor({value}, torch.device("{device}"), {dtype})'
            )

    def _generate_extern_kernel_alloc_helper(self, extern_kernel, args):
        """Mark calls whose eager result is consumed inside this graph.

        Spyre eager kernels normally preserve their actual device layout at an
        eager boundary.  A ``FallbackKernel`` output, however, is assigned a
        canonical layout by Inductor before the runtime eager call occurs.  The
        marker lets the eager dispatcher normalize only this in-graph case.
        """
        if not isinstance(extern_kernel, ir.FallbackKernel):
            return super()._generate_extern_kernel_alloc_helper(extern_kernel, args)

        no_return = isinstance(extern_kernel.layout, ir.NoneLayout)
        output_name = extern_kernel.get_name()
        origin_node = extern_kernel.get_origin_node()
        kernel_name = extern_kernel.get_kernel_name()
        ending = self.ending
        if config.memory_planning and "view_as_complex" in kernel_name:
            ending = f".clone(){ending}"

        comma = ", " if args else ""
        call = f"_run_compiled_fallback({kernel_name}{comma}{', '.join(args)}){ending}"
        if no_return:
            self.writeline(f"{self.declare}{call}")
        else:
            self.writeline(f"{self.declare}{output_name} = {call}")
            if (
                self.supports_intermediate_hooks
                and config.generate_intermediate_hooks
                and origin_node is not None
            ):
                counters["inductor"]["intermediate_hooks"] += 1
                self.writeline(
                    f"run_intermediate_hooks({origin_node.name!r}, {output_name})"
                )

    def generate_fallback_kernel_with_runtime_lookup(
        self,
        buf_name,
        python_kernel_name,
        get_args,
        op_overload,
        raw_args,
        outputs,
    ) -> None:
        """Apply the same marker to runtime-looked-up fallback overloads."""
        args = list(get_args())
        comma = ", " if args else ""
        self.writeline(
            f"{buf_name} = _run_compiled_fallback("
            f"{python_kernel_name}{comma}{', '.join(args)})"
        )

    def _is_hbm_pool_buffer(self, buffer: BufferLike) -> bool:
        layout = buffer.get_layout()
        return isinstance(layout, FixedTiledLayout) and "hbm_pool" in layout.allocation

    def codegen_free_buffer(self, buffer: BufferLike) -> None:
        if not self._is_hbm_pool_buffer(buffer):
            super().codegen_free_buffer(buffer)

    def make_buffer_reuse(self, old: BufferLike, new: BufferLike, delete_old: bool):
        assert old.get_dtype() == new.get_dtype()
        old_name = old.get_name()
        new_name = new.get_name()
        del_line = ";"
        if old_name not in V.graph.get_output_names() and delete_old:
            del_line = f"; {self.make_buffer_free(old)}"

        new_offset = new.get_layout().offset or 0
        old_offset = old.get_layout().offset or 0
        if (
            old.get_size() == new.get_size()
            and old.get_stride() == new.get_stride()
            and old_offset == new_offset
        ):
            return self.codegen_exact_buffer_reuse(old_name, new_name, del_line)

        new_stl = new.get_layout().device_layout
        # reinterpret_tensor_with_layout's offset arg is added to old_name's
        # *current* runtime storage_offset (see spyre_views.cpp), not treated
        # as an absolute target. Passing old's offset directly double-counts
        # it; the delta to new's offset is what actually shifts to the right
        # place.
        offset_increment = new_offset - old_offset
        # Static-shape paths only: a symbolic offset would render a bare
        # sympy expression into the generated source below.
        if getattr(offset_increment, "free_symbols", None):
            raise Unsupported(
                f"symbolic storage_offset not supported in buffer-reuse codegen: "
                f"{offset_increment!r}"
            )
        reinterpret_view = f"reinterpret_tensor_with_layout({old_name}, {new.get_size()}, {new.get_stride()}, {offset_increment}, {new_stl!r})"
        return f"{self.declare}{new_name} = {reinterpret_view}{del_line}  {self.comment} reuse"


class SpyrePythonWrapperCodegen(_SpyreWrapperCodegenMixin, PythonWrapperCodegen):
    def __init__(self):
        super().__init__()
        self._patch_sizevars()

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: Optional[str],
        parent_wrapper: Optional[PythonWrapperCodegen],
        partition_signatures: Optional[GraphPartitionSignature] = None,
    ):
        if is_subgraph:
            assert subgraph_name is not None
            assert parent_wrapper is not None
            return SpyreSubgraphPythonWrapperCodegen(
                subgraph_name, parent_wrapper, partition_signatures
            )
        return SpyrePythonWrapperCodegen()

    def write_header(self) -> None:
        super().write_header()
        self.imports.splice(
            """
                from sympy import sympify
                from torch_spyre._inductor.op_spec import TensorArg, TensorWorkDivision, OpSpec, UnimplementedOp, LoopSpec, spyre_constant_tensor, IndirectAccess, DebugHandle, SourceLoc, ProvenanceTransform
                from torch_spyre.execution.async_compile import SpyreAsyncCompile
                from torch_spyre._C import DataFormats, ElementArrangement, SpyreTensorLayout, spyre_empty_with_layout, set_spyre_tensor_layout
                import subprocess
                from torch_spyre.ops.eager import _run_compiled_fallback
            """,
            strip=True,
        )
        self.header.writeline(
            "from torch_spyre._C import reinterpret_tensor as reinterpret_tensor"
        )
        self.header.writeline(
            "from torch_spyre._C import reinterpret_tensor_with_layout"
        )
        self.header.writeline("del async_compile")
        self.header.writeline("async_compile = SpyreAsyncCompile()")


class SpyreSubgraphPythonWrapperCodegen(
    _SpyreWrapperCodegenMixin, SubgraphPythonWrapperCodegen
):
    """Spyre wrapper for an invoke_subgraph body (e.g. a nested_compile_region
    decoder block reused across layers).

    Inherits the stock subgraph plumbing (launcher naming, empty ``write_header``
    so imports are not re-emitted, input/output signature handling) from
    ``SubgraphPythonWrapperCodegen`` and layers the Spyre buffer/op codegen on
    top via the mixin. Without this, subgraph codegen fell back to the stock
    wrapper and hit ``AttributeError`` the moment it emitted a Spyre buffer
    (e.g. a SpyreConstantFallback materialized by split_multi_ops).
    """

    def __init__(
        self,
        subgraph_name: str,
        parent_wrapper: PythonWrapperCodegen,
        partition_signatures: Optional[GraphPartitionSignature] = None,
    ):
        super().__init__(subgraph_name, parent_wrapper, partition_signatures)
        self._patch_sizevars()


def noop_simplify_loops_impl(
    self, index_vars: list[sympy.Symbol], sizes, index_formulas
):
    """
    This is a noop implementation of SizeVarAllocator._simplify_loops_impl.

    We do this because the memory layout of tensors on the Spyre device is not
    entirely visible to Inductor.  Therefore Inductor's understanding of which
    tensor dimensions are actually contiguous is not accurate.
    """
    return sizes, lambda x: x, lambda x: x