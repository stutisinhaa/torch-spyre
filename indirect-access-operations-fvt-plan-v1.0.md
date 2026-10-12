# Indirect Access Operations FVT Plan — torch.gather / torch.index_select / torch.nn.Embedding

---

| Field | Value |
|---|---|
| **Document Title** | FVT Plan — Indirect Access Operations: torch.gather / torch.index_select / torch.nn.Embedding |
| **Version** | 1.0 |
| **Status** | Draft — Initial — Pending Developer Review |
| **Date** | 2026-06-14 |
| **Author** | Mahalakshmi Pathivada |
| **Reviewers** | Shruthip Venkatesh, Tanushree Haldar |
| **Approver** | Albin Joy, Pradipta Ghosh, ANTO JOHN |

---

## Document History

| Version | Date | Author | Changes |
|---|---|---|---|
| 1.0 | 2026-06-14 | Mahalakshmi Pathivada | Initial draft: FVT coverage for torch.gather, torch.index_select, and torch.nn.Embedding indirect access operations on torch-spyre. Section 1 tracks ~385 upstream YAML tests; Section 3 documents 63 developer unit tests; Sections 4–9 define 345 net-new FVT test cases (gather: 146, index_select: 86, embedding: 52, cross-op: 14, paged attention: 32, unit: 15), each count verified directly against the implemented test files in tests/indirect_access/. |

---

## Table of Contents

| Section | Title | Test Count |
|---|---|---|
| 1 | Upstream PyTorch Test Coverage and YAML Configurations | ~385 tracked (4 upstream YAML files) |
| 2 | torch-spyre Existing Tests | 1 |
| 3 | torch-spyre Unit Tests (`tests/indirect_access/`) | 63 |
| 4 | torch.gather — Test Cases | 146 net-new |
| 5 | torch.index_select — Test Cases | 86 net-new |
| 6 | torch.nn.Embedding — Test Cases | 52 net-new |
| 7 | Cross-Operation Tests | 14 net-new |
| 8 | Paged Attention Comprehensive Coverage | 32 net-new |
| 9 | Unit Tests | 15 net-new |
| — | Summary, Coverage Matrix, Risk Register, Sign-off | — |

**Net-new FVT test cases to write: 345** (146 gather + 86 index_select + 52 embedding + 14 cross-op + 32 paged attention + 15 unit)
**Grand total tracked: ~794** (~385 upstream + 1 existing + 63 developer + 345 net-new)

> **Explicitly excluded from scope:**
> - Training / backward pass — torch-spyre is inference-only
> - `torch.jit.script` / JIT path — not a supported execution mode
> - Performance benchmarks / latency — owned by the performance team

---

## Reference Documents

| Tag | Document | Location | Relevance |
|---|---|---|---|
| `FRS-GG` | FRS — torch.gather Operation | `Epic-866/FRS/FRS torch.gather.docx` | **Primary** — FR-1 through FR-8 for gather behavior; all functional coverage traces here |
| `FRS-AC` | FRS — Index-to-Address Translation | `Epic-866/FRS/FRS Address Computation.docx` | **Primary** — Address computation pipeline requirements; all AC and SD tests trace here |

---

## Test Environment and Prerequisites

### Environment Variables

| Variable | Default | Value Used | When Required |
|---|---|---|---|
| `SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS` | `0` | `1` | Master gate for all indirect access ops |
| `SENCORES` | `32` | `1` (single-core), `4`, `32` | Set to `1` for all current tests; MC tests use `4` or `32` |
| `LX_PLANNING` | `1` | `1` | LX scratchpad planning |
| `CO_OPTIMIZING_LX_PLANNING` | `0` | `1` | Co-optimizing LX allocator; requires `LX_PLANNING=1` |
| `CHUNK_LARGE_TENSORS` | `0` | `1` | Chunk path — NOT yet integrated with indirect access |
| `GLOBAL_STICK_OPTIMIZER` | `1` | `0` | Disable for GSO-off comparison tests |
| `SPYRE_INDUCTOR_ENABLE_FUSION` | `1` | `0` | Disable fusion for isolation tests |
| `SPYRE_INDUCTOR_IGNORE_HINTS` | `0` | `1` | Ignore coarse-tiling annotations |
| `BUNDLE_SYMBOLIC_ARGS` | `0` | `1` | Emit HBM addresses as runtime symbols |
| `UNROLL_LOOPS` | `1` | `0` | Disable loop unrolling for LoopSpec pass-through |
| `DXP_LX_FRAC_AVAIL` | `0.2` | `0.1`, `0.2`, `0.5`, `0.8` | Scratchpad budget fraction |
| `TORCH_SPYRE_DEBUG` | `0` | `1` | Enable C++ debug logging |
| `TORCH_COMPILE_DEBUG` | `0` | `1` | Dump Inductor debug artifacts |

### Environment Shorthand

| Shorthand | Expands To |
|---|---|
| `ENV_BASE` | `SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS=1 SENCORES=1` |
| `ENV_GATE_OFF` | `SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS=0 SENCORES=1` |
| `ENV_DYN` | `ENV_BASE` + `torch.compile(dynamic=True)` |
| `ENV_LX` | `ENV_BASE` + `LX_PLANNING=1` |
| `ENV_LX_CO` | `ENV_BASE` + `LX_PLANNING=1 CO_OPTIMIZING_LX_PLANNING=1` |
| `ENV_MC4` | `SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS=1 SENCORES=4` |
| `ENV_MC32` | `SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS=1 SENCORES=32` |
| `ENV_CHUNK` | `ENV_BASE` + `CHUNK_LARGE_TENSORS=1` |
| `ENV_GSO_OFF` | `ENV_BASE` + `GLOBAL_STICK_OPTIMIZER=0` |
| `ENV_NO_FUSION` | `ENV_BASE` + `SPYRE_INDUCTOR_ENABLE_FUSION=0` |
| `ENV_NO_HINTS` | `ENV_BASE` + `SPYRE_INDUCTOR_IGNORE_HINTS=1` |
| `CPU_ONLY` | No Spyre device; unit tests using mocks or direct C++ op calls |

### Status Definitions

> **Nothing in this plan has been executed.** Every status below other than `READY` is a *plan-level* risk annotation derived from developer comments, FRS notes, or dev-team-reported issues — it is not a confirmed execution result, and the actual test implementations in `tests/indirect_access/` do **not** mark any of these as `xfail`. Every written test runs as a normal test; its real pass/fail outcome is unknown until someone actually runs it against real hardware. Treat the `BLOCKED:*` labels below as "here's why we suspect this might not pass yet," not "this is confirmed to fail."

| Status | Meaning |
|---|---|
| `READY` | Can be written and run today; no known reason to suspect failure — not the same as confirmed-passing, since nothing has been executed |
| `BLOCKED:HW` | Developer comment indicates potential hardware issue (rank>2 tensors); suspected, not independently confirmed by running this suite |
| `BLOCKED:MC` | Developer TODO comment indicates multi-core path is incomplete; suspected, not independently confirmed |
| `BLOCKED:CHUNK` | Developer TODO indicates chunk pass not yet integrated with indirect access; suspected, not independently confirmed |
| `NEEDS:INFRA` | Requires SDSC JSON capture tooling before test can execute |
| `BLOCKED:FX` | PyTorch-valid pattern; FX pass / address-computation kernel does not yet implement it per direct code review (software gap, not a hardware rank limitation) — the code-level gap itself was verified by reading source, but the test's actual runtime behavior has not been executed/confirmed |
| `BLOCKED:#2662` | Tracked bug, **confirmed by the dev team at one specific reported configuration** (single-SDSC bundle, repro `sdsc_fused_index_0_9nrav2b4`, shape mb=3/out=1024): `RuntimeError: Failed to parse SpyreCode command: Device pointer ... out of program segment bounds`. Whether this reproduces for *other* single-SDSC shapes/tests is an open question — nothing in this FVT suite has been executed against real hardware/the `dxp` toolchain to confirm that, so this status is not applied beyond the one tagged test until someone actually runs it. https://github.com/torch-spyre/torch-spyre/issues/2662 |
| `BLOCKED:#2661` | Tracked bug, **confirmed by the dev team at one specific reported configuration** (multi-SDSC bundle, repro `sdsc_fused_exp_index_0_7vgnfcm1`, 3D gather/index → separate `exp` kernel): job plan missing the host-side address-computation/transfer step. Whether this reproduces for other multi-SDSC shapes/tests is unverified — same caveat as `BLOCKED:#2662`. https://github.com/torch-spyre/torch-spyre/issues/2661 |
| `BLOCKED:#2654` | Tracked bug, confirmed by the dev team via the `dxp` CLI directly on a bundle: `primaryDsInfo_` dimension order is incorrectly significant to the compiler (should be a set). Not reachable from `torch.compile` — the order is chosen internally by codegen, not by any Python-level argument — so no FVT-level test in this suite can exercise it either way. https://github.com/torch-spyre/torch-spyre/issues/2654 |
| `BLOCKED:#2653` | Tracked bug, confirmed by the dev team via `dxp_standalone` directly on a bundle: corelet splitting (`SENCORELETS` unset/>1, the default) crashes for indirect-access SDSCs (`DtException: A transfer cannot move more than one page at time`); only passes with `SENCORELETS=1`. Verified by source inspection that `SENCORELETS` is not read anywhere in torch-spyre's Python code, so no `torch.compile`-level FVT test can exercise this either. https://github.com/torch-spyre/torch-spyre/issues/2653 |
| `PASS` | Currently passing |
| `FAIL` | Currently failing |
| `DISABLED` | Assertion commented out |
| `XFAIL` | Expected failure with linked bug |

### Test Type Definitions

| Type | Meaning |
|---|---|
| `UNIT` | Pure Python / mock-based; no Spyre hardware required |
| `FUNC` | Functional correctness test; requires Spyre hardware |
| `INTG` | Integration test; verifies two or more components together |
| `E2E` | End-to-end compiled inference; requires full torch.compile pipeline |
| `INSPECT` | Inspects compiled IR, FX graph, or SDSC JSON output |
| `CPP` | Directly calls registered C++ op; requires Spyre build |

---

## Section 1 — Upstream PyTorch Test Coverage and YAML Configurations

> Four upstream YAML files are tracked in `tests/configs/upstream_tests_beta/`. Each YAML maps 1:1 to one upstream PyTorch test file. Tables below list **every test declared in each YAML** with its configured mode.
>
> **Mode legend:** `mandatory_success` — must pass post-merge; `xfail` — expected failure (CI green when failing). All 4 files use `unlisted_test_mode: skip` — tests not listed are excluded.
>
> **Supported ops:** `torch.gather`, `torch.index_select`, `torch.nn.Embedding` / `F.embedding` → `mandatory_success` where implemented on Spyre. FRS Ref column identifies limitations for xfail entries.

---

### 1.1 `test_scatter_gather_ops.yaml`

> **Upstream file:** `${TORCH_ROOT}/test/test_scatter_gather_ops.py`
> **Global dtypes:** float16, float32, int64, bfloat16, bool
> **Primary CI gate for:** `torch.gather` (aten.gather.default → `_add_indices_to_address_gather`)
> **Total:** 16 tests (2 mandatory_success + 14 xfail)

| Test Name | Op | YAML Mode | Notes / FRS Ref |
|---|---|---|---|
| `TestScatterGather::test_gather` | torch.gather | mandatory_success | float32 (complex64 excluded); shapes (1,1),(10,5),(100,50); FRS-GG FR-1,FR-3 |
| `TestScatterGather::test_gather_bool` | torch.gather | mandatory_success | bool dtype; FRS-GG FR-1 |
| `TestScatterGather::test_gather_large` | torch.gather | xfail | bfloat16 (int8 excluded); stick expr `2*(Mod(d1,32))` rejected by `_check_stick_expr_supported`; FRS-GG FR-1,FR-6 |
| `TestScatterGather::test_gather_expanded_index` | torch.gather | xfail | float32,bfloat16; zero-stride expanded index not integrated in restickify pipeline; FRS-GG FR-2,FR-3 |
| `TestScatterGather::test_scatter_` | torch.scatter_ | xfail | Write-direction indirect access; not in Epic-866 scope |
| `TestScatterGather::test_scatter__scalar` | torch.scatter_ | xfail | Scalar src variant; not in scope |
| `TestScatterGather::test_scatter__reductions` | torch.scatter_ | xfail | Reduce-mode scatter write; not in scope |
| `TestScatterGather::test_scatter_expanded_index` | torch.scatter | xfail | Zero-stride expanded index scatter; not in scope |
| `TestScatterGather::test_scatter_add_` | torch.scatter_add_ | xfail | Accumulate scatter; not in scope |
| `TestScatterGather::test_scatter_add_mult_index_base` | torch.scatter_add_ | xfail | Multi-index base accumulate; not in scope |
| `TestScatterGather::test_scatter_add_broadcasted_index_deterministic` | torch.scatter_add_ | xfail | Deterministic accumulate; not in scope |
| `TestScatterGather::test_scatter_reduce_sum` | torch.scatter_reduce | xfail | Sum reduce scatter; not in scope |
| `TestScatterGather::test_scatter_reduce_prod` | torch.scatter_reduce | xfail | Product reduce; not in scope |
| `TestScatterGather::test_scatter_reduce_mean` | torch.scatter_reduce | xfail | Mean reduce; not in scope |
| `TestScatterGather::test_scatter_reduce_amax` | torch.scatter_reduce | xfail | Amax reduce; not in scope |
| `TestScatterGather::test_scatter_reduce_amin` | torch.scatter_reduce | xfail | Amin reduce; not in scope |

---

### 1.2 `test_indexing.yaml`

> **Upstream file:** `${TORCH_ROOT}/test/test_indexing.py`
> **Global dtypes:** float16, float32, int64, bfloat16, bool
> **Primary CI gate for:** `torch.index_select` (aten.index.Tensor → `_add_indices_to_address_index`)
> **Total:** 82 tests (22 mandatory_success + 60 xfail)

| Test Name | Op | YAML Mode | Notes / FRS Ref |
|---|---|---|---|
| `NumpyTests::test_boolean_assignment_value_mismatch` | General indexing | mandatory_success | Error fires at Python layer before device dispatch |
| `NumpyTests::test_everything_returns_views` | General indexing | mandatory_success | Slice always returns a view |
| `NumpyTests::test_index_no_floats` | General indexing | mandatory_success | Float rejected as index type |
| `NumpyTests::test_none_index` | General indexing | mandatory_success | None as index dimension |
| `NumpyTests::test_truncate_leading_1s` | General indexing | mandatory_success | Leading size-1 dim truncation |
| `TestIndexing::test_empty_slice` | General indexing | mandatory_success | Empty slice → empty tensor |
| `TestIndexing::test_index_put_accumulate_empty` | General indexing | mandatory_success | `index_put` with empty index (no-op) |
| `TestIndexing::test_invalid_index` | General indexing | mandatory_success | Invalid index type → error |
| `TestIndexing::test_multiple_int` | General indexing | mandatory_success | Multiple integer indices |
| `TestIndexing::test_none` | General indexing | mandatory_success | None as index |
| `TestIndexing::test_out_of_bound_index` | General indexing | mandatory_success | OOB slice error; Python/ATen layer check |
| `TestIndexing::test_setitem_expansion_error` | General indexing | mandatory_success | Setitem shape expansion error |
| `TestIndexing::test_single_int` | General indexing | mandatory_success | Single integer index |
| `TestIndexing::test_step` | General indexing | mandatory_success | Step slicing `[::2]`, `[1::3]` semantics |
| `TestIndexing::test_variable_slicing` | General indexing | mandatory_success | Variable start/stop/step slice indices |
| `TestIndexing::test_unravel_index_errors` | General indexing | mandatory_success | `torch.unravel_index` invalid input errors |
| `TestIndexing::test_zero_dim_index` | General indexing | mandatory_success | 0-dim tensor as index |
| `TestIndexing::test_index_select` | torch.index_select | mandatory_success | aten.index.Tensor → `_add_indices_to_address_index`; all supported dtypes; FRS-GG FR-1,FR-3 |
| `NumpyTests::test_boolean_shape_mismatch` | General indexing | mandatory_success | Bool mask shape mismatch → IndexError at Python/ATen layer |
| `TestIndexing::test_index_tensor_empty_indices` | General indexing | mandatory_success | `t[tensor([])]` → IndexError before kernel dispatch |
| `TestIndexing::test_empty_ndim_index_bool` | General indexing | mandatory_success | Empty uint8 N-dim index → IndexError before dispatch |
| `TestIndexing::test_index_limits` | General indexing | mandatory_success | `t[INT64_MIN]`/`t[INT64_MAX]` → IndexError before dispatch |
| `TestIndexing::test_step_assignment` | General indexing | xfail | Setitem write path (slice_scatter / index_put_) not yet verified on Spyre |
| `TestIndexing::test_int_assignment` | General indexing | xfail | Integer row setitem write path |
| `TestIndexing::test_index` | aten::index.Tensor | xfail | Basic tensor-index indirect read; not yet implemented on Spyre |
| `TestIndexing::test_advancedindex` | aten::index.Tensor | xfail | Advanced/fancy indexing |
| `TestIndexing::test_advancedindex_big` | aten::index.Tensor | xfail | Advanced indexing large tensor |
| `TestIndexing::test_int_indices` | aten::index.Tensor | xfail | Integer tensor indices |
| `TestIndexing::test_int_indices2d` | aten::index.Tensor | xfail | 2D integer indices |
| `TestIndexing::test_int_indices_broadcast` | aten::index.Tensor | xfail | Broadcast integer indices |
| `TestIndexing::test_list_indices` | aten::index.Tensor | xfail | List as index |
| `TestIndexing::test_basic_advanced_combined` | aten::index.Tensor | xfail | Basic + advanced combined indexing |
| `TestIndexing::test_ellipsis_tensor` | aten::index.Tensor | xfail | Ellipsis + tensor index |
| `TestIndexing::test_bool_indices` | Boolean mask | xfail | Boolean mask indexing |
| `TestIndexing::test_bool_indices_accumulate` | Boolean mask | xfail | Boolean mask accumulate |
| `TestIndexing::test_multiple_bool_indices` | Boolean mask | xfail | Multiple boolean masks |
| `TestIndexing::test_multi_dimensional_bool_mask` | Boolean mask | xfail | N-dim boolean mask |
| `TestIndexing::test_bool_mask_assignment` | Boolean mask | xfail | Boolean mask setitem |
| `TestIndexing::test_multi_dimensional_bool_mask_assignment` | Boolean mask | xfail | N-dim bool mask setitem |
| `TestIndexing::test_byte_mask` | Boolean mask | xfail | uint8 byte mask |
| `TestIndexing::test_byte_mask_accumulate` | Boolean mask | xfail | Byte mask accumulate |
| `TestIndexing::test_byte_mask2d` | Boolean mask | xfail | 2D byte mask |
| `TestIndexing::test_multiple_byte_mask` | Boolean mask | xfail | Multiple byte masks |
| `TestIndexing::test_byte_tensor_assignment` | Boolean mask | xfail | Byte tensor setitem |
| `TestIndexing::test_index_scalar_with_bool_mask` | Boolean mask | xfail | Scalar index + bool mask |
| `TestIndexing::test_index_getitem_copy_bools_slices` | Boolean mask | xfail | Getitem copy with bool slices |
| `TestIndexing::test_index_setitem_bools_slices` | Boolean mask | xfail | Setitem with bool slices |
| `TestIndexing::test_index_put_accumulate_large_tensor` | aten::index_put | xfail | Large tensor index_put accumulate |
| `TestIndexing::test_index_put_src_datatype` | aten::index_put | xfail | Dtype coercion for index_put src |
| `TestIndexing::test_index_src_datatype` | aten::index_put | xfail | Dtype of source tensor |
| `TestIndexing::test_take_along_dim` | torch.take_along_dim | xfail | Gather along dim with index tensor |
| `TestIndexing::test_take_along_dim_invalid` | torch.take_along_dim | xfail | Invalid inputs for take_along_dim |
| `TestIndexing::test_index_reduce` | torch.index_reduce_ | xfail | Scatter-reduce via index; not yet implemented |
| `TestIndexing::test_index_copy` | torch.index_copy | xfail | Copy values at index positions; not yet implemented |
| `TestIndexing::test_index_fill` | torch.index_fill | xfail | Fill values at index positions; not yet implemented |
| `NumpyTests::test_empty_tuple_index` | NumPy-compat indirect | xfail | Empty tuple index |
| `NumpyTests::test_empty_fancy_index` | NumPy-compat indirect | xfail | Empty fancy index |
| `NumpyTests::test_ellipsis_index` | NumPy-compat indirect | xfail | Ellipsis index |
| `NumpyTests::test_single_int_index` | NumPy-compat indirect | xfail | Single int index (NumPy compat) |
| `NumpyTests::test_single_bool_index` | NumPy-compat indirect | xfail | Single bool index |
| `NumpyTests::test_boolean_indexing_onedim` | NumPy-compat indirect | xfail | 1D boolean indexing |
| `NumpyTests::test_boolean_indexing_twodim` | NumPy-compat indirect | xfail | 2D boolean indexing |
| `NumpyTests::test_boolean_indexing_weirdness` | NumPy-compat indirect | xfail | Boolean indexing edge cases |
| `NumpyTests::test_boolean_indexing_weirdness_tensors` | NumPy-compat indirect | xfail | Boolean indexing edge cases (tensors) |
| `NumpyTests::test_boolean_indexing_alldims` | NumPy-compat indirect | xfail | Boolean indexing across all dims |
| `NumpyTests::test_boolean_list_indexing` | NumPy-compat indirect | xfail | Boolean list as index |
| `NumpyTests::test_broaderrors_indexing` | NumPy-compat indirect | xfail | Broadcast error cases |
| `NumpyTests::test_index_is_larger` | NumPy-compat indirect | xfail | Index larger than tensor |
| `NumpyTests::test_broadcast_subspace` | NumPy-compat indirect | xfail | Broadcast subspace indexing |
| `NumpyTests::test_trivial_fancy_out_of_bounds` | NumPy-compat indirect | xfail | Out-of-bounds fancy index |
| `TestIndexing::test_empty_index` | aten::index.Tensor | xfail | Empty index read |
| `TestIndexing::test_empty_ndim_index` | aten::index.Tensor | xfail | Empty N-dim index |
| `TestIndexing::test_getitem_scalars` | aten::index.Tensor | xfail | Scalar-tensor getitem |
| `TestIndexing::test_setitem_scalars` | aten::index.Tensor | xfail | Scalar-tensor setitem |
| `TestIndexing::test_index_put_accumulate_expanded_values` | aten::index_put | xfail | Expanded values accumulate |
| `TestIndexing::test_index_put_accumulate_duplicate_indices` | aten::index_put | xfail | Duplicate indices accumulate |
| `TestIndexing::test_index_ind_dtype` | aten::index_put | xfail | Index int/long dtype variants |
| `TestIndexing::test_index_put_non_accumulate_deterministic` | aten::index_put | xfail | Non-accumulate deterministic mode |
| `TestIndexing::test_index_copy_scalars` | torch.index_copy | xfail | Scalar input index_copy |
| `TestIndexing::test_index_copy_deterministic` | torch.index_copy | xfail | Deterministic index_copy |
| `TestIndexing::test_index_add_large_index_vectorized_path` | torch.index_add | xfail | Large index vectorized accumulation |
| `TestIndexing::test_index_add_deterministic` | torch.index_add | xfail | Deterministic index_add |

---

### 1.3 `test_nn_test_embedding.yaml`

> **Upstream file:** `${TORCH_ROOT}/test/nn/test_embedding.py`
> **Global dtypes:** float16, float32, int64, bfloat16
> **Primary CI gate for:** `torch.nn.Embedding` / `F.embedding` (aten.embedding.default → `_add_indices_to_address_embedding`)
> **Total:** 39 tests (26 mandatory_success + 13 xfail)

| Test Name | Op | YAML Mode | Notes / FRS Ref |
|---|---|---|---|
| `TestEmbeddingNN::test_embedding_bag_from_pretrained` | EmbeddingBag CPU | mandatory_success | CPU-only; from_pretrained factory |
| `TestEmbeddingNN::test_embedding_bag_from_pretrained_padding_idx` | EmbeddingBag CPU | mandatory_success | CPU-only; from_pretrained with padding_idx |
| `TestEmbeddingNN::test_embedding_bag_functional` | EmbeddingBag CPU | mandatory_success | CPU-only; F.embedding_bag functional API |
| `TestEmbeddingNN::test_embedding_bag_padding_idx_error` | EmbeddingBag CPU | mandatory_success | CPU-only; padding_idx error validation |
| `TestEmbeddingNN::test_embedding_from_pretrained_float32` | Embedding CPU | mandatory_success | CPU-only; float32 pretrained loading |
| `TestEmbeddingNN::test_embedding_from_pretrained_float64` | Embedding CPU | mandatory_success | CPU-only; float64 pretrained loading |
| `TestEmbeddingNN::test_embedding_from_pretrained_int16` | Embedding CPU | mandatory_success | CPU-only; int16 pretrained loading |
| `TestEmbeddingNN::test_embedding_from_pretrained_int32` | Embedding CPU | mandatory_success | CPU-only; int32 pretrained loading |
| `TestEmbeddingNN::test_embedding_from_pretrained_int64` | Embedding CPU | mandatory_success | CPU-only; int64 pretrained loading |
| `TestEmbeddingNN::test_embedding_from_pretrained_int8` | Embedding CPU | mandatory_success | CPU-only; int8 pretrained loading |
| `TestEmbeddingNN::test_embedding_from_pretrained_options` | Embedding CPU | mandatory_success | CPU-only; pretrained with freeze/sparse options |
| `TestEmbeddingNN::test_embedding_from_pretrained_padding_idx` | Embedding CPU | mandatory_success | CPU-only; pretrained with padding_idx |
| `TestEmbeddingNN::test_embedding_from_pretrained_uint8` | Embedding CPU | mandatory_success | CPU-only; uint8 pretrained loading |
| `TestEmbeddingNN::test_embedding_functional` | Embedding CPU | mandatory_success | CPU-only; F.embedding functional API |
| `TestEmbeddingNN::test_embedding_max_norm` | Embedding CPU | mandatory_success | CPU-only; max_norm weight clipping |
| `TestEmbeddingNN::test_embedding_sparse_basic` | Embedding CPU | mandatory_success | CPU-only; sparse embedding weights |
| `TestEmbeddingNN::test_embedding_sparse_empty_tensor` | Embedding CPU | mandatory_success | CPU-only; sparse embedding empty tensor |
| `TestEmbeddingNN::test_embeddingbag_2d_include_last_offset` | EmbeddingBag CPU | mandatory_success | CPU-only; 2D include_last_offset flag |
| `TestEmbeddingNN::test_embeddingbag_from_pretrained` | EmbeddingBag CPU | mandatory_success | CPU-only; EmbeddingBag pretrained |
| `TestEmbeddingNN::test_embeddingbag_from_pretrained_options` | EmbeddingBag CPU | mandatory_success | CPU-only; EmbeddingBag pretrained options |
| `TestEmbeddingNN::test_embeddingbag_include_last_offset` | EmbeddingBag CPU | mandatory_success | CPU-only; include_last_offset |
| `TestEmbeddingNN::test_move_sparse_half_embedding` | Embedding CPU | mandatory_success | CPU-only; device move sparse half embedding |
| `TestEmbeddingNNDeviceType::test_embedding_padding_idx` | torch.nn.Embedding | mandatory_success | float32; F.embedding with padding_idx on Spyre; FRS-GG FR-1,FR-3 |
| `TestEmbeddingNNDeviceType::test_embedding_max_norm_device` | torch.nn.Embedding | mandatory_success | float32; max_norm applied before lookup; FRS-GG FR-3 |
| `TestEmbeddingNNDeviceType::test_embedding_bag_out_of_bounds_idx` | EmbeddingBag | mandatory_success | 12 parametrized variants; OOB check fires at CPU fallback before Spyre dispatch |
| `TestEmbeddingNNDeviceType::test_embedding_scalar_weight_error` | torch.nn.Embedding | mandatory_success | weight must be 2-D check fires in F.embedding before device dispatch |
| `TestEmbeddingNNDeviceType::test_embedding_bag_device` | EmbeddingBag | xfail | F.embedding_bag not yet implemented on Spyre |
| `TestEmbeddingNNDeviceType::test_embedding_bag_empty_input` | EmbeddingBag | xfail | Empty input bag; not implemented |
| `TestEmbeddingNNDeviceType::test_embedding_bag_non_contiguous_weight` | EmbeddingBag | xfail | Non-contiguous weight; not implemented |
| `TestEmbeddingNNDeviceType::test_embedding_bag_1D_padding_idx` | EmbeddingBag | xfail | 1D padding_idx; float32,float64 (float64 excluded); not implemented |
| `TestEmbeddingNNDeviceType::test_embedding_bag_2D_padding_idx` | EmbeddingBag | xfail | 2D padding_idx; not implemented |
| `TestEmbeddingNNDeviceType::test_embedding_bag_dimension_errors` | EmbeddingBag | xfail | Dimension error validation; not implemented |
| `TestEmbeddingNNDeviceType::test_embedding_bag_half` | EmbeddingBag | xfail | float16 bag; not implemented |
| `TestEmbeddingNNDeviceType::test_embedding_bag_bfloat16` | EmbeddingBag | xfail | bfloat16 bag; not implemented |
| `TestEmbeddingNNDeviceType::test_EmbeddingBag_per_sample_weights_and_offsets` | EmbeddingBag | xfail | Per-sample weights + offsets; not implemented |
| `TestEmbeddingNNDeviceType::test_EmbeddingBag_per_sample_weights_and_new_offsets` | EmbeddingBag | xfail | Per-sample weights + new offsets; not implemented |
| `TestEmbeddingNNDeviceType::test_EmbeddingBag_per_sample_weights_and_no_offsets` | EmbeddingBag | xfail | Per-sample weights; no offsets; not implemented |
| `TestEmbeddingNNDeviceType::test_EmbeddingBag_empty_per_sample_weights_and_offsets` | EmbeddingBag | xfail | Empty per-sample weights; not implemented |
| `TestEmbeddingNNDeviceType::test_EmbeddingBag_per_sample_weights_failures` | EmbeddingBag | xfail | Per-sample weights error cases; not implemented |

---

### 1.4 `test_torch.yaml`

> **Upstream file:** `${TORCH_ROOT}/test/test_torch.py`
> **Global dtypes:** float16, float32, int64, bfloat16, bool
> **Indirect access gates:** `torch.gather` (test_gather_neg_dim), `torch.index_select` (test_index_select, test_index_select_neg_dim)
> **Total:** ~248 tests (~220 mandatory_success + 28 xfail)
> **Note:** Most mandatory_success entries here are general TestTorch tests unrelated to indirect access. Indirect-access-relevant tests are listed individually; general tests are grouped by class.

| Test Name | Op | YAML Mode | Notes / FRS Ref |
|---|---|---|---|
| `TestBasicVitalSigns::test_basic_vitals_read_write` | General | mandatory_success | Vital-signs read/write |
| `TestBasicVitalSigns::test_dataloader_vitals` | General | mandatory_success | Dataloader vitals |
| `TestTorch::test_gather_neg_dim` | torch.gather | mandatory_success | Negative dim wrapping (`dim=-1 = dim=ndim-1`); FRS-GG FR-3 |
| `TestTorch::test_index_add` | torch.index_add | mandatory_success | Dim-0 accumulation via index |
| `TestTorch::test_index_add_all_dtypes` | torch.index_add | mandatory_success | All dtypes variant |
| `TestTorch::test_index_add_cornercase` | torch.index_add | mandatory_success | Corner cases |
| `TestTorch::test_index_add_correctness` | torch.index_add | mandatory_success | Correctness across shapes |
| `TestTorch::test_index_add_neg_dim` | torch.index_add | mandatory_success | Negative dim normalization |
| `TestTorch::test_index_copy_neg_dim` | torch.index_copy | mandatory_success | Negative dim normalization only (full impl is xfail) |
| `TestTorch::test_index_fill_neg_dim` | torch.index_fill | mandatory_success | Negative dim normalization only |
| `TestTorch::test_index_select_neg_dim` | torch.index_select | mandatory_success | Negative dim normalization; FRS-GG FR-3 |
| `TestTorch::test_index_select` | torch.index_select | mandatory_success | Forward pass on Spyre; aten.index.Tensor → `_add_indices_to_address_index`; FRS-GG FR-1,FR-3 |
| `TestTorch::test_RNGState` | General | mandatory_success | |
| `TestTorch::test_RNGStateAliasing` | General | mandatory_success | |
| `TestTorch::test_RNG_after_pickle` | General | mandatory_success | |
| `TestTorch::test_Size` | General | mandatory_success | |
| `TestTorch::test_Size_concat_non_tuple_sequence` | General | mandatory_success | |
| `TestTorch::test_Size_concat_wildcard` | General | mandatory_success | |
| `TestTorch::test_Size_iter` | General | mandatory_success | |
| `TestTorch::test_Size_scalar` | General | mandatory_success | |
| `TestTorch::test_add_meta_scalar` | General | mandatory_success | |
| `TestTorch::test_allow_tensor_metadata_change` | General | mandatory_success | |
| `TestTorch::test_apply` | General | mandatory_success | |
| `TestTorch::test_as_subclass` | General | mandatory_success | |
| `TestTorch::test_assert_async` | General | mandatory_success | |
| `TestTorch::test_backward_hooks_traverse` | General | mandatory_success | |
| `TestTorch::test_batch_norm_cpu_inference` | General | mandatory_success | |
| `TestTorch::test_bf16_supported_on_cpu` | General | mandatory_success | |
| `TestTorch::test_bmm_multithreaded` | General | mandatory_success | |
| `TestTorch::test_boxMullerState` | General | mandatory_success | |
| `TestTorch::test_cat_neg_dim` | General | mandatory_success | |
| `TestTorch::test_check` | General | mandatory_success | |
| `TestTorch::test_chunk_neg_dim` | General | mandatory_success | |
| `TestTorch::test_conj_neg_tolist` | General | mandatory_success | |
| `TestTorch::test_conj_physical_meta_stride` | General | mandatory_success | |
| `TestTorch::test_contains` | General | mandatory_success | |
| `TestTorch::test_copy_dtypes` | General | mandatory_success | |
| `TestTorch::test_copy_float16` | General | mandatory_success | |
| `TestTorch::test_copy_many_to_one` | General | mandatory_success | |
| `TestTorch::test_copy_transpose` | General | mandatory_success | |
| `TestTorch::test_cuda_not_built` | General | mandatory_success | |
| `TestTorch::test_cummax_neg_dim` | General | mandatory_success | |
| `TestTorch::test_cummin_neg_dim` | General | mandatory_success | |
| `TestTorch::test_cumprod_neg_dim` | General | mandatory_success | |
| `TestTorch::test_cumsum_neg_dim` | General | mandatory_success | |
| `TestTorch::test_cxx_flags` | General | mandatory_success | |
| `TestTorch::test_data_ptr_of_empty_tensor_with_storage` | General | mandatory_success | |
| `TestTorch::test_data_ptr_of_empty_view_with_storage` | General | mandatory_success | |
| `TestTorch::test_deepcopy_gradient` | General | mandatory_success | |
| `TestTorch::test_deepcopy_parameter` | General | mandatory_success | |
| `TestTorch::test_deterministic_fill_uninitialized_memory` | General | mandatory_success | |
| `TestTorch::test_deterministic_flag` | General | mandatory_success | |
| `TestTorch::test_device` | General | mandatory_success | |
| `TestTorch::test_dim_order` | General | mandatory_success | |
| `TestTorch::test_dir` | General | mandatory_success | |
| `TestTorch::test_doc` | General | mandatory_success | |
| `TestTorch::test_doc_template` | General | mandatory_success | |
| `TestTorch::test_dot_data_use` | General | mandatory_success | |
| `TestTorch::test_dtype_is_signed` | General | mandatory_success | |
| `TestTorch::test_element_size` | General | mandatory_success | |
| `TestTorch::test_empty_meta` | General | mandatory_success | |
| `TestTorch::test_empty_storage_view` | General | mandatory_success | |
| `TestTorch::test_equal` | General | mandatory_success | |
| `TestTorch::test_error_msg_type_translation` | General | mandatory_success | |
| `TestTorch::test_fill_diagonal` | General | mandatory_success | |
| `TestTorch::test_format_scalar_meta` | General | mandatory_success | |
| `TestTorch::test_from_buffer` | General | mandatory_success | |
| `TestTorch::test_from_file` | General | mandatory_success | |
| `TestTorch::test_generator_cpu` | General | mandatory_success | |
| `TestTorch::test_get_cpu_capability` | General | mandatory_success | |
| `TestTorch::test_has_internal_overlap` | General | mandatory_success | |
| `TestTorch::test_has_storage` | General | mandatory_success | |
| `TestTorch::test_invalid_arg_error_handling` | General | mandatory_success | |
| `TestTorch::test_invalid_generator_raises` | General | mandatory_success | |
| `TestTorch::test_is_nonzero` | General | mandatory_success | |
| `TestTorch::test_is_same_size` | General | mandatory_success | |
| `TestTorch::test_iter` | General | mandatory_success | |
| `TestTorch::test_kthvalue_neg_dim` | General | mandatory_success | |
| `TestTorch::test_linspace_logspace` | General | mandatory_success | |
| `TestTorch::test_logcumsumexp_neg_dim` | General | mandatory_success | |
| `TestTorch::test_manual_seed` | General | mandatory_success | |
| `TestTorch::test_map` | General | mandatory_success | |
| `TestTorch::test_map2` | General | mandatory_success | |
| `TestTorch::test_max_neg_dim` | General | mandatory_success | |
| `TestTorch::test_mean_neg_dim` | General | mandatory_success | |
| `TestTorch::test_median_neg_dim` | General | mandatory_success | |
| `TestTorch::test_memory_format` | General | mandatory_success | |
| `TestTorch::test_memory_format_contiguous_returns_same_tensor_if_already_satisfies` | General | mandatory_success | |
| `TestTorch::test_memory_format_empty` | General | mandatory_success | |
| `TestTorch::test_min_neg_dim` | General | mandatory_success | |
| `TestTorch::test_mode_neg_dim` | General | mandatory_success | |
| `TestTorch::test_nanmedian_neg_dim` | General | mandatory_success | |
| `TestTorch::test_narrow_neg_dim` | General | mandatory_success | |
| `TestTorch::test_nbytes` | General | mandatory_success | |
| `TestTorch::test_ndim` | General | mandatory_success | |
| `TestTorch::test_new` | General | mandatory_success | |
| `TestTorch::test_newaxis_numpy_comparison` | General | mandatory_success | |
| `TestTorch::test_newindex` | General | mandatory_success | |
| `TestTorch::test_no_cuda_monkeypatch` | General | mandatory_success | |
| `TestTorch::test_norm_neg_dim` | General | mandatory_success | |
| `TestTorch::test_normal_shape` | General | mandatory_success | |
| `TestTorch::test_numel` | General | mandatory_success | |
| `TestTorch::test_parallel_info` | General | mandatory_success | |
| `TestTorch::test_parsing_double` | General | mandatory_success | |
| `TestTorch::test_parsing_int64` | General | mandatory_success | |
| `TestTorch::test_parsing_intlist` | General | mandatory_success | |
| `TestTorch::test_permute` | General | mandatory_success | |
| `TestTorch::test_pickle` | General | mandatory_success | |
| `TestTorch::test_pickle_dtype` | General | mandatory_success | |
| `TestTorch::test_pickle_function` | General | mandatory_success | |
| `TestTorch::test_pickle_generator` | General | mandatory_success | |
| `TestTorch::test_pickle_parameter` | General | mandatory_success | |
| `TestTorch::test_pickle_parameter_no_requires_grad` | General | mandatory_success | |
| `TestTorch::test_pickle_size` | General | mandatory_success | |
| `TestTorch::test_pin_memory` | General | mandatory_success | |
| `TestTorch::test_print` | General | mandatory_success | |
| `TestTorch::test_prod_neg_dim` | General | mandatory_success | |
| `TestTorch::test_pyobj_preserved` | General | mandatory_success | |
| `TestTorch::test_qengine` | General | mandatory_success | |
| `TestTorch::test_renorm_neg_dim` | General | mandatory_success | |
| `TestTorch::test_resizable` | General | mandatory_success | |
| `TestTorch::test_reversed` | General | mandatory_success | |
| `TestTorch::test_scatter_neg_dim` | General | mandatory_success | |
| `TestTorch::test_select_neg_dim` | General | mandatory_success | |
| `TestTorch::test_setting_real_imag_to_a_number` | General | mandatory_success | |
| `TestTorch::test_show_config` | General | mandatory_success | |
| `TestTorch::test_size_neg_dim` | General | mandatory_success | |
| `TestTorch::test_size_stride` | General | mandatory_success | |
| `TestTorch::test_sizeof` | General | mandatory_success | |
| `TestTorch::test_slice` | General | mandatory_success | |
| `TestTorch::test_sobolengine_bounds` | General | mandatory_success | |
| `TestTorch::test_sobolengine_continuing` | General | mandatory_success | |
| `TestTorch::test_sobolengine_default_dtype` | General | mandatory_success | |
| `TestTorch::test_sobolengine_distribution` | General | mandatory_success | |
| `TestTorch::test_sobolengine_draw` | General | mandatory_success | |
| `TestTorch::test_sobolengine_draw_base2` | General | mandatory_success | |
| `TestTorch::test_sobolengine_fast_forward` | General | mandatory_success | |
| `TestTorch::test_sobolengine_first_point` | General | mandatory_success | |
| `TestTorch::test_sobolengine_high_dim` | General | mandatory_success | |
| `TestTorch::test_sobolengine_raise` | General | mandatory_success | |
| `TestTorch::test_sobolengine_reset` | General | mandatory_success | |
| `TestTorch::test_sort_neg_dim` | General | mandatory_success | |
| `TestTorch::test_split_neg_dim` | General | mandatory_success | |
| `TestTorch::test_split_with_sizes_copy_out` | General | mandatory_success | |
| `TestTorch::test_squeeze_neg_dim` | General | mandatory_success | |
| `TestTorch::test_std_neg_dim` | General | mandatory_success | |
| `TestTorch::test_storage_base_init` | General | mandatory_success | |
| `TestTorch::test_storage_base_new` | General | mandatory_success | |
| `TestTorch::test_storage_byteswap` | General | mandatory_success | |
| `TestTorch::test_storage_casts` | General | mandatory_success | |
| `TestTorch::test_storage_cycle_via_dict` | General | mandatory_success | |
| `TestTorch::test_storage_cycle_via_slots` | General | mandatory_success | |
| `TestTorch::test_storage_dead_weak_ref` | General | mandatory_success | |
| `TestTorch::test_storage_dealloc` | General | mandatory_success | |
| `TestTorch::test_storage_dealloc_resurrected` | General | mandatory_success | |
| `TestTorch::test_storage_dealloc_subclass_resurrected` | General | mandatory_success | |
| `TestTorch::test_storage_dealloc_subclass_zombie` | General | mandatory_success | |
| `TestTorch::test_storage_dict_dealloc` | General | mandatory_success | |
| `TestTorch::test_storage_error` | General | mandatory_success | |
| `TestTorch::test_storage_error_no_attribute` | General | mandatory_success | |
| `TestTorch::test_storage_finalizer_dealloc` | General | mandatory_success | |
| `TestTorch::test_storage_fix_weakref_no_leak` | General | mandatory_success | |
| `TestTorch::test_storage_from_tensor_dealloc` | General | mandatory_success | |
| `TestTorch::test_storage_from_tensor_dealloc_resurrected` | General | mandatory_success | |
| `TestTorch::test_storage_from_tensor_dealloc_zombie` | General | mandatory_success | |
| `TestTorch::test_storage_preserve_nonhermetic_in_hermetic_context` | General | mandatory_success | |
| `TestTorch::test_storage_resurrected_weak_ref` | General | mandatory_success | |
| `TestTorch::test_storage_slot_dealloc` | General | mandatory_success | |
| `TestTorch::test_storage_thread_safety` | General | mandatory_success | |
| `TestTorch::test_storage_weakref_dealloc` | General | mandatory_success | |
| `TestTorch::test_structseq_repr` | General | mandatory_success | |
| `TestTorch::test_sum_neg_dim` | General | mandatory_success | |
| `TestTorch::test_swap_basic` | General | mandatory_success | |
| `TestTorch::test_swap_fail_slots` | General | mandatory_success | |
| `TestTorch::test_t_not_2d_error` | General | mandatory_success | |
| `TestTorch::test_tensor_ctor_scalar` | General | mandatory_success | |
| `TestTorch::test_tensor_cycle_via_dict` | General | mandatory_success | |
| `TestTorch::test_tensor_dead_weak_ref` | General | mandatory_success | |
| `TestTorch::test_tensor_dict_dealloc` | General | mandatory_success | |
| `TestTorch::test_tensor_fix_weakref_no_leak` | General | mandatory_success | |
| `TestTorch::test_tensor_item_no_warning` | General | mandatory_success | |
| `TestTorch::test_tensor_ressurecting_clear` | General | mandatory_success | |
| `TestTorch::test_tensor_resurrected_weak_ref` | General | mandatory_success | |
| `TestTorch::test_tensor_set` | General | mandatory_success | |
| `TestTorch::test_tensor_set_errors` | General | mandatory_success | |
| `TestTorch::test_tensor_weakref_dealloc` | General | mandatory_success | |
| `TestTorch::test_tensor_where_scalar` | General | mandatory_success | |
| `TestTorch::test_tensor_with_grad_to_scalar_warning` | General | mandatory_success | |
| `TestTorch::test_tensoriterator_output_setup` | General | mandatory_success | |
| `TestTorch::test_to` | General | mandatory_success | |
| `TestTorch::test_to_with_tensor` | General | mandatory_success | |
| `TestTorch::test_topk_neg_dim` | General | mandatory_success | |
| `TestTorch::test_torch_from_file` | General | mandatory_success | |
| `TestTorch::test_transpose_neg_dim` | General | mandatory_success | |
| `TestTorch::test_type` | General | mandatory_success | |
| `TestTorch::test_type_alias` | General | mandatory_success | |
| `TestTorch::test_type_conversion_via_dtype_name` | General | mandatory_success | |
| `TestTorch::test_typed_storage_deprecation_warning` | General | mandatory_success | |
| `TestTorch::test_typed_storage_internal_no_warning` | General | mandatory_success | |
| `TestTorch::test_unbind_neg_dim` | General | mandatory_success | |
| `TestTorch::test_unflatten` | General | mandatory_success | |
| `TestTorch::test_unfold_neg_dim` | General | mandatory_success | |
| `TestTorch::test_unsqueeze_neg_dim` | General | mandatory_success | |
| `TestTorch::test_upsample_nearest1d_meta` | General | mandatory_success | |
| `TestTorch::test_upsample_nearest2d_meta` | General | mandatory_success | |
| `TestTorch::test_var_neg_dim` | General | mandatory_success | |
| `TestTorch::test_warn_types` | General | mandatory_success | |
| `TestTorch::test_wildcard_import` | General | mandatory_success | |
| `TestTorchDeviceType::test_clone_not_memory_dense` | General | mandatory_success | |
| `TestTorchDeviceType::test_clone_zero_stride_dim` | General | mandatory_success | |
| `TestTorchDeviceType::test_expected_failure_xla` | General | mandatory_success | |
| `TestTorchDeviceType::test_memory_format_consistency` | General | mandatory_success | |
| `TestTorchDeviceType::test_narrow_empty` | General | mandatory_success | |
| `TestTorchDeviceType::test_nullary_op_mem_overlap` | General | mandatory_success | |
| `TestTorchDeviceType::test_set_default_tensor_type_warnings` | General | mandatory_success | |
| `TestTorchDeviceType::test_skip_xla` | General | mandatory_success | |
| `TestTorchDeviceType::test_tensor_type` | General | mandatory_success | |
| `TestTorchDeviceType::test_multinomial_cpu` | General | mandatory_success | |
| `TestTorchDeviceType::test_storage_meta_errors` | General | mandatory_success | |
| `TestTorchDeviceType::test_ternary_op_mem_overlap` | General | mandatory_success | |
| `TestTorchDeviceType::test_cov` | General | xfail | Spyre cannot lower int64 constant to lesserequal in codegen; float32 only after dtype filter |
| `TestTorchDeviceType::test_take` | torch.take | xfail | Flat 1D-index gather; not yet implemented on Spyre |
| `TestTorchDeviceType::test_take_empty` | torch.take | xfail | Empty input; not implemented |
| `TestTorchDeviceType::test_scatter_zero_size_index` | torch.scatter | xfail | Zero-size index scatter; not implemented |
| `TestTorchDeviceType::test_scatter_add_non_unique_index` | torch.scatter_add | xfail | Non-unique index scatter_add; not implemented |
| `TestTorchDeviceType::test_scatter_to_large_input` | torch.scatter | xfail | Oversized input scatter; not implemented |
| `TestTorchDeviceType::test_scatter_add_to_large_input` | torch.scatter_add | xfail | Oversized input scatter_add; not implemented |
| `TestTorchDeviceType::test_scatter_bool` | torch.scatter | xfail | Bool dtype scatter; not implemented |
| `TestTorchDeviceType::test_scatter_add_bool` | torch.scatter_add | xfail | Bool dtype scatter_add; not implemented |
| `TestTorchDeviceType::test_scatter_reduce_operations_to_large_input` | torch.scatter_reduce | xfail | Large input; float32 only after dtype filter; not implemented |
| `TestTorchDeviceType::test_scatter_reduce_scalar` | torch.scatter_reduce | xfail | Scalar variant; not implemented |
| `TestTorchDeviceType::test_scatter_reduce_non_unique_index` | torch.scatter_reduce | xfail | Non-unique index; not implemented |
| `TestTorchDeviceType::test_masked_scatter` | torch.masked_scatter | xfail | Write at True-mask positions; not implemented |
| `TestTorchDeviceType::test_masked_scatter_bool_tensor` | torch.masked_scatter | xfail | Bool tensor masked_scatter; not implemented |
| `TestTorchDeviceType::test_masked_scatter_large_tensor` | torch.masked_scatter | xfail | Large tensor; not implemented |
| `TestTorchDeviceType::test_masked_scatter_mem_overlap` | torch.masked_scatter | xfail | Memory overlap check; not implemented |
| `TestTorchDeviceType::test_masked_scatter_inplace_noncontiguous` | torch.masked_scatter | xfail | Inplace noncontiguous; not implemented |
| `TestTorchDeviceType::test_masked_select` | torch.masked_select | xfail | Gather at True positions; not implemented |
| `TestTorchDeviceType::test_masked_select_discontiguous` | torch.masked_select | xfail | Discontiguous input; not implemented |
| `TestTorchDeviceType::test_masked_fill` | torch.masked_fill | xfail | Partial CPU fallback; not confirmed supported; FRS-GG |
| `TestTorchDeviceType::test_masked_fill_bool_tensor` | torch.masked_fill | xfail | Bool tensor; no fallback path |
| `TestTorchDeviceType::test_masked_fill_mem_overlap` | torch.masked_fill | xfail | Memory overlap; not implemented |
| `TestTorchDeviceType::test_scatter_add_one_dim_deterministic` | torch.scatter_add | xfail | Deterministic mode; not implemented |
| `TestTorchDeviceType::test_index_add_mem_overlap` | torch.index_add | xfail | Memory overlap detection; not implemented |
| `TestTorchDeviceType::test_index_copy_mem_overlap` | torch.index_copy | xfail | Memory overlap detection; not implemented |
| `TestTorchDeviceType::test_index_fill_mem_overlap` | torch.index_fill | xfail | Memory overlap detection; not implemented |
| `TestTorchDeviceType::test_index_put_mem_overlap` | torch.index_put | xfail | Memory overlap detection; not implemented |
| `TestTorchDeviceType::test_scatter_mem_overlap` | torch.scatter | xfail | Memory overlap detection; not implemented |

---

### 1.5 Upstream Test Tracking Summary

> `test_index_select` appears in both `test_indexing.yaml` (TestIndexing::test_index_select) and `test_torch.yaml` (TestTorch::test_index_select) — both `mandatory_success`. There is no upstream `test_index_select.py`; all index_select tests live in `test_indexing.py` and `test_torch.py`. torch-spyre unit tests (`tests/indirect_access/`) are not upstream tests and are documented in Section 3.

| YAML File | Upstream File | mandatory_success | xfail | Total |
|---|---|---|---|---|
| `test_scatter_gather_ops.yaml` | `test/test_scatter_gather_ops.py` | 2 | 14 | 16 |
| `test_indexing.yaml` | `test/test_indexing.py` | 22 | 60 | 82 |
| `test_nn_test_embedding.yaml` | `test/nn/test_embedding.py` | 26 | 13 | 39 |
| `test_torch.yaml` | `test/test_torch.py` | ~220 | 28 | ~248 |
| **Total** | | **~270** | **~115** | **~385** |

---

## Section 2 — torch-spyre Existing Tests

> Tests present in the `main` branch before the indirect access feature. These confirm that embedding-like operations and basic inductor ops were already tracked in CI before the feature was introduced.

| Test ID | Test Function | Source File | Test Type | What It Tests | Env | YAML Mode | FRS Ref | Notes |
|---|---|---|---|---|---|---|---|---|
| TST-01 | `TestOps::test_embedding_cpu` | `tests/inductor/test_inductor_ops.py` | `FUNC` | `F.embedding` as CPU reference within inductor ops test suite; validates embedding-table lookup shape/value | `ENV_BASE` | mandatory_success | `FRS-GG FR-3` | CPU-side validation; not a gather-on-Spyre test |

>
> Before the indirect access feature, torch-spyre had **zero direct `torch.gather` or `torch.index_select` functional tests on Spyre hardware**.

---

## Section 3 — torch-spyre Unit and Integration Tests

> All tests introduced by the developer across five test files in `tests/indirect_access/`.

| # | File | Config YAML | Type | Scope |
|---|---|---|---|---|
| 3.1 | `test_detect_indirect_access.py` | `test_detect_indirect_access_config.yaml` | UNIT | Detection pass |
| 3.2 | `test_indirect_access_codegen.py` | _(no config yet)_ | UNIT | Codegen helpers |
| 3.3 | `test_indices_to_address.py` | `test_indices_to_address_custom_op_config.yaml` | UNIT+CPP | C++ address computation |
| 3.4 | `test_indices_to_address_pass.py` | `test_indices_to_address_pass_config.yaml` | UNIT+INTG | FX graph transformation |
| 3.5 | `test_spyre_kernel_indirect_store.py` | _(no config yet)_ | UNIT | `SpyreKernel.store()` indirect path |

### 3.1 Compiler Pass Unit Tests — `test_detect_indirect_access.py`

| Test ID | Class::Method | Test Type | What It Tests | Env | FRS Ref |
|---|---|---|---|---|---|
| DPR-01 | `TestFindIndirectPattern::test_no_reads` | UNIT | Empty reads list → returns `(None, None)` | CPU_ONLY | FRS-AC 5.1 |
| DPR-02 | `TestFindIndirectPattern::test_single_read` | UNIT | Single read (< 2 required) → returns `(None, None)` | CPU_ONLY | FRS-AC 5.1 |
| DPR-03 | `TestFindIndirectPattern::test_no_tmp_variables` | UNIT | Multiple reads, no `tmp*` symbol → `(None, None)` | CPU_ONLY | FRS-AC 5.1 |
| DPR-04 | `TestFindIndirectPattern::test_indirect_pattern_detected` | UNIT | Index read + value read (value has `tmp0`) → `(0, 1)` | CPU_ONLY | FRS-AC 5.1 |
| DPR-05 | `TestFindIndirectPattern::test_indirect_pattern_reversed_order` | UNIT | Value read before index → `(1, 0)` | CPU_ONLY | FRS-AC 5.1 |
| DPR-06 | `TestFindIndirectPattern::test_multiple_tmp_variables` | UNIT | Multiple `tmp*` symbols in value read → `(0, 1)` | CPU_ONLY | FRS-AC 5.1 |
| DPR-07 | `TestMarkIndirectAccess::test_mark_indirect_access_basic` | UNIT | `op_info` set with `op=identity`, `index_args`, `tensor_names`, `index_value_pairs` | CPU_ONLY | FRS-AC 5.1 |
| DPR-08 | `TestMarkIndirectAccess::test_mark_indirect_access_with_extra_tensors` | UNIT | Extra tensors appended after value/index in `tensor_names` | CPU_ONLY | FRS-AC 5.1 |
| DPR-09 | `TestDetectIndirectAccess::test_empty_operations` | UNIT | Empty operations list → no error | CPU_ONLY | FRS-AC 5.1 |
| DPR-10 | `TestDetectIndirectAccess::test_non_computed_buffer` | UNIT | Non-`ComputedBuffer` op silently skipped | CPU_ONLY | FRS-AC 5.1 |
| DPR-11 | `TestDetectIndirectAccess::test_operation_with_existing_op_info` | UNIT | Already-annotated op left unchanged | CPU_ONLY | FRS-AC 5.1 |
| DPR-12 | `TestDetectIndirectAccess::test_operation_with_single_read` | UNIT | Op with 1 read → skipped (need ≥ 2) | CPU_ONLY | FRS-AC 5.1 |
| DPR-13 | `TestDetectIndirectAccess::test_indirect_pattern_detected_and_marked` | UNIT | End-to-end: `_find_indirect_pattern` + `_mark_indirect_access` called together | CPU_ONLY | FRS-AC 5.1 |

### 3.2 Codegen Unit Tests — `test_indirect_access_codegen.py`

| Test ID | Class::Method | Test Type | What It Tests | Env | FRS Ref |
|---|---|---|---|---|---|
| DPR-14 | `TestMaxDimSizesComputation::test_value_tensor_maxdim_sizes_4d_value_3d_index` | UNIT | `compute_indirect_max_dim_sizes` for 4D value + 3D index: all dims return `-1` (dynamic) | CPU_ONLY | FRS-AC 5.2 |
| DPR-15 | `TestMaxDimSizesComputation::test_value_tensor_maxdim_sizes_2d_value_1d_index` | UNIT | `compute_indirect_max_dim_sizes` for 2D value + 1D index with no dim overlap | CPU_ONLY | FRS-AC 5.2 |
| DPR-16 | `TestLayoutLabelAssignment::test_index_tensor_gets_kernel_idx_label` | UNIT | `TensorArg.is_index_tensor=True`, `related_value_tensor_idx=0` correctly set | CPU_ONLY | FRS-AC 5.2 |
| DPR-17 | `TestLayoutLabelAssignment::test_value_tensor_gets_input_label` | UNIT | Value `TensorArg`: `is_index_tensor=False`, `is_input=True` | CPU_ONLY | FRS-AC 5.2 |
| DPR-18 | `TestIndirectAccessHelpers::test_is_indirect_value_tensor` | UNIT | `is_indirect_value_tensor(op_spec, 0)=True`; index(1) and output(2) return False | CPU_ONLY | FRS-AC 5.2 |
| DPR-19 | `TestIndirectAccessHelpers::test_get_indirect_tensor_address` | UNIT | Value→`SEGMENT_OFFSETS[0]`, index→`SEGMENT_OFFSETS[1]`, output→`SEGMENT_OFFSETS[2]` | CPU_ONLY | FRS-AC 5.2 |
| DPR-20 | `TestIndirectAccessHelpers::test_get_positive_indirect_dims` | UNIT | Equal-size dims → empty set; mb index<value → `{"mb"}` in positive dims | CPU_ONLY | FRS-AC 5.2 |
| DPR-21 | `TestIndirectAccessHelpers::test_get_active_indirect_dims` | UNIT | `get_active_indirect_dims` returns Symbol("mb") when mb is positive indirect dim | CPU_ONLY | FRS-AC 5.2 |
| DPR-22 | `TestDataFormatHandling::test_index_tensor_uses_senuint32` | UNIT | Index tensor `device_dtype = DataFormats.SENUINT32` | CPU_ONLY | FRS-AC 5.2 |
| DPR-23 | `TestDataFormatHandling::test_value_tensor_preserves_original_dtype` | UNIT | Value tensor keeps original dtype `SEN169_FP16` | CPU_ONLY | FRS-AC 5.2 |
| DPR-24 | `TestStrideAndOffsetComputation::test_index_tensor_stride_zeroing` | UNIT | Placeholder for stride zeroing test — currently `pass` pending multi-core fix | CPU_ONLY | FRS-AC 5.2 |
| DPR-25 | `TestScatterOperation::test_scatter_has_correct_indirect_metadata` | UNIT | Scatter: `index.related_value_tensor_idx=2`; `is_indirect_value_tensor(op_spec, 2)=True` | CPU_ONLY | FRS-AC 5.2 |

### 3.3 Address Computation Unit Tests — `test_indices_to_address.py`

| Test ID | Class::Method | Test Type | What It Tests | Env | FRS Ref |
|---|---|---|---|---|---|
| DPR-26 | `TestIndicesToAddress::test_1d_indexing_basic` | CPP | 1D mode: output shape = indices shape; dtype = int64 | CPU_ONLY | FRS-AC 3.4 |
| DPR-27 | `TestIndicesToAddress::test_1d_indexing_with_stride` | CPP | `dim=1` on 3D tensor: stride parameter correctly passed | CPU_ONLY | FRS-AC 3.4 |
| DPR-28 | `TestIndicesToAddress::test_2d_multidim_indexing` | CPP | Multi-dim mode: output shape = indices[0..-2]; dtype = float32 | CPU_ONLY | FRS-AC 3.4 |
| DPR-29 | `TestIndicesToAddress::test_3d_multidim_indexing` | CPP | Multi-dim on 3D tensor with 3-coord indices | CPU_ONLY | FRS-AC 3.4 |
| DPR-30 | `TestIndicesToAddress::test_4d_multidim_indexing` | CPP | Multi-dim on 4D tensor | CPU_ONLY | FRS-AC 3.4 |
| DPR-31 | `TestIndicesToAddress::test_virtual_offset_basic` | CPP | `virtual_offset=1024` accepted; no crash | CPU_ONLY | FRS-AC 3.4 |
| DPR-32 | `TestIndicesToAddress::test_virtual_offset_multidim` | CPP | virtual_offset with multi-dim indexing | CPU_ONLY | FRS-AC 3.4 |
| DPR-33 | `TestIndicesToAddress::test_batch_indexing` | CPP | 8 indices processed in one call | CPU_ONLY | FRS-AC 3.4 |
| DPR-34 | `TestIndicesToAddress::test_different_dtypes` | CPP | float16, float32, int32, int64 — different element_size paths | CPU_ONLY | FRS-GG FR-1, FRS-AC 3.4 |
| DPR-35 | `TestIndicesToAddress::test_reshape_indices` | CPP | Output shape matches reshaped input shape | CPU_ONLY | FRS-AC 3.4 |
| DPR-36 | `TestIndicesToAddress::test_fake_mode` | CPP | Meta device (Dynamo tracing) — `register_fake` shape inference | CPU_ONLY | FRS-AC 3.4 |
| DPR-37 | `TestIndicesToAddress::test_edge_case_single_index` | CPP | Single-element index → shape `(1,)` | CPU_ONLY | FRS-AC 3.4 |
| DPR-38 | `TestIndicesToAddress::test_edge_case_zero_dim_index` | CPP | Scalar (0-dim) index → scalar output | CPU_ONLY | FRS-AC 3.4 |
| DPR-39 | `TestIndicesToAddress::test_index_bounds_validation` | CPP | OOB index → `RuntimeError("out of bounds")` | CPU_ONLY | FRS-AC 3.4 |
| DPR-40 | `TestIndicesToAddress::test_negative_index_validation` | CPP | Negative index → `RuntimeError` (no wraparound) | CPU_ONLY | FRS-AC 3.4 |
| DPR-41 | `TestIndicesToAddress::test_dim_out_of_bounds` | CPP | `dim=2` on 2D tensor → `RuntimeError` | CPU_ONLY | FRS-AC 3.4 |
| DPR-42 | `TestIndicesToAddress::test_consistency_1d_vs_multidim` | CPP | 1D and multi-dim both produce valid address tensors | CPU_ONLY | FRS-AC 3.4 |
| DPR-43 | `test_integration_with_gather` | CPP | Gather-like row selection: 4 row addresses computed | CPU_ONLY | FRS-GG FR-3 |
| DPR-44 | `test_integration_with_paged_attention` | CPP | Paged attention simulation: 2D page indices + virtual_offset | CPU_ONLY | FRS-GG FR-5 |

### 3.4 FX Pass Unit and Integration Tests — `test_indices_to_address_pass.py`

| Test ID | Class::Method | Test Type | What It Tests | Env | FRS Ref |
|---|---|---|---|---|---|
| DPR-45 | `TestIndicesToAddressPass::test_gather_basic_2d` | INTG | 2D gather transformed; output numerically correct | ENV_BASE | FRS-GG FR-3, FRS-AC 5.1 |
| DPR-46 | `TestIndicesToAddressPass::test_gather_3d_dim1` | INTG | 3D gather along dim=1; expanded zero-stride index | ENV_BASE | FRS-GG FR-6 |
| DPR-47 | `TestIndicesToAddressPass::test_gather_different_dtypes` | INTG | float16, float32 — transformation fires for each | ENV_BASE | FRS-GG FR-1 |
| DPR-48 | `TestIndicesToAddressPass::test_gather_disabled_pass` | INTG | Gate=0 → no `indices_to_address` node in generated code | ENV_GATE_OFF | FRS-GG FR-7 |
| DPR-49 | `TestIndicesToAddressPass::test_embedding_basic` | INTG | `torch.embedding` transformed; correct output | ENV_BASE | FRS-GG FR-3 |
| DPR-50 | `TestIndicesToAddressPass::test_embedding_2d_indices` | INTG | Embedding with 2D batch×seq indices | ENV_BASE | FRS-GG FR-3 |
| DPR-51 | `TestIndicesToAddressPass::test_embedding_large_vocab` | INTG | vocab_size=10000; large gather | ENV_BASE | FRS-GG FR-6 |
| DPR-52 | `TestIndicesToAddressPass::test_index_select_basic` | INTG | `torch.index_select` → `aten.index.Tensor` path transformed | ENV_BASE | FRS-GG FR-3 |
| DPR-53 | `TestIndicesToAddressPass::test_index_select_dim1` | INTG | `index_select` dim=1 on 3D tensor | ENV_BASE | FRS-GG FR-3 |
| DPR-54 | `TestIndicesToAddressPass::test_index_tensor_basic` | INTG | `input[indices]` subscript → `aten.index.Tensor` transformed | ENV_BASE | FRS-GG FR-3 |
| DPR-55 | `TestIndicesToAddressPass::test_gather_single_element` | INTG | Single-element index `[[5]]` | ENV_BASE | FRS-GG FR-3 |
| DPR-56 | `TestIndicesToAddressPass::test_gather_large_batch` | INTG | Large random index `(50,64)` | ENV_BASE | FRS-GG FR-3 |
| DPR-57 | `TestIndicesToAddressPass::test_multiple_gather_operations` | INTG | Two gathers in one graph both transformed | ENV_BASE | FRS-GG FR-3 |
| DPR-58 | `TestIndicesToAddressPass::test_gather_with_computation` | INTG | `gather → * 2.0 + 1.0`; downstream computation correct | ENV_BASE | FRS-GG FR-7 |
| DPR-59 | `TestIndicesToAddressPass::test_embedding_with_linear` | INTG | `embedding → matmul`; LLM pattern | ENV_BASE | FRS-GG FR-7 |
| DPR-60 | `TestIndicesToAddressPass::test_cpu_tensor_not_transformed` | INTG | CPU tensor skips transformation; still correct | CPU_ONLY | FRS-GG FR-7 |
| DPR-61 | `TestIndicesToAddressPassIntegration::test_simple_embedding_layer` | INTG | `nn.Embedding` module compiled; assertion-enabled | ENV_BASE | FRS-GG FR-3 |
| DPR-62 | `TestIndicesToAddressPassIntegration::test_attention_gather_pattern` | INTG | `gather` with `unsqueeze(-1).expand(...)` index — attention pattern | ENV_BASE | FRS-GG FR-3 |

### 3.5 Kernel Store Test — `test_spyre_kernel_indirect_store.py`

| Test ID | Test Function | Test Type | What It Tests | Env | FRS Ref |
|---|---|---|---|---|---|
| DPR-63 | `test_store_indirect_access_builds_index_tensor_args_inline` | UNIT | `SpyreKernel.store()` with indirect `op_info`: (1) `enrich_indirect_index_value_pairs` called; (2) OpSpec with `op="identity"`; (3) value TensorArg: `is_input=True, is_index_tensor=False`; (4) index TensorArg: `is_input=True, is_index_tensor=True, related_value_tensor_idx=0`; (5) output TensorArg: `is_input=False`; (6) args ordered `[value, index, output]` | CPU_ONLY | FRS-AC 5.1, 5.2 |

---

## Section 4 — torch.gather FVT Test Cases

> **FX pass handler:** `_add_indices_to_address_gather` — inserts `spyre::indices_to_address` node for `aten.gather.default`.
>
> **Address formula:** `address = virtual_offset + Σᵢ (coord[i] × device_stride[i]) × element_size`
>
> **Shape coverage:** 2D tests are the primary axis for non-rank-specific categories. 3D/4D tests use BLOCKED:HW status (known hardware limitation in progress); valid test designs that execute once the HW limitation is resolved.

---

### 4.1 Functional Correctness — Shape and Dimension

| ID | Test Name | Input Shape | Index Shape | dim | dtype | What It Validates | Pass Criteria | Tol | Env | Priority | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FC-01 | `test_gather_2d_dim1` | (4,64) | (4,8) | 1 | float16 | Column-axis selection on 2D tensor | `result[r,i]==src[r,idx[r,i]]`; allclose | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| FC-02 | `test_gather_3d_dim0` | (8,4,64) | (3,4,64) | 0 | float16 | Page-axis selection on 3D tensor | `result[p,r,c]==src[idx[p,r,c],r,c]`; allclose | 1e-2/1e-2 | ENV_BASE | P1 | BLOCKED:HW |
| FC-03 | `test_gather_3d_dim1` | (2,8,64) | (2,3,64) | 1 | float16 | Row-axis selection on 3D tensor | `result[b,i,c]==src[b,idx[b,i,c],c]`; allclose | 1e-2/1e-2 | ENV_BASE | P1 | BLOCKED:HW |
| FC-04 | `test_gather_3d_dim2` | (2,4,64) | (2,4,16) | 2 | float16 | Feature-axis selection on 3D tensor | `result[b,r,i]==src[b,r,idx[b,r,i]]`; allclose | 1e-2/1e-2 | ENV_BASE | P1 | BLOCKED:HW |
| FC-05 | `test_gather_4d_dim0` | (8,2,4,64) | (3,2,4,64) | 0 | float16 | Page-axis selection on 4D tensor | allclose vs CPU | 1e-2/1e-2 | ENV_BASE | P1 | BLOCKED:HW |
| FC-06 | `test_gather_4d_dim1` | (2,8,4,64) | (2,3,4,64) | 1 | float16 | Sequence-axis selection on 4D (KV cache pattern) | allclose vs CPU | 1e-2/1e-2 | ENV_BASE | P1 | BLOCKED:HW |
| FC-07 | `test_gather_4d_dim3` | (2,2,4,64) | (2,2,4,32) | 3 | float16 | Head feature dim selection on 4D | allclose vs CPU | 1e-2/1e-2 | ENV_BASE | P1 | BLOCKED:HW |
| FC-08 | `test_gather_5d` | (2,2,4,4,64) | (2,2,4,4,16) | 4 | float16 | 5D N-D support (FRS-GG FR-6) | allclose vs CPU | 1e-2/1e-2 | ENV_BASE | P2 | BLOCKED:HW |
| FC-09 | `test_gather_2d_neg_dim` | (4,64) | (4,8) | -1 | float16 | `dim=-1` == `dim=1` on 2D | Identical to FC-01; allclose | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| FC-10 | `test_gather_3d_neg_dim` | (2,4,64) | (2,4,16) | -1 | float16 | `dim=-1` == `dim=2` on 3D | Identical to FC-04; allclose | 1e-2/1e-2 | ENV_BASE | P1 | BLOCKED:HW |
| FC-11 | `test_gather_4d_neg_dim_minus2` | (2,2,4,64) | (2,2,3,64) | -2 | float16 | `dim=-2` on 4D equals `dim=2` | allclose vs CPU | 1e-2/1e-2 | ENV_BASE | P2 | BLOCKED:HW |
| FC-12 | `test_gather_2d_large_dim` | (4,256) | (4,64) | 1 | float16 | Wide tensor (256 cols); multi-stick stride formula | `result[r,i]==src[r,idx[r,i]]`; allclose | 1e-2/1e-2 | ENV_BASE | P2 | READY |

---

### 4.2 Index Access Patterns

| ID | Test Name | Input Shape | Index Shape | dim | Index Pattern | What It Validates | Pass Criteria | Tol | Env | Priority | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| IP-01 | `test_gather_index_subset` | (64,64) | (4,64) | 0 | 4 of 64 rows | Sparse read | `result[i,:]==src[idx[i,0],:]` | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| IP-02 | `test_gather_index_same_size` | (32,64) | (32,64) | 0 | All 32 rows, identity | Full copy baseline | allclose(result, src_cpu) | 1e-2/1e-2 | ENV_BASE | P2 | READY |
| IP-03 | `test_gather_index_oversampling` | (16,64) | (32,64) | 0 | Each row repeated twice | Index larger than source dim | Shape (32,64); rows match | 1e-2/1e-2 | ENV_BASE | P2 | READY |
| IP-04 | `test_gather_hotspot` | (64,64) | (8,64) | 0 | All values = 5 | All output rows identical to src[5,:] | all_rows_equal | 1e-2/1e-2 | ENV_BASE | P2 | READY |
| IP-05 | `test_gather_sequential_ascending` | (64,64) | (8,64) | 0 | [0,1,2,3,4,5,6,7] | Best-case locality | result[:8,:]==src_cpu[:8,:] | 1e-2/1e-2 | ENV_BASE | P2 | READY |
| IP-06 | `test_gather_sequential_descending` | (64,64) | (8,64) | 0 | [7,6,5,4,3,2,1,0] | Reverse-order read | allclose vs reversed reference | 1e-2/1e-2 | ENV_BASE | P2 | READY |
| IP-07 | `test_gather_random_permuted` | (64,64) | (64,64) | 0 | Random permutation [0..63] | Worst-case HBM access locality | allclose vs CPU same index | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| IP-08 | `test_gather_interleaved` | (64,64) | (16,64) | 0 | [0,32,1,33,2,34,...] | Cache-unfriendly stride pattern | allclose vs CPU reference | 1e-2/1e-2 | ENV_BASE | P2 | READY |
| IP-09 | `test_gather_dim1_all_cols` | (8,64) | (8,64) | 1 | All 64 cols in order | Full column gather | allclose(result, src_cpu) | 1e-2/1e-2 | ENV_BASE | P2 | READY |
| IP-10 | `test_gather_dim1_sparse_cols` | (8,64) | (8,8) | 1 | 8 of 64 cols, scattered | Sparse column selection | `result[r,i]==src_cpu[r,idx[r,i]]` | 1e-2/1e-2 | ENV_BASE | P1 | READY |

---

### 4.3 API Variants

| ID | Test Name | Input Shape | Index Shape | dim | Call Form | What It Validates | Pass Criteria | Tol | Env | Priority | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AV-01 | `test_gather_method_form_2d` | (64,64) | (4,64) | 0 | `input.gather(dim, index)` | Method form identical to functional form | allclose | 1e-2/1e-2 | ENV_BASE | P2 | READY |
| AV-02 | `test_gather_method_form_3d` | (2,8,64) | (2,3,64) | 1 | `input.gather(dim, index)` | Method form on 3D | allclose vs CPU | 1e-2/1e-2 | ENV_BASE | P2 | BLOCKED:HW |
| AV-03 | `test_gather_out_parameter` | (64,64) | (4,64) | 0 | `torch.gather(..., out=buf)` | Pre-allocated out buffer | `result is buf`; allclose | 1e-2/1e-2 | ENV_BASE | P2 | READY |
| AV-04 | `test_gather_out_wrong_shape` | (64,64) | (4,64) | 0 | `out=` wrong shape | Shape validation at ATen layer | RuntimeError raised | — | ENV_BASE | P2 | READY |
| AV-05 | `test_gather_keyword_args` | (64,64) | (4,64) | 0 | All keyword args | Keyword argument form | allclose vs positional | 1e-2/1e-2 | ENV_BASE | P2 | READY |

---

### 4.4 Dtype Coverage

| ID | Test Name | dtype | element_size | What It Validates | Pass Criteria | Tol | Env | Priority | Status |
|---|---|---|---|---|---|---|---|---|---|
| DT-01 | `test_gather_bfloat16` | bfloat16 | 2 | bf16 stick-expression fix (UPT-S02 root cause) at 2D | allclose vs CPU | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| DT-02 | `test_gather_float32` | float32 | 4 | stick width = 32 elems; catches hardcoded fp16 element_size bug | allclose vs CPU | 1e-5/1e-5 | ENV_BASE | P1 | READY |
| DT-03 | `test_gather_int64_exact` | int64 | 8 | 16 elements/stick; integer gather — no FP rounding | exact equality | exact | ENV_BASE | P2 | READY |
| DT-04 | `test_gather_int32_exact` | int32 | 4 | same stick width as float32; validates integer path | exact equality | exact | ENV_BASE | P2 | READY |
| DT-05 | `test_gather_bool_dtype` | bool | 1 | 128 elements/stick; element_size=1; FRS-GG FR-1 covers bool | exact bool equality | exact | ENV_BASE | P2 | READY |
| DT-06 | `test_gather_unsupported_dtype` | float64 | 8 | FRS-GG FR-1 unsupported dtype must NOT silently produce wrong results | RuntimeError OR CPU fallback + FallbackWarning; no silent wrong result | — | ENV_BASE | P1 | READY |
| DT-07 | `test_gather_fp32_vs_fp16_addresses` | fp32 vs fp16 | 4 vs 2 | Validates `device_stride` scaling by `element_size` correct | allclose(fp32_result.half(), fp16_result) | 1e-2/1e-2 | ENV_BASE | P2 | READY |

---

### 4.5 Execution Mode

| ID | Test Name | Mode | What It Validates | Pass Criteria | Tol | Env | Priority | Status |
|---|---|---|---|---|---|---|---|---|
| EM-01 | `test_gather_eager_1d` | Eager (no compile) | Eager path: correct result or unsupported error | Correct result OR RuntimeError; no silent wrong | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| EM-02 | `test_gather_eager_2d` | Eager | 2D eager path | Same as EM-01 | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| EM-03 | `test_gather_fullgraph` | `torch.compile(fullgraph=True)` | No graph break in gather lowering | No Unsupported exception; allclose | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| EM-04 | `test_gather_graph_softmax` | Compiled | gather → softmax in single graph | allclose vs CPU gather+softmax | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| EM-05 | `test_gather_graph_layernorm` | Compiled | gather → layernorm layout interaction | allclose vs CPU reference | 1e-2/1e-2 | ENV_BASE | P2 | READY |
| EM-06 | `test_gather_graph_matmul` | Compiled | gather → matmul; output layout compatible | allclose vs CPU; no SDSC shape mismatch | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| EM-07 | `test_gather_graph_add` | Compiled | gather → elementwise add | allclose vs CPU reference | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| EM-08 | `test_gather_repeated_calls` | Compiled × 3 calls | No state leakage across successive calls | All 3 calls correct | 1e-2/1e-2 | ENV_BASE | P1 | READY |

---

### 4.6 Dynamic / Symbolic Shapes

| ID | Test Name | Dynamic Dims | What It Validates | Pass Criteria | Tol | Env | Priority | Status |
|---|---|---|---|---|---|---|---|---|
| DS-01 | `test_gather_dynamic_index_rows` | Index dim-0 | Kernel handles different numbers of gathered rows | Both calls correct; ≤1 recompile | 1e-2/1e-2 | ENV_DYN | P1 | READY |
| DS-02 | `test_gather_dynamic_source_rows` | Source dim-0 | Different source → different strides → address adapts | Both calls correct | 1e-2/1e-2 | ENV_DYN | P1 | READY |
| DS-03 | `test_gather_dynamic_batch` | Batch dim B | Realistic LLM: batch varies per request | Correct for B=1,2,4 | 1e-2/1e-2 | ENV_DYN | P1 | BLOCKED:HW |
| DS-04 | `test_gather_dynamic_seq_len` | S and T dynamic | Decode phase: KV cache grows while T=1 per step | Correct for all (S,T) pairs | 1e-2/1e-2 | ENV_DYN | P1 | BLOCKED:HW |
| DS-05 | `test_gather_dynamic_all_dims` | All batch+seq dims | Fully symbolic; only last dim static | Correct for multiple (B,S,T) | 1e-2/1e-2 | ENV_DYN | P2 | BLOCKED:HW |
| DS-06 | `test_gather_no_recompile_in_guard` | dim-0 of index range [1,8] | Once compiled for range, no recompile per size | Single graph; all 8 sizes correct | 1e-2/1e-2 | ENV_DYN | P2 | READY |
| DS-07 | `test_gather_export_dynamic` | N via `torch.export` | Exported program handles any N | Correct for N=2,4,8,16 | 1e-2/1e-2 | ENV_BASE | P2 | READY |
| DS-08 | `test_gather_dynamic_chain` | N and M independent | Two chained gathers with independent dynamic sizes | Both correct for (N,M)=(4,2),(8,4) | 1e-2/1e-2 | ENV_DYN | P2 | READY |
| DS-09 | `test_gather_dynamic_index_dtype_stable` | N dynamic | Index dtype stable as int64 | `index.dtype == torch.int64` for all N | exact | ENV_DYN | P2 | READY |

---

### 4.7 Memory Layout — Contiguous Source

| ID | Test Name | Input Shape | Layout Property | What It Validates | Pass Criteria | Tol | Env | Priority | Status |
|---|---|---|---|---|---|---|---|---|---|
| ML-01 | `test_gather_stick_aligned_64` | (32,64) | 1 stick/row | No padding artifacts; clean stick boundary | allclose; shape (4,64) | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| ML-02 | `test_gather_stick_aligned_128` | (32,128) | 2 sticks/row | Address spans two sticks; no boundary corruption | allclose | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| ML-03 | `test_gather_non_aligned_50` | (32,50) padded to 64 | 50 cols non-aligned | Output shape (4,50); padding does not leak | Shape (4,50) not (4,64); values correct | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| ML-04 | `test_gather_non_aligned_100` | (16,100) padded to 128 | 64 < 100 < 128; spans 2 sticks partial | No zero-padding in output | Shape (4,100) correct | 1e-2/1e-2 | ENV_BASE | P2 | READY |
| ML-05 | `test_gather_single_row_source` | (1,64) | Minimal source | Single-row source | allclose(result.cpu(), src_cpu[0:1,:]) | 1e-2/1e-2 | ENV_BASE | P2 | READY |
| ML-06 | `test_gather_large_baseline` | (512,64) | Large contiguous | No OOB in address computation | allclose vs CPU | 1e-2/1e-2 | ENV_BASE | P1 | READY |

---

### 4.8 Memory Layout — Non-Contiguous Source

| ID | Test Name | Source Preparation | Non-Contig Type | What It Validates | Pass Criteria | Tol | Env | Priority | Status |
|---|---|---|---|---|---|---|---|---|---|
| NCS-01 | `test_gather_transposed_src` | `src.t()` | Transposed | Restickify inserted before gather | allclose vs `src_cpu.t()[idx]` | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| NCS-02 | `test_gather_strided_src` | `src[::2,:]` | Row stride=2 | Non-unit stride; restickify makes contiguous | allclose | 1e-2/1e-2 | ENV_BASE | P2 | READY |
| NCS-03 | `test_gather_permuted_src_3d` | `src.permute(1,0,2)` | Permuted 3D | Restickify fires on permuted 3D | allclose | 1e-2/1e-2 | ENV_BASE | P2 | BLOCKED:HW |
| NCS-04 | `test_gather_sliced_offset` | `src[:,4:20]` | Non-zero storage offset | Storage offset in restickify; no off-by-one | allclose | 1e-2/1e-2 | ENV_BASE | P2 | READY |
| NCS-05 | `test_gather_col_strided_src` | `src[:,::2]` | Column stride=2 | Non-unit stride on non-gather dim | allclose | 1e-2/1e-2 | ENV_BASE | P2 | READY |
| NCS-06 | `test_gather_contig_vs_noncontig` | contig and `src.t().t()` | Non-contig but equal | Both paths produce identical results | allclose(result_contig, result_noncontig) | 1e-2/1e-2 | ENV_BASE | P1 | READY |

---

### 4.9 Memory Layout — Non-Contiguous Index

| ID | Test Name | Index Preparation | Non-Contig Type | What It Validates | Pass Criteria | Tol | Env | Priority | Status |
|---|---|---|---|---|---|---|---|---|---|
| NCI-01 | `test_gather_expanded_index` | `idx_1d.unsqueeze(1).expand(4,64)` zero stride | Zero-stride expansion | No double-read from address aliasing | allclose; no aliasing artifacts | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| NCI-02 | `test_gather_strided_index` | `full_idx[::2,:]` | Row-strided index | Non-unit index stride; correct subset used | allclose vs CPU with same strided index | 1e-2/1e-2 | ENV_BASE | P2 | READY |
| NCI-03 | `test_gather_unsqueezed_index` | `idx_1d.unsqueeze(-1)` → (4,1) | Extra singleton dim | Singleton dimension handled | Correct output; no shape mismatch | 1e-2/1e-2 | ENV_BASE | P2 | READY |
| NCI-04 | `test_gather_index_on_cpu` | index on "cpu" | Device mismatch | Device mismatch detected; not silent | RuntimeError raised | — | ENV_BASE | P1 | READY |

---

### 4.10 Restickify Behavior

| ID | Test Name | Setup | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|---|
| RS-01 | `test_no_restickify_on_gather_output` | Contiguous (64,64) source; gather dim=0; inspect IR | NO restickify op on gather output | No restickify node between gather and downstream consumer | ENV_BASE | P1 | READY |
| RS-02 | `test_restickify_on_noncontig_input` | `src.t()` transposed; inspect IR | Restickify IS inserted before gather | restickify node present before gather | ENV_BASE | P1 | READY |
| RS-03 | `test_shared_restickify_two_gathers` | `gather(src.t(), idx1)` and `gather(src.t(), idx2)` | Single shared restickify; not duplicated | Exactly 1 restickify; both gathers consume it | ENV_BASE | P2 | READY |
| RS-04 | `test_gather_output_is_contiguous` | Transposed source; gather | Output always contiguous | `result.is_contiguous() == True` | ENV_BASE | P1 | READY |
| RS-05 | `test_gather_then_matmul_no_restickify` | gather → matmul; inspect IR | No restickify between gather output and matmul | No restickify node between gather and matmul | ENV_BASE | P2 | READY |
| RS-06 | `test_restickify_before_not_after` | `lambda s,i: gather(s.t(), 0, i)` compiled | IR order: restickify before gather | restickify precedes gather; none after | ENV_BASE | P1 | READY |
| RS-07 | `test_restickify_fp32_noncontig` | fp32 source `src.t()` transposed; gather | Restickify fires for fp32 using fp32 stick width (32 el/stick) | restickify present; allclose; output fp32-contiguous | ENV_BASE | P1 | READY |

---

### 4.11 Compiler Pass Pipeline

| ID | Test Name | Setup | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|---|
| CP-01 | `test_add_index_pass_fires` | Compile gather; inspect FX graph | `add_index_to_address_pass` inserts `spyre.indices_to_address` node | Node present in FX graph | ENV_BASE | P1 | READY |
| CP-02 | `test_add_index_pass_gated_off` | Gate=0; inspect FX graph | Pass does NOT fire | No `spyre.indices_to_address` node | ENV_GATE_OFF | P1 | READY |
| CP-03 | `test_add_index_pass_cpu_skipped` | Input on "cpu" | Pass skips non-Spyre tensors | No node; CPU gather runs | CPU_ONLY | P1 | READY |
| CP-04 | `test_detect_no_false_positive_matmul` | Compile matmul only | detect pass does NOT mark matmul | No matmul op has `index_args` | CPU_ONLY | P1 | READY |
| CP-05 | `test_pass_order_detect_before_restickify` | Non-contig source; trace pass order | detect fires before restickify | detect (line 260) before restickify (line 264) | ENV_BASE | P1 | READY |
---

### 4.12 SDSC JSON / Kernel Spec Validation

> **Capture path (corrected against source):** the SDSC bundle dict is the direct return value of `generate_sdsc()`/`compile_op_spec()` (`torch_spyre/_inductor/codegen/superdsc.py`, `compute_ops.py`) — `compile_op_spec(idx, op_spec, symbols)[0]` returns it immediately given a hand-built `OpSpec`/`TensorArg`, with no `torch.compile`, device, or `TORCH_COMPILE_DEBUG` capture needed. This is the same direct-call technique already used by `tests/inductor/test_coarse_tiling.py`. The indirect-access fields (`indirectAllocType_`, `relatedIndirectAccessAlloc_`, `indirectAccessIndexLabeledDs`) are produced by `_build_indirect_access_fields()` purely from `TensorArg.is_index_tensor` / `related_value_tensor_idx` — no feature-gate env var or hardware involved at this layer.
>

| ID | Test Name | What to Inspect | Expected Value | Env | Priority | Status |
|---|---|---|---|---|---|---|
| SD-01 | `test_sdsc_index_alloc_type` | `indirectAllocType_` for the index tensor arg | `"index_tensor"` | CPU_ONLY | P1 | READY |
| SD-02 | `test_sdsc_value_alloc_type` | `indirectAllocType_` for the value (source) tensor arg | `"value_tensor"` | CPU_ONLY | P1 | READY |
| SD-03 | `test_sdsc_no_false_positive_pointwise` | Plain elementwise `add` OpSpec; check all `indirectAllocType_` fields | All `"no_indirection"` — indirect fields must NOT appear for non-indirect ops | CPU_ONLY | P1 | READY |
| SD-04 | `test_sdsc_indirect_labeled_ds` | `indirectAccessIndexLabeledDs` list in the kernel spec | Non-empty list; one entry per index-tensor argument (`["Tensor1-idx1"]` for a single-index op) | CPU_ONLY | P1 | READY |
| SD-05 | `test_sdsc_related_value_tensor_idx` | `relatedIndirectAccessAlloc_` on the index tensor's allocation | Points to the value tensor's allocation name (`"allocate-Tensor0_hbm"`) | CPU_ONLY | P1 | READY |
| SD-06 | `test_sdsc_op_field_identity` | `opFuncName` in the kernel spec's `computeOp_` | `"identity"` (the op `detect_indirect_access` marks gather/index_select/embedding with) | CPU_ONLY | P1 | READY |
| SD-07 | `test_sdsc_no_indirect_fields_when_unmarked` | OpSpec with no `is_index_tensor=True` arg at all (the gate-off case: `detect_indirect_access` never marked anything) | No `indirectAccessIndexLabeledDs` key present anywhere in the bundle | CPU_ONLY | P0 | READY |


---

### 4.13 Multi-Core / SEN Cores

| ID | Test Name | SENCORES | What It Validates | Pass Criteria | Tol | Env | Priority | Status |
|---|---|---|---|---|---|---|---|---|
| MC-01 | `test_gather_sencores4_1d` | 4 | SENCORES=4 == SENCORES=1 result | allclose(result_4c, result_1c) | 1e-2/1e-2 | ENV_MC4 | P1 | BLOCKED:MC |
| MC-02 | `test_gather_sencores4_2d` | 4 | SENCORES=4 2D gather | allclose | 1e-2/1e-2 | ENV_MC4 | P1 | BLOCKED:MC |
| MC-03 | `test_gather_sencores32_2d` | 32 | Default SENCORES=32 matches SENCORES=1 | allclose | 1e-2/1e-2 | ENV_MC32 | P1 | BLOCKED:MC |
| MC-04 | `test_gather_multicore_random` | 4 | Worst-case random indices across cores | allclose | 1e-2/1e-2 | ENV_MC4 | P1 | BLOCKED:MC |
| MC-05 | `test_gather_multicore_large` | 32 | Large input across 32 cores; no cross-core address collision | allclose | 1e-2/1e-2 | ENV_MC32 | P1 | BLOCKED:MC |
| MC-06 | `test_gather_multicore_matmul` | 4 | Multi-core gather feeding matmul | allclose | 1e-2/1e-2 | ENV_MC4 | P2 | BLOCKED:MC |

---

### 4.14 LX Planning / Scratchpad Allocation

| ID | Test Name | Config | What It Validates | Pass Criteria | Tol | Env | Priority | Status |
|---|---|---|---|---|---|---|---|---|
| LX-01 | `test_gather_lx_planning_compiles` | LX=1 | Scratchpad planner does not crash on gather | Compilation succeeds; allclose | 1e-2/1e-2 | ENV_LX | P1 | READY |
| LX-02 | `test_gather_lx_output_scratchpad` | LX=1 | Gather output allocated to scratchpad | Allocation visible in IR; correct result | 1e-2/1e-2 | ENV_LX | P1 | READY |
| LX-03 | `test_gather_co_optimizing_lx` | LX=1, CO_OPT=1 | `StrategyBCoOptimizingAllocator` does not crash | Compilation + correct result | 1e-2/1e-2 | ENV_LX_CO | P2 | READY |
| LX-04 | `test_gather_dxp_frac_low` | LX=1, DXP=0.1 | Tight scratchpad budget | allclose or clear OOM; no silent wrong | 1e-2/1e-2 | ENV_LX | P2 | READY |
| LX-05 | `test_gather_dxp_frac_high` | LX=1, DXP=0.8 | Larger budget; result unchanged | allclose; identical to DXP=0.2 | 1e-2/1e-2 | ENV_LX | P2 | READY |
| LX-06 | `test_gather_lx_non_aligned_shape` | LX=1, (32,50) | LX allocates rounded to stick boundary | result.shape==(4,50); allclose; no scratchpad overrun | 1e-2/1e-2 | ENV_LX | P1 | READY |

---

### 4.15 Chunk Large Tensors

| ID | Test Name | What It Validates | Pass Criteria | Tol | Env | Priority | Status |
|---|---|---|---|---|---|---|---|
| CL-01 | `test_gather_no_chunk_baseline` | Baseline gather without chunking; establish reference | allclose vs CPU; no CHUNK_LARGE_TENSORS in env | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| CL-02 | `test_gather_chunk_span_threshold` | Chunk triggers at span threshold; indirect access + chunk path | allclose or clear error; no silent wrong | 1e-2/1e-2 | ENV_CHUNK | P1 | BLOCKED:CHUNK |
| CL-03 | `test_gather_chunk_total_threshold` | Chunk triggers at total-size threshold | allclose or clear error | 1e-2/1e-2 | ENV_CHUNK | P1 | BLOCKED:CHUNK |

---

### 4.16 Edge Cases and Error Handling

#### 4.16a — ATen-Layer Error Validation

| ID | Test Name | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|
| EC-01 | `test_gather_index_out_of_bounds` | OOB index value >= source dim → RuntimeError | RuntimeError raised | ENV_BASE | P0 | READY |
| EC-02 | `test_gather_empty_index` | Zero-size index tensor → empty output | Output shape (0, K); no error | ENV_BASE | P1 | READY |
| EC-03 | `test_gather_negative_index` | Negative index value; no wraparound | RuntimeError raised | ENV_BASE | P0 | READY |
| EC-04 | `test_gather_wrong_index_dtype` | Float index → TypeError at ATen | TypeError raised | ENV_BASE | P1 | READY |

#### 4.16b — Valid Boundary Scenarios

| ID | Test Name | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|
| EC-05 | `test_gather_dim0_first_row` | Index = [0]; first row only | output[0] == src[0]; allclose | ENV_BASE | P1 | READY |
| EC-06 | `test_gather_dim0_last_row` | Index = [N-1]; last row only | output[0] == src[N-1]; allclose | ENV_BASE | P1 | READY |
| EC-07 | `test_gather_single_element_output` | Output size = 1 | Output shape (1,1); value correct | ENV_BASE | P2 | READY |

#### 4.16c — Spyre-Specific Error Paths

| ID | Test Name | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|
| EC-09 | `test_gather_gate_off_1d_dim0` | Gate=0, 1D padded, dim=0 — gate-off baseline safety | Fallback+warning OR RuntimeError; no wrong result | ENV_GATE_OFF | P0 | READY |
| EC-09b | `test_gather_gate_off_2d_dim0` | Gate=0, 2D aligned 64 cols, dim=0 | No wrong result | ENV_GATE_OFF | P0 | READY |
| EC-09c | `test_gather_gate_off_2d_dim1` | Gate=0, 2D, dim=1 | No wrong result | ENV_GATE_OFF | P0 | READY |
| EC-09d | `test_gather_gate_off_50cols` | Gate=0, 2D, 50 cols non-aligned | No wrong result | ENV_GATE_OFF | P0 | READY |
| EC-09e | `test_gather_gate_off_fp32` | Gate=0, float32 | No wrong result | ENV_GATE_OFF | P0 | READY |
| EC-09f | `test_gather_gate_off_gather_matmul` | Gate=0, composed gather→matmul | gather: fallback or error; matmul: allclose vs CPU | ENV_GATE_OFF | P1 | READY |

#### 4.16d — Cache Invalidation and Misc

| ID | Test Name | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|
| EC-09g | `test_gather_cache_invalidation` | Compile gate=1 then toggle to gate=0 | Recompile observed OR correct CPU-fallback; no stale kernel | ENV_BASE / ENV_GATE_OFF | P1 | READY |
| EC-10 | `test_gather_empty_source_dim` | source dim=0 size is 0 | Empty output; no crash | ENV_BASE | P2 | READY |
| EC-11 | `test_gather_unsupported_dtype_gate_off` | Float64 with gate=0 | No wrong result; consistent with gate=1 | ENV_GATE_OFF | P2 | READY |
| EC-12 | `test_gather_non_contiguous_output_rejected` | Non-contiguous out= buffer | RuntimeError or auto-allocation; no corruption | ENV_BASE | P2 | READY |
| EC-13 | `test_gather_large_index_values` | Very large valid index near dim-size boundary | Correct fetch; no address overflow | ENV_BASE | P2 | READY |

---

### 4.17 Model Scenarios — Embedding Table Lookup

| ID | Test Name | What It Models | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|---|
| EMB-01 | `test_gather_embedding_style_fp16` | Embedding table lookup: (vocab,embed) weight, row selection via gather dim=0 | allclose vs F.embedding reference | ENV_BASE | P1 | READY |
| EMB-02 | `test_gather_embedding_large_vocab` | LLM-scale vocab: (32768, 64) | allclose vs CPU; no address overflow | ENV_BASE | P2 | READY |
| EMB-03 | `test_gather_embedding_then_linear` | Token embedding → linear projection | Linear output matches CPU reference | ENV_BASE | P1 | READY |
| EMB-04 | `test_gather_embedding_batch_seqlen` | Batched embedding: (B, seq_len) index → (B, seq_len, embed) | Output shape correct; all values match | ENV_BASE | P2 | READY |
---

### 4.18 Model Scenarios — Unique Index Patterns

> One representative test per production model usage pattern. Each captures a distinct index-access shape or routing structure not exercised by sections 4.1–4.11.
>
> - **4.26 GQA**: covered by Section 8 PA-G-03 (`test_paged_gather_gqa`) and PA-EXT-05 (`test_paged_attn_mqa_shared_kv`) — no duplicate added here.

#### 4.18a — Sparse Attention Block Selection

| ID | Test Name | Unique Index Pattern | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|---|
| SA-01 | `test_sparse_attn_gather_block_mask` | Non-sequential block IDs from a binary block mask; sparse, non-contiguous index tensor | gather with sparse block indices: only non-zero block positions accessed; tests the index path with large gaps between valid positions | allclose vs dense reference with mask applied | ENV_BASE | P1 | READY |

#### 4.18b — Positional Embedding Table Lookup

| ID | Test Name | Unique Index Pattern | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|---|
| PE-01 | `test_rotary_pos_gather` | 1D position ID tensor (seq_len,) into a (max_seq_len, head_dim) sin/cos table; many queries into a small table | gather from a smaller-than-output source; tests stride formula when N_queries > N_source_rows | allclose vs CPU reference | ENV_BASE | P1 | READY |

#### 4.18c — Beam Search Row Expansion

| ID | Test Name | Unique Index Pattern | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|---|
| BS-01 | `test_beam_expand_gather` | Repeated index pattern: each source row gathered B times (beam_width=4); index = repeat([0,1,…,N-1], B) | gather with duplicate indices; validates no aliasing or overwrite between repeated accesses; same source row fetched to multiple output rows | allclose; each beam copy identical | ENV_BASE | P1 | READY |

#### 4.18d — Mixture of Experts Top-K Routing

| ID | Test Name | Unique Index Pattern | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|---|
| MOE-01 | `test_moe_topk_routing_gather` | Per-token top-k index: shape (batch_tokens, top_k) indexes into (num_experts, expert_dim) weight matrix | 2D index tensor (non-1D) selecting K rows per token from E expert rows; validates 2D index → 3D output address formula | allclose vs CPU reference | ENV_BASE | P2 | READY |


---

### 4.19 Chained Gather Patterns

| ID | Test Name | Graph Structure | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|---|
| CH-01 | `test_chained_gather_2d` | gather(src, 0, idx1) → gather(g_out, 0, idx2) | Two-level indirect access | Final output matches CPU two-step reference | ENV_BASE | P1 | READY |
| CH-02 | `test_chained_gather_intermediate_op` | gather → relu → gather | relu between two gathers | allclose | ENV_BASE | P2 | READY |
| CH-03 | `test_chained_gather_address_independence` | Two chained gathers; different shapes | Second gather uses first's output shape for addresses | allclose for both gather outputs | ENV_BASE | P1 | BLOCKED:HW |

---

### 4.20 Exact Value Roundtrip

| ID | Test Name | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|
| RT-01 | `test_gather_exact_known_value` | Known values in source; verify gather returns exact values | Element-wise exact match (fp16 atol=0) for known seeds | ENV_BASE | P0 | READY |
| RT-02 | `test_gather_identity_index` | Identity permutation index; gather output == source | allclose(result, source) | ENV_BASE | P1 | READY |

---

### 4.21 LX + virtual_offset Correctness

| ID | Test Name | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|
| LX-07 | `test_lx_virtual_offset_scratchpad_source` | LX allocates gather source to scratchpad at non-zero offset; address computation must use that offset | allclose vs CPU; no silent wrong result from wrong virtual_offset | ENV_LX | P1 | READY |
| LX-08 | `test_lx_virtual_offset_direct_call` | Direct call to `_C.indices_to_addresses_nd` with known `virtual_offset`; address = virtual_offset + stride × index | Exact address matches formula with explicit virtual_offset | CPU_ONLY | P0 | READY |

---

### 4.22 Code Behavior Confirmations

| ID | Test Name | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|
| CB-02 | `test_gate_in_cache_key` | Gate env var is included in Inductor compilation cache key; toggling gate forces recompile | Recompile triggered when gate changes; no stale kernel served | ENV_BASE / ENV_GATE_OFF | P1 | READY |

---

### 4.23 Symbolic Bundle Args

| ID | Test Name | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|
| BA-01 | `test_bundle_symbolic_args_gather` | `BUNDLE_SYMBOLIC_ARGS=1`: HBM addresses emitted as runtime symbols rather than baked integers | Compiles and runs correctly | ENV_BASE + BUNDLE_SYMBOLIC_ARGS=1 | P2 | READY |

---

### 4.24 Chained Ops with Layout-Changing Upstream

| ID | Test Name | Graph Structure | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|---|
| CNW-01 | `test_gather_on_matmul_output` | matmul → gather | `device_stride` derived from matmul output layout | allclose | ENV_BASE | P1 | READY |
| CNW-02 | `test_gather_on_layernorm_output` | layernorm → gather | `device_stride` from layernorm output | allclose | ENV_BASE | P2 | READY |
| CNW-03 | `test_gather_on_relu_output` | relu → gather | relu output is gather source; address formula uses relu output strides | allclose | ENV_BASE | P1 | READY |

---

### 4.25 Paged Attention Token Position Correctness

| ID | Test Name | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|
| PATC-01 | `test_paged_attn_token_exact_position` | Sentinel values at known token positions; gather at those positions returns exact sentinel values | Bit-exact sentinel value match at each position | ENV_BASE | P1 | BLOCKED:HW |
| PATC-02 | `test_paged_attn_page_boundary_tokens` | Pool with sentinel values at first and last token of each page; gather at boundary positions | Boundary token values match sentinels; no off-by-one | ENV_BASE | P1 | BLOCKED:HW |

---

### 4.26 Multi-Source Independent Gather

| ID | Test Name | Setup | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|---|
| MP-01 | `test_two_gathers_same_source_different_indices` | `gather(src, 0, idx1)` and `gather(src, 0, idx2)` in one compiled graph; idx1 ≠ idx2 | Two independent address nodes; no cross-gather aliasing | allclose(out1, ref1) and allclose(out2, ref2) independently | ENV_BASE | P1 | READY |
| MP-02 | `test_two_gathers_different_sources` | `gather(src1 (64,64), 0, idx1)` and `gather(src2 (32,64), 0, idx2)` | Each gather uses its own source's device_stride | Both outputs allclose vs independent CPU references | ENV_BASE | P1 | READY |
| MP-03 | `test_two_gathers_different_dtypes` | `gather(src_fp16 (64,64), 0, idx1)` and `gather(src_fp32 (32,32), 0, idx2)` | Each uses own element_size (2 vs 4); dtype-specific address scaling | Both allclose; dtypes preserved | ENV_BASE | P2 | READY |
---

### 4.27 Key Downstream Chain Patterns — torch.gather

> These two tests validate that the gather output is correctly consumed by the downstream ops that appear in production graphs. Validates layout compatibility between the address-based gather output and the consumer's input expectations.

| ID | Test Name | Graph Pattern | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|---|
| OC-09 | `test_gather_then_softmax` | gather(2D, dim=0) → softmax(dim=-1) | Attention score pattern: gather selects rows then softmax normalizes; validates output layout flows correctly into softmax | allclose vs CPU; softmax output sums to 1.0 per row | ENV_BASE | P1 | READY |
| OC-11 | `test_gather_then_matmul` | gather(2D, dim=0) → matmul(gathered, weight) | Token embedding → projection pattern: gather selects embeddings then matmul projects; validates device_stride of gather output used correctly as matmul input | allclose vs CPU | ENV_BASE | P1 | READY |


---

## Section 5 — torch.index_select FVT Test Cases

> **FX pass handler:** `_add_indices_to_address_index` — inserts `spyre::indices_to_address` node for `aten.index.Tensor`.
>
> **Address formula:** `address = virtual_offset + Σᵢ (coord[i] × device_stride[i]) × element_size`
>
> **API signature:** `torch.index_select(input, dim, index)` — index is always a 1D int64 tensor; output shape replaces `input.size(dim)` with `len(index)`.
>
> **FX op:** `aten.index.Tensor` (advanced indexing `input[index_list]`) — the same op dispatched by `torch.index_select` and also by Python `[]` advanced indexing.
>
> **Count correction:** this section was previously stated as 112 net-new tests; that figure did not match a row-by-row count of the tables below (a stale/approximate value, not derived from the actual rows). Implementing every row directly against source — including catching the `IS-FC-12`/`IS-AV-04` duplicate — gives an actual count of **86** (77 from Sections 5.1–5.12, plus the 9 from Section 5.13 added in the gap-closure revision). All summary tables below reflect 86.

---

### 5.1 Functional Correctness — Shape and Dimension

| ID | Test Name | Input Shape | Index | dim | dtype | What It Validates | Pass Criteria | Tol | Env | Priority | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| IS-FC-01 | `test_is_2d_dim0` | (64,64) | 1D len=4 | 0 | float16 | Row selection on 2D tensor; output shape (4,64) | `result[i,:]==src[idx[i],:]`; allclose | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| IS-FC-02 | `test_is_2d_dim1` | (4,64) | 1D len=8 | 1 | float16 | Column selection on 2D tensor; output shape (4,8) | `result[:,j]==src[:,idx[j]]`; allclose | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| IS-FC-03 | `test_is_3d_dim0` | (8,4,64) | 1D len=3 | 0 | float16 | Page selection on 3D tensor | allclose vs CPU | 1e-2/1e-2 | ENV_BASE | P1 | BLOCKED:HW |
| IS-FC-04 | `test_is_3d_dim1` | (2,8,64) | 1D len=3 | 1 | float16 | Row selection on 3D tensor (batch dim preserved) | allclose vs CPU | 1e-2/1e-2 | ENV_BASE | P1 | BLOCKED:HW |
| IS-FC-05 | `test_is_3d_dim2` | (2,4,64) | 1D len=16 | 2 | float16 | Feature selection on 3D tensor | allclose vs CPU | 1e-2/1e-2 | ENV_BASE | P1 | BLOCKED:HW |
| IS-FC-06 | `test_is_4d_dim0` | (8,2,4,64) | 1D len=3 | 0 | float16 | Page-axis selection on 4D tensor | allclose vs CPU | 1e-2/1e-2 | ENV_BASE | P1 | BLOCKED:HW |
| IS-FC-07 | `test_is_4d_dim1` | (2,8,4,64) | 1D len=3 | 1 | float16 | Sequence-axis selection on 4D (KV cache) | allclose vs CPU | 1e-2/1e-2 | ENV_BASE | P1 | BLOCKED:HW |
| IS-FC-08 | `test_is_4d_dim3` | (2,2,4,64) | 1D len=32 | 3 | float16 | Feature-axis selection on 4D | allclose vs CPU | 1e-2/1e-2 | ENV_BASE | P1 | BLOCKED:HW |
| IS-FC-09 | `test_is_2d_neg_dim` | (4,64) | 1D len=8 | -1 | float16 | `dim=-1` == `dim=1` on 2D | Identical to IS-FC-02; allclose | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| IS-FC-10 | `test_is_3d_neg_dim` | (2,4,64) | 1D len=16 | -1 | float16 | `dim=-1` == `dim=2` on 3D | allclose vs CPU | 1e-2/1e-2 | ENV_BASE | P1 | BLOCKED:HW |
| IS-FC-11 | `test_is_2d_large_cols` | (4,256) | 1D len=64 | 1 | float16 | Wide tensor (256 cols); multi-stick column stride | allclose | 1e-2/1e-2 | ENV_BASE | P2 | READY |

---

### 5.2 Index Access Patterns

| ID | Test Name | Input Shape | Index Pattern | dim | What It Validates | Pass Criteria | Tol | Env | Priority | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| IS-IP-01 | `test_is_index_subset` | (64,64) | 4 of 64 rows | 0 | Sparse read via index_select | `result[i,:]==src[idx[i],:]` | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| IS-IP-02 | `test_is_index_all_rows` | (32,64) | All 32 rows, identity | 0 | Full copy baseline | allclose(result, src_cpu) | 1e-2/1e-2 | ENV_BASE | P2 | READY |
| IS-IP-03 | `test_is_index_single_row` | (64,64) | Index = [5] | 0 | Single-row selection | output.shape == (1,64); output[0]==src[5] | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| IS-IP-04 | `test_is_index_repeated` | (16,64) | Index = [2,2,2,2] | 0 | Same row repeated | All output rows identical to src[2,:] | 1e-2/1e-2 | ENV_BASE | P2 | READY |
| IS-IP-05 | `test_is_index_sequential_asc` | (64,64) | [0,1,2,...,7] | 0 | Sequential ascending | result[:8,:]==src[:8,:] | 1e-2/1e-2 | ENV_BASE | P2 | READY |
| IS-IP-06 | `test_is_index_sequential_desc` | (64,64) | [7,6,5,...,0] | 0 | Sequential descending | allclose vs reversed reference | 1e-2/1e-2 | ENV_BASE | P2 | READY |
| IS-IP-07 | `test_is_index_random_permuted` | (64,64) | Random permutation [0..63] | 0 | Worst-case HBM locality | allclose vs CPU | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| IS-IP-08 | `test_is_index_interleaved` | (64,64) | [0,32,1,33,...] | 0 | Cache-unfriendly stride pattern | allclose vs CPU | 1e-2/1e-2 | ENV_BASE | P2 | READY |
| IS-IP-09 | `test_is_dim1_sparse_cols` | (8,64) | 8 of 64 cols | 1 | Sparse column selection via dim=1 | `result[:,j]==src[:,idx[j]]` | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| IS-IP-10 | `test_is_dim1_all_cols` | (8,64) | All 64 cols | 1 | Full column identity | allclose(result, src_cpu) | 1e-2/1e-2 | ENV_BASE | P2 | READY |

---

### 5.3 API Variants

| ID | Test Name | Call Form | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|---|
| IS-AV-01 | `test_is_functional_form` | `torch.index_select(input, dim, index)` | Functional form baseline | allclose | ENV_BASE | P1 | READY |
| IS-AV-02 | `test_is_method_form` | `input.index_select(dim, index)` | Method form identical to functional | allclose vs IS-AV-01 | ENV_BASE | P2 | READY |
| IS-AV-03 | `test_is_out_parameter` | `torch.index_select(..., out=buf)` | Pre-allocated out buffer | `result is buf`; allclose | ENV_BASE | P2 | READY |
| IS-AV-04 | `test_is_advanced_index_brackets` | `src[index_1d]` | Python advanced indexing syntax dispatches to aten.index.Tensor | allclose vs torch.index_select | ENV_BASE | P1 | READY |
| IS-AV-05 | `test_is_keyword_args` | `torch.index_select(input=.., dim=.., index=..)` | Keyword argument form | allclose vs positional | ENV_BASE | P2 | READY |

---

### 5.4 Dtype Coverage

| ID | Test Name | dtype | element_size | What It Validates | Pass Criteria | Tol | Env | Priority | Status |
|---|---|---|---|---|---|---|---|---|---|
| IS-DT-01 | `test_is_bfloat16` | bfloat16 | 2 | bf16 stick-expression for index_select | allclose vs CPU | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| IS-DT-02 | `test_is_float32` | float32 | 4 | stick width = 32 elems for index_select | allclose vs CPU | 1e-5/1e-5 | ENV_BASE | P1 | READY |
| IS-DT-03 | `test_is_int64_exact` | int64 | 8 | 16 elements/stick; integer index_select | exact equality | exact | ENV_BASE | P2 | READY |
| IS-DT-04 | `test_is_int32_exact` | int32 | 4 | same stick width as float32 | exact equality | exact | ENV_BASE | P2 | READY |
| IS-DT-05 | `test_is_bool_exact` | bool | 1 | 128 elements/stick | exact bool equality | exact | ENV_BASE | P2 | READY |
| IS-DT-06 | `test_is_unsupported_dtype` | float64 | 8 | Unsupported dtype must NOT silently produce wrong results | RuntimeError OR CPU fallback + FallbackWarning | — | ENV_BASE | P1 | READY |
| IS-DT-07 | `test_is_fp32_vs_fp16_addresses` | fp32 vs fp16 | 4 vs 2 | `device_stride` scaling by `element_size` correct for index_select | allclose(fp32_result.half(), fp16_result) | 1e-2/1e-2 | ENV_BASE | P2 | READY |

---

### 5.5 Execution Mode

| ID | Test Name | Mode | What It Validates | Pass Criteria | Tol | Env | Priority | Status |
|---|---|---|---|---|---|---|---|---|
| IS-EM-01 | `test_is_eager_2d` | Eager (no compile) | Eager path: correct result or clear error | Correct result OR RuntimeError | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| IS-EM-02 | `test_is_fullgraph` | `torch.compile(fullgraph=True)` | No graph break in index_select lowering | No Unsupported exception; allclose | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| IS-EM-03 | `test_is_graph_softmax` | Compiled | index_select → softmax in single graph | allclose vs CPU | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| IS-EM-04 | `test_is_graph_matmul` | Compiled | index_select → matmul | allclose vs CPU | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| IS-EM-05 | `test_is_graph_layernorm` | Compiled | index_select → layernorm | allclose vs CPU | 1e-2/1e-2 | ENV_BASE | P2 | READY |
| IS-EM-06 | `test_is_graph_add` | Compiled | index_select → elementwise add | allclose vs CPU | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| IS-EM-07 | `test_is_repeated_calls` | Compiled × 3 calls | No state leakage across successive calls | All 3 calls correct | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| IS-EM-08 | `test_is_composed_with_gather` | Compiled | index_select + gather in same compiled graph | Both ops correct; allclose | 1e-2/1e-2 | ENV_BASE | P1 | READY |

---

### 5.6 Dynamic / Symbolic Shapes

| ID | Test Name | Dynamic Dims | What It Validates | Pass Criteria | Tol | Env | Priority | Status |
|---|---|---|---|---|---|---|---|---|
| IS-DS-01 | `test_is_dynamic_index_len` | len(index) dynamic | Kernel handles different numbers of selected rows | Both calls correct; ≤1 recompile | 1e-2/1e-2 | ENV_DYN | P1 | READY |
| IS-DS-02 | `test_is_dynamic_source_rows` | Source dim-0 | Different source → different strides → address adapts | Both calls correct | 1e-2/1e-2 | ENV_DYN | P1 | READY |
| IS-DS-03 | `test_is_dynamic_batch` | Batch dim B (3D input) | Realistic LLM: batch varies per request | Correct for B=1,2,4 | 1e-2/1e-2 | ENV_DYN | P1 | BLOCKED:HW |
| IS-DS-04 | `test_is_dynamic_seq_len` | S dynamic | KV selection with dynamic sequence length | Correct for multiple S values | 1e-2/1e-2 | ENV_DYN | P1 | BLOCKED:HW |
| IS-DS-05 | `test_is_no_recompile_in_guard` | dim-0 of index range [1,8] | Once compiled for range, no recompile per size | Single graph; all 8 sizes correct | 1e-2/1e-2 | ENV_DYN | P2 | READY |
| IS-DS-06 | `test_is_export_dynamic` | N via `torch.export` | Exported program handles any N | Correct for N=2,4,8,16 | 1e-2/1e-2 | ENV_BASE | P2 | READY |
| IS-DS-07 | `test_is_dynamic_chain` | N and M independent | index_select then gather with independent dynamic sizes | Both correct for various (N,M) | 1e-2/1e-2 | ENV_DYN | P2 | READY |

---

### 5.7 Memory Layout

| ID | Test Name | Input Shape | Layout / Non-Contig Type | What It Validates | Pass Criteria | Tol | Env | Priority | Status |
|---|---|---|---|---|---|---|---|---|---|
| IS-ML-01 | `test_is_stick_aligned_64` | (32,64) | 1 stick/row | No padding artifacts | allclose; shape (4,64) | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| IS-ML-02 | `test_is_non_aligned_50` | (32,50) padded to 64 | 50 cols | Output shape (4,50); padding does not leak | Shape (4,50); values correct | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| IS-ML-03 | `test_is_transposed_src` | `src.t()` (64,32) | Transposed | Restickify before index_select | allclose | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| IS-ML-04 | `test_is_strided_src` | `src[::2,:]` | Row stride=2 | Non-unit stride; restickify fires | allclose | 1e-2/1e-2 | ENV_BASE | P2 | READY |
| IS-ML-05 | `test_is_sliced_offset` | `src[:,4:20]` | Non-zero storage offset | Storage offset in restickify | allclose | 1e-2/1e-2 | ENV_BASE | P2 | READY |
| IS-ML-06 | `test_is_large_baseline` | (512,64) | Large contiguous | No OOB in address computation | allclose vs CPU | 1e-2/1e-2 | ENV_BASE | P1 | READY |

---

### 5.8 Edge Cases and Error Handling

| ID | Test Name | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|
| IS-EC-01 | `test_is_index_out_of_bounds` | OOB index value >= source dim → RuntimeError | RuntimeError raised | ENV_BASE | P0 | READY |
| IS-EC-02 | `test_is_empty_index` | Zero-size index tensor → empty output | Output shape (0, K); no error | ENV_BASE | P1 | READY |
| IS-EC-03 | `test_is_negative_index` | Negative index value | RuntimeError raised | ENV_BASE | P0 | READY |
| IS-EC-04 | `test_is_2d_index_rejected` | 2D index tensor to index_select (requires 1D) | RuntimeError raised at ATen layer | ENV_BASE | P1 | READY |
| IS-EC-05 | `test_is_index_device_mismatch` | index on CPU, input on Spyre | RuntimeError with clear message | ENV_BASE | P1 | READY |
| IS-EC-06 | `test_is_gate_off_2d` | Gate=0, 2D input | Fallback+warning OR RuntimeError; no wrong result | ENV_GATE_OFF | P0 | READY |
| IS-EC-07 | `test_is_empty_source_dim` | source dim=0 size is 0 | Empty output; no crash | ENV_BASE | P2 | READY |
| IS-EC-08 | `test_is_single_element_output` | Index = [N-1]; output shape (1, ...) | Correct single element; no crash | ENV_BASE | P2 | READY |
| IS-EC-09 | `test_is_first_row` | Index = [0]; first row only | output[0]==src[0]; allclose | ENV_BASE | P1 | READY |
| IS-EC-10 | `test_is_last_row` | Index = [N-1]; last row only | output[0]==src[N-1]; allclose | ENV_BASE | P1 | READY |

---

### 5.9 Model Scenarios — KV Cache and Paged Attention

| ID | Test Name | What It Models | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|
| IS-KV-01 | `test_is_kv_cache_row_select_2d` | 2D flat KV cache row selection via index_select dim=0; validates production KV access pattern | allclose vs CPU | ENV_BASE | P0 | READY |
| IS-KV-04 | `test_is_paged_attn_2d_small` | 2D paged attention hardware-ready verification via index_select | allclose; assertion enabled | ENV_BASE | P0 | READY |
| IS-KV-06 | `test_is_embedding_table_lookup` | Embedding table (vocab, embed_dim): row selection via index_select validates the op as embedding lookup | allclose vs nn.Embedding | ENV_BASE | P1 | READY |

---

### 5.10 Compiler Pass and Exact Value Roundtrip

| ID | Test Name | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|
| IS-RT-01 | `test_is_exact_known_value` | Known values in source; verify index_select returns exact values | Element-wise exact match for known seeds | ENV_BASE | P0 | READY |
| IS-RT-02 | `test_is_identity_index` | Identity permutation index; output == source | allclose(result, source) | ENV_BASE | P1 | READY |
| IS-RT-03 | `test_is_pass_fires` | Compile index_select; inspect FX graph | `spyre.indices_to_address` node present for index_select | Node present | ENV_BASE | P1 | READY |
| IS-RT-04 | `test_is_pass_gated_off` | Gate=0; inspect FX graph | Pass does NOT fire | No `spyre.indices_to_address` node | ENV_GATE_OFF | P1 | READY |
| IS-RT-05 | `test_is_dynamic_shapes_mc4` | SENCORES=4, dynamic index_select | allclose(mc4, mc1) | ENV_MC4 | P2 | BLOCKED:MC |
| IS-RT-06 | `test_is_lx_planning` | LX scratchpad + index_select | Compilation succeeds; allclose | ENV_LX | P2 | READY |
---

### 5.11 Unique Index Patterns — Model Scenarios

> index_select equivalent of Section 4.22 unique patterns. Only the patterns where index_select's semantics differ materially from gather are included (index_select always uses a 1D index on one dimension).

| ID | Test Name | Unique Index Pattern | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|---|
| IS-MP-01 | `test_is_sparse_block_select` | Non-sequential 1D block IDs from sparse mask; large gaps between selected indices | index_select with sparse non-contiguous indices; validates address spacing in non-sequential access pattern | allclose vs CPU | ENV_BASE | P1 | READY |
| IS-MP-02 | `test_is_moe_expert_select` | 1D top-k expert IDs per batch step; index into (num_experts, expert_dim) | index_select dim=0 selecting K rows from expert matrix; validates row-selection address formula at expert-routing scale | allclose vs CPU | ENV_BASE | P2 | READY |

---

### 5.12 Key Downstream Chain Patterns — torch.index_select

| ID | Test Name | Graph Pattern | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|---|
| IS-OC-01 | `test_is_then_softmax` | index_select(2D, dim=0) → softmax(dim=-1) | KV selection → attention normalization; validates output layout compatible with softmax | allclose vs CPU; softmax sums to 1.0 per row | ENV_BASE | P1 | READY |
| IS-OC-02 | `test_is_then_matmul` | index_select(2D, dim=0) → matmul(selected, weight) | KV row selection → projection; validates index_select output strides used correctly as matmul input | allclose vs CPU | ENV_BASE | P1 | READY |

---

### 5.13 Advanced Indexing (`tensor[index]`) — PyTorch Contract Coverage

> `aten.index.Tensor` is the FX op shared by both `torch.index_select` (Sections 5.1–5.12, always a 1D index — a hard PyTorch API constraint, see IS-EC-04) and Python `[]` advanced indexing (`tensor[index_tensor]`), whose PyTorch contract is far broader: arbitrary index-tensor rank, multiple simultaneous tensor indices with broadcasting, negative index values, and boolean masks. Per the principle that every PyTorch-valid pattern gets a test regardless of current torch-spyre support, this subsection covers the patterns unique to advanced indexing (i.e. not reachable through `torch.index_select` itself). Status reflects direct code review of `_add_indices_to_address_index` (`torch_spyre/_inductor/indices_to_address_pass.py`) and `indices_to_addresses_nd` (`torch_spyre/csrc/spyre_address_computation.cpp`), not an assumption — so unsupported patterns are marked `BLOCKED:FX`/`BLOCKED:HW` rather than omitted. Boolean-mask indexing (`test_bool_indices`, `test_byte_mask2d`, `test_multi_dimensional_bool_mask`, etc.) and basic+advanced combined indexing (`test_basic_advanced_combined`, `test_ellipsis_tensor`) are PyTorch-valid `aten.index.Tensor` patterns already tracked as upstream xfail rows in Section 1 (lines ~194–217) and are not duplicated here.

#### 5.13a Index Tensor Rank Matrix

| ID | Test Name | Source Shape | Index Shape | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|---|---|
| IX-RANK-01 | `test_ix_rank_0d_scalar` | (64,64) | `()` (0D scalar tensor) | `src[idx_scalar]` — 0D index tensor selects a single row; no rank check exists in the FX pass, so this is mechanically supported | allclose vs `src[int(idx_scalar)]` | ENV_BASE | P2 | READY |
| IX-RANK-02 | `test_ix_rank_2d_generic` | (10,8) | (4,3) | 2D index into 2D source (general case, not tied to a specific paged-attention shape — contrast with PA-ND-01/PA-SH-01); output (4,3,8) per PyTorch advanced-indexing broadcast rule | allclose vs CPU `src[idx]` | ENV_BASE | P1 | READY |
| IX-RANK-03 | `test_ix_rank_3d_index` | (10,8) | (2,4,3) | 3D index tensor into 2D source; output (2,4,3,8) | allclose vs CPU | ENV_BASE | P2 | BLOCKED:HW |
| IX-RANK-04 | `test_ix_rank_4d_index` | (10,8) | (2,2,4,3) | 4D index tensor into 2D source; output (2,2,4,3,8) | allclose vs CPU | ENV_BASE | P2 | BLOCKED:HW |
| IX-RANK-05 | `test_ix_rank_ambiguity_2d_source` | (10,8) | (5,2) | **Finding from this review:** the C++ kernel branches on `indices_shape[-1] == value_ndim` to decide between "coordinate-tuple indexing" and "single-axis batched indexing." Here `value_ndim=2` and the index's last dim is also `2` — an accidental collision that risks misrouting correct batched indexing into coordinate-tuple semantics | allclose vs CPU `src[idx]` (must NOT match coordinate-tuple/`nonzero`-style gather semantics) | ENV_BASE | P0 | READY |

#### 5.13b Multiple Simultaneous Tensor Indices — PyTorch-valid, not implemented in FX pass

> **Finding:** `_add_indices_to_address_index` (`indices_to_address_pass.py:188-198`) loops over the `indices: List[Optional[Tensor]]` argument of `aten.index.Tensor` and `break`s on the **first** non-None entry — any second or third tensor index is silently left unconverted. This is not a raised error; it risks a silent incorrect result rather than a clean failure, which is why these are `BLOCKED:FX` at P0 rather than lower priority.

| ID | Test Name | Pattern | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|---|
| IX-MULTI-01 | `test_ix_multi_two_simultaneous_indices` | `src3d[idx0, idx1]` — two tensor indices on a 3D source's first two axes | PyTorch contract: both axes indexed simultaneously, broadcast together; output shape = `broadcast(idx0,idx1).shape + src.shape[2:]` | allclose vs CPU `src3d[idx0, idx1]` — must catch silent fallback to first-index-only semantics | ENV_BASE | P0 | BLOCKED:FX |
| IX-MULTI-02 | `test_ix_multi_broadcast_shapes` | `src3d[idx0, idx1]` with idx0 shape (3,1), idx1 shape (1,4) — broadcast to (3,4) | PyTorch broadcasting semantics between two index tensors of different shape | allclose vs CPU; output shape (3,4,*) | ENV_BASE | P1 | BLOCKED:FX |

#### 5.13c Negative Index Values — PyTorch-valid, intentionally rejected by the address kernel

> **Correction (verified directly against source, not from the earlier code-review pass):** `indices_to_addresses_nd` (`spyre_address_computation.cpp:128`) **does** guard this: `TORCH_CHECK(index >= 0 && index < value_shape[dim], "Index ", index, " out of bounds for dimension ", dim, ...)`. This is exercised today by the existing unit test `test_negative_index_validation` (`tests/indirect_access/test_indices_to_address.py:286-296`), which asserts a `RuntimeError` with `"out of bounds"` for a `-1` index. There is no memory-safety risk — the kernel deliberately rejects any negative value rather than normalizing it. The PyTorch-contract gap is semantic, not a safety bug: PyTorch's advanced indexing wraps negative values (`-1` == last row) and returns a result; torch-spyre raises instead. These tests document that divergence at the full compiled `tensor[idx]` path (the existing unit test only exercises the custom op directly, bypassing the FX-detection pass), so `Status: READY` — they execute and pass today, asserting current intentional behavior.

| ID | Test Name | Pattern | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|---|
| IX-NEG-01 | `test_ix_negative_single_index` | `src[torch.tensor([-1, 0, 2])]` (compiled, via `tensor[idx]`) | Single negative index value (`-1` == last row in PyTorch); confirms the FX-inserted address node inherits the same bounds check as the direct custom-op call | `RuntimeError` raised with `"out of bounds"`; documents divergence from PyTorch's wraparound semantics (FRS note, not a bug) | ENV_BASE | P2 | READY |
| IX-NEG-02 | `test_ix_negative_mixed_indices` | `src[torch.tensor([-3, 1, -1, 4])]` (compiled) | Mixed positive/negative indices in one index tensor — confirms the check fires per-element, not just on an all-negative tensor | `RuntimeError` raised with `"out of bounds"` | ENV_BASE | P2 | READY |

---

## Section 6 — torch.nn.Embedding / F.embedding FVT Test Cases

> **FX pass handler:** `_add_indices_to_address_embedding` — inserts `spyre::indices_to_address` node for `aten.embedding.default`.
>
> **Address formula:** `address = virtual_offset + idx × device_stride[0] × element_size` (dim=0; index is a flat or batched integer tensor; output shape = `(*index.shape, embedding_dim)`).
>
> **API:** `nn.Embedding(num_embeddings, embedding_dim)` and `F.embedding(input, weight)` — both dispatch to `aten.embedding.default`.
>
> **Key difference from gather/index_select:** embedding index is typically a long (int64) tensor of shape `(*)` with arbitrary leading batch dimensions; weight is (num_embeddings, embedding_dim).
>
> **Count correction:** this section was previously stated as 71 net-new tests; a row-by-row count of the tables below (the same check applied to Section 5) gives **52**. All summary tables below reflect 52.

---

### 6.1 Functional Correctness — Shape and Index Dimension

| ID | Test Name | Weight Shape | Index Shape | What It Validates | Pass Criteria | Tol | Env | Priority | Status |
|---|---|---|---|---|---|---|---|---|---|
| EM-FC-01 | `test_emb_1d_index` | (64,64) | (8,) | 1D index; output (8,64) | `result[i]==weight[idx[i]]`; allclose | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| EM-FC-02 | `test_emb_2d_index` | (64,64) | (4,8) | 2D batch+seq index; output (4,8,64) | allclose vs F.embedding(cpu) | 1e-2/1e-2 | ENV_BASE | P1 | BLOCKED:HW |
| EM-FC-03 | `test_emb_3d_index` | (64,64) | (2,4,8) | 3D index; output (2,4,8,64) | allclose vs CPU | 1e-2/1e-2 | ENV_BASE | P2 | BLOCKED:HW |
| EM-FC-04 | `test_emb_large_vocab` | (32768,64) | (8,) | LLM-scale vocab (32768 tokens) | allclose vs CPU; no address overflow | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| EM-FC-05 | `test_emb_large_embed_dim` | (64,256) | (8,) | Wide embedding dimension (256); multi-stick row | allclose | 1e-2/1e-2 | ENV_BASE | P2 | READY |
| EM-FC-06 | `test_emb_small_vocab` | (4,64) | (4,) | Minimal vocab | allclose | 1e-2/1e-2 | ENV_BASE | P2 | READY |
| EM-FC-07 | `test_emb_single_token` | (64,64) | (1,) | Single token lookup | result.shape==(1,64); allclose | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| EM-FC-08 | `test_emb_seq_batch` | (64,64) | (2,16) | Batch=2, seq_len=16; output (2,16,64) | allclose vs CPU | 1e-2/1e-2 | ENV_BASE | P1 | BLOCKED:HW |

---

### 6.2 Embedding Module API Variants

| ID | Test Name | Call Form | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|---|
| EM-AV-01 | `test_emb_nn_embedding_module` | `nn.Embedding(N,D)(idx)` | Module forward → aten.embedding.default dispatch | allclose vs F.embedding | ENV_BASE | P1 | READY |
| EM-AV-02 | `test_emb_functional_form` | `F.embedding(idx, weight)` | Functional form identical to module form | allclose vs EM-AV-01 | ENV_BASE | P1 | READY |
| EM-AV-03 | `test_emb_padding_idx` | `nn.Embedding(N, D, padding_idx=0)` | padding_idx=0: position 0 always zero in output | result[result_idx_is_0] == zeros | ENV_BASE | P2 | READY |
| EM-AV-04 | `test_emb_max_norm` | `nn.Embedding(N, D, max_norm=1.0)` | max_norm renormalizes weight rows before lookup | allclose; all output rows have L2 norm ≤ 1.0 | ENV_BASE | P2 | READY |
| EM-AV-05 | `test_emb_from_pretrained` | `nn.Embedding.from_pretrained(weight)` | from_pretrained loads weight; embedding lookup correct | allclose vs weight[idx] | ENV_BASE | P2 | READY |
| EM-AV-06 | `test_emb_sparse_param` | `nn.Embedding(N, D, sparse=True)` | sparse=True flag set; embedding op still runs | allclose; no crash | ENV_BASE | P2 | READY |

---

### 6.3 Dtype Coverage

| ID | Test Name | dtype | element_size | What It Validates | Pass Criteria | Tol | Env | Priority | Status |
|---|---|---|---|---|---|---|---|---|---|
| EM-DT-01 | `test_emb_float16` | float16 | 2 | Primary dtype; 64 elements/stick | allclose vs CPU | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| EM-DT-02 | `test_emb_bfloat16` | bfloat16 | 2 | bf16 stick-expression fix for embedding | allclose vs CPU | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| EM-DT-03 | `test_emb_float32` | float32 | 4 | 32 elements/stick; weight dtype float32 | allclose vs CPU | 1e-5/1e-5 | ENV_BASE | P1 | READY |
| EM-DT-04 | `test_emb_int64_weight` | int64 | 8 | 16 elements/stick; integer embedding table | exact equality | exact | ENV_BASE | P2 | READY |
| EM-DT-05 | `test_emb_unsupported_dtype` | float64 | 8 | Unsupported dtype must NOT silently produce wrong results | RuntimeError OR CPU fallback; no silent wrong | — | ENV_BASE | P1 | READY |

---

### 6.4 Execution Mode

| ID | Test Name | Mode | What It Validates | Pass Criteria | Tol | Env | Priority | Status |
|---|---|---|---|---|---|---|---|---|
| EM-EM-01 | `test_emb_eager_1d` | Eager (no compile) | Eager path | Correct result OR RuntimeError | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| EM-EM-02 | `test_emb_fullgraph` | `torch.compile(fullgraph=True)` | No graph break in embedding lowering | No Unsupported exception; allclose | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| EM-EM-03 | `test_emb_graph_linear` | Compiled | embedding → linear projection (standard LLM front-end) | allclose vs CPU | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| EM-EM-04 | `test_emb_graph_layernorm` | Compiled | embedding → layernorm | allclose vs CPU | 1e-2/1e-2 | ENV_BASE | P2 | READY |
| EM-EM-05 | `test_emb_graph_add_pos_emb` | Compiled | embedding + positional embedding | allclose vs CPU | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| EM-EM-06 | `test_emb_repeated_calls` | Compiled × 3 calls | No state leakage across successive calls | All 3 calls correct | 1e-2/1e-2 | ENV_BASE | P1 | READY |

---

### 6.5 Dynamic / Symbolic Shapes

| ID | Test Name | Dynamic Dims | What It Validates | Pass Criteria | Tol | Env | Priority | Status |
|---|---|---|---|---|---|---|---|---|
| EM-DS-01 | `test_emb_dynamic_seq_len` | seq_len dynamic | Different seq lengths produce correct outputs | Both calls correct; ≤1 recompile | 1e-2/1e-2 | ENV_DYN | P1 | READY |
| EM-DS-02 | `test_emb_dynamic_batch` | Batch dim B | Batch varies per request | Correct for B=1,2,4 | 1e-2/1e-2 | ENV_DYN | P1 | BLOCKED:HW |
| EM-DS-03 | `test_emb_dynamic_vocab_size` | num_embeddings | Different vocab sizes | Correct for vocab=64,128,256 | 1e-2/1e-2 | ENV_DYN | P2 | READY |
| EM-DS-04 | `test_emb_export_dynamic` | N via `torch.export` | Exported program handles any seq_len | Correct for N=4,8,16 | 1e-2/1e-2 | ENV_BASE | P2 | READY |

---

### 6.6 Memory Layout

| ID | Test Name | Weight Shape | Layout / Non-Contig Type | What It Validates | Pass Criteria | Tol | Env | Priority | Status |
|---|---|---|---|---|---|---|---|---|---|
| EM-ML-01 | `test_emb_stick_aligned_64` | (32,64) | 1 stick/row | Clean stick boundary | allclose; shape (8,64) | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| EM-ML-02 | `test_emb_non_aligned_50` | (32,50) padded to 64 | 50-col embedding dim | Output shape (8,50); padding does not leak | Shape (8,50); values correct | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| EM-ML-03 | `test_emb_transposed_weight` | `weight.t()` (64,32) transposed | Transposed weight | Restickify before embedding | allclose | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| EM-ML-04 | `test_emb_large_baseline` | (512,64) | Large contiguous weight table | No OOB in address computation | allclose vs CPU | 1e-2/1e-2 | ENV_BASE | P1 | READY |

---

### 6.7 Edge Cases and Error Handling

| ID | Test Name | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|
| EM-EC-01 | `test_emb_index_out_of_bounds` | Index value >= num_embeddings → RuntimeError | RuntimeError raised | ENV_BASE | P0 | READY |
| EM-EC-02 | `test_emb_negative_index` | Negative index; nn.Embedding does NOT support wraparound (unlike Python lists) | RuntimeError raised | ENV_BASE | P0 | READY |
| EM-EC-03 | `test_emb_empty_index` | Zero-length index | Output shape (0, embed_dim); no error | ENV_BASE | P1 | READY |
| EM-EC-04 | `test_emb_index_device_mismatch` | index on CPU, weight on Spyre | RuntimeError with clear message | ENV_BASE | P1 | READY |
| EM-EC-05 | `test_emb_gate_off` | Gate=0, embedding | Fallback+warning OR RuntimeError; no wrong result | ENV_GATE_OFF | P0 | READY |
| EM-EC-06 | `test_emb_scalar_weight_error` | 1D weight tensor (not 2D) | RuntimeError raised at ATen layer | ENV_BASE | P1 | READY |
| EM-EC-07 | `test_emb_weight_non_contiguous` | `weight[::2,:]` non-contiguous | Restickify fires; correct lookup | allclose | ENV_BASE | P2 | READY |

---

### 6.8 Model Scenarios — Language Model Pipeline

| ID | Test Name | What It Models | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|
| EM-LM-01 | `test_emb_lm_token_lookup` | Standard LM token embedding: (vocab, embed) weight, (batch, seq_len) index | allclose vs CPU | ENV_BASE | P1 | READY |
| EM-LM-02 | `test_emb_lm_pos_emb_add` | Token embed + positional embed → first transformer input | allclose vs CPU | ENV_BASE | P1 | READY |
| EM-LM-03 | `test_emb_lm_type_embed` | Token type embedding (BERT-style): separate embedding table for token types | allclose vs CPU | ENV_BASE | P2 | READY |
| EM-LM-04 | `test_emb_lm_full_front_end` | Token embed + type embed + pos embed all added → LayerNorm | allclose vs CPU | ENV_BASE | P1 | BLOCKED:HW |

---

### 6.9 Exact Value Roundtrip and Pass Validation

| ID | Test Name | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|
| EM-RT-01 | `test_emb_exact_known_value` | Known sentinel values in weight; verify embedding returns exact values | Element-wise exact match | ENV_BASE | P0 | READY |
| EM-RT-02 | `test_emb_identity_index` | Identity index (idx[i]=i for all i); output rows == weight rows | allclose(result, weight_cpu) | ENV_BASE | P1 | READY |
| EM-RT-03 | `test_emb_pass_fires` | Compile embedding; inspect FX graph | `spyre.indices_to_address` node present for embedding | Node present | ENV_BASE | P1 | READY |
| EM-RT-04 | `test_emb_pass_gated_off` | Gate=0; inspect FX graph | No `spyre.indices_to_address` node for embedding | Node absent | ENV_GATE_OFF | P1 | READY |
| EM-RT-05 | `test_emb_lx_planning` | LX scratchpad + embedding | Compilation succeeds; allclose | ENV_LX | P2 | READY |
| EM-RT-06 | `test_emb_mc4` | SENCORES=4 embedding | allclose(mc4, mc1) | ENV_MC4 | P2 | BLOCKED:MC |
---

### 6.10 Key Downstream Chain Patterns — torch.nn.Embedding

> Validates that the embedding output is correctly consumed by the downstream ops that always follow it in transformer graphs.

| ID | Test Name | Graph Pattern | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|---|
| EM-OC-01 | `test_emb_then_layernorm` | nn.Embedding(vocab, dim)(idx) → LayerNorm(dim) | Token embedding → layer normalization (standard transformer input block); validates embedding output layout compatible with LayerNorm | allclose vs CPU | ENV_BASE | P1 | READY |
| EM-OC-02 | `test_emb_then_add` | token_emb = nn.Embedding(…)(idx); out = token_emb + pos_emb | Token embedding + positional embedding; two different embedding lookups then add; validates two independent address computations in one graph | allclose vs CPU | ENV_BASE | P1 | READY |


---

## Section 7 — Cross-Operation Tests

> These tests exercise scenarios where torch.gather, torch.index_select, and torch.nn.Embedding are used together in a single compiled graph or contrasted against each other. An **Op** column identifies which indirect access operation(s) each test exercises.
>
> **Count correction:** this section was previously stated as 22 net-new tests; a row-by-row count of the tables below (the same check applied to Sections 5 and 6) gives **14**. All summary tables below reflect 14.

---

### 7.1 Combined-Category Cross-Op Tests

| ID | Test Name | Op(s) | What It Validates | Pass Criteria | Tol | Env | Priority | Status |
|---|---|---|---|---|---|---|---|---|
| CO-GIS-01 | `test_gather_then_index_select` | gather + index_select | Two indirect ops chained: gather output fed into index_select | allclose vs CPU two-step reference | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| CO-GIS-02 | `test_index_select_then_gather` | index_select + gather | Reversed chaining: index_select selects rows, gather selects columns | allclose vs CPU reference | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| CO-GIS-03 | `test_gather_vs_index_select_equivalence` | gather / index_select | For dim=0, 1D index: `torch.index_select(src, 0, idx)` == `torch.gather(src, 0, idx.unsqueeze(1).expand(...))` | allclose between gather and index_select results | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| CO-EMB-01 | `test_emb_then_gather` | Embedding + gather | Token embed → gather selects subset of token positions | allclose vs CPU | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| CO-EMB-02 | `test_emb_then_index_select` | Embedding + index_select | Token embed → index_select picks specific tokens | allclose vs CPU | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| CO-EMB-03 | `test_gather_vs_emb_equivalence` | gather / Embedding | gather(weight, 0, idx.unsqueeze(1).expand(...)) should match F.embedding(idx, weight) for simple case | allclose between gather and embedding results | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| CO-ALL-01 | `test_all_three_ops_in_graph` | gather + index_select + Embedding | All 3 indirect ops in single compiled graph | All 3 outputs allclose vs CPU | 1e-2/1e-2 | ENV_BASE | P2 | READY |

---

### 7.2 Per-Op Category Tests

#### 7.2a — Address Node Independence (One Table per Op)

**torch.gather address node independence:**

| ID | Test Name | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|
| CO-ADDR-G-01 | `test_gather_address_node_separate_from_is` | In a graph with both gather and index_select, gather's `indices_to_address` node is a separate node from index_select's | Two distinct `indices_to_address` nodes in FX graph | ENV_BASE | P1 | READY |

**torch.index_select address node independence:**

| ID | Test Name | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|
| CO-ADDR-IS-01 | `test_is_address_node_separate_from_emb` | index_select's address node separate from embedding's | Two distinct `indices_to_address` nodes | ENV_BASE | P1 | READY |

**torch.nn.Embedding address node independence:**

| ID | Test Name | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|
| CO-ADDR-E-01 | `test_emb_address_node_separate_from_gather` | Embedding's address node separate from gather's | Two distinct `indices_to_address` nodes | ENV_BASE | P1 | READY |

---

### 7.3 Shared Source Tensor Cross-Op Tests

| ID | Test Name | Op(s) | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|---|
| CO-SRC-01 | `test_gather_and_is_same_source` | gather + index_select | Both ops reading from same source tensor | Both results allclose vs independent CPU references | 1e-2/1e-2 | ENV_BASE | P1 | READY |
| CO-SRC-02 | `test_emb_and_gather_same_weight` | Embedding + gather | Embedding and gather from same weight tensor | Both allclose; weight not corrupted | 1e-2/1e-2 | ENV_BASE | P2 | READY |

---

### 7.4 Graph Fusion Interaction

| ID | Test Name | Op(s) | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|---|
| CO-FI-01 | `test_no_cross_op_fusion` | gather + index_select | Gather and index_select kernels must not be fused into one kernel; each has its own address node | Two separate kernel dispatch events | ENV_BASE | P2 | READY |
| CO-FI-02 | `test_gather_emb_interleaved_schedule` | gather + Embedding | Compile gather and embedding with interleaved operations; validate schedule produces correct results | allclose for both outputs | 1e-2/1e-2 | ENV_BASE | P2 | READY |

---

## Section 8 — Paged Attention Comprehensive Coverage

> These tests focus on paged attention as a cross-operation use case, encompassing gather, index_select, and embedding patterns. An **Op** column identifies which indirect access op is used in each scenario.
>
> **Why dedicated section:** Paged attention is the primary production use case for indirect access on Spyre. It exercises gather/index_select with 3D/4D KV pools, dynamic block tables, multi-core distribution, and online softmax — a superset of the individual op scenarios.

---

### 8.1 Paged Attention Core Patterns (Op-Specific Tables)

#### 8.1a — torch.gather Paged Attention Patterns

| ID | Test Name | Op | KV Pool Shape | Index Shape | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|---|---|---|
| PA-G-01 | `test_paged_gather_2d_flat` | gather | (N_pages×page_size, head_dim) | (seq_len,) | 2D flat KV gather — hardware-ready baseline | allclose vs CPU | ENV_BASE | P0 | READY |
| PA-G-02 | `test_paged_gather_4d` | gather | (N_pages, heads, page_size, head_dim) | block_table | Full 4D gather with block_table | allclose vs CPU | ENV_BASE | P1 | BLOCKED:HW |
| PA-G-03 | `test_paged_gather_gqa` | gather | (N_pages, kv_heads, page_size, head_dim) | kv block_table | GQA: fewer KV heads; gather then expand | allclose vs reference | ENV_BASE | P1 | BLOCKED:HW |
| PA-G-04 | `test_paged_gather_decode` | gather | (pool_size, head_dim) | slot_idxs | Single decode step: token selects KV via gather | allclose vs reference | ENV_BASE | P1 | BLOCKED:HW |
| PA-G-05 | `test_paged_gather_multi_step` | gather | (pool_size, head_dim) | slot_idxs per step | Multi-step decode: KV grows per step | Each step correct; earlier not corrupted | ENV_BASE | P2 | BLOCKED:HW |

#### 8.1b — torch.index_select Paged Attention Patterns

| ID | Test Name | Op | KV Pool Shape | Index Shape | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|---|---|---|
| PA-IS-01 | `test_paged_is_2d_flat` | index_select | (N_pages×page_size, head_dim) | 1D block_table | 2D flat KV index_select — hardware-ready | allclose vs CPU | ENV_BASE | P0 | READY |
| PA-IS-02 | `test_paged_is_3d` | index_select | (N_pages, page_size, head_dim) | 1D block_table | 3D KV with page selection via index_select dim=0 | allclose vs CPU | ENV_BASE | P1 | BLOCKED:HW |
| PA-IS-03 | `test_paged_is_decode` | index_select | (pool_size, head_dim) | slot_idx 1D | Single decode via index_select | allclose vs reference | ENV_BASE | P1 | BLOCKED:HW |

#### 8.1c — Comparison: gather vs index_select for Paged Attention

| ID | Test Name | Op(s) | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|---|
| PA-CMP-01 | `test_paged_gather_vs_is_2d` | gather / index_select | For 2D flat KV pool: gather and index_select produce equivalent results | allclose(gather_result, is_result) | ENV_BASE | P1 | READY |
| PA-CMP-02 | `test_paged_gather_vs_is_3d` | gather / index_select | For 3D KV pool: gather dim=0 vs index_select dim=0 equivalence | allclose | ENV_BASE | P1 | BLOCKED:HW |

---

### 8.2 Paged Attention Advanced Scenarios

| ID | Test Name | Op | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|---|
| PA-ADV-01 | `test_paged_online_softmax_2d` | gather | Online softmax + paged KV gather; 2D baseline | Max diff < 1e-2 vs offline softmax | ENV_BASE | P0 | READY |
| PA-ADV-02 | `test_paged_online_softmax_4d` | gather | Full 4D online softmax paged attention | allclose vs CPU reference | ENV_BASE | P1 | BLOCKED:HW |
| PA-ADV-03 | `test_paged_vllm_e2e` | gather + index_select | vLLM-style e2e: slot_mapping → KV pool → attention → output | allclose vs reference implementation | ENV_BASE | P1 | BLOCKED:HW |
| PA-ADV-04 | `test_paged_continuous_batch` | gather | Continuous batching: variable context lengths per sequence | All sequences correct; no cross-sequence contamination | ENV_BASE | P1 | BLOCKED:HW |
| PA-ADV-05 | `test_paged_dynamic_block_table` | gather | Block table grows dynamically; validate after each growth step | Correct for each step | ENV_DYN | P2 | BLOCKED:HW |
| PA-ADV-06 | `test_paged_sparse_block_table` | index_select | Non-sequential physical pages; sparse block_table | All fetched pages correct | ENV_BASE | P2 | BLOCKED:HW |
| PA-ADV-07 | `test_paged_large_pool` | gather | Large KV pool (256 pages×32 tokens×8 heads×64 head_dim) | No OOB; correct subset fetched | ENV_BASE | P2 | BLOCKED:HW |
| PA-ADV-08 | `test_paged_attn_lx_sencores4` | gather | LX_PLANNING=1 + SENCORES=4 + 4D KV | allclose(mc4, mc1) | ENV_LX + ENV_MC4 | P2 | BLOCKED:MC |

---

### 8.3 Additional Paged Attention Scenarios (Consolidated from Former Section 4.20/4.21)

> Nine scenarios from the former Section 4 model-scenario subsections that had no equivalent in Sections 8.1–8.2. Consolidated here to keep all paged-attention coverage in one section.

| ID | Test Name | Op | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|---|
| PA-EXT-01 | `test_gather_kv_cache_3d` | gather | 3D KV cache `(pages, page_size, head_dim)`; gather selects pages | allclose vs CPU | ENV_BASE | P1 | BLOCKED:HW |
| PA-EXT-02 | `test_fms_kv_cache_int32` | gather | FMS-style int32 block_table KV cache lookup | allclose vs CPU | ENV_BASE | P1 | BLOCKED:HW |
| PA-EXT-03 | `test_fms_5d_kv` | gather | FMS 5D KV cache `[pages, 2, heads, page_size, head_dim]` | allclose vs CPU | ENV_BASE | P2 | BLOCKED:HW |
| PA-EXT-04 | `test_paged_attn_int32_vs_int64_index` | gather | int32 vs int64 block_table produce identical results | allclose(int32_result, int64_result) | ENV_BASE | P2 | BLOCKED:HW |
| PA-EXT-05 | `test_paged_attn_mqa_shared_kv` | gather | MQA: 1 KV head shared across 8 Q heads via gather | All 8 Q heads get correct KV | ENV_BASE | P2 | BLOCKED:HW |
| PA-EXT-06 | `test_paged_attn_reuse_page` | gather | Same physical page referenced in multiple sequences' block_tables | Each sequence gets shared correct values | ENV_BASE | P1 | BLOCKED:HW |
| PA-EXT-07 | `test_paged_attn_page_reorder` | gather | block_table permutation changes physical access order | Correct logical ordering after reorder | ENV_BASE | P2 | BLOCKED:HW |
| PA-EXT-08 | `test_paged_attn_lx_planning` | gather | LX_PLANNING=1 + paged attention KV fetch; no scratchpad vs pool aliasing | No crash; correct output | ENV_LX | P2 | BLOCKED:HW |
| PA-EXT-09 | `test_paged_attn_bundle_symbolic` | gather | BUNDLE_SYMBOLIC_ARGS=1 + paged attention; symbolic block_table shape | Compiles and runs correctly | ENV_BASE | P2 | BLOCKED:HW |

---

### 8.4 Production Example Coverage — gaps identified from `examples/test_paged.py`

> `test_paged.py` is the canonical production paged attention example in the torch-spyre repo. It uses `name_tensor_dims` on all KV and index tensors, and wraps the indirect access inside nested `spyre_hint(tiles={...})`. Cross-checking that file against Sections 8.1–8.3 identified two patterns with no existing test coverage. Both are testable at 2D scale (READY — no 3D HW fix required).

#### 8.4a — Named Tensor Dims + Indirect Access

> **Why this matters:** `test_paged.py` calls `name_tensor_dims(keys, ["cache","H","D"])` and `name_tensor_dims(slot_idxs, ["B","Lk"])` before `torch.compile`. Named-dim labels propagate through the FX graph into the SDSC JSON (`indirectAccessIndexLabeledDs`). If the indirect access FX pass does not correctly read the named-dim stride for the index tensor, the address formula uses the wrong stride and produces silently wrong results.

| ID | Test Name | Op | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|---|
| PA-ND-01 | `test_paged_named_dims_kv_2d` | index_select / index.Tensor | 2D KV cache (cache_size, head_dim); `name_tensor_dims(kv, ["cache","D"])` and `name_tensor_dims(slot_idxs, ["B","Lk"])`; compile and run indirect access | allclose vs CPU reference; named dims do not corrupt address computation | ENV_BASE | P0 | READY |
| PA-ND-02 | `test_paged_named_dims_keys_values` | index.Tensor | 2D K and V tensors both carry named dims `["cache","D"]` (corrected from an earlier draft's `["cache","H","D"]`, which would have required the 3D shape this section's intro says is out of scope); two independent indirect accesses in one compiled graph with named dims | allclose(K_out, K_ref) and allclose(V_out, V_ref); no cross-tensor stride aliasing from named dims | ENV_BASE | P1 | READY |

#### 8.4b — `spyre_hint` Tiles + Indirect Access

> **Why this matters:** `test_paged.py` wraps indirect access inside nested `spyre_hint(tiles={"B": B//2})`, `spyre_hint(tiles={"Lq": ...})`, `spyre_hint(tiles={"H": ...})`, `spyre_hint(tiles={"Lk": ...})`. These hints control how the compiler tiles the loop nest. When the Lk (sequence) dimension is tiled, each tile sees a different slice of `slot_idxs` — the base pointer of the address computation shifts per tile. If the indirect access pass does not account for the tile offset in the address base, each tile after the first fetches from the wrong KV page.

| ID | Test Name | Op | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|---|
| PA-SH-01 | `test_paged_spyre_hint_batch_tile` | index.Tensor | 2D KV; `spyre_hint(tiles={"B": 2})`; batch dimension tiled; each batch tile uses its own slice of slot_idxs | allclose vs CPU; no batch-tile cross-contamination in address computation | ENV_BASE | P0 | READY |
| PA-SH-02 | `test_paged_spyre_hint_seq_tile` | index.Tensor | 2D KV; `spyre_hint(tiles={"Lk": 4})`; sequence dimension tiled; each tile covers a different block of slot_idxs | allclose vs CPU; tile-offset shift in slot_idxs correctly reflected in addresses per tile | ENV_BASE | P0 | READY |
| PA-SH-03 | `test_paged_spyre_hint_nested` | index.Tensor | Nested hints matching `test_paged.py` pattern: `spyre_hint(tiles={"B": 2})` → `spyre_hint(tiles={"Lk": 4})`; 2D KV; both K and V indirect | allclose vs `paged_cpu` reference; validates the exact tiling structure from `examples/test_paged.py` at small scale | ENV_BASE | P0 | READY |

---

## Section 9 — Unit Tests

> These are CPU-only or mock unit tests for the individual compiler pass and address computation components. They do NOT require Spyre hardware. Op column indicates which indirect access operation the unit test covers.
>
> **FX pass architecture:** `detect_indirect_access` → `_add_indices_to_address_gather/index/embedding` — three separate pass handlers sharing the same `spyre::indices_to_address` kernel insertion pattern.
>

### 9.1 Address Computation Unit Tests

| ID | Test Name | Op | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|---|
| GA-AC-IS-01 | `test_address_formula_gather_2d` | any | Formula: `addr = v_offset + coord[0]×stride[0]×elem_sz + coord[1]×stride[1]×elem_sz` for 2D | Exact address match for known shape | CPU_ONLY | P0 | READY |
| GA-AC-IS-02 | `test_address_formula_gather_3d` | any | Same formula extended to 3D | Exact address match | CPU_ONLY | P1 | READY |
| GA-AC-IS-03 | `test_address_stride_from_tensor_layout` | any | `device_stride` extracted from tensor's stride() attribute and scaled by element_size | stride values match tensor.stride() × elem_sz | CPU_ONLY | P1 | READY |
| EM-AC-02 | `test_address_embedding_large_vocab` | any | Large vocab (32768): no integer overflow in address | Address fits in int64 | CPU_ONLY | P1 | READY |

---

### 9.2 FX Pass — detect_indirect_access Unit Tests

| ID | Test Name | Op(s) | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|---|
| GA-DT-06 | `test_detect_all_three_in_one_graph` | gather + index_select + Embedding | All three marked; no duplicates; no false positives | Exactly 3 nodes marked with `op_info`; matmul unmarked | CPU_ONLY | P1 | READY |

---

### 9.3 FX Pass — add_index_to_address Unit Tests

| ID | Test Name | Op | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|---|
| GA-FX-02 | `test_gather_handler_node_args_order` | gather | Inserted call's args end with the hardcoded virtual_offset (0) | Trailing arg == "0" in generated code | CPU_ONLY | P1 | READY |
| GA-FX-03 | `test_gather_handler_dim_arg` | gather | Handler passes `dim` correctly for dim=0 and dim=1 | dim arg == 0 or 1 in node args | CPU_ONLY | P1 | READY |
| IS-FX-02 | `test_is_handler_node_args_order` | index_select | Args order correct (same virtual_offset=0 check) | Node args match expected order | CPU_ONLY | P1 | READY |
| EM-FX-02 | `test_emb_handler_node_args_order` | Embedding | Args order correct (same virtual_offset=0 check) | Node args match | CPU_ONLY | P1 | READY |

---

### 9.4 Stick Formula Unit Tests

| ID | Test Name | Op | dtype | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|---|---|
| GA-SF-01 | `test_stick_expr_fp16_gather` | any | float16 | Stick formula `2*(Mod(d1, 64))` accepted and generated | Expression accepted; no rejection | CPU_ONLY | P0 | READY |
| GA-SF-02 | `test_stick_expr_fp32_gather` | any | float32 | Stick formula `4*(Mod(d1, 32))` accepted | Expression accepted | CPU_ONLY | P0 | READY |
| GA-SF-03 | `test_stick_expr_bf16_gather` | any | bfloat16 | `2*(Mod(d1, 32))` accepted (fix for UPT-S02) | Expression accepted | CPU_ONLY | P0 | READY |


---

### 9.6 Flow Logic Unit Tests

| ID | Test Name | Op(s) | What It Validates | Pass Criteria | Env | Priority | Status |
|---|---|---|---|---|---|---|---|
| GA-FL-01 | `test_is_indirect_access_function_true_case` | any | `is_indirect_access_operation(node)` returns True once `op_info` carries index metadata | True | CPU_ONLY | P0 | READY |
| GA-FL-04 | `test_is_indirect_access_function_matmul` | N/A | Same function returns False for an operation with no `op_info` (e.g. matmul) | False | CPU_ONLY | P0 | READY |
| GA-FL-EM-01 | `test_all_three_handlers_called` | gather + index_select + Embedding | All 3 handler functions fire in a real compiled graph with all 3 ops present | >= 3 `indices_to_address` call sites in generated code | CPU_ONLY | P1 | READY |

---

## Conclusion

### Summary Table — Test Count by Section

| Section | Scope | Op(s) | Test Count | READY | BLOCKED:HW | BLOCKED:MC | BLOCKED:CHUNK | NEEDS:INFRA | BLOCKED:FX |
|---|---|---|---|---|---|---|---|---|---|
| **1** | Upstream PyTorch Tests (4 YAML files) | gather / index_select / embedding | ~385 | — | — | — | — | — | — |
| **2** | torch-spyre Existing Tests | gather / embedding | 1 | — | — | — | — | — | — |
| **3** | Developer Unit Tests (`tests/indirect_access/`) | gather / index_select / embedding | 63 | — | — | — | — | — | — |
| **4** | torch.gather FVT Test Cases | gather | 146 | 121 | 17 | 6 | 2 | 0 | 0 |
| **5** | torch.index_select FVT Test Cases | index_select / index.Tensor | 86 | 72 | 11 | 1 | 0 | 0 | 2 |
| **6** | torch.nn.Embedding FVT Test Cases | embedding | 52 | 46 | 5 | 1 | 0 | 0 | 0 |
| **7** | Cross-Operation Tests | gather + index_select + embedding | 14 | 14 | 0 | 0 | 0 | 0 | 0 |
| **8** | Paged Attention Comprehensive Coverage | gather + index_select | 32 | 9 | 22 | 1 | 0 | 0 | 0 |
| **9** | Unit Tests | gather + index_select + embedding | 15 | 15 | 0 | 0 | 0 | 0 | 0 |
| **Total** | | | **345** | **277** | **55** | **9** | **2** | **0** | **2** |

> Net-new FVT test cases (Sections 4–9): **532**. Tests tracked from prior baselines (Sections 1–3): **449** (~385 upstream + 1 existing + 63 developer).
> Sections 1–3 READY/BLOCKED/NEEDS columns are blank — those tests are tracked by upstream CI run results after code merge, not pre-classified here.
> `BLOCKED:FX` (Section 5.13 only): PyTorch-valid `aten.index.Tensor` pattern — multiple simultaneous tensor indices (IX-MULTI-01/02) — confirmed by direct source review to be unimplemented in `_add_indices_to_address_index` (`indices_to_address_pass.py:188-198` breaks on the first non-None index in the list). Negative-index tests (IX-NEG-01/02) were re-verified directly against `spyre_address_computation.cpp:128` and moved to `READY`: the kernel already has a `TORCH_CHECK` bounds guard (confirmed by the existing passing unit test `test_negative_index_validation`), so there is no safety gap — only a documented semantic divergence from PyTorch's wraparound behavior.

---

### Coverage Matrix

| Coverage Axis | torch.gather | torch.index_select | torch.nn.Embedding | Coverage Notes |
|---|---|---|---|---|
| 2D — dim=0 (row selection) | Sec 4.1 FC-01, FC-09, FC-12 | Sec 5.1 IS-FC-01 | Sec 6.1 EM-FC-01..07 | Full 2D READY coverage for all three ops |
| 2D — dim=1 (column selection) | Sec 4.1 FC-09, 4.2 IP-10 | Sec 5.1 IS-FC-02, IS-FC-09 | N/A (embedding always dim=0) | READY coverage |
| 3D (any dim) | Sec 4.1 FC-02..04, FC-10 | Sec 5.1 IS-FC-03..05, IS-FC-10 | Sec 6.1 EM-FC-02, EM-FC-03 | BLOCKED:HW; tests written; executes after HW fix |
| 4D / 5D | Sec 4.1 FC-05..FC-08, FC-11 | Sec 5.1 IS-FC-06..08 | Sec 6.1 EM-FC-02, EM-FC-08 | BLOCKED:HW; design validated at 2D |
| dtype: float16 | Sec 4.4 DT-07 | Sec 5.4 IS-DT-07 | Sec 6.3 EM-DT-01 | READY |
| dtype: bfloat16 | Sec 4.4 DT-01 | Sec 5.4 IS-DT-01 | Sec 6.3 EM-DT-02 | READY (bf16 stick fix validated) |
| dtype: float32 | Sec 4.4 DT-02 | Sec 5.4 IS-DT-02 | Sec 6.3 EM-DT-03 | READY |
| dtype: int64/int32/bool | Sec 4.4 DT-03..05 | Sec 5.4 IS-DT-03..05 | Sec 6.3 EM-DT-04 | READY |
| dtype: float64 (unsupported) | Sec 4.4 DT-06 | Sec 5.4 IS-DT-06 | Sec 6.3 EM-DT-05 | READY — validates no silent wrong result |
| Eager mode | Sec 4.5 EM-01..02 | Sec 5.5 IS-EM-01 | Sec 6.4 EM-EM-01 | READY |
| Compiled (torch.compile) | Sec 4.5 EM-03..08 | Sec 5.5 IS-EM-02..08 | Sec 6.4 EM-EM-02..06 | READY |
| Dynamic shapes | Sec 4.6 DS-01..09 | Sec 5.6 IS-DS-01..07 | Sec 6.5 EM-DS-01..04 | READY (2D); BLOCKED:HW (3D/4D) |
| torch.export | Sec 4.6 DS-07 | Sec 5.6 IS-DS-06 | Sec 6.5 EM-DS-04 | READY |
| Memory layout (aligned) | Sec 4.7 ML-01..02 | Sec 5.7 IS-ML-01 | Sec 6.6 EM-ML-01 | READY |
| Memory layout (non-aligned padding) | Sec 4.7 ML-03..04 | Sec 5.7 IS-ML-02 | Sec 6.6 EM-ML-02 | READY |
| Non-contiguous source (restickify) | Sec 4.8 NCS-01..06 | Sec 5.7 IS-ML-03..05 | Sec 6.6 EM-ML-03 | READY (2D); BLOCKED:HW (3D) |
| Non-contiguous index | Sec 4.9 NCI-01..04 | N/A (index_select index always 1D) | N/A | READY for gather |
| Restickify pass validation | Sec 4.10 RS-01..07 | N/A | N/A | READY (gather-specific) |
| Compiler pass pipeline | Sec 4.11 CP-01..06 | Sec 5.10 IS-RT-03..04 | Sec 6.9 EM-RT-03..04 | READY |
| SDSC / kernel spec JSON | Sec 4.12 SD-01..SD-07 (READY — direct compile_op_spec() call, no compile/device needed) | Sec 9.5 IS-KS-01 | Sec 9.5 EM-KS-01 | READY (Sec 4.12); NEEDS:INFRA (Sec 9.5) |
| LX planning / scratchpad | Sec 4.15 LX-01..06, 4.34 LX-07..08 | Sec 5.10 IS-RT-06 | Sec 6.9 EM-RT-05 | READY |
| Multi-core (SENCORES > 1) | Sec 4.14 MC-01..06 | Sec 5.10 IS-RT-05 | Sec 6.9 EM-RT-06 | BLOCKED:MC |
| Chunk large tensors | Sec 4.16 CL-01..03 | N/A | N/A | BLOCKED:CHUNK |
| Error handling (OOB, device mismatch) | Sec 4.18 EC-01..13 | Sec 5.8 IS-EC-01..10 | Sec 6.7 EM-EC-01..07 | READY |
| Gate env var safety | Sec 4.18 EC-09..EC-09g | Sec 5.8 IS-EC-06 | Sec 6.7 EM-EC-05 | READY |
| KV cache / paged attention | Sec 8 (PA-G, PA-IS, PA-CMP, PA-ADV, PA-EXT) | Sec 5.9 IS-KV-01..08 | N/A | READY (2D); BLOCKED:HW (3D/4D) |
| Model: embedding table | Sec 4.19 EMB-01..04 | Sec 5.9 IS-KV-06 | Sec 6.8 EM-LM-01..04 | READY |
| Model: RoPE / positional emb | Sec 4.22b PE-01 (gather from sin/cos table) | N/A | Sec 6.8 EM-LM-02 | READY |
| Model: MoE expert routing | Sec 4.22d MOE-01 | Sec 5.11 IS-MP-02 | N/A | READY |
| Model: GQA / MQA | Sec 8 PA-G-03, PA-EXT-05 | N/A | N/A | BLOCKED:HW |
| Cross-op tests (multi-op in graph) | Sec 7 CO-GIS-01..CO-FI-02 | — | — | READY (2D) |
| Unit: address formula | Sec 9.1 GA-AC-IS-01..07 | Sec 9.1 IS-AC-01..03 | Sec 9.1 EM-AC-01..02 | READY |
| Unit: FX pass detect | Sec 9.2 GA-DT-01..06 | — | — | READY |
| Unit: FX pass handler | Sec 9.3 GA-FX-01..03 | Sec 9.3 IS-FX-01..02 | Sec 9.3 EM-FX-01..02 | READY |
| Unit: stick formula | Sec 9.4 GA-SF-01..03 | Sec 9.4 IS-SF-01 | Sec 9.4 EM-SF-01 | READY |

---

### Risk Register

| Risk ID | Description | Affects | Severity | Likelihood | Mitigation | FRS Ref |
|---|---|---|---|---|---|---|
| RISK-01 | 3D gather produces wrong values on hardware (DPR-66 comment: "Values wrongly fetched") | Sec 4.1 FC-02..04, 4.32, 4.45; Sec 8 | HIGH | Confirmed (FAIL) | ~35 tests marked BLOCKED:HW in Section 4; 22 in Section 8; 2D baseline (FC-01) passes; unblock after HW fix | FRS-GG FR-5 |
| RISK-02 | 4D gather produces runtime error on hardware (DPR-67 comment) | Sec 4.1 FC-05..08; Sec 8 paged attention | HIGH | Confirmed (FAIL) | Tests marked BLOCKED:HW; 2D fallback PA-G-01, PA-ADV-01 in Section 8 cover assertion-enabled path | FRS-GG FR-5 |
| RISK-03 | bfloat16 stick expression `2*(Mod(d1,32))` rejected by `_check_stick_expr_supported` | Sec 4.4 DT-01, upstream test_gather_large | MEDIUM | Confirmed (xfail) | UPT-S02 tracked; fix = add bf16 pattern to supported set; DT-01 validates fix | FRS-AC 5.1 |
| RISK-04 | virtual_offset hardcoded to 0 in FX pass; incorrect for LX scratchpad sources at non-zero offset | Sec 4.34 LX-07..08, all LX tests | HIGH | Suspected | LX-07 and LX-08 explicitly cover this; any READY+LX tests will catch regression | FRS-AC 5.2 |
| RISK-05 | Zero-stride expanded index not yet integrated in restickify + indirect access pipeline | Sec 4.9 NCI-01, upstream test_gather_expanded_index | MEDIUM | Confirmed (xfail) | UPT-S04a/b tracked; NCI-01 validates fix once integrated | FRS-GG FR-4 |
| RISK-06 | Multi-core (SENCORES > 1) gather/index_select not validated on hardware | Sec 4.14 MC-01..06, 5.10 IS-RT-05, 8.2 PA-ADV-08 | HIGH | Untested | BLOCKED:MC on all multi-core tests; unblock after MC validation milestone | FRS-AC 5.3 |
| RISK-07 | SDSC JSON kernel spec validation for index_select and embedding still NEEDS:INFRA (only gather covered by Sec 4.12) | Sec 9.5 IS-KS-01, EM-KS-01 | LOW | Partial coverage | Sec 4.12 covers gather SDSC via TORCH_COMPILE_DEBUG path; IS and Embedding kernel spec validation (9.5) remain deferred until introspection API available | FRS-AC 5.2 |
| RISK-08 | Silent wrong result risk when gate disabled (gate=0 without recompile) | Sec 4.18 EC-09..EC-09g | HIGH | Latent | CB-02 validates gate in cache key; EC-09g covers stale-kernel scenario | FRS-GG FR-2 |
| RISK-09 | index_select 3D/4D blocked by same HW limitation as 3D/4D gather | Sec 5.1 IS-FC-03..08, 5.9 IS-KV-02..03 | HIGH | Inferred from RISK-01/02 | Same BLOCKED:HW status applied; 2D tests cover assertion-enabled baseline | FRS-GG FR-5 |
| RISK-10 | Chunk large tensors path for indirect access not validated | Sec 4.16 CL-02..03 | MEDIUM | Unknown | BLOCKED:CHUNK; unblock when CHUNK_LARGE_TENSORS + indirect access path tested | Internal |
| RISK-11 | Embedding batch index shape (2D/3D index) blocked by HW limitation (multi-dim embedding output) | Sec 6.1 EM-FC-02..08 | MEDIUM | Inferred | BLOCKED:HW; EM-FC-01 (1D index, 2D weight) is READY baseline | FRS-GG FR-5 |
| RISK-12 | **#2662** — a compiled graph with exactly one SDSC (the indirect-access op alone, no fusion) failed at runtime in the dev team's report: `RuntimeError: Failed to parse SpyreCode command: Device pointer ... out of program segment bounds`. **Confirmed only** for the dev-shared repro `sdsc_fused_index_0_9nrav2b4` (mb=3, out=1024) — no test in this suite has been executed against real hardware or the `dxp` toolchain, so whether other single-SDSC shapes/tests reproduce the same failure is an open question, not a confirmed fact | HIGH (at confirmed shape); UNKNOWN (generalized) | Confirmed (FAIL) at the one reported shape only; NOT independently verified at any other shape — no execution has been performed in this work | `test_gather_2d_dim1` carries a code comment naming this issue for visibility; it is *not* marked `xfail`, since we have not verified that this specific test (different shape than the confirmed repro) actually fails. Re-confirm against real hardware before assuming broader impact | Internal (torch-spyre #2662) |
| RISK-13 | **#2661** — a compiled graph with more than one SDSC (the indirect-access op fused/composed with any other op) produced a job plan missing the host-side address-computation/transfer step in the dev team's report. **Confirmed only** for the dev-shared repro `sdsc_fused_exp_index_0_7vgnfcm1` (3D gather/index → separate `exp` kernel) — same caveat as RISK-12: not executed or independently verified for any other shape/test in this suite | HIGH (at confirmed shape); UNKNOWN (generalized) | Confirmed (FAIL) at the one reported shape only; NOT independently verified at any other shape — no execution has been performed in this work | `test_gather_graph_softmax` carries a code comment naming this issue for visibility; it is *not* marked `xfail`, since we have not verified that this specific test (different shape than the confirmed repro) actually fails. Re-confirm against real hardware before assuming broader impact | Internal (torch-spyre #2661) |
| RISK-14 | **#2654** — `primaryDsInfo_` dimension order is incorrectly significant to the backend compiler (documented as a set, should be order-independent); reordering `out`/`mb` in the same SDSC causes a crash. This is chosen internally by codegen, not by any Python-level `torch.compile` argument — **no FVT-level test in this suite can directly force one ordering over the other** | MEDIUM | Confirmed (FAIL via `dxp`), open issue | Not testable at this layer; tracked here for visibility only. Would need a direct SDSC-bundle-level regression test in the backend's own test suite, not torch-spyre's | Internal (torch-spyre #2654) |
| RISK-15 | **#2653** — corelet splitting (`SENCORELETS` unset/>1, the default) crashes for indirect-access SDSCs (`DtException: A transfer cannot move more than one page at time`); only passes with `SENCORELETS=1`. Confirmed: `SENCORELETS` is not read anywhere in torch-spyre's Python code (verified via `grep -r` across `torch_spyre/`) — it is consumed only by the external `dxp`/`dxp_standalone` CLI when replaying an already-generated bundle directly, **so no `torch.compile`-level FVT test can exercise this knob either** | MEDIUM | Confirmed (FAIL via `dxp_standalone`), open issue | Not testable at this layer; tracked here for visibility only, same reasoning as RISK-14 | Internal (torch-spyre #2653) |

---

### Sign-off Gates

> A test gate defines the set of tests that must pass for a given milestone. Tests with BLOCKED status do not count against a gate until unblocked.

| Gate ID | Gate Name | Required Tests | Blocked Tests (excluded until unblocked) | Milestone |
|---|---|---|---|---|
| **G-1** | Upstream Integration Baseline | Section 1 all mandatory_success; Section 2 TST-01; Section 3 DPR-01..DPR-65 (PASS) | DPR-66..71 (FAIL/DISABLED) | Existing CI green |
| **G-2** | 2D gather correctness on HW | FC-01, FC-09, FC-12; IP-01..IP-10; AV-01..AV-05; DT-01..DT-07; EM-01..EM-08; DS-01, DS-02, DS-06, DS-07; ML-01..ML-06; NCS-01..NCS-06; NCI-01..NCI-04; EC-01..EC-13; PA-G-01, PA-IS-01, PA-ADV-01 | All BLOCKED:HW tests | HW bring-up gate |
| **G-3** | index_select 2D on HW | IS-FC-01..02, IS-FC-09, IS-FC-11..12; IS-IP-01..IS-IP-10; IS-AV-01..05; IS-DT-01..07; IS-EM-01..08; IS-EC-01..10; IS-KV-01, IS-KV-04, IS-KV-06..08 | IS-FC-03..IS-FC-08, IS-KV-02..05, IS-RT-05 | index_select HW bring-up |
| **G-4** | embedding 2D on HW | EM-FC-01, EM-FC-04..07; EM-AV-01..06; EM-DT-01..05; EM-EM-01..06; EM-EC-01..07; EM-LM-01..03; EM-RT-01..05 | EM-FC-02..03, EM-FC-08, EM-DS-02, EM-LM-04, EM-RT-06 | embedding HW bring-up |
| **G-5** | Cross-op baseline | All Section 7 tests (CO-GIS-01..CO-FI-02) | None (all 2D READY) | Cross-op validation |
| **G-6** | Paged attention 2D baseline | PA-G-01, PA-IS-01, PA-CMP-01, PA-ADV-01 | All BLOCKED:HW PA tests | Paged attention milestone |
| **G-7** | 3D/4D gather unblocked | FC-02..FC-08, FC-10..FC-11; all BLOCKED:HW in Secs 4, 5, 6, 8 | None (all unblocked) | HW limitation resolved |
| **G-8** | Multi-core validation | All BLOCKED:MC tests: MC-01..06, IS-RT-05, EM-RT-06, PA-ADV-08 | None | Multi-core milestone |
| **G-9** | Unit test baseline | All Section 9 READY tests (GA-AC-IS-*, GA-DT-*, GA-FX-*, GA-SF-*) | All NEEDS:INFRA tests | Unit test gate (no HW required) |
| **G-10** | Full FVT sign-off | All gates G-1 through G-9 passing; all NEEDS:INFRA tests covered by SDSC tooling; all RISK entries resolved or accepted | None | Release sign-off |

---

### Blocking Issue Tracker

| Block ID | Type | Affected Test Count | Description | Resolution Path |
|---|---|---|---|---|
| BLOCKED:HW | Hardware | ~73 | 3D/4D indirect access produces wrong values or runtime errors on Spyre hardware (DPR-66/67 root cause not yet resolved) | HW team investigation; unblock once 3D address formula validated on chip |
| BLOCKED:MC | Multi-core | ~18 | Multi-core (SENCORES > 1) indirect access not yet validated; results diverge from SENCORES=1 | MC work division for indirect access; unblock after per-core address partitioning validated |
| BLOCKED:CHUNK | Chunking | ~3 | CHUNK_LARGE_TENSORS + indirect access interaction not validated | Chunking path + indirect access; unblock after chunk-span + gather tested |
| NEEDS:INFRA | Infrastructure | ~3 | Kernel spec (SDSC JSON) validation for index_select and embedding (Sec 9.5 IS-KS-01, EM-KS-01, EM-KS-01) lacks introspection tooling. Gather SDSC (Sec 4.12) is now READY via TORCH_COMPILE_DEBUG capture path. | Add SDSC bundle introspection API for IS/embedding kernel spec; unblock once API available |

---

### Abbreviations and Environment Shorthands

| Shorthand | Expansion |
|---|---|
| `ENV_BASE` | `SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS=1 SENCORES=1` |
| `ENV_GATE_OFF` | `SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS=0 SENCORES=1` |
| `ENV_DYN` | `ENV_BASE` + dynamic shapes enabled via `torch._dynamo.config.assume_static_by_default=False` |
| `ENV_LX` | `ENV_BASE LX_PLANNING=1` |
| `ENV_LX_CO` | `ENV_LX SPYRE_INDUCTOR_CO_OPTIMIZING_ALLOC=1` |
| `ENV_MC4` | `SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS=1 SENCORES=4` |
| `ENV_MC32` | `SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS=1 SENCORES=32` |
| `ENV_CHUNK` | `ENV_BASE CHUNK_LARGE_TENSORS=1` |
| `ENV_GSO_OFF` | `ENV_BASE SPYRE_INDUCTOR_DISABLE_GLOBAL_STICK_OPTIMIZER=1` |
| `ENV_NO_FUSION` | `ENV_BASE SPYRE_INDUCTOR_ENABLE_FUSION=0` |
| `ENV_NO_HINTS` | `ENV_BASE SPYRE_INDUCTOR_IGNORE_HINTS=1` |
| `CPU_ONLY` | No Spyre device required; test runs on CPU / mock FX graph |
| `FRS-GG` | Functional Requirements Specification — torch.gather |
| `FRS-AC` | Functional Requirements Specification — Address Computation |

---

*Document status: v1.0 — Initial Draft — Pending Developer Review*

