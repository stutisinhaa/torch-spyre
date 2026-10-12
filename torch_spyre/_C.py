# Copyright 2025 The Torch-Spyre Authors.
# Wrapper for _C module that falls back to stub implementation

"""
This module provides a compatibility layer for the _C extension.

When the package is built with USE_STUBS=0 (real build):
  - The C++ extension torch_spyre._C will be compiled
  - This file won't be used (the compiled .so/.pyd takes precedence)

When the package is built with USE_STUBS=1 (stub build):
  - No C++ extension is compiled
  - This file provides the _C module by importing from the stub
  - All imports from torch_spyre._C will get the stub implementation
"""

# When built with stubs, this file provides the _C module
# by re-exporting everything from the stub implementation
from torch_spyre.mock_device_integration._C_stub import *  # noqa: F401, F403
