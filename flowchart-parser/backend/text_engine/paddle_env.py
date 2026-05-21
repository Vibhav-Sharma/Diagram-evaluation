"""
Configure Paddle runtime BEFORE paddle/paddleocr is imported.

PaddlePaddle 3.3+ on Windows can crash in OneDNN PIR conversion:
  ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<...>]

Disabling MKLDNN/OneDNN avoids this path. Import this module first in main.py.
"""

from __future__ import annotations

import os


def configure_paddle_runtime() -> None:
    # Must be set before `import paddle`
    os.environ["FLAGS_use_mkldnn"] = "0"
    os.environ["FLAGS_use_onednn"] = "0"
    os.environ["FLAGS_enable_mkldnn"] = "0"
    # Prefer legacy executor on some Windows CPU builds
    os.environ.setdefault("FLAGS_enable_pir_api", "0")
    os.environ.setdefault("FLAGS_pir_apply_inplace_pass", "0")


# Apply immediately on import
configure_paddle_runtime()
