# Stratum — Multi-GPU Layer-Parallel Training
#
# Based on the proven qz-roundpipe build chain:
#   PyTorch 2.12.0+cu126 / CUDA 12.6
#   flash-attention-v100 (patched for sm_70)
#   causal-conv1d (patched for sm_70)
#   bitsandbytes, transformers, peft
#
# Build:
#   docker build -t stratum:latest .
#
# Run:
#   scripts/run-unified.sh python scripts/doctor.py
#   STRATUM_DATA_DIR=/path/to/training_data \
#     scripts/run-unified.sh python scripts/train.py ...

FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

SHELL ["/bin/bash", "-lc"]

ENV MAX_JOBS=4
ENV NVCC_THREADS=4
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ninja-build build-essential \
    wget curl pkg-config \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Python 3.12 via deadsnakes
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /workspace/venv
ENV PATH=/workspace/venv/bin:$PATH
ENV LD_LIBRARY_PATH=/workspace/venv/lib/python3.12/site-packages/torch/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
RUN pip install --upgrade pip setuptools wheel

# PyTorch 2.12 + CUDA 12.6 (matching flash-attn-v100 requirements)
RUN pip install --no-cache-dir \
    torch==2.12.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126

# Core ML dependencies
RUN pip install --no-cache-dir \
    transformers peft bitsandbytes datasets \
    accelerate sentencepiece protobuf ninja packaging psutil tqdm

# causal-conv1d: sm_70 only build
RUN git clone --depth 1 https://github.com/Dao-AILab/causal-conv1d.git /tmp/causal-conv1d

RUN cd /tmp/causal-conv1d && python - <<'PY'
from pathlib import Path
p = Path("setup.py")
s = p.read_text()
old = '''        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_75,code=sm_75")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_80,code=sm_80")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_87,code=sm_87")
        if bare_metal_version >= Version("11.8"):
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_90,code=sm_90")
        if bare_metal_version >= Version("12.8"):
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_100,code=sm_100")
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_120,code=sm_120")
        if bare_metal_version >= Version("13.0"):
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_103,code=sm_103")
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_110,code=sm_110")
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_121,code=sm_121")
'''
new = '''        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_70,code=sm_70")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_86,code=sm_86")
'''
if old not in s:
    raise SystemExit("could not find causal-conv1d CUDA arch block in setup.py")
p.write_text(s.replace(old, new))
print("patched causal-conv1d setup.py for sm_70 + sm_86")
PY

RUN TORCH_CUDA_ARCH_LIST="7.0;8.6" \
    CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    pip install --no-cache-dir --no-build-isolation -v /tmp/causal-conv1d

RUN rm -rf /tmp/causal-conv1d

# FlashAttention QK-Clip stats patches are applied to the cloned upstream trees
# before the existing build compatibility edits below.
COPY patches/ /tmp/patches/

# flash-attention-v100: patched for CUDA 12.6 + sm_70
RUN git clone https://github.com/ai-bond/flash-attention-v100.git /tmp/fav100 && \
    cd /tmp/fav100 && git checkout 6d53118 && \
    git apply /tmp/patches/flash-attn-v100-6d53118-qk-max-logits.patch

RUN cd /tmp/fav100 && python - <<'PY'
from pathlib import Path
p = Path("setup.py")
s = p.read_text()
old_ver = 'if parse(torch.version.cuda) < parse("12.9"):'
new_ver = 'if parse(torch.version.cuda) < parse("12.6"):'
assert old_ver in s, "CUDA version check not found in setup.py"
s = s.replace(old_ver, new_ver)
old_gpu = '''            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required but not available.")
'''
assert old_gpu in s, "GPU availability check not found in setup.py"
s = s.replace(old_gpu, "")
p.write_text(s)
print("patched flash-attn-v100 for CUDA 12.6; removed GPU check")
PY

RUN cd /tmp/fav100 && python - <<'PY'
from pathlib import Path
p = Path("pyproject.toml")
lines = p.read_text().splitlines()
patched = []
for line in lines:
    if "torch" in line and ("download.pytorch.org" in line or line.strip().startswith('"torch')):
        continue  # remove hardcoded torch wheel URL
    if line.strip() == 'license = "BSD-3-Clause"':
        patched.append('license = {text = "BSD-3-Clause"}')  # fix license format
        continue
    if line.strip().startswith("license-files"):
        continue  # remove legacy license-files key
    patched.append(line)
p.write_text("\n".join(patched) + "\n")
print("patched flash-attn-v100 pyproject.toml: removed torch URL, fixed license")
PY

RUN cd /tmp/fav100 && python - <<'PY'
from pathlib import Path
p = Path("include/mat_mul.h")
s = p.read_text()
assert "__tanhf" in s, "__tanhf not found in include/mat_mul.h"
p.write_text(s.replace("__tanhf", "tanhf"))
print("patched flash-attn-v100 include/mat_mul.h: __tanhf -> tanhf")
PY

RUN pip install --no-cache-dir --no-build-isolation --no-deps -v /tmp/fav100

RUN rm -rf /tmp/fav100

# The V100 fork also creates a flash_attn/ module (same name as standard FA).
# Remove it so standard flash-attn can install cleanly into that namespace.
RUN rm -rf /workspace/venv/lib/python3.12/site-packages/flash_attn/ && \
    rm -rf /workspace/venv/lib/python3.12/site-packages/flash_attn-*.dist-info

# Standard flash-attention 2.8.3 (Ampere+, targets sm_86 for RTX 3080).
# Build from source restricted to sm_86. No pre-built wheel for torch2.12.
RUN git clone --depth 1 --branch v2.8.3 https://github.com/Dao-AILab/flash-attention.git /tmp/fa2 && \
    cd /tmp/fa2 && \
    git apply /tmp/patches/flash-attn-2.8.3-qk-max-logits.patch

RUN cd /tmp/fa2 && python - <<'PY'
from pathlib import Path
p = Path("setup.py")
s = p.read_text()
# Add sm_86 to default archs and to gencode flags
s = s.replace(
    'return os.getenv("FLASH_ATTN_CUDA_ARCHS", "80;90;100;120").split(";")',
    'return os.getenv("FLASH_ATTN_CUDA_ARCHS", "86;80;90;100;120").split(";")',
)
old = '''if "80" in cuda_archs():
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_80,code=sm_80")'''
new = '''if "86" in cuda_archs():
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_86,code=sm_86")
    if "80" in cuda_archs():
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_80,code=sm_80")'''
assert old in s, "could not find sm_80 block in setup.py"
s = s.replace(old, new)
p.write_text(s)
print("patched flash-attn setup.py: added sm_86 support")
PY

RUN FLASH_ATTN_CUDA_ARCHS="86" pip install --no-cache-dir --no-build-isolation -v /tmp/fa2

RUN rm -rf /tmp/fa2

# Flash Linear Attention: Qwen3.5 linear-attention layers require FLA's
# non-quadratic gated-delta kernels. Without this, Transformers falls back to
# torch_chunk_gated_delta_rule and long-context backward recompute OOMs on the
# 10 GiB RTX 3080 path.
RUN pip install --no-cache-dir "flash-linear-attention[cuda]"

# Stratum install — MUST come AFTER all CUDA builds so source file changes
# don't invalidate the expensive CUDA kernel compilation layers.
# Keep docs/tests/git/cache/data out of the final runtime image. The package
# build gets a tiny generated README so real documentation is not baked into
# wheel metadata.
RUN rm -rf /workspace/stratum /tmp/stratum-src
WORKDIR /tmp/stratum-src
COPY pyproject.toml /tmp/stratum-src/
RUN printf 'Stratum runtime image package build.\n' > README.md
COPY stratum /tmp/stratum-src/stratum
RUN pip install --no-cache-dir /tmp/stratum-src && rm -rf /tmp/stratum-src

WORKDIR /workspace/stratum
COPY scripts/train.py scripts/doctor.py /workspace/stratum/scripts/

# Unified cache directory
ENV STRATUM_CACHE=/var/cache/stratum
ENV HF_HOME=$STRATUM_CACHE/huggingface
ENV TORCH_EXTENSIONS_DIR=$STRATUM_CACHE/torch_extensions
ENV CUDA_CACHE_PATH=$STRATUM_CACHE/cuda
ENV XDG_CACHE_HOME=$STRATUM_CACHE/xdg
ENV CUDA_CACHE_MAXSIZE=2147483648
ENV PYTORCH_ALLOC_CONF=expandable_segments:True

# Verify imports
RUN python - <<'PY'
import torch
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
from flash_attn_v100 import flash_attn_func
print("flash-attn-v100: OK")
from flash_attn import flash_attn_func as fa2
print("flash-attn (standard): OK")
import fla
print("flash-linear-attention: OK")
import bitsandbytes
import transformers, peft
import stratum
print("All imports OK")
print("Stratum import OK:", stratum.__file__)
PY

LABEL description="Stratum: multi-GPU layer-parallel training"
LABEL base_image="nvidia/cuda:12.6.3-devel-ubuntu22.04"
LABEL torch_version="2.12.0+cu126"
