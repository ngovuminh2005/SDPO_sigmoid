#!/bin/bash
set -euo pipefail

echo "Checking core stack first..."
python - << 'EOF'
import torch, vllm, flash_attn
import vllm._C
import flash_attn_2_cuda
from flash_attn import flash_attn_func

print("torch:", torch.__version__, torch.version.cuda)
print("vllm:", vllm.__version__)
print("flash_attn:", flash_attn.__version__)
print("core ABI OK")
EOF

echo "Installing remaining packages with --no-deps..."

python -m pip install --no-deps \
  absl-py \
  accelerate \
  aiohappyeyeballs \
  aiohttp \
  aiohttp-cors \
  aiosignal \
  annotated-doc \
  annotated-types \
  antlr4-python3-runtime \
  anyio \
  apache-tvm-ffi \
  astor \
  attrs \
  blake3 \
  cachetools \
  cbor2 \
  certifi \
  cffi \
  charset-normalizer \
  click \
  cloudpickle \
  codetiming \
  colorful \
  compressed-tensors \
  datasets \
  depyf \
  dill \
  diskcache \
  distro \
  dnspython \
  docstring_parser \
  einops \
  email-validator \
  fastapi \
  fastapi-cli \
  fastapi-cloud-cli \
  filelock \
  frozenlist \
  fsspec \
  gguf \
  GitPython \
  gitdb \
  google-api-core \
  google-auth \
  googleapis-common-protos \
  grpcio \
  h11 \
  hf-xet \
  httpcore \
  httptools \
  httpx \
  huggingface-hub \
  hydra-core \
  idna \
  importlib_metadata \
  interegular \
  Jinja2 \
  jiter \
  jmespath \
  jsonschema \
  jsonschema-specifications \
  lark \
  latex2sympy2 \
  latex2sympy2_extended \
  liger-kernel \
  llguidance \
  llvmlite \
  lm-format-enforcer \
  loguru \
  Markdown \
  markdown-it-py \
  MarkupSafe \
  math-verify \
  mdurl \
  mistral_common \
  model-hosting-container-standards \
  mpmath \
  msgpack \
  msgspec \
  multidict \
  multiprocess \
  networkx \
  ninja \
  numba \
  omegaconf \
  openai \
  openai-harmony \
  opencensus \
  opencensus-context \
  opencv-python-headless \
  opentelemetry-api \
  opentelemetry-exporter-prometheus \
  opentelemetry-proto \
  opentelemetry-sdk \
  opentelemetry-semantic-conventions \
  orjson \
  outlines \
  outlines_core \
  packaging \
  pandas \
  partial-json-parser \
  peft \
  pillow \
  platformdirs \
  prometheus-fastapi-instrumentator \
  prometheus_client \
  propcache \
  proto-plus \
  protobuf \
  psutil \
  py-cpuinfo \
  py-spy \
  pyarrow \
  pyasn1 \
  pyasn1_modules \
  pybase64 \
  pybind11 \
  pycountry \
  pycparser \
  pydantic \
  pydantic-extra-types \
  pydantic-settings \
  pydantic_core \
  Pygments \
  pylatexenc \
  python-dateutil \
  python-dotenv \
  python-json-logger \
  python-multipart \
  pytz \
  pyvers \
  PyYAML \
  pyzmq \
  ray \
  referencing \
  regex \
  requests \
  rich \
  rich-toolkit \
  rignore \
  rpds-py \
  rsa \
  safetensors \
  scipy \
  sentencepiece \
  sentry-sdk \
  setproctitle \
  shellingham \
  six \
  smart_open \
  smmap \
  sniffio \
  soundfile \
  soxr \
  starlette \
  supervisor \
  sympy \
  tabulate \
  tensorboard \
  tensorboard-data-server \
  tensordict \
  tiktoken \
  tokenizers \
  torchdata \
  tornado \
  tqdm \
  transformers \
  typer \
  typing-inspection \
  typing_extensions \
  tzdata \
  urllib3 \
  uvicorn \
  uvloop \
  virtualenv \
  wandb \
  watchfiles \
  websockets \
  Werkzeug \
  word2number \
  wrapt \
  xgrammar \
  xxhash \
  yarl \
  zipp

echo "Installing local project with --no-deps..."
python -m pip install -e . --no-deps

echo "Checking imports..."
python - << 'EOF'
import traceback

def check(name, code):
    try:
        exec(code, {})
        print(f"{name}: OK")
    except Exception as e:
        print(f"{name}: FAIL -> {repr(e)}")
        traceback.print_exc()

check("torch", "import torch; print(' ', torch.__version__, torch.version.cuda)")
check("vllm", "import vllm; print(' ', vllm.__version__)")
check("vllm ABI", "import vllm._C")
check("flash_attn", "import flash_attn; print(' ', flash_attn.__version__)")
check("flash_attn ABI", "import flash_attn_2_cuda")
check("flash_attn func", "from flash_attn import flash_attn_func")

check("ray", "import ray; print(' ', ray.__version__)")
check("transformers", "import transformers; print(' ', transformers.__version__)")
check("datasets", "import datasets; print(' ', datasets.__version__)")
check("pandas", "import pandas; print(' ', pandas.__version__)")
check("wandb", "import wandb; print(' ', wandb.__version__)")
check("protobuf", "import google.protobuf.descriptor_pb2")

check("runtime extras", "import pyvers, mpmath, msgspec, cpuinfo, openai, sniffio, distro, multiprocess, dateutil, pytz")
check("web/runtime", "import aiohttp, httpx, fastapi, uvicorn, starlette")
check("math/runtime", "import sympy, mpmath, latex2sympy2_extended, math_verify, word2number")
EOF

echo "Done."