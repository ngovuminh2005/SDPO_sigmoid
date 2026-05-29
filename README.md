# Setup thư viện

## 1. Tạo môi trường

```bash
conda create --prefix /path/to/mbo python=3.12 -y
conda activate /path/to/mbo
python -m pip install --upgrade pip setuptools wheel
python -m pip install "numpy==1.26.4" "packaging==25.0"
```

## 2. Cài vLLM trước

```bash
python -m pip install "vllm==0.10.2"
```

Kiểm tra stack mà vLLM kéo vào:

```bash
python - << 'EOF'
import torch, vllm
print("torch:", torch.__version__, torch.version.cuda)
print("vllm:", vllm.__version__)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
EOF
```

## 3. Cài nvcc theo CUDA của torch

Ví dụ torch dùng CUDA 12.8:

```bash
conda install --prefix /path/to/mbo -c nvidia cuda-nvcc=12.8 -y
export CUDA_HOME=/path/to/mbo
export PATH=$CUDA_HOME/bin:$PATH
nvcc -V
```

## 4. Build flash-attn

Với H100:

```bash
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=4
python -m pip install "flash-attn==2.8.3" --no-build-isolation --no-cache-dir --no-deps
```

Kiểm tra ABI:

```bash
python - << 'EOF'
import torch, vllm, flash_attn
import vllm._C
import flash_attn_2_cuda
from flash_attn import flash_attn_func

print("torch:", torch.__version__, torch.version.cuda)
print("vllm:", vllm.__version__)
print("flash_attn:", flash_attn.__version__)
print("ABI OK")
EOF
```

## 5. Cài các thư viện còn lại

Sau khi `vllm` và `flash-attn` đã OK, cài các package còn lại bằng script `install_no_deps_fixed.sh`:

```bash
bash install_no_deps.sh
```

## 6. Chạy training

Từ repo root:

```bash
bash experiments/generalization/run_sdpo_local.sh

Script này dùng `--no-deps`, không đụng lại `torch`, `vllm`, `flash-attn`, `numpy`.
