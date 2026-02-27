"""
embed-server — 极简 Qwen3 embedding 服务
OpenAI /v1/embeddings 兼容接口，运行在 RS1000 (AMD EPYC Zen4)

关键优化：
- zentorch.zentorch_compiler_noinductor 绕过 AVX-512 codegen bug
- OMP_NUM_THREADS=4 对齐 vCPU 数量
- 模块级别初始化 + 预热，保证 healthcheck 时模型已就绪
"""

import os
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── 推理依赖（模块级别，服务启动时执行一次）──────────────────────────────────

import torch
import zentorch  # noqa: F401 — 注册 zentorch backend，必须在 compile 前 import

torch.set_num_threads(4)

logger.info("Loading model from %s ...", os.environ.get("MODEL_PATH", "/models/Qwen3-Embedding-0.6B"))
_t0 = time.time()

from sentence_transformers import SentenceTransformer  # noqa: E402

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/Qwen3-Embedding-0.6B")
model = SentenceTransformer(MODEL_PATH)

_load_time = time.time() - _t0
logger.info("Model loaded in %.2fs", _load_time)

# zentorch 编译
# ⚠️  必须用 zentorch_compiler_noinductor：
#     - backend="inductor"           → AVX-512 codegen bug，崩溃
#     - backend="zentorch"           → 底层仍走 inductor，同样崩溃
#     - zentorch.optimize(model)     → Qwen3 动态控制流无法 FX trace，报错
#     ✅ zentorch_compiler_noinductor → 绕过 inductor，走 ZENDNN/BLIS，正常
logger.info("Applying torch.compile(backend=zentorch_compiler_noinductor) ...")
_t1 = time.time()
model[0].auto_model = torch.compile(
    model[0].auto_model,
    backend=zentorch.zentorch_compiler_noinductor,
)
_compile_time = time.time() - _t1
logger.info("Compile done in %.2fs", _compile_time)

# 预热：触发实际 JIT 编译（约 8s），之后 healthcheck 才返回 ok
logger.info("Warming up (first compile pass ~8s) ...")
_t2 = time.time()
model.encode(["warmup"], normalize_embeddings=True)
_warmup_time = time.time() - _t2
logger.info("Warmup done in %.2fs", _warmup_time)

model_ready = True
logger.info(
    "Model ready! load=%.2fs compile=%.2fs warmup=%.2fs total=%.2fs",
    _load_time, _compile_time, _warmup_time, time.time() - _t0,
)

# ── FastAPI ────────────────────────────────────────────────────────────────────

from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from typing import List  # noqa: E402

app = FastAPI(title="embed-server", version="1.0.0")


class EmbedRequest(BaseModel):
    model: str = "Qwen3-Embedding-0.6B"  # 原样传入，服务侧忽略，用挂载的模型
    input: List[str]


@app.post("/v1/embeddings")
def embeddings(req: EmbedRequest):
    """OpenAI-compatible embeddings endpoint."""
    if not req.input:
        return {"data": []}

    _t = time.time()
    vecs = model.encode(req.input, normalize_embeddings=True)
    _elapsed_ms = (time.time() - _t) * 1000

    logger.info("Encoded %d text(s) in %.1fms", len(req.input), _elapsed_ms)

    return {
        "data": [{"embedding": v.tolist()} for v in vecs]
    }


@app.get("/health")
def health():
    """Health check — returns loading until warmup completes."""
    if model_ready:
        return {"status": "ok", "model_ready": True}
    return {"status": "loading", "model_ready": False}
