# embed-server

极简 embedding 服务，运行 **Qwen3-Embedding-0.6B**，提供 **OpenAI `/v1/embeddings` 兼容接口**。

专为 RS1000（AMD EPYC Zen4, AVX-512）优化，使用 zentorch + PyTorch CPU build，供 LE-B 上的 OpenClaw 跨机调用。

---

## 快速部署（3 步）

### Step 1：克隆代码库

```bash
git clone <repo_url>
cd embed-server
```

### Step 2：下载模型

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-Embedding-0.6B \
  --local-dir /home/aichan/models/Qwen3-Embedding-0.6B
```

> 模型约 600MB，下载到宿主机后挂载进容器（不打包进镜像）。

### Step 3：启动服务

```bash
docker compose up -d
```

服务监听 `0.0.0.0:8765`，首次启动预热约 **8–10 秒**（zentorch JIT 编译），`healthcheck` 的 `start_period: 30s` 已留足余量。

---

## 换更大的模型

只需修改 `MODEL_PATH` 环境变量，无需重新构建镜像：

```bash
# 方式 1：docker compose 覆盖
MODELS_DIR=/data/models \
MODEL_PATH=/models/Qwen3-Embedding-4B \
docker compose up -d

# 方式 2：修改 docker-compose.yml 中的 MODEL_PATH
```

---

## OpenClaw 配置示例

在 LE-B 的 OpenClaw 配置中添加：

```json
{
  "agents": {
    "defaults": {
      "memorySearch": {
        "provider": "openai",
        "model": "Qwen3-Embedding-0.6B",
        "remote": {
          "baseUrl": "http://RS1000_IP:8765/v1",
          "apiKey": ""
        }
      }
    }
  }
}
```

将 `RS1000_IP` 替换为 RS1000 的实际 IP 地址。

---

## 测试命令

### 健康检查

```bash
curl http://RS1000_IP:8765/health
# 就绪时返回：{"status":"ok","model_ready":true}
# 预热中返回：{"status":"loading","model_ready":false}
```

### 向量生成

```bash
curl -X POST http://RS1000_IP:8765/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-Embedding-0.6B","input":["hello world","你好世界"]}'

# 返回示例：
# {
#   "data": [
#     {"embedding": [0.123, -0.456, ...]},
#     {"embedding": [0.789,  0.012, ...]}
#   ]
# }
```

---

## 性能基准（RS1000 实测）

| 指标 | 数值 |
|------|------|
| CPU | AMD EPYC 9634，4 vCPU |
| 单条 encode P50 | **40ms** |
| 单条 encode P95 | **43ms** |
| 批量吞吐（batch=8） | **25.5 条/s** |
| 首次 zentorch 编译 | ~8s（一次性） |
| 内存占用 | ~1.5GB（限制 4g） |

**优化说明：**
- `torch==2.7.0+cpu`：纯 CPU 构建，比 CUDA build 快 2x（无 CUDA 初始化开销）
- `zentorch_compiler_noinductor`：绕过 AVX-512 inductor codegen bug，走 ZENDNN/BLIS
- `OMP_NUM_THREADS=4`：对齐 vCPU 数量，避免线程争抢
- BF16 未启用：短文本场景 cast overhead 抵消收益（慢 21%）

---

## 目录结构

```
embed-server/
├── Dockerfile           # python:3.11-slim，torch 单独一层
├── docker-compose.yml   # 端口 8765，模型挂载，内存限制 4g
├── .dockerignore
├── server.py            # FastAPI 主程序（zentorch 推理）
├── requirements.txt     # 依赖（torch 需单独安装 CPU 版）
└── README.md
```

---

## 注意事项

- **端口 8765** 需在 RS1000 防火墙/安全组中对 LE-B IP 开放
- `docker compose up -d` 后用 `docker compose logs -f` 观察预热日志
- 内存限制设为 `4g`（RS1000 总内存 7.8GB），防止 OOM 影响其他服务
