# 运行与部署

## 本地演示

独立 Python 3.10—3.12 环境安装 `requirements-dev.txt`，执行 `python demo_web.py --redis-simulated`，访问 `http://127.0.0.1:8765`。演示不发送平台消息或调用模型 API。

仅使用核心流程时安装 `requirements-core.txt`。fakeredis、pytest 和截图工具属于开发依赖。

## Redis

```powershell
docker compose -f docker-compose.redis.yml up -d
```

配置 `REDIS_URL=redis://127.0.0.1:6379/0`，不使用 `--redis-simulated`。Compose 仅向本机暴露 Redis，启用 AOF 和健康检查。截图中的 Redis 是 API 模拟，不是已部署服务。

## 完整模型与平台模式

安装 `requirements.txt`，配置 `API_KEY`、`MODEL_BASE_URL`、`MODEL_NAME`、`BGE_MODEL_PATH`。基础模型可使用本地目录；微调分类还需要 `models/output/train_model/intent_model/` 的匹配权重。

`LLM_SLOT_EXTRACTION=true` 启用模型抽取，`INTENT_BACKEND=bge` 启用微调分类。新图默认使用确定性业务回复，原有生成式图保留供对照。

配置本人闲鱼 Cookie 后运行 `python run_xianyu.py`。新入口保留心跳、Token 刷新、人工接管与商品信息读取；新业务库在 `data/runtime/`，兼容历史在 `data/chat_history.db`。

`LIVE_PRICE_ENABLED=true` 开启网页报价，还需安装 Playwright Chromium。预算按每晚解释，网页报价与确认库存分开。没有房态接口时不承诺能直接订房。

默认并发 4、TTL 1800 秒、工具等待 20 秒、最多 2 次重试。多酒店多晚查价可能需更长超时；取消线程等待不等于底层浏览器已终止。

## 容器演示

```powershell
docker compose up --build
```

默认镜像运行展示页面并连接 Compose Redis，只向本机暴露 8765。默认安装核心依赖。使用 `INCLUDE_MODELS=true` 构建参数可安装完整依赖，还需模型挂载、真实配置和平台入口设置。

本次没有 Docker 服务，配置仅做静态路径检查，不宣称镜像构建已通过。

## 部署范围

当前按单进程运行，多个 worker 不能共享进程内锁。扩展前需分布式租约、共享检查点和外部数据库。展示页不是带鉴权的生产后台，默认只监听本机。不要混用演示与生产运行库，不提交 `.env` 和数据库。
