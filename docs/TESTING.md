# 测试与展示证据

## 本次实际运行

日期：2026-10-13。本地自动业务测试：**51 passed，1 skipped**；跳过项需要真实 Redis，CI 已配置该服务。Windows 工作区 Python 3.9 兼容路径；推荐新环境 Python 3.10—3.12。

实际执行 **60/60 条合成业务轨迹**与 **11 次浏览器业务响应验证**，生成 **9 张截图**，浏览器无 JavaScript 错误。使用系统 Edge 与 Playwright；未生成或伪造运行图片。

测试使用临时 SQLite 与合成酒店/房态。页面使用真实目录名、合成库存、合成偏好证据、哈希向量及 fakeredis；未进行模型语言质量或生产成交评测。轨迹结果见 [evaluation-results.json](evaluation-results.json)，机制与修复记录见 [AI_NATIVE_IMPLEMENTATION.md](AI_NATIVE_IMPLEMENTATION.md)。

## 自动断言覆盖

- 槽位缺失时追问，且不运行检索。
- 完整条件过滤城市、房型、每晚预算和整个住宿区间。
- 城市/日期/预算/房型变更后旧候选失效。
- 选择酒店后确认，再修改条件时清除选择。
- 低预算无匹配、缺少某晚报价、库存售罄。
- 同会话入口串行、检索期间新需求可提交、旧结果不得发布；不同任务受并发上限控制。
- 消息去重不增加历史；不同正文不能复用 ID。
- Redis TTL、缓存缺失回填、故障回退与旧缓存拒绝。
- 有限重试、实际等待超时和相同消息恢复。
- 重建服务后恢复失败检索节点，不重跑意图节点。
- 旧失败任务不能覆盖后续消息的槽位。
- 离店单独修改、相对日期顺序和日期不误识别为预算。
- 参考报价不能虚构库存或变成已确认房源。

## 执行命令

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/evaluate_ai_native.py
python demo_booking.py --scripted --output data/demo_runtime/showcase.json
python demo_web.py --redis-simulated
```

另一个终端执行：

```powershell
python scripts/capture_demo.py --channel msedge
```

截图脚本检查 HTTP 成功、任务状态、去重、历史长度和无匹配清空候选，再截图。运行 JSON 与 JUnit XML 存放于被 Git 忽略的本地运行目录。

## 人工复核提纲

检查日期与预算语义、候选的城市/房型/版本、条件修改后是否引用旧结果、无匹配时是否虚构推荐、失败原因是否可追踪、恢复是否重复写历史，以及报价是否保持房态未确认说明。

这些检查点不等于已收集独立模型标注集。当前不报告 BGE-M3 精确率、用户转化率、生产吞吐量或真实 Redis 压测结果。

## 尚未验证的外部链路

真实模型接口集成检查遇到 APIConnectionError，未取得有效响应；公开截图使用离线规则。

本次未登录真实闲鱼店铺、向买家发消息或创建订单。未运行 BGE-M3 质量评测，未连接真实 Redis 服务，也未构建 Docker 镜像。真实网页查询受页面结构、网络与验证页影响，部署后需单独联调。

## 持续集成

GitHub Actions 使用 Python 3.11 和 Redis 7.4，运行全部业务断言以及真实 Redis TTL / 状态回填测试。本地没有 `REDIS_TEST_URL` 时真实服务测试明确跳过；本机的 fakeredis 结果不代替这项集成测试。

历史工程版本已在 Python 3.11 + Redis 7.4 下通过全部 32 项测试。当前目标仓库的运行记录见 [GitHub Actions](https://github.com/spark216/YF-agent/actions)。本机没有真实 Redis 的说明不影响独立服务验证结果。
