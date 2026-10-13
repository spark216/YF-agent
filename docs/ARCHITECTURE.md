# 架构与状态一致性

## 工作流

`BookingService` 使用异步上下文管理器持有 LangGraph 和 SQLite checkpointer。图依次运行意图识别、槽位更新、条件校验、检索与回复；可选网页查价位于检索之前。

缺少条件时不检索。选择当前版本候选后直接进入确认引导；修改条件会清除选择。

## 状态划分

会话保存 `history`、`slots`、`revision`、`candidates`、`candidate_revision`、`selected_hotel`、`quotes`、`task_status`。业务历史保留最近 20 条，兼容平台历史仍单独存储。

主要任务状态为 `collecting`、`ready`、`no_match`、`quote_pending`、`awaiting_confirmation`。失败请求另存为 `failed`。

条件修改先清空候选和选择、递增版本并持久化，再检索。候选通过 `candidate_revision` 绑定产生它的需求版本。

## Redis 与持久层

Redis 使用带 TTL 的 `SET ... EX` 写入会话快照，命中后刷新 TTL。[Redis SET 文档](https://redis.io/docs/latest/commands/set/)

SQLite 始终保存会话真值。缓存封装包含持久层更新时间，读取时与 SQLite 元数据核对；过期、故障或不一致时重新读取并回填。回填读取最新持久快照，避免早期读请求覆盖新缓存。请求完成时，会话和去重结果在一个事务中提交。

缓存减少完整 JSON/历史读取，仍保留一次轻量元数据查询保证当前一致性；不是完全脱离数据库的读路径。

## 检索与房态

酒店目录包含 ID、名称、城市与品牌。`offers` 另存逐日房型、价格、库存、来源和更新时间。查询要求每个住宿夜都有数据、库存大于零且价格不超每晚预算。

SQL 使用 `COUNT(DISTINCT night)` 校验住宿区间，`MAX(price)` 控制每晚预算，`SUM(price)` 计算整段价格。当前约 1 万条目录使用 `IndexFlatIP`，取得全目录相似度排序后执行业务过滤，避免小 Top-K 提前排除合格酒店。

BGE-M3 和离线哈希编码器通过 `encode()` 注入。演示哈希向量不替代 BGE 质量评测。网页价格不等于确认库存，`LivePriceProvider` 不将未知库存写成 1，参考报价进入 `quote_pending`。

## 并发与消息去重

会话锁保护入口需求事务，工具执行在锁外进行；等待者使用引用计数清理空闲锁。独立入口限流与工具 Semaphore 防止慢工具占满容量后阻塞新需求提交。发布时校验需求版本、消息轮次与选择版本。AI-native 事务、证据与审批细节见 [实施说明](AI_NATIVE_IMPLEMENTATION.md)。

`requests` 以 `(chat_id,message_id)` 为主键。相同 ID 不同内容被拒绝；完成后的重复请求返回已有回复与最新会话状态。平台优先使用明确消息 ID，缺少时使用会话、发送者、时间戳和正文的稳定哈希。

平台发送结果另存。发送和 SQLite 不具备跨系统事务，在发送成功但未记录的窗口仍可能重发，不承诺严格 exactly-once。多进程部署需要分布式会话锁和任务租约。

## 检查点与恢复

每个请求用会话 ID 与消息 ID 计算独立 `thread_id`。失败后重放相同消息，从 `checkpoint.next` 继续；图已完成但最终事务未提交时，可以使用最终图状态完成提交。采用官方异步 SQLite saver。[LangGraph SQLite checkpoint 文档](https://reference.langchain.com/python/langgraph.checkpoint.sqlite)

更新消息将同会话旧中断请求标为 `superseded`，不允许旧任务恢复后覆盖新需求。超时与有限重试限制等待；线程中的同步工具不能强制取消，应保持读取幂等。

## 追踪

`traces` 保存节点、状态、耗时、结构化结果与异常类型，不主动记录凭据或请求头。真实会话文本仍是业务数据，需管理保留周期和访问范围。
