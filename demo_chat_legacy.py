"""
终端多轮对话测试脚本
模拟实际运行环境，在终端中进行对话测试
"""
import os
import sys
import json
from dotenv import load_dotenv
from loguru import logger

from legacy_agents import XianyuReplyBot
from context_manager import ChatContextManager


def setup_logger(level: str = "DEBUG"):
    """配置日志输出"""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )


def load_config():
    """加载环境变量配置"""
    if os.path.exists(".env"):
        load_dotenv()
        logger.info("已加载 .env 配置")

    required = ["API_KEY", "MODEL_BASE_URL", "MODEL_NAME"]
    for key in required:
        val = os.getenv(key)
        if not val:
            logger.error(f"缺少必要配置: {key}")
            sys.exit(1)
    logger.info(f"模型: {os.getenv('MODEL_NAME')} @ {os.getenv('MODEL_BASE_URL')}")


def build_item_description() -> str:
    """构建测试用商品信息描述"""
    return json.dumps({
        "title": "全季酒店上海虹桥火车站店",
        "desc": "华住会旗下全季品牌酒店，位于上海虹桥火车站附近，交通便利",
        "price_range": "¥300-500",
        "total_stock": 50,
        "sku_details": [
            {"spec": "大床房", "price": 350, "stock": 20},
            {"spec": "双床房", "price": 400, "stock": 15},
            {"spec": "豪华套房", "price": 500, "stock": 5},
        ]
    }, ensure_ascii=False)


def print_banner():
    """打印欢迎信息"""
    print("=" * 60)
    print("  闲鱼智能客服 Agent - 终端对话测试")
    print("=" * 60)
    print("输入 /reset 重置对话 | 输入 /quit 退出")
    print("-" * 60)


def main():
    setup_logger("DEBUG")
    load_config()

    # 初始化 Bot（使用独立的测试数据库）
    bot = XianyuReplyBot()
    context_manager = ChatContextManager(
        max_history=100,
        db_path="data/demo_chat_history.db"
    )

    # 对话参数
    chat_id = "test_chat_001"
    user_id = "test_user_001"
    item_id = "test_item_001"
    item_desc = build_item_description()

    print_banner()
    logger.info(f"会话已创建: {chat_id}")
    print()

    while True:
        try:
            user_msg = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not user_msg:
            continue

        # 控制命令
        if user_msg.lower() in ("/quit", "/exit"):
            print("退出测试")
            break

        if user_msg == "/reset":
            # 清空当前会话
            context_manager = ChatContextManager(
                max_history=100,
                db_path="data/demo_chat_history.db"
            )
            logger.info(f"对话已重置: {chat_id}")
            print("[系统] 对话已重置，上下文已清空\n")
            continue

        # 获取对话上下文
        context = context_manager.get_context_by_chat(chat_id)

        # 生成回复
        try:
            bot_reply = bot.generate_reply(
                user_msg=user_msg,
                item_desc=item_desc,
                context=context
            )
        except Exception as e:
            logger.error(f"生成回复失败: {e}")
            bot_reply = f"抱歉，系统出现错误: {e}"

        # 检查是否需要回复
        if bot_reply == "-":
            logger.info("消息被识别为无需回复类型")
            print("[系统] 无需回复\n")
            continue

        # 保存消息到上下文
        context_manager.add_message_by_chat(chat_id, user_id, item_id, "user", user_msg)
        context_manager.add_message_by_chat(chat_id, "seller", item_id, "assistant", bot_reply)

        # 检查议价意图
        if bot.last_intent == "price":
            context_manager.increment_bargain_count_by_chat(chat_id)
            bargain_count = context_manager.get_bargain_count_by_chat(chat_id)
            logger.info(f"议价次数: {bargain_count}")

        # 输出回复
        print(f"\n客服: {bot_reply}")
        print(f"意图: {bot.last_intent}\n")


if __name__ == "__main__":
    main()
