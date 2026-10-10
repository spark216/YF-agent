"""
脚本2: BGE-M3 微调训练 - 意图分类模型

流程:
  1. 加载标注数据集 (prepare_intent_dataset.py 输出)
  2. 构建 PyTorch Dataset + DataLoader
  3. 加载 BGE-M3 + 分类头
  4. 微调训练 (Early Stopping + 学习率 Warmup)
  5. 评估训练集准确率
  6. 保存微调后模型到 output/train_model/intent_model/

注意: 训练时会在内存中存储优化器状态(momentum+variance)，
      约占用 8GB GPU显存/系统内存，这是正常现象。
      所有临时缓存已设置到 D 盘，避免占用 C 盘。
"""
import os

# === 设置所有缓存到 D 盘，避免占用 C 盘 ===
os.environ["HF_HOME"] = r"D:\config\huggingface_cache"
os.environ["HF_HUB_CACHE"] = r"D:\config\huggingface_cache\hub"
os.environ["HF_TOKENIZERS_CACHE"] = r"D:\config\huggingface_cache\tokenizers"
os.environ["TORCH_HOME"] = r"D:\config\torch_cache"

import json
import time
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, accuracy_score

import intent_config as config


# ============================================================
# 1. 数据集定义
# ============================================================

class IntentDataset(Dataset):
    """意图分类数据集"""

    def __init__(self, texts, labels, tokenizer, max_length=config.MAX_LENGTH):
        """
        Args:
            texts: 文本列表
            labels: 标签列表 (0~3)
            tokenizer: AutoTokenizer
            max_length: 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def collate_fn(batch):
    """DataLoader collate function"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ============================================================
# 2. 模型定义
# ============================================================

class IntentClassifier(torch.nn.Module):
    """
    基于BGE-M3的意图分类器

    架构: BGE-M3 (mean pooling) -> Linear -> softmax

    推理速度与原始BGE-M3 embedding完全一致
    (仅增加一层线性投影，开销可忽略)
    """

    def __init__(self, model_name=config.MODEL_NAME, num_labels=config.NUM_INTENTS):
        super().__init__()

        # 加载预训练BGE-M3
        self.bert = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.config = self.bert.config

        # 分类头
        hidden_size = self.config.hidden_size
        self.classifier = torch.nn.Linear(hidden_size, num_labels)
        self.dropout = torch.nn.Dropout(0.1)

        # 初始化分类头
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        self.classifier.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        """前向传播"""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Mean pooling over valid tokens
        token_embeddings = outputs.last_hidden_state  # (batch, seq_len, hidden)
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        # 分类
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

    def predict(self, input_ids, attention_mask):
        """预测 (返回概率分布)"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return preds, probs


# ============================================================
# 3. 训练流程
# ============================================================

def train_model():
    """完整训练流程"""

    # 设置随机种子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载标注数据
    print(f"\n加载标注数据: {config.SAMPLED_LABELS_PATH}")
    with open(config.SAMPLED_LABELS_PATH, "r", encoding="utf-8") as f:
        labeled_data = json.load(f)

    print(f"  标注数据总量: {len(labeled_data)} 条")
    for intent in config.INTENT_NAMES:
        cnt = sum(1 for d in labeled_data if d["intent"] == intent)
        print(f"    [{intent}] {cnt} 条")

    # 划分训练集/验证集 (80/20)
    random.shuffle(labeled_data)
    split = int(len(labeled_data) * 0.8)
    train_data = labeled_data[:split]
    val_data = labeled_data[split:]

    print(f"\n  训练集: {len(train_data)} 条")
    print(f"  验证集: {len(val_data)} 条")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)

    # 构建数据集
    train_dataset = IntentDataset(
        texts=[d["text"] for d in train_data],
        labels=[config.INTENT_TO_IDX[d["intent"]] for d in train_data],
        tokenizer=tokenizer,
    )
    val_dataset = IntentDataset(
        texts=[d["text"] for d in val_data],
        labels=[config.INTENT_TO_IDX[d["intent"]] for d in val_data],
        tokenizer=tokenizer,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn,
    )

    # 加载模型
    print(f"\n加载模型: {config.MODEL_NAME}")
    model = IntentClassifier()
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params / 1e6:.1f}M")
    print(f"  可训练参数: {trainable_params / 1e6:.1f}M")

    # 优化器 + Scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": config.TRAIN_WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.TRAIN_LR)

    total_steps = len(train_loader) * config.TRAIN_EPOCHS
    warmup_steps = int(total_steps * config.TRAIN_WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    criterion = torch.nn.CrossEntropyLoss()

    # 训练循环
    print(f"\n开始训练: {config.TRAIN_EPOCHS} epochs, {total_steps} steps")
    print(f"  学习率: {config.TRAIN_LR}, Warmup: {warmup_steps} steps")
    print("-" * 60)

    best_val_acc = 0.0
    best_model_state = None
    patience = 3
    no_improve = 0

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(config.TRAIN_EPOCHS):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        epoch_start = time.time()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN_MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            if (step + 1) % 10 == 0 or step == 0:
                avg_loss = train_loss / (step + 1)
                acc = train_correct / train_total
                print(f"    Epoch {epoch+1} Step {step+1}/{len(train_loader)} "
                      f"Loss: {avg_loss:.4f} Train Acc: {acc:.4f}")

        train_acc = train_correct / train_total
        train_time = time.time() - epoch_start
        print(f"  Epoch {epoch+1} 完成 | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Time: {train_time:.1f}s")

        # --- 验证阶段 ---
        val_acc = evaluate(model, val_loader, device)
        print(f"  验证集准确率: {val_acc:.4f}")

        # Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
            print(f"  ** 新最佳验证准确率: {best_val_acc:.4f} **")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early Stopping: 验证准确率 {patience} 个epoch未提升")
                break

        # 保存checkpoint
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)

    # 恢复最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"\n恢复最佳模型 (验证准确率: {best_val_acc:.4f})")

    # --- 最终评估 (完整训练集) ---
    print("\n" + "=" * 60)
    print("  最终评估")
    print("=" * 60)

    full_train_dataset = IntentDataset(
        texts=[d["text"] for d in labeled_data],
        labels=[config.INTENT_TO_IDX[d["intent"]] for d in labeled_data],
        tokenizer=tokenizer,
    )
    full_loader = DataLoader(
        full_train_dataset, batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn,
    )

    # 收集所有预测
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in full_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        labels=list(range(config.NUM_INTENTS)),
        target_names=config.INTENT_NAMES,
        digits=4,
    )

    print(f"\n训练集总准确率: {acc:.4f}")
    print(f"\n分类报告:\n{report}")

    # 保存最终模型
    save_model(model, tokenizer)

    print("\n" + "=" * 60)
    print("  模型训练完成!")
    print(f"  最终准确率: {acc:.4f}")
    print(f"  模型保存路径: {config.MODEL_SAVE_DIR}")
    print("=" * 60)
    print("\n  下一步: 运行 python intent_predictor.py 进行推理测试")


def evaluate(model, data_loader, device):
    """在验证集上评估"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0


def save_model(model, tokenizer):
    """
    保存模型

    保存内容:
      - pytorch_model.bin: 完整模型权重
      - config.json: 模型配置
      - intent_config.json: 意图分类配置 (名称映射、max_length等)
      - tokenizer相关文件
    """
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)

    # 保存PyTorch模型权重
    torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_DIR, "pytorch_model.bin"))
    print(f"  模型权重已保存: {config.MODEL_SAVE_DIR}/pytorch_model.bin")

    # 保存BGE-M3配置
    model.config.save_pretrained(config.MODEL_SAVE_DIR)
    print(f"  模型配置已保存")

    # 保存Tokenizer (用于推理)
    tokenizer.save_pretrained(config.MODEL_SAVE_DIR)
    print(f"  Tokenizer已保存")

    # 保存意图分类配置
    intent_config = {
        "intent_names": config.INTENT_NAMES,
        "intent_to_idx": config.INTENT_TO_IDX,
        "idx_to_intent": {str(v): k for k, v in config.INTENT_TO_IDX.items()},
        "num_intents": config.NUM_INTENTS,
        "max_length": config.MAX_LENGTH,
        "model_name": config.MODEL_NAME,
    }
    with open(os.path.join(config.MODEL_SAVE_DIR, "intent_config.json"), "w", encoding="utf-8") as f:
        json.dump(intent_config, f, ensure_ascii=False, indent=2)
    print(f"  意图配置已保存")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    train_model()
