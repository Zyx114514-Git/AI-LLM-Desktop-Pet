# -*- coding: utf-8 -*-
"""
LLM 管理器：开源模型接入 + QLoRA 微调
======================================

功能：
  1. 加载本地开源模型（Qwen / Llama / ChatGLM / Mistral 等，HuggingFace 格式）
  2. 4bit 量化推理（bitsandbytes），降低显存占用
  3. QLoRA 微调：用户上传数据集，微调 LoRA 适配器
  4. 加载微调后的适配器，用于推理
  5. 支持多种数据集格式：JSONL / CSV，instruction / messages / prompt-completion

依赖（可选，未安装时核心桌宠仍可运行，AI 功能提示安装）：
  pip install torch transformers peft bitsandbytes accelerate datasets trl

注意：
  - 微调需要 NVIDIA GPU（建议 >= 8GB 显存）
  - bitsandbytes 在 Windows 上需安装对应版本：pip install bitsandbytes-windows 或使用 WSL
  - 模型需为 HuggingFace 格式（包含 config.json、tokenizer.json、pytorch_model.bin 等）
"""

import json
import os
import sys
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal


# ==================== 依赖检测 ====================

def check_dependencies() -> tuple:
    """检测可选依赖是否安装，返回 (是否可用, 缺失列表)。"""
    missing = []
    for pkg, import_name in [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("peft", "peft"),
        ("accelerate", "accelerate"),
        ("datasets", "datasets"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)
    # bitsandbytes 是可选的（不装也可以不量化推理，但微调建议装）
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        missing.append("bitsandbytes (可选，用于4bit量化)")
    return (len(missing) == 0, missing)


def app_dir() -> str:
    """程序所在目录（打包后=exe目录，源码运行=脚本目录）。"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


# ==================== 默认微调参数 ====================

DEFAULT_FINETUNE_PARAMS = {
    # 训练参数
    "learning_rate": 2e-4,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 3,
    "max_steps": -1,               # -1 表示不限制，按 epochs 走
    "warmup_ratio": 0.03,
    "weight_decay": 0.0,
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "save_steps": 100,
    # LoRA 参数
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_bias": "none",
    # 量化与序列
    "load_in_4bit": True,
    "max_seq_length": 512,
    "gradient_checkpointing": True,
}


# ==================== 数据集加载 ====================

def load_dataset_file(filepath: str) -> list:
    """
    加载数据集文件，返回统一格式的样本列表。
    支持格式：
      - JSONL：每行一个 JSON 对象
      - JSON：整个文件是 JSON 数组
      - CSV：含 instruction/input/output 或 prompt/completion 列
    每个样本统一为 {"prompt": str, "response": str} 格式。
    """
    samples = []
    ext = os.path.splitext(filepath)[1].lower()

    if ext in (".jsonl", ".json"):
        with open(filepath, "r", encoding="utf-8") as f:
            if ext == ".jsonl":
                lines = [line.strip() for line in f if line.strip()]
                raw_samples = [json.loads(line) for line in lines]
            else:
                raw_samples = json.load(f)
                if isinstance(raw_samples, dict):
                    raw_samples = [raw_samples]
    elif ext == ".csv":
        import csv
        with open(filepath, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            raw_samples = list(reader)
    else:
        raise ValueError(f"不支持的文件格式: {ext}（支持 .jsonl / .json / .csv）")

    for raw in raw_samples:
        prompt, response = _parse_sample(raw)
        if prompt and response:
            samples.append({"prompt": prompt, "response": response})

    return samples


def _parse_sample(raw: dict) -> tuple:
    """将各种格式的样本解析为 (prompt, response)。"""
    # 格式1: messages 对话格式
    if "messages" in raw and isinstance(raw["messages"], list):
        user_text = ""
        assistant_text = ""
        for msg in raw["messages"]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                user_text = content
            elif role == "assistant":
                assistant_text = content
        return (user_text, assistant_text)

    # 格式2: instruction + input + output
    if "instruction" in raw or "output" in raw:
        instruction = raw.get("instruction", "")
        inp = raw.get("input", "")
        output = raw.get("output", "")
        if inp:
            prompt = f"{instruction}\n\n{inp}"
        else:
            prompt = instruction
        return (prompt, output)

    # 格式3: prompt + completion
    if "prompt" in raw and "completion" in raw:
        return (raw["prompt"], raw["completion"])

    # 格式4: question + answer
    if "question" in raw and "answer" in raw:
        return (raw["question"], raw["answer"])

    return ("", "")


# ==================== LLM 管理器 ====================

class LLMManager:
    """
    LLM 管理器：负责模型加载、推理、QLoRA 微调。
    所有耗时操作在调用方的线程中执行（UI 层用 QThread 包装）。
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_path = None
        self.adapter_path = None  # 当前加载的微调适配器路径
        self.device = "cuda" if self._cuda_available() else "cpu"
        self.is_loaded = False
        self.system_prompt = ""  # 系统提示词（由 pet.py 从 config.json 注入）

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    # ---------- 模型加载 ----------

    def load_model(self, model_path: str, load_in_4bit: bool = True) -> None:
        """
        加载基础模型。
        model_path: 本地模型目录路径（HuggingFace 格式）。
        load_in_4bit: 是否 4bit 量化加载（需要 bitsandbytes）。
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        }

        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["quantization_config"] = bnb_config
        else:
            kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        self.model.eval()
        self.is_loaded = True
        self.adapter_path = None

    def load_adapter(self, adapter_path: str) -> None:
        """加载微调后的 LoRA 适配器。"""
        from peft import PeftModel

        if not self.is_loaded:
            raise RuntimeError("请先加载基础模型")

        # 如果已加载适配器，先卸载
        if self.adapter_path:
            self.model = self.model.unload()

        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.adapter_path = adapter_path

    def unload_model(self) -> None:
        """卸载模型，释放显存。"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.is_loaded = False
        self.system_prompt = ""  # 系统提示词（由 pet.py 从 config.json 注入）
        self.adapter_path = None
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

    # ---------- 推理 ----------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        screen_context: str = "",
    ) -> str:
        """
        推理生成回复。
        自动添加对话模板（如果 tokenizer 有 chat_template）。
        screen_context: 当前屏幕OCR文字，作为 system 上下文。
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载")

        # 构造 messages：system_prompt + 屏幕内容 + 用户输入
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if screen_context:
            sys_extra = f"以下是用户当前屏幕上显示的文字（OCR识别），可参考回答：\n{screen_context}"
            if messages and messages[-1]["role"] == "system":
                messages[-1]["content"] += "\n\n" + sys_extra
            else:
                messages.append({"role": "system", "content": sys_extra})
        messages.append({"role": "user", "content": prompt})
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # 没有 chat template 时用简单格式
            ctx = f"\n[屏幕内容] {screen_context}\n" if screen_context else ""
            text = f"用户：{ctx}{prompt}\n助手："

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with __import__("torch").no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # 只取生成的部分（去掉输入）
        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        return response.strip()

    # ---------- 微调 ----------

    def finetune(
        self,
        model_path: str,
        dataset_path: str,
        output_dir: str,
        params: Optional[dict] = None,
        progress_callback=None,
    ) -> str:
        """
        QLoRA 微调，返回输出目录路径。
        progress_callback(step, total, message) 用于进度更新。
        """
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from datasets import Dataset

        p = {**DEFAULT_FINETUNE_PARAMS, **(params or {})}

        # 1. 加载数据集
        samples = load_dataset_file(dataset_path)
        if not samples:
            raise ValueError("数据集中没有有效样本，请检查格式")

        # 2. 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 3. 格式化数据集
        def format_example(example):
            messages = [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["response"]},
            ]
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False)
            except Exception:
                text = f"用户：{example['prompt']}\n助手：{example['response']}{tokenizer.eos_token}"
            return {"text": text}

        dataset = Dataset.from_list(samples)
        dataset = dataset.map(format_example, remove_columns=dataset.column_names)

        # 4. 加载模型（4bit 量化）
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=p["load_in_4bit"],
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)

        # 5. 配置 LoRA
        lora_config = LoraConfig(
            r=p["lora_r"],
            lora_alpha=p["lora_alpha"],
            lora_dropout=p["lora_dropout"],
            bias=p["lora_bias"],
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)

        if p["gradient_checkpointing"]:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

        # 6. 训练参数
        os.makedirs(output_dir, exist_ok=True)
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=p["learning_rate"],
            per_device_train_batch_size=p["per_device_train_batch_size"],
            gradient_accumulation_steps=p["gradient_accumulation_steps"],
            num_train_epochs=p["num_train_epochs"],
            max_steps=p["max_steps"],
            warmup_ratio=p["warmup_ratio"],
            weight_decay=p["weight_decay"],
            lr_scheduler_type=p["lr_scheduler_type"],
            logging_steps=p["logging_steps"],
            save_steps=p["save_steps"],
            save_total_limit=2,
            fp16=True,
            report_to="none",
            remove_unused_columns=False,
        )

        # 7. 自定义 Trainer（支持进度回调）
        from transformers import Trainer, DataCollatorForLanguageModeling

        class ProgressTrainer(Trainer):
            def __init__(self, *args, **kwargs):
                self._progress_cb = kwargs.pop("progress_callback", None)
                super().__init__(*args, **kwargs)

            def log(self, logs, **kwargs):
                super().log(logs, **kwargs)
                if self._progress_cb and "loss" in logs:
                    step = logs.get("step", 0)
                    self._progress_cb(step, -1, f"step {step}, loss={logs['loss']:.4f}")

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer = ProgressTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            progress_callback=progress_callback,
        )

        # 8. 开始训练
        if progress_callback:
            total_steps = p["max_steps"] if p["max_steps"] > 0 else int(
                len(dataset) * p["num_train_epochs"] / (p["per_device_train_batch_size"] * p["gradient_accumulation_steps"])
            )
            progress_callback(0, total_steps, f"开始微调，共 {len(samples)} 条样本，约 {total_steps} 步")

        trainer.train()

        # 9. 保存适配器
        final_dir = os.path.join(output_dir, "final")
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)

        if progress_callback:
            progress_callback(-1, -1, f"微调完成，适配器已保存到 {final_dir}")

        # 释放显存
        del model, trainer
        torch.cuda.empty_cache()

        return final_dir


# ==================== 微调工作线程 ====================

class FinetuneWorker(QThread):
    """后台执行 QLoRA 微调，通过信号更新进度。"""
    progress = pyqtSignal(int, int, str)   # current_step, total_steps, message
    finished_ok = pyqtSignal(str)           # 输出目录
    failed = pyqtSignal(str)                # 错误信息

    def __init__(
        self,
        manager: LLMManager,
        model_path: str,
        dataset_path: str,
        output_dir: str,
        params: Optional[dict] = None,
    ):
        super().__init__()
        self._manager = manager
        self._model_path = model_path
        self._dataset_path = dataset_path
        self._output_dir = output_dir
        self._params = params

    def run(self):
        try:
            def _cb(step, total, msg):
                self.progress.emit(step, total, msg)

            result = self._manager.finetune(
                model_path=self._model_path,
                dataset_path=self._dataset_path,
                output_dir=self._output_dir,
                params=self._params,
                progress_callback=_cb,
            )
            self.finished_ok.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


# ==================== 模型加载工作线程 ====================

class ModelLoadWorker(QThread):
    """后台加载模型，避免阻塞 UI。"""
    finished_ok = pyqtSignal(str)   # 模型路径
    failed = pyqtSignal(str)

    def __init__(self, manager: LLMManager, model_path: str, load_in_4bit: bool = True):
        super().__init__()
        self._manager = manager
        self._model_path = model_path
        self._load_in_4bit = load_in_4bit

    def run(self):
        try:
            self._manager.load_model(self._model_path, self._load_in_4bit)
            self.finished_ok.emit(self._model_path)
        except Exception as exc:
            self.failed.emit(str(exc))


# ==================== 推理工作线程 ====================

class GenerateWorker(QThread):
    """后台推理，避免阻塞 UI。"""
    finished_ok = pyqtSignal(str)   # 回复文本
    failed = pyqtSignal(str)

    def __init__(self, manager: LLMManager, prompt: str, max_new_tokens: int = 200, screen_context: str = ""):
        super().__init__()
        self._manager = manager
        self._prompt = prompt
        self._max_new_tokens = max_new_tokens
        self._screen = screen_context

    def run(self):
        try:
            response = self._manager.generate(self._prompt, self._max_new_tokens, screen_context=self._screen)
            self.finished_ok.emit(response)
        except Exception as exc:
            self.failed.emit(str(exc))
