from __future__ import annotations


def _detect_chat_format(model_type: str, tok_cfg: dict, chat_format: str) -> str:
    if chat_format != "auto":
        return chat_format

    model_type = (model_type or "").lower()
    template = str(tok_cfg.get("chat_template", "")).lower()

    if "im_start" in template or "qwen" in model_type:
        return "chatml"
    if "llama" in model_type and "<|start_header_id|>" in template:
        return "llama3"
    if "mistral" in model_type or "[inst]" in template:
        return "alpaca"
    return "plain"


def build_prompt(prompt: str, model_type: str, tok_cfg: dict, lang_mode: str, chat_format: str) -> str:
    fmt = _detect_chat_format(model_type, tok_cfg, chat_format)

    if lang_mode == "vi":
        sys_line = "Bạn là trợ lý AI. Trả lời ngắn gọn, rõ ràng, tự nhiên bằng tiếng Việt. Không trộn ngôn ngữ khác nếu không được yêu cầu."
    elif lang_mode == "en":
        sys_line = "You are an AI assistant. Reply briefly and clearly in English."
    else:
        sys_line = f"You are an AI assistant. Reply in user language ({lang_mode})."

    if fmt == "chatml":
        return (
            "<|im_start|>system\n"
            f"{sys_line}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    if fmt == "llama3":
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{sys_line}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    if fmt == "alpaca":
        return f"[INST] {prompt} [/INST]"

    return prompt
