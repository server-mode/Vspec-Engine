from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from vspec_runner import VspecRunArgs, run_once


def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    if not messages:
        return ""
    lines: list[str] = []
    for m in messages:
        role = str(m.get("role", "user")).strip()
        content = str(m.get("content", "")).strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


class VspecOpenAIHandler(BaseHTTPRequestHandler):
    server_version = "VspecServer/1.0"

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            app = self.server.app
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": app.model_label,
                            "object": "model",
                            "owned_by": "vspec",
                        }
                    ],
                },
            )
            return

        if self.path == "/health":
            self._send_json(200, {"ok": True})
            return

        self._send_json(404, {"error": {"message": "Not found", "type": "not_found"}})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._send_json(404, {"error": {"message": "Not found", "type": "not_found"}})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            self._send_json(400, {"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}})
            return

        app = self.server.app
        messages = payload.get("messages", [])
        prompt = _messages_to_prompt(messages)
        if not prompt:
            prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            self._send_json(400, {"error": {"message": "messages or prompt is required", "type": "invalid_request_error"}})
            return

        max_tokens = int(payload.get("max_tokens", app.max_tokens))
        temperature = float(payload.get("temperature", app.temperature))
        top_k = int(payload.get("top_k", app.top_k))

        req = VspecRunArgs(
            model=app.model,
            prompt=prompt,
            device=app.device,
            fused_bits=app.fused_bits,
            target_bits=app.target_bits,
            max_layers=app.max_layers,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=app.repetition_penalty,
            repeat_window=app.repeat_window,
            no_repeat_ngram=app.no_repeat_ngram,
            speed_preset=app.speed_preset,
            lang=app.lang,
            stream=False,
        )
        result = run_once(req)
        if not result["ok"]:
            self._send_json(
                500,
                {
                    "error": {
                        "message": "runtime generation failed",
                        "type": "server_error",
                        "details": result["stderr"][-800:],
                    }
                },
            )
            return

        text = str(result["text"] or "").strip()
        response = {
            "id": "chatcmpl-vspec",
            "object": "chat.completion",
            "created": 0,
            "model": app.model_label,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": int(result["metrics"].get("tokens_prompt", "0") or 0),
                "completion_tokens": int(result["metrics"].get("tokens_generated", "0") or 0),
                "total_tokens": int(result["metrics"].get("tokens_total", "0") or 0),
            },
        }
        self._send_json(200, response)


class VspecServerApp:
    def __init__(self, args):
        self.model = args.model
        self.model_label = args.model_label or args.model
        self.device = args.device
        self.fused_bits = args.fused_bits
        self.target_bits = args.target_bits
        self.max_layers = args.max_layers
        self.max_tokens = args.max_tokens
        self.temperature = args.temperature
        self.top_k = args.top_k
        self.repetition_penalty = args.repetition_penalty
        self.repeat_window = args.repeat_window
        self.no_repeat_ngram = args.no_repeat_ngram
        self.speed_preset = args.speed_preset
        self.lang = args.lang


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI-compatible HTTP server for Vspec runtime")
    parser.add_argument("--model", required=True, help="Path to .vspec or model directory")
    parser.add_argument("--model-label", default="", help="Model label returned by /v1/models")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)

    parser.add_argument("--device", default="", choices=["", "cpu", "cuda", "cuda-native", "torch-cuda"])
    parser.add_argument("--fused-bits", type=int, choices=[0, 3, 4], default=None)
    parser.add_argument("--target-bits", type=int, choices=[0, 2, 3, 4], default=None)
    parser.add_argument("--max-layers", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--repeat-window", type=int, default=64)
    parser.add_argument("--no-repeat-ngram", type=int, default=3)
    parser.add_argument("--speed-preset", default="fast", choices=["normal", "fast", "ultra"])
    parser.add_argument("--lang", default="auto", choices=["auto", "vi", "en"])
    args = parser.parse_args()

    app = VspecServerApp(args)
    httpd = ThreadingHTTPServer((args.host, args.port), VspecOpenAIHandler)
    httpd.app = app

    print(f"[vspec-server] listening=http://{args.host}:{args.port}")
    print(f"[vspec-server] model={args.model}")
    print("[vspec-server] endpoints=/v1/models,/v1/chat/completions,/health")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
