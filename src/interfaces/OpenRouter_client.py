import hashlib
import itertools
import json
import os
import time
import threading

from openai import APIError, OpenAI

from interfaces.call_context import current_tags


class OpenRouterClient:
    """
    OpenRouter client via OpenAI-compatible Chat Completions API.
    Tracks cumulative token usage and API call counts (thread-safe).

    Env vars:
      - OPENROUTER_API_KEY: required
      - OPENROUTER_BASE_URL: optional (default https://openrouter.ai/api/v1)
      - OPENROUTER_SEND_SEED: "0" to omit the per-request seed (default: send)
      - OPENROUTER_ENABLE_THINKING: "0"/"1" passed as chat_template_kwargs.enable_thinking
        (vLLM-style servers only; unset = server default)
      - OPENROUTER_CALL_LOG: path of a JSONL file receiving one record per request attempt, with
        provenance (call_id, prompt_sha256, the sampling parameters actually sent, response id and
        call_context tags such as task_id / operator)
      - OPENROUTER_LOG_METHOD: method label written to every call-log record
      - OPENROUTER_MAX_ATTEMPTS: cap on attempts for retryable transport/server errors
        (unset = retry indefinitely, the historical behaviour)
    """

    def __init__(self, model_name: str, tokenizer_name=None):
        del tokenizer_name
        if not model_name:
            raise ValueError("model_name is required (OpenRouter model id).")

        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OpenRouter_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set in environment/.env (also tried OpenRouter_API_KEY)"
            )

        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        self.model_name = model_name
        self.call_log = os.getenv("OPENROUTER_CALL_LOG") or None
        self.log_method = os.getenv("OPENROUTER_LOG_METHOD") or None
        max_attempts = os.getenv("OPENROUTER_MAX_ATTEMPTS")
        self.max_attempts = int(max_attempts) if max_attempts else None
        # With a call log, SDK-internal retries are disabled so every attempt is recorded.
        client_kwargs = {"api_key": api_key, "base_url": base_url}
        if self.call_log:
            client_kwargs["max_retries"] = 0
        self.client = OpenAI(**client_kwargs)
        self.default_timeout = 120
        self.default_max_retries = 4

        self._lock = threading.Lock()
        self._api_calls = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0

    def _record_usage(self, completion):
        usage = getattr(completion, "usage", None)
        if usage is None:
            return
        with self._lock:
            self._api_calls += 1
            self._prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self._completion_tokens += getattr(usage, "completion_tokens", 0) or 0

    _call_ids = itertools.count()
    _call_id_lock = threading.Lock()

    _request_tls = threading.local()

    def _log_attempt(self, record):
        if not self.call_log:
            return
        request = getattr(OpenRouterClient._request_tls, "request", None) or {}
        record = {"ts": time.time(), "model": self.model_name, "method": self.log_method,
                  **current_tags(), **request, **record}
        with self._lock:
            with open(self.call_log, "a") as f:
                f.write(json.dumps(record) + "\n")

    def get_usage(self) -> dict:
        with self._lock:
            return {
                "api_calls": self._api_calls,
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "total_tokens": self._prompt_tokens + self._completion_tokens,
            }

    def generate(self, prompt, **generation_kwargs):
        return self.generate_n(prompt, n=1, **generation_kwargs)[0]

    def generate_n(self, prompt, n=1, **generation_kwargs):
        n = max(1, int(n))
        max_tokens = generation_kwargs.get("max_tokens", 1024)
        temperature = generation_kwargs.get("temperature", 0.7)
        top_p = generation_kwargs.get("top_p", 0.95)
        stop = generation_kwargs.get("stop", None)
        response_format = generation_kwargs.get("response_format")

        create_kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "n": n,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop,
        }
        if response_format is not None:
            create_kwargs["response_format"] = response_format
        # OpenAI-compatible seed (best-effort; providers may ignore). A server that honors it
        # returns identical samples for repeated n=1 calls on the same prompt.
        send_seed = os.getenv("OPENROUTER_SEND_SEED", "1") != "0"
        if send_seed and generation_kwargs.get("seed") is not None:
            create_kwargs["seed"] = int(generation_kwargs["seed"])
        thinking = os.getenv("OPENROUTER_ENABLE_THINKING")
        if thinking in ("0", "1"):
            create_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": thinking == "1"}}

        request = None
        if self.call_log:
            with OpenRouterClient._call_id_lock:
                call_id = next(OpenRouterClient._call_ids)
            request = {
                "call_id": f"{os.getpid()}-{call_id}",
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                if isinstance(prompt, str) else None,
                "seed": create_kwargs.get("seed"),
                "temperature": temperature,
                "top_p": top_p,
                "stop": stop,
            }
        OpenRouterClient._request_tls.request = request
        ans = None
        timeout = 2
        attempt = 0
        while ans is None:
            attempt += 1
            try:
                completion = self.client.chat.completions.create(**create_kwargs)
                self._record_usage(completion)
                choices = getattr(completion, "choices", None) or []
                usage = getattr(completion, "usage", None)
                if not choices:
                    raise RuntimeError("OpenRouter returned empty choices (choices is None or [])")
                outs = []
                n_reasoning_fallback = 0
                for c in choices:
                    msg = getattr(c, "message", None)
                    if msg is None:
                        outs.append("")
                        continue
                    content = getattr(msg, "content", None)
                    if content is None or content == "":
                        content = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
                        n_reasoning_fallback += bool(content)
                    outs.append(content or "")

                self._log_attempt({
                    "attempt": attempt,
                    "status": "ok",
                    "n": n,
                    "max_tokens": max_tokens,
                    "finish_reasons": [getattr(c, "finish_reason", None) for c in choices],
                    "n_empty": sum(1 for o in outs if not o),
                    "n_reasoning_fallback": n_reasoning_fallback,
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "provider": getattr(completion, "provider", None),
                    "system_fingerprint": getattr(completion, "system_fingerprint", None),
                    "seed_sent": "seed" in create_kwargs,
                    "response_id": getattr(completion, "id", None),
                    "n_choices": len(choices),
                })
                if all(o == "" for o in outs):
                    try:
                        finish_reasons = [getattr(c, "finish_reason", None) for c in choices]
                        print(
                            f"OpenRouter empty text output for model={self.model_name}, "
                            f"finish_reasons={finish_reasons}, usage={usage}"
                        )
                    except Exception:
                        pass
                return outs
            except APIError as e:
                code = getattr(e, "status_code", None)
                err_s = str(e).lower()
                if code == 401:
                    self._log_attempt({"attempt": attempt, "status": "api_error_final", "code": code, "n": n})
                    print("OpenRouter APIError 401 (unauthorized). Check OPENROUTER_API_KEY. Not retrying.")
                    raise
                if code == 403 and (
                    "limit" in err_s
                    or "quota" in err_s
                    or "exceeded" in err_s
                    or "billing" in err_s
                ):
                    self._log_attempt({"attempt": attempt, "status": "api_error_final", "code": code, "n": n})
                    print(
                        "OpenRouter 403: key limit / quota exceeded. "
                        "Add credits at https://openrouter.ai/settings/keys — not retrying."
                    )
                    raise RuntimeError(
                        "OpenRouter key limit exceeded (403). Stop the job and fix billing/limits."
                    ) from e
                if self.max_attempts is not None and code is not None and 400 <= code < 500 and code not in (408, 429):
                    # Request-level errors (e.g. context too long) do not change on retry.
                    self._log_attempt({"attempt": attempt, "status": "request_error", "code": code, "n": n,
                                       "error": str(e)[:300]})
                    print(f"Non-retryable request error {code}: {e}")
                    return ["" for _ in range(n)]
                self._log_attempt({"attempt": attempt, "status": "api_error_retry", "code": code, "n": n,
                                   "error": str(e)[:300]})
                print(e)
            except Exception as e:
                self._log_attempt({"attempt": attempt, "status": "api_error_retry", "code": None, "n": n,
                                   "error": str(e)[:300]})
                print(e)
            if self.max_attempts is not None and attempt >= self.max_attempts:
                self._log_attempt({"attempt": attempt, "status": "api_failed_final", "n": n})
                print(f"Giving up after {attempt} attempts.")
                return ["" for _ in range(n)]
            timeout = min(timeout * 2, 120)
            print(f"Will retry after {timeout} seconds ...")
            time.sleep(timeout)

        return ["" for _ in range(n)]

    def close(self):
        pass
