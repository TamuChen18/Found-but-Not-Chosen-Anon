import os
import time
import threading
from openai import OpenAI


class OpenAIClient:
    """
    A client wrapper for interacting with OpenAI chat models (e.g., GPT-4o).
    Tracks cumulative token usage and API call counts (thread-safe).
    """

    def __init__(
        self,
        model_name="gpt-4o-mini",
        tokenizer_name = None
    ):
        self.api_key = None or os.getenv("OPENAI_API_KEY")
        
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            raise Exception("OpenAI api key not set. Set API key using export OPENAI_API_KEY=<api key value>")

        self.model_name = model_name
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

    def get_usage(self) -> dict:
        with self._lock:
            return {
                "api_calls": self._api_calls,
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "total_tokens": self._prompt_tokens + self._completion_tokens,
            }

    def generate(self, prompt, **generation_kwargs):
        """Generate a single response from the LLM."""
        messages = [{"role": "user", "content": prompt}]
        ans, timeout = "", self.default_timeout
        retries = 0
        while not ans and retries <= self.default_max_retries:
            try:
                completion = self.client.with_options(
                    timeout=timeout,
                    max_retries=0,
                ).chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **generation_kwargs,
                )
                self._record_usage(completion)
                ans = completion.choices[0].message.content
                if ans:
                    return ans
            except Exception as e:
                msg = str(e)
                print(e)
                # Do not spin on billing / permanent auth failures
                if any(
                    s in msg
                    for s in (
                        "insufficient_quota",
                        "credit_balance_exhausted",
                        "invalid_api_key",
                        "Incorrect API key",
                    )
                ):
                    raise
                timeout = min(timeout * 2, 120)
                retries += 1
                print(f"Will retry after {timeout} seconds ...")
                import time as _t; _t.sleep(min(2, timeout))
        return ans

    def generate_n(self, prompt, n=1, **generation_kwargs):
        """Generate n responses from the LLM."""
        messages = [{"role": "user", "content": prompt}]
        ans, timeout = None, self.default_timeout
        retries = 0
        while ans is None and retries <= self.default_max_retries:
            try:
                completion = self.client.with_options(
                    timeout=timeout,
                    max_retries=0,
                ).chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    n=n,
                    **generation_kwargs,
                )
                self._record_usage(completion)
                ans = [choice.message.content for choice in completion.choices]
                if ans:
                    return ans
            except Exception as e:
                print(e)
                timeout = min(timeout * 2, 120)
                retries += 1
                print(f"Will retry after {timeout} seconds ...")
                import time as _t; _t.sleep(min(2, timeout))
        return ans or ["" for _ in range(n)]

    def close(self):
        pass
