import torch
import copy
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
)

class HuggingFaceClient:
    def __init__(self, model_name=None, tokenizer_name=None):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

            print(f"[HuggingFaceClient] Loading model with standard Hugging Face transformers.")
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                # quantization_config=quantization_config,
                device_map = "auto",
                dtype=torch.bfloat16,
                trust_remote_code=True,
            )

            self.model.eval()

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.model.config.eos_token_id

            self.base_generation_config = GenerationConfig.from_pretrained(self.model_name)

        except Exception as e:
            print(f"[Error] Failed to load model or tokenizer: {e}")
            raise

    def _map_generation_config(self, n, **kwargs):
        config = copy.deepcopy(self.base_generation_config)
        config.do_sample = True
        if isinstance(n, int) and n > 0:
            config.num_return_sequences = n
        config.temperature = kwargs.get("temperature", config.temperature)
        config.top_p = kwargs.get("top_p", config.top_p)
        config.top_k = kwargs.get("top_k", config.top_k)
        config.max_new_tokens = kwargs.get("max_tokens", config.max_new_tokens)
        config.stop_strings = kwargs.get("stop", getattr(config, "stop_strings", None))
        return config

    def generate(self, prompt, **generation_kwargs):
        return self.generate_n(prompt, n=1, **generation_kwargs)[0]

    def generate_n(self, prompt, n=1, **generation_kwargs):
        results = []
        batch_size = 8
        total_batches = (n + batch_size - 1) // batch_size

        try:
            # Tokenize once outside the loop
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)
            prompt_len = input_ids.shape[-1]

            for batch_idx in range(total_batches):
                current_n = min(batch_size, n - batch_idx * batch_size)
                timeout = 5
                batch_results = []

                while not batch_results:
                    try:
                        current_generation_config = self._map_generation_config(current_n, **generation_kwargs)

                        with torch.no_grad():
                            outputs = self.model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                tokenizer=self.tokenizer,  # needed for stop strings
                                generation_config=current_generation_config,
                            )

                        batch_results = [
                            self.tokenizer.decode(output[prompt_len:], skip_special_tokens=True).strip()
                            for output in outputs
                        ]
                        results.extend(batch_results)

                    except Exception as e:
                        print(f"[Batch {batch_idx+1}/{total_batches}] Generation failed: {e}")
                        timeout = min(timeout * 2, 120)
                        print(f"Retrying batch {batch_idx+1} in {timeout} seconds...")
                        time.sleep(timeout)

        except Exception as e:
            print(f"[Error] Tokenization or setup failed: {e}")

        finally:
            torch.cuda.empty_cache()

        return results

    def close(self):
        try:
            if hasattr(self, "model"):
                del self.model
            if hasattr(self, "tokenizer"):
                del self.tokenizer
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            print("[HuggingFaceClient] Cleanup complete.")
        except Exception as e:
            print(f"[Error] Failed to close HuggingFaceClient cleanly: {e}")
