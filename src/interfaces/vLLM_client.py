import time
from openai import OpenAI

class VLLMClient:
    """
    A client wrapper for interacting with vLLM models.

    This class handles model initialization, prompt generation,
    and supports both single and multiple completions.

    Attributes:
        model_name (str): Path or name of the model to load.
    """

    def __init__(self, model_name: str):
        """
        Initialize the vLLM client.

        Args:
            model_name (str): HuggingFace model ID or local path.
        """
        try:
            self.client = OpenAI(base_url = "http://localhost:8000/v1", api_key = "token123")
            self.model_name = model_name

        except Exception as e:
            print(f"Error loading vLLM model: {e}")
            raise

    def _build_sampling_params(self, n=1, **generation_kwargs):
        """
        Build vLLM SamplingParams from kwargs.
        """
        return SamplingParams(
            n=n,
            temperature=generation_kwargs.get("temperature", 1.0),
            top_p=generation_kwargs.get("top_p", 1.0),
            max_tokens=generation_kwargs.get("max_tokens", 4096),
            stop=generation_kwargs.get("stop", None),
        )

    def generate(self, prompt: str, **generation_kwargs) -> str:
        """
        Generate a single response.

        Args:
            prompt (str): User input string.

        Returns:
            str: Model-generated response.
        """
        messages = [{"role": "user", "content": prompt}]

        ans, timeout = "", 2
        while not ans:
            try:
                time.sleep(timeout)
                completion = self.client.chat.completions.create(model=self.model_name, messages=messages, **generation_kwargs)
                ans = completion.choices[0].message.content
            except Exception as e:
                print(e)
                timeout = min(timeout * 2, 120)
                print(f"Will retry after {timeout} seconds ...")
        return ans


    def generate_n(self, prompt: str, n=1, **generation_kwargs):
        """
        Generate multiple responses.

        Args:
            prompt (str): User input string.
            n (int): Number of completions to return.

        Returns:
            List[str]: List of generated completions.
        """
        messages = [{"role": "user", "content": prompt}]
        all_answers = []

        batch_size = 16
        total_batches = (n + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            current_batch_n = min(batch_size, n - batch_idx * batch_size)
            timeout = 1
            ans = []

            while not ans:
                try:
                    # Wait briefly before retrying
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        n=current_batch_n,
                        **generation_kwargs,
                    )
                    ans = [choice.message.content for choice in completion.choices]
                    all_answers.extend(ans)

                except Exception as e:
                    print(f"[Batch {batch_idx+1}/{total_batches}] Error: {e}")
                    sleep(timeout)
                    print(f"Will retry batch {batch_idx+1} after {timeout} seconds...")

        return all_answers
    
    def close(self):
        """
        Gracefully shuts down the vLLM engine and cleans up distributed resources.
        Safe to call multiple times or across multiple processes.
        """
        pass
