# src/langchain/llms.py
from transformers import pipeline
import transformers
from torch import cuda, bfloat16

class HuggingFacePipeline:
    def __init__(self, model_id, hf_auth = "HF_AUTH"):
        self.model_id = model_id
        self.hf_auth = hf_auth
        self.pipeline = self._initialize_pipeline()

    def _initialize_pipeline(self):
        
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
        )
                
        model_config = transformers.AutoConfig.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            use_auth_token=self.hf_auth
        )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            config=model_config,
            #quantization_config=bnb_config,
            device_map='auto',
            use_auth_token=self.hf_auth
        )

        model.eval()

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            use_auth_token=self.hf_auth
        )

        return pipeline(
            model=model,
            tokenizer=tokenizer,
            return_full_text=True,
            task='text-generation',
            temperature=0.4,
            max_new_tokens=256,
            repetition_penalty=1.1
        )

    def process_data(self, prompt):
        # Example method for processing data using the initialized pipeline
        res = self.pipeline(prompt)
        return res[0]["generated_text"]

# Usage:
# llm = HuggingFacePipeline(model_id="meta-llama/Llama-2-13b-chat-hf", hf_auth="YOUR_HF_AUTH_TOKEN")
# result = llm.process_data("What are some applications of Large Language Models?")
# print(result)
