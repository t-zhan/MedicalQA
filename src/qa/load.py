import os
import torch
# from modelscope import snapshot_download
from huggingface_hub import snapshot_download
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)


def load_model(mdoel_name):
    # 设置可见的 GPU
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    # model_id = snapshot_download('qwen/Qwen2.5-14B-Instruct',
    #                              local_dir='models/Qwen2.5-14B-Instruct', 
    #                              cache_dir='models/Qwen2.5-14B-Instruct/cache')
    if mdoel_name == 'Qwen2.5-14B':
        model_id = snapshot_download("qwen/Qwen2.5-14B-Instruct", 
                                    local_dir='models/Qwen2.5-14B-Instruct', 
                                    cache_dir='models/Qwen2.5-14B-Instruct/cache')
    elif mdoel_name == 'Llama3.1':
        model_id = snapshot_download("LLM-Research/Meta-Llama-3.1-8B-Instruct", 
                                    local_dir='models/Meta-Llama-3.1-8B-Instruct', 
                                    cache_dir='models/Meta-Llama-3.1-8B-Instruct/cache')
    elif mdoel_name == 'Qwen2.5-7B-LoRA':
        model_id = "models/qwen2.5-7B-LoRA"
    elif mdoel_name == 'Huatuo':
        model_id = snapshot_download("FreedomIntelligence/HuatuoGPT2-7B", 
                                    local_dir='models/HuatuoGPT2-7B', 
                                    cache_dir='models/HuatuoGPT2-7B/cache')
    else:
        raise ValueError('Invalid model name, must be "Qwen 2.5", "Llama3.1" or "Huatuo".')
    print(f"model_id: {model_id}")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map='balanced_low_0',
    )
    # 配置生成参数
    generation_config = GenerationConfig(
        temperature=0.1,
        do_sample=True,
        top_p=0.75,
        top_k=40,
        repetition_penalty=1.1,
        max_new_tokens=256
    )
    model.eval()
    model.requires_grad_(False)
    return model,tokenizer,generation_config
