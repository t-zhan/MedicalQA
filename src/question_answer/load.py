import os
import torch
from modelscope import snapshot_download
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
    if mdoel_name == 'Qwen 2.5':
        model_id = snapshot_download("qwen/Qwen2.5-14B-Instruct", 
                                    local_dir='models/Qwen2.5-14B-Instruct', 
                                    cache_dir='models/Qwen2.5-14B-Instruct/cache')
    elif mdoel_name == 'Llama 3.1':
        model_id = snapshot_download("LLM-Research/Meta-Llama-3.1-8B-Instruct", 
                                    local_dir='models/Meta-Llama-3.1-8B-Instruct', 
                                    cache_dir='models/Meta-Llama-3.1-8B-Instruct/cache')
    else:
        raise ValueError('Invalid model name, must be either "Qwen 2.5" or "Llama3.1"')
    config = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
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

    return model,tokenizer,generation_config
