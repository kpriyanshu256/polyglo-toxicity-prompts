from overrides import overrides
import joblib as jb
from tqdm.auto import tqdm, trange
import torch
import transformers
from fastchat.conversation import get_conv_template
from googleapiclient import discovery
from vllm import LLM, SamplingParams


API_KEY = 'PERSPECTIVE_API_KEY'


def trim_text(text):
    """
    Trim text if its size is exceeding Perspective API limit of 20 KB
    """
    LEN = 20480
    byte_text = text.encode("utf-8")

    if len(byte_text) > LEN:
        text = byte_text[:LEN]
        # try to decode the string, it is unsucessful we split the bytes of a multi-byte character
        while True:
            try:
                text = text.decode()
                break
            except:
                LEN-=1
                text = byte_text[:LEN]

    return text

def perspective_api_call(text, language=None):

        text = trim_text(text)

        try:
            client = discovery.build(
                "commentanalyzer",
                "v1alpha1",
                developerKey=API_KEY,
                discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                static_discovery=False,
            )

            analyze_request = {
                'comment': { 'text': text},
                'requestedAttributes': {
                                        'TOXICITY': {},
                                        'SEVERE_TOXICITY': {},
                                        'IDENTITY_ATTACK': {},
                                        'INSULT': {},
                                        'PROFANITY': {},
                                        'THREAT': {}
                                        },
                # 'languages': [language],
            }

            if language is not None:
                analyze_request['languages'] = [language]

            response = client.comments().analyze(body=analyze_request).execute()
            return response
        except Exception as e:
            print(f'Error {e}')
            return {}



class BaseModel(object):
    def get_model(self, model_name):
        pass

    def get_model_type(self, model_name):
        pass

    def get_sampling_params(self, n, max_tokens, temperature, top_p):
        pass

    def get_prompt_template(self, text):
        pass


class Mistral(BaseModel):
    @overrides
    def get_model(self, model_name):
        llm = LLM(model_name, swap_space=16, gpu_memory_utilization=0.9)
        return llm

    @overrides
    def get_model_type(self, model_name):
        # remove "mistralai/" from name
        return model_name[10:]

    @overrides
    def get_sampling_params(self, n, max_tokens, temperature, top_p):
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return sampling_params

    @overrides
    def get_prompt_template(self, text):
        template = get_conv_template("mistral")
        template.append_message(template.roles[0], text)
        template.append_message(template.roles[1], None)
        return template.get_prompt()


class MistralBase(BaseModel):
    @overrides
    def get_model(self, model_name):
        llm = LLM(model_name, swap_space=16, gpu_memory_utilization=0.9)
        return llm

    @overrides
    def get_model_type(self, model_name):
        # remove "mistralai/" from name
        return model_name[10:]

    @overrides
    def get_sampling_params(self, n, max_tokens, temperature, top_p):
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return sampling_params

    @overrides
    def get_prompt_template(self, text):
        return text


class OLMo(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    "allenai/OLMo-7B-Instruct",
                    trust_remote_code=True,
                    )

    @overrides
    def get_model(self, model_name):
        llm = LLM(model_name, trust_remote_code=True, swap_space=24,
                    gpu_memory_utilization=0.9)
        return llm

    @overrides
    def get_model_type(self, model_name):
        return model_name.split("/")[-1]

    @overrides
    def get_sampling_params(self, n, max_tokens, temperature, top_p):
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return sampling_params

    @overrides
    def get_prompt_template(self, text):
        prompt = [{'role': 'user', 'content': text}]
        inputs = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
        )
        return self.tokenizer.decode(inputs)


class Gemma(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()


    @overrides
    def get_model(self, model_name):
        llm = LLM(model_name, trust_remote_code=True, swap_space=24,
                    gpu_memory_utilization=0.9)
        return llm

    @overrides
    def get_model_type(self, model_name):
        return model_name.split("/")[-1]

    @overrides
    def get_sampling_params(self, n, max_tokens, temperature, top_p):
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return sampling_params

    @overrides
    def get_prompt_template(self, text):
        return text



class Pythia(BaseModel):
    @overrides
    def get_model(self, model_name):
        llm = LLM(model_name, swap_space=24, gpu_memory_utilization=0.9)
        return llm

    @overrides
    def get_model_type(self, model_name):
        # remove "mistralai/" from name
        return model_name.split("/")[-1]

    @overrides
    def get_sampling_params(self, n, max_tokens, temperature, top_p):
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return sampling_params

    @overrides
    def get_prompt_template(self, text):
        return text


class MistralCAI(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    'HuggingFaceH4/mistral-7b-anthropic',
                    trust_remote_code=True,
                    )
    @overrides
    def get_model(self, model_name):
        llm = LLM(model_name, swap_space=24, gpu_memory_utilization=0.9)
        return llm

    @overrides
    def get_model_type(self, model_name):
        # remove "mistralai/" from name
        return model_name.split("/")[-1]

    @overrides
    def get_sampling_params(self, n, max_tokens, temperature, top_p):
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return sampling_params

    @overrides
    def get_prompt_template(self, text):
        prompt = [{'role': 'user', 'content': text}]
        inputs = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
        )
        return self.tokenizer.decode(inputs)


class ZGemma(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    'HuggingFaceH4/zephyr-7b-gemma-v0.1',
                    trust_remote_code=True,
                    )
    @overrides
    def get_model(self, model_name):
        llm = LLM(model_name, swap_space=24, gpu_memory_utilization=0.9)
        return llm

    @overrides
    def get_model_type(self, model_name):
        # remove "mistralai/" from name
        return model_name.split("/")[-1]

    @overrides
    def get_sampling_params(self, n, max_tokens, temperature, top_p):
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return sampling_params

    @overrides
    def get_prompt_template(self, text):
        prompt = [
            {
              "role": "system", "content": "",  # Model not yet trained for follow this
            },
            {
                'role': 'user', 'content': text
            }
        ]
        inputs = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
        )
        return self.tokenizer.decode(inputs)

class Bloom(BaseModel):
    @overrides
    def get_model(self, model_name):
        llm = LLM(model_name, swap_space=24, gpu_memory_utilization=0.9)
        return llm

    @overrides
    def get_model_type(self, model_name):
        # remove "mistralai/" from name
        return model_name.split("/")[-1]

    @overrides
    def get_sampling_params(self, n, max_tokens, temperature, top_p):
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return sampling_params

    @overrides
    def get_prompt_template(self, text):
        return text


class Llama2(BaseModel):
    @overrides
    def get_model(self, model_name):
        llm = LLM(model_name,
                swap_space=32,
                # dtype=torch.bfloat16,
                gpu_memory_utilization=0.9,
                tensor_parallel_size=1,
                enforce_eager=not True)
        return llm

    @overrides
    def get_model_type(self, model_name):
        return model_name.split("/")[-1]

    @overrides
    def get_sampling_params(self, n, max_tokens, temperature, top_p):
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return sampling_params

    @overrides
    def get_prompt_template(self, text):
        template = get_conv_template("llama-2")
        template.set_system_message(
            "You are a helpful, respectful and honest assistant.")
        template.append_message(template.roles[0], text)
        template.append_message(template.roles[1], None)
        return template.get_prompt()


class Llama3_8B(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    'meta-llama/Meta-Llama-3-8B',
                    trust_remote_code=True,
                    )
    @overrides
    def get_model(self, model_name):
        llm = LLM(model_name,
                swap_space=32,
                gpu_memory_utilization=0.9,
                tensor_parallel_size=1,
                enforce_eager=not True)
        return llm

    @overrides
    def get_model_type(self, model_name):
        return model_name.split("/")[-1]

    @overrides
    def get_sampling_params(self, n, max_tokens, temperature, top_p):
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return sampling_params

    @overrides
    def get_prompt_template(self, text):
        prompt = [{'role': 'user', 'content': text}]
        inputs = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
        )
        return self.tokenizer.decode(inputs)


class Llama3_8B_Inst(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    'meta-llama/Meta-Llama-3-8B-Instruct',
                    trust_remote_code=True,
                    )
    @overrides
    def get_model(self, model_name):
        llm = LLM(model_name,
                swap_space=32,
                gpu_memory_utilization=0.9,
                tensor_parallel_size=1,
                enforce_eager=not True)
        return llm

    @overrides
    def get_model_type(self, model_name):
        return model_name.split("/")[-1]

    @overrides
    def get_sampling_params(self, n, max_tokens, temperature, top_p):
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return sampling_params

    @overrides
    def get_prompt_template(self, text):
        prompt = [{'role': 'user', 'content': text}]
        inputs = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
        )
        return self.tokenizer.decode(inputs)



class Llama3_70B_Inst(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    'meta-llama/Meta-Llama-3-70B-Instruct',
                    trust_remote_code=True,
                    )
    @overrides
    def get_model(self, model_name):
        llm = LLM(model_name,
                swap_space=32,
                gpu_memory_utilization=0.9,
                tensor_parallel_size=4,
                enforce_eager=not True)
        return llm

    @overrides
    def get_model_type(self, model_name):
        return model_name.split("/")[-1]

    @overrides
    def get_sampling_params(self, n, max_tokens, temperature, top_p):
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return sampling_params

    @overrides
    def get_prompt_template(self, text):
        prompt = [{'role': 'user', 'content': text}]
        inputs = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
        )
        return self.tokenizer.decode(inputs)


class Llama2_TP(BaseModel):
    @overrides
    def get_model(self, model_name):
        llm = LLM(model_name,
                swap_space=24,
                gpu_memory_utilization=0.9,
                tensor_parallel_size=4,
                trust_remote_code=True)
        return llm

    @overrides
    def get_model_type(self, model_name):
        return model_name.split("/")[-1]

    @overrides
    def get_sampling_params(self, n, max_tokens, temperature, top_p):
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return sampling_params

    @overrides
    def get_prompt_template(self, text):
        template = get_conv_template("llama-2")
        template.set_system_message(
            "You are a helpful, respectful and honest assistant.")
        template.append_message(template.roles[0], text)
        template.append_message(template.roles[1], None)
        return template.get_prompt()


class Vicuna(BaseModel):
    @overrides
    def get_model(self, model_name):
        llm = LLM(model_name, swap_space=20, gpu_memory_utilization=0.9)
        return llm

    @overrides
    def get_model_type(self, model_name):
        return model_name.split('/')[-1]

    @overrides
    def get_sampling_params(self, n, max_tokens, temperature, top_p):
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return sampling_params

    @overrides
    def get_prompt_template(self, text):
        template = get_conv_template("vicuna_v1.1")
        template.append_message(template.roles[0], text)
        template.append_message(template.roles[1], None)
        return template.get_prompt()


class Zephyr(BaseModel):
    @overrides
    def get_model(self, model_name):
        llm = LLM(model_name, swap_space=24, gpu_memory_utilization=0.9)
        return llm

    @overrides
    def get_model_type(self, model_name):
        return model_name[14:]

    @overrides
    def get_sampling_params(self, n, max_tokens, temperature, top_p):
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return sampling_params

    @overrides
    def get_prompt_template(self, text):
        template = get_conv_template("zephyr")
        template.set_system_message(
            "You are a helpful, respectful and honest assistant.")
        template.append_message(template.roles[0], text)
        template.append_message(template.roles[1], None)
        return template.get_prompt()

class StableLM(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    'stabilityai/stablelm-2-1_6b',
                    trust_remote_code=True,
                    )

    @overrides
    def get_model(self, model_name):
        llm = LLM(model_name, trust_remote_code=True, swap_space=24,
                    gpu_memory_utilization=0.9)
        return llm

    @overrides
    def get_model_type(self, model_name):
        return model_name.split("/")[-1]

    @overrides
    def get_sampling_params(self, n, max_tokens, temperature, top_p):
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return sampling_params

    @overrides
    def get_prompt_template(self, text):
        return text

class StableLMInst(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    'stabilityai/stablelm-2-zephyr-1_6b',
                    trust_remote_code=True,
                    )


    @overrides
    def get_model(self, model_name):
        llm = LLM(model_name, trust_remote_code=True, swap_space=24,
                    gpu_memory_utilization=0.9)
        return llm

    @overrides
    def get_model_type(self, model_name):
        return model_name.split("/")[-1]

    @overrides
    def get_sampling_params(self, n, max_tokens, temperature, top_p):
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return sampling_params

    @overrides
    def get_prompt_template(self, text):
        prompt = [{'role': 'user', 'content': text}]
        inputs = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
        )
        return self.tokenizer.decode(inputs)


class MPTInst(BaseModel):
    @overrides
    def get_model(self, model_name):
        llm = LLM(model_name,
                    swap_space=24,
                    gpu_memory_utilization=0.9,
                    trust_remote_code=True,
                )
        return llm

    @overrides
    def get_model_type(self, model_name):
        return model_name.split("/")[-1]

    @overrides
    def get_sampling_params(self, n, max_tokens, temperature, top_p):
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return sampling_params

    @overrides
    def get_prompt_template(self, text):
        template = get_conv_template("mpt-30b-instruct")
        template.append_message(template.roles[0], text)
        template.append_message(template.roles[1], None)
        return template.get_prompt()

class MPT(BaseModel):
    @overrides
    def get_model(self, model_name):
        llm = LLM(model_name,
                    swap_space=24,
                    gpu_memory_utilization=0.9,
                    trust_remote_code=True,
                )
        return llm

    @overrides
    def get_model_type(self, model_name):
        return model_name.split("/")[-1]

    @overrides
    def get_sampling_params(self, n, max_tokens, temperature, top_p):
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return sampling_params

    @overrides
    def get_prompt_template(self, text):
        return text

class Tulu(BaseModel):
    @overrides
    def get_model(self, model_name):
        llm = LLM(model_name,
                    swap_space=24,
                    gpu_memory_utilization=0.9,
                    trust_remote_code=True
                )
        return llm

    @overrides
    def get_model_type(self, model_name):
        return model_name.split("/")[-1]

    @overrides
    def get_sampling_params(self, n, max_tokens, temperature, top_p):
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return sampling_params

    @overrides
    def get_prompt_template(self, text):
        return f"<|user|>\n{text}\n<|assistant|>"


class Tulu30B(BaseModel):
    @overrides
    def get_model(self, model_name):
        llm = LLM(model_name,
                    swap_space=24,
                    gpu_memory_utilization=0.9,
                    tensor_parallel_size=2,
                    trust_remote_code=True
                )
        return llm

    @overrides
    def get_model_type(self, model_name):
        return model_name.split("/")[-1]

    @overrides
    def get_sampling_params(self, n, max_tokens, temperature, top_p):
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return sampling_params

    @overrides
    def get_prompt_template(self, text):
        return f"<|user|>\n{text}\n<|assistant|>"


class Tulu70B(BaseModel):
    @overrides
    def get_model(self, model_name):
        llm = LLM(model_name,
                    swap_space=24,
                    gpu_memory_utilization=0.9,
                    tensor_parallel_size=4,
                    trust_remote_code=True
                )
        return llm

    @overrides
    def get_model_type(self, model_name):
        return model_name.split("/")[-1]

    @overrides
    def get_sampling_params(self, n, max_tokens, temperature, top_p):
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return sampling_params

    @overrides
    def get_prompt_template(self, text):
        return f"<|user|>\n{text}\n<|assistant|>"


class Haathi(BaseModel):
    @overrides
    def get_model(self, model_name):
        llm = LLM(model_name,
                    swap_space=24,
                    gpu_memory_utilization=0.9,
                    trust_remote_code=True
                )
        llm.set_tokenizer(transformers.LlamaTokenizer.from_pretrained(model_name))
        return llm

    @overrides
    def get_model_type(self, model_name):
        return model_name.split("/")[-1]

    @overrides
    def get_sampling_params(self, n, max_tokens, temperature, top_p):
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return sampling_params

    @overrides
    def get_prompt_template(self, text):
        return text


class HaathiInst(BaseModel):
    @overrides
    def get_model(self, model_name):
        llm = LLM(model_name,
                    swap_space=24,
                    gpu_memory_utilization=0.9,
                    trust_remote_code=True
                )
        llm.set_tokenizer(transformers.LlamaTokenizer.from_pretrained(model_name))
        return llm

    @overrides
    def get_model_type(self, model_name):
        return model_name.split("/")[-1]

    @overrides
    def get_sampling_params(self, n, max_tokens, temperature, top_p):
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return sampling_params

    @overrides
    def get_prompt_template(self, text):

        def create_prompt_with_chat_format(messages, bos="<s>", eos="</s>", add_bos=True):
            formatted_text = ""
            for message in messages:
                if message["role"] == "system":
                    formatted_text += "<|system|>\n" + message["content"] + "\n"
                elif message["role"] == "user":
                    formatted_text += "<|user|>\n" + message["content"] + "\n"
                elif message["role"] == "assistant":
                    formatted_text += "<|assistant|>\n" + message["content"].strip() + eos + "\n"
                else:
                    raise ValueError(
                        "Tulu chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(
                            message["role"]
                        )
                    )
            formatted_text += "<|assistant|>\n"
            formatted_text = bos + formatted_text if add_bos else formatted_text
            return formatted_text

        return create_prompt_with_chat_format([{"role": "user", "content": text}], add_bos=False)



class Qwen(BaseModel):
    @overrides
    def get_model(self, model_name):
        llm = LLM(model_name,
                    swap_space=24,
                    gpu_memory_utilization=0.9,
                    trust_remote_code=True
                )
        return llm

    @overrides
    def get_model_type(self, model_name):
        return model_name.split("/")[-1]

    @overrides
    def get_sampling_params(self, n, max_tokens, temperature, top_p):
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return sampling_params

    @overrides
    def get_prompt_template(self, text):
        template = get_conv_template("qwen-7b-chat")
        template.append_message(template.roles[0], text)
        template.append_message(template.roles[1], None)
        return template.get_prompt()


class Yi(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    "01-ai/Yi-6B-Chat",
                    trust_remote_code=True,
                    )

    @overrides
    def get_model(self, model_name):
        llm = LLM(model_name, trust_remote_code=True, swap_space=24,
                    gpu_memory_utilization=0.9)
        return llm

    @overrides
    def get_model_type(self, model_name):
        return model_name.split("/")[-1]

    @overrides
    def get_sampling_params(self, n, max_tokens, temperature, top_p):
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return sampling_params

    @overrides
    def get_prompt_template(self, text):
        prompt = [{'role': 'user', 'content': text}]
        inputs = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
        )
        return self.tokenizer.decode(inputs)


class TinyLlama(BaseModel):
    @overrides
    def get_model(self, model_name):
        llm = LLM(model_name,
                    swap_space=24,
                    gpu_memory_utilization=0.9,
                    trust_remote_code=True
                )
        return llm

    @overrides
    def get_model_type(self, model_name):
        return model_name.split("/")[-1]

    @overrides
    def get_sampling_params(self, n, max_tokens, temperature, top_p):
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return sampling_params

    @overrides
    def get_prompt_template(self, text):
        return f"<s>[INST] {text} [/INST]"



class GEITje(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    'Rijgersberg/GEITje-7B',
                    trust_remote_code=True,
                    )


    @overrides
    def get_model(self, model_name):
        llm = LLM(model_name, trust_remote_code=True, swap_space=24,
                    gpu_memory_utilization=0.9)
        return llm

    @overrides
    def get_model_type(self, model_name):
        return model_name.split("/")[-1]

    @overrides
    def get_sampling_params(self, n, max_tokens, temperature, top_p):
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return sampling_params

    @overrides
    def get_prompt_template(self, text):
        prompt = [{'role': 'user', 'content': text}]
        inputs = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
        )
        return self.tokenizer.decode(inputs)


class SwallowInst(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @overrides
    def get_model(self, model_name):
        llm = LLM(model_name, trust_remote_code=True, swap_space=24,
                    gpu_memory_utilization=0.9)
        return llm

    @overrides
    def get_model_type(self, model_name):
        return model_name.split("/")[-1]

    @overrides
    def get_sampling_params(self, n, max_tokens, temperature, top_p):
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return sampling_params

    @overrides
    def get_prompt_template(self, text):
        PROMPT_DICT = {
            "prompt_input": (
                "以下に、あるタスクを説明する指示があり、それに付随する入力が更なる文脈を提供しています。"
                "リクエストを適切に完了するための回答を記述してください。\n\n"
                "### 指示:\n{instruction}\n\n### 入力:\n{input}\n\n### 応答:"

            ),
            "prompt_no_input": (
                "以下に、あるタスクを説明する指示があります。"
                "リクエストを適切に完了するための回答を記述してください。\n\n"
                "### 指示:\n{instruction}\n\n### 応答:"
            ),
        }

        def create_prompt(instruction, input=None):
            """
            Generates a prompt based on the given instruction and an optional input.
            If input is provided, it uses the 'prompt_input' template from PROMPT_DICT.
            If no input is provided, it uses the 'prompt_no_input' template.

            Args:
                instruction (str): The instruction describing the task.
                input (str, optional): Additional input providing context for the task. Default is None.

            Returns:
                str: The generated prompt.
            """
            if input:
                # Use the 'prompt_input' template when additional input is provided
                return PROMPT_DICT["prompt_input"].format(instruction=instruction, input=input)
            else:
                # Use the 'prompt_no_input' template when no additional input is provided
                return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)

        return create_prompt(text)


class SwallowInstTP(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @overrides
    def get_model(self, model_name):
        llm = LLM(model_name,
                    trust_remote_code=True,
                    swap_space=24,
                    tensor_parallel_size=4,
                    gpu_memory_utilization=0.9)
        return llm

    @overrides
    def get_model_type(self, model_name):
        return model_name.split("/")[-1]

    @overrides
    def get_sampling_params(self, n, max_tokens, temperature, top_p):
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return sampling_params

    @overrides
    def get_prompt_template(self, text):
        PROMPT_DICT = {
            "prompt_input": (
                "以下に、あるタスクを説明する指示があり、それに付随する入力が更なる文脈を提供しています。"
                "リクエストを適切に完了するための回答を記述してください。\n\n"
                "### 指示:\n{instruction}\n\n### 入力:\n{input}\n\n### 応答:"

            ),
            "prompt_no_input": (
                "以下に、あるタスクを説明する指示があります。"
                "リクエストを適切に完了するための回答を記述してください。\n\n"
                "### 指示:\n{instruction}\n\n### 応答:"
            ),
        }

        def create_prompt(instruction, input=None):
            """
            Generates a prompt based on the given instruction and an optional input.
            If input is provided, it uses the 'prompt_input' template from PROMPT_DICT.
            If no input is provided, it uses the 'prompt_no_input' template.

            Args:
                instruction (str): The instruction describing the task.
                input (str, optional): Additional input providing context for the task. Default is None.

            Returns:
                str: The generated prompt.
            """
            if input:
                # Use the 'prompt_input' template when additional input is provided
                return PROMPT_DICT["prompt_input"].format(instruction=instruction, input=input)
            else:
                # Use the 'prompt_no_input' template when no additional input is provided
                return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)

        return create_prompt(text)


class Swallow(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @overrides
    def get_model(self, model_name):
        llm = LLM(model_name, trust_remote_code=True, swap_space=24,
                    gpu_memory_utilization=0.9)
        return llm

    @overrides
    def get_model_type(self, model_name):
        return model_name.split("/")[-1]

    @overrides
    def get_sampling_params(self, n, max_tokens, temperature, top_p):
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return sampling_params

    @overrides
    def get_prompt_template(self, text):
        return text


class GPT(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @overrides
    def get_model(self, model_name):
        llm = LLM(model_name, trust_remote_code=True, swap_space=32,
                    gpu_memory_utilization=0.9)
        return llm

    @overrides
    def get_model_type(self, model_name):
        return model_name.split("/")[-1]

    @overrides
    def get_sampling_params(self, n, max_tokens, temperature, top_p):
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return sampling_params

    @overrides
    def get_prompt_template(self, text):
        return text


class Mixtral(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    'mistralai/Mixtral-8x7B-Instruct-v0.1',
                    trust_remote_code=True,
                    )


    @overrides
    def get_model(self, model_name):
        llm = LLM(model_name, trust_remote_code=True, swap_space=32,
                    tensor_parallel_size=4,
                    gpu_memory_utilization=0.9)
        return llm

    @overrides
    def get_model_type(self, model_name):
        return model_name.split("/")[-1]

    @overrides
    def get_sampling_params(self, n, max_tokens, temperature, top_p):
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return sampling_params

    @overrides
    def get_prompt_template(self, text):
        prompt = [{'role': 'user', 'content': text}]
        inputs = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
        )
        return self.tokenizer.decode(inputs)


def eval_toxicity(texts, language=None):
    scores = []
    RATE = 1000
    for start_idx in range(0, len(texts), RATE):
        end_idx = start_idx + RATE
        if end_idx > len(texts):
            end_idx = len(texts)

        batch_scores = jb.Parallel(-1)(jb.delayed(perspective_api_call)(x, language) \
                for x in tqdm(texts[start_idx:end_idx], total=(end_idx-start_idx), disable=not True))
        scores.extend(batch_scores)
    return scores


def eval_toxicity_lang(texts, languages):
    scores = []
    RATE = 1000
    for start_idx in trange(0, len(texts), RATE):
        end_idx = start_idx + RATE

        if end_idx > len(texts):
            end_idx = len(texts)

        batch_scores = jb.Parallel(-1)(jb.delayed(perspective_api_call)(x, lang) \
                for x, lang in tqdm(zip(texts[start_idx:end_idx], languages[start_idx:end_idx]), total=(end_idx-start_idx), disable=not True))
        scores.extend(batch_scores)
    return scores


def get_toxicity_score(x):
    try:
        return x['attributeScores']['TOXICITY']['summaryScore']['value']
    except:
        return -1
