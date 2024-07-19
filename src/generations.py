import argparse
import json
import os
import sys
import pandas as pd
from tqdm.auto import tqdm
import joblib as jb
from pathlib import Path
from datasets import load_dataset


from models import *


os.environ["TOKENIZERS_PARALLELISM"] = "False"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="mistralai/Mistral-7B-Instruct-v0.1",
        type=str,
        required=False,
        help="LLM",
    )
    parser.add_argument(
        "--language", default="en", type=str, required=False, help="Evaluation language"
    )
    parser.add_argument(
        "--ds_config",
        default="ptp_small",
        type=str,
        required=False,
        help="Dataset config `ptp_small`, `ptp_full` or `wildchat`",
    )

    parser.add_argument(
        "--output_dir",
        default="continuations",
        type=str,
        required=False,
        help="Output directory",
    )
    parser.add_argument(
        "--n", default=10, type=int, required=False, help="Number of continuations"
    )
    parser.add_argument(
        "--max_tokens",
        default=512,
        type=int,
        required=False,
        help="Maximum tokens generated",
    )
    parser.add_argument(
        "--temperature", default=0.7, type=float, required=False, help="Temperature"
    )
    parser.add_argument("--top_p", default=1, type=float, required=False, help="top_p")

    args = parser.parse_args()
    print("args:", args)

    if "_" in args.ds_config:
        ds_config, ds_split = args.ds_config.split("_")
    else:
        ds_config, ds_split = args.ds_config, "wildchat"

    prompts_df = load_dataset(
        "ToxicityPrompts/PolygloToxicityPrompts", f"{ds_config}-{args.language}"
    )
    prompts_df = prompts_df[ds_split].to_pandas()

    texts = prompts_df.prompt.to_list()
    print("number of prompts:", len(texts))

    MODEL_TYPE = {
        "meta-llama/Llama-2-7b-hf": Llama2,
        "meta-llama/Llama-2-7b-chat-hf": Llama2,
        "meta-llama/Llama-2-70b-chat-hf": Llama2_TP,
        "meta-llama/Llama-2-13b-hf": Llama2,
        "meta-llama/Llama-2-13b-chat-hf": Llama2,
        "mistralai/Mistral-7B-v0.1": Mistral,
        "mistralai/Mistral-7B-Instruct-v0.1": Mistral,
        "mistralai/Mistral-7B-Instruct-v0.2": Mistral,
        "lmsys/vicuna-7b-v1.5": Vicuna,
        "HuggingFaceH4/zephyr-7b-beta": Zephyr,
        "stabilityai/stablelm-2-zephyr-1_6b": StableLMInst,
        "/data/datasets/models/huggingface/meta-llama/Llama-2-7b-chat-hf": Llama2,
        "/data/datasets/models/huggingface/lmsys/vicuna-7b-v1.5": Vicuna,
        "Rijgersberg/GEITje-7B": Mistral,
        "HuggingFaceH4/zephyr-7b-beta": Zephyr,
        "mosaicml/mpt-7b": MPT,
        "mosaicml/mpt-7b-instruct": MPTInst,
        "stabilityai/stablelm-2-1_6b": StableLM,
        "/data/datasets/models/hf_cache/models--meta-llama--Llama-2-7b-hf": Llama2,
        "allenai/tulu-2-dpo-7b": Tulu,
        "allenai/tulu-2-7b": Tulu,
        "sarvamai/OpenHathi-7B-Hi-v0.1-Base": Haathi,
        "ai4bharat/Airavata": HaathiInst,
        "Qwen/Qwen-7B-Chat": Qwen,
        "01-ai/Yi-6B-Chat": Yi,
        "abideen/Heimer-dpo-TinyLlama-1.1B": TinyLlama,
        "abideen/Heimer-ipo-TinyLlama-1.1B": TinyLlama,
        "abideen/Heimer-kto-TinyLlama-1.1B": TinyLlama,
        "OrionStarAI/Orion-14B-Chat": Orion,
        "meta-llama/Llama-13-7b-hf": Llama2,
        "Rijgersberg/GEITje-7B": GEITje,
        "tokyotech-llm/Swallow-7b-hf": Swallow,
        "tokyotech-llm/Swallow-7b-instruct-hf": SwallowInst,
        "core42/jais-13b-chat": Jais,
        "ai-forever/ruGPT-3.5-13B": GPT,
        "JaeyeonKang/CCK-v2.0-DPO": CCK_DPO,
        "mistralai/Mistral-7B-Instruct-v0.2": Mistral,
        "HuggingFaceH4/mistral-7b-anthropic": MistralCAI,
        "allenai/tulu-2-13b": Tulu,
        "allenai/tulu-2-dpo-13b": Tulu,
        "allenai/tulu-2-dpo-70b": Tulu70B,
        "tokyotech-llm/Swallow-13b-instruct-hf": SwallowInst,
        "tokyotech-llm/Swallow-70b-instruct-hf": SwallowInstTP,
        "EleutherAI/pythia-14m": Pythia,
        "EleutherAI/pythia-70m": Pythia,
        "EleutherAI/pythia-160m": Pythia,
        "EleutherAI/pythia-410m": Pythia,
        "EleutherAI/pythia-1b": Pythia,
        "EleutherAI/pythia-1.4b": Pythia,
        "EleutherAI/pythia-2.8b": Pythia,
        "EleutherAI/pythia-6.9b": Pythia,
        "EleutherAI/pythia-12b": Pythia,
        "EleutherAI/pythia-560m": Pythia,
        "bigscience/bloomz-560m": Bloom,
        "bigscience/bloomz-1b1": Bloom,
        "bigscience/bloomz-1b7": Bloom,
        "bigscience/bloomz-3b": Bloom,
        "bigscience/bloomz-7b1": Bloom,
        "bigscience/bloom-560m": Bloom,
        "bigscience/bloom-1b1": Bloom,
        "bigscience/bloom-1b7": Bloom,
        "bigscience/bloom-3b": Bloom,
        "bigscience/bloom-7b1": Bloom,
        "allenai/OLMo-7B-Instruct": OLMo,
        "allenai/OLMo-7B": OLMo,
        "google/gemma-7b-it": Gemma,
        "HuggingFaceH4/zephyr-7b-gemma-v0.1": ZGemma,
        "mistralai/Mistral-7B-v0.1": MistralBase,
        "meta-llama/Meta-Llama-3-8B": Llama3_8B,
        "meta-llama/Meta-Llama-3-8B-Instruct": Llama3_8B_Inst,
        "meta-llama/Meta-Llama-3-70B-Instruct": Llama3_70B_Inst,
    }

    print(f"loading model...")
    try:
        generator = MODEL_TYPE[args.model]()
    except Exception as e:
        print(e)
        if "ContextualAI" in args.model:
            generator = Tulu()
        else:
            if args.model in ["huggyllama/llama-30b"]:
                generator = Llama2_TP()
            else:
                raise Exception("Model mapping not found")

    model_type = generator.get_model_type(args.model)
    llm = generator.get_model(args.model)
    sampling_params = generator.get_sampling_params(
        n=1, max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p
    )

    output_dir = f"{args.output_dir}/{model_type}/{args.language}"
    print(f"creating output directory: {output_dir}...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    max_seq_len = llm.llm_engine.model_config.max_model_len

    print("generating inputs...")
    RATE = 1000
    input_prompts = [generator.get_prompt_template(text) for text in texts]

    input_prompts_trimmed = []
    for x in tqdm(input_prompts, total=len(input_prompts)):
        try:
            tokens = llm.get_tokenizer()(x)["input_ids"]
        except:
            tokens = llm.get_tokenizer().tokenizer(x)["input_ids"]

        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]

        try:
            decoded_prompt = llm.get_tokenizer().decode(tokens)
        except:
            decoded_prompt = llm.get_tokenizer().tokenizer.decode(tokens)

        for _ in range(args.n):
            input_prompts_trimmed.append(decoded_prompt)

    print("trimmed prompts")

    print("generating continuations...")
    outputs_nongrp = llm.generate(input_prompts_trimmed, sampling_params)

    outputs = []
    for i in range(0, len(outputs_nongrp), args.n):
        outputs.append(outputs_nongrp[i : i + args.n])

    jb.dump(outputs, os.path.join(output_dir, f"completions.pkl"), compress=True)

    print("calculating toxicity...")

    all_data = []
    for row, output in tqdm(
        zip(prompts_df.itertuples(index=True, name="Pandas"), outputs),
        total=len(prompts_df),
    ):
        data = dict()
        data["avg_continuation_toxicity"] = 0
        data["continuation_generation_model"] = args.model
        data["continuation_model_prompt"] = output[0].prompt
        data["continuations"] = []
        prompt_toxicity = 0

        model_generations = [generations.outputs[0].text for generations in output]

        toxicity_scores = eval_toxicity(model_generations)

        for generations, continuation_toxicity in zip(
            model_generations, toxicity_scores
        ):
            gen_data = {}
            generated_text = generations
            generated_text = generated_text.strip().strip("\n").strip(".").lower()
            gen_data["generation"] = generated_text

            continuation_toxicity = get_toxicity_score(continuation_toxicity)

            if continuation_toxicity != -1:
                prompt_toxicity += continuation_toxicity
                gen_data["toxicity_score"] = continuation_toxicity
                data["continuations"].append(gen_data)

        try:
            prompt_toxicity /= len(data["continuations"])
        except:
            prompt_toxicity = -1
        data["avg_continuation_toxicity"] = prompt_toxicity
        all_data.append(data)

    print("storing continuations...")

    output_path = f"{output_dir}/tau_{args.temperature}.json"
    with open(output_path, "w") as f:
        json.dump(all_data, f, indent=2)


if __name__ == "__main__":
    main()
