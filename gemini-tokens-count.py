from google import genai
from google.genai import types
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
from prompts import format_correctness_prompt, format_efficiency_prompt, format_readability_prompt
import asyncio
import os
import pandas as pd
import json
import argparse

load_dotenv()

parser = argparse.ArgumentParser(description="Tokens count for different prompt types")
parser.add_argument("--input", '-i', type=str, required=True, help="Path to input JSONL file containing the data.")
parser.add_argument("--model", '-m', type=str, default="gemini-3-flash-preview", help="Gemini model to use for token counting (default: gemini-3-flash-preview).")
parser.add_argument("--type", '-t', choices=["readability", "correctness", "efficiency", "all"], default="all", help="Type of prompt to count tokens (default: all).")

args = parser.parse_args()
INPUT_DIR = "./data"
df = pd.read_json(args.input, lines=True)
metadata_df = pd.read_json(f"{INPUT_DIR}/metadata.jsonl", lines=True)

df['temp_id'] = df['sub_id'].str.rsplit('_', n=1).str[0]

cols_to_get = ['id', 'description', 'time_limit', 'memory_limit']

df = df.merge(
    metadata_df[cols_to_get],
    left_on='temp_id',
    right_on='id',
    how='left'
)

df.drop(columns=['temp_id'], inplace=True)

df['readability_prompt'] = df.apply(format_readability_prompt, axis=1)
df['correctness_prompt'] = df.apply(format_correctness_prompt, axis=1)
df['efficiency_prompt'] = df.apply(format_efficiency_prompt, axis=1)

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

async def count_tokens(prompt, semaphore):
    async with semaphore:
        try:
            response = await client.aio.models.count_tokens(
                model=args.model,
                contents=prompt
            )
            return response.total_tokens
        except Exception as e:
            return len(prompt)//4

async def count_all_tokens(prompts):
    semaphore = asyncio.Semaphore(100)
    tasks = [count_tokens(p, semaphore) for p in prompts]
    token_counts = await tqdm_asyncio.gather(*tasks, desc="Counting tokens")
    return token_counts

# Chạy
if __name__ == "__main__":
    if args.type == "all":
        prompts_dict = {
            "readability": df['readability_prompt'].tolist(),
            "correctness": df['correctness_prompt'].tolist(),
            "efficiency": df['efficiency_prompt'].tolist(),
        }
        
        total_tokens = 0
        for prompt_type, prompts in prompts_dict.items():
            print(f"Counting {prompt_type.upper()} ({len(prompts)} prompts)...")
            results = asyncio.run(count_all_tokens(prompts))
            df[f'{prompt_type}_tokens'] = results
            type_total = sum(results)
            print(f"{prompt_type.upper()}: {type_total:,} tokens")
            total_tokens += type_total
        df[['sub_id', 'readability_tokens', 'correctness_tokens', 'efficiency_tokens']].to_json(f"prompt_tokens.jsonl", orient='records', lines=True, force_ascii=False)
        print(f"Total: {total_tokens:,} tokens")
    else:
        prompts = df[f'{args.type}_prompt'].tolist()
        print(f"Counting {args.type.upper()} prompts ({len(prompts)} prompts)...")
        tokens = asyncio.run(count_all_tokens(prompts))
        print(f'Tokens: {sum(tokens)}')
        print(f'Max tokens: {max(tokens)}')
        print(f'Min tokens: {min(tokens)}')
