from google import genai
from google.genai import types
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
from aiolimiter import AsyncLimiter
from prompts import format_readability_prompt, format_correctness_prompt, format_efficiency_prompt
import asyncio
import os
import pandas as pd
import json
import argparse

parser = argparse.ArgumentParser(description="Call inference for different prompt types")
parser.add_argument("--input", '-i', type=str, required=True, help="Path to input JSONL file containing the data.")
parser.add_argument("--outptut", '-o', type=str, required=True, help="Path to output JSONL file to save the results.")
parser.add_argument(
        "--type", 
        "-t",
        choices=["readability", "correctness", "efficiency"],
        required=True,
        help="Type of prompt to call inference."
)

args = parser.parse_args()
load_dotenv()


INPUT_DIR = './data'
INPUT = args.input
OUTPUT = args.outptut
output_dir = os.path.dirname(OUTPUT)
os.makedirs(output_dir, exist_ok=True)

df = pd.read_json(INPUT, lines=True)
metadata_df = pd.read_json(f"{INPUT_DIR}/metadata.jsonl", lines=True)
tokens_df = pd.read_json(f"{INPUT_DIR}/prompt_tokens.jsonl", lines=True)
failed_ids = json.loads(open(f"{INPUT_DIR}/failed_chunk_4.json", 'r').read())

df['temp_id'] = df['sub_id'].str.rsplit('_', n=1).str[0]

cols_to_get = ['id', 'description', 'time_limit', 'memory_limit']

df = df.merge(
    metadata_df[cols_to_get],
    left_on='temp_id',
    right_on='id',
    how='left'
)

df = df.merge(
    tokens_df,
    left_on='sub_id',
    right_on='sub_id',
    how='left'
)


df.drop(columns=['temp_id'], inplace=True)
df = df[df['sub_id'].isin(failed_ids)]

df['readability_prompt'] = df.apply(format_readability_prompt, axis=1)
df['correctness_prompt'] = df.apply(format_correctness_prompt, axis=1)
df['efficiency_prompt'] = df.apply(format_efficiency_prompt, axis=1)

# api_key = os.getenv("GEMINI_API_KEY")
api_key = ""
client = genai.Client(api_key=api_key)
rpm_limiter = AsyncLimiter(max_rate=950, time_period=60)
tpm_limiter = AsyncLimiter(max_rate=1980000, time_period=60)

def parse_response(response_text):
    try:
        if response_text.startswith("Error:"):
            return None
        data = json.loads(response_text)
        return data.get("score")
    except (json.JSONDecodeError, TypeError):
        return response_text

async def call_gemini_async(prompt, prompt_tokens, semaphore):
    await tpm_limiter.acquire(prompt_tokens)
    await rpm_limiter.acquire(1)
    async with semaphore:
        try:
            response = await client.aio.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=128,
                    response_mime_type="application/json",
                    response_json_schema={
                        "type": "object",
                        "properties": {
                            "score": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5
                            }
                        },
                        "required": ["score"]
                    },
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.MINIMAL,
                    )
                )
            )
            return response.text
        except Exception as e:
            return f"Error: {e}"

async def process_all(data, max_concurrent=10):
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [call_gemini_async(prompt, tokens, semaphore) for prompt, tokens in data]
    results = await tqdm_asyncio.gather(*tasks)
    return results

# Chạy
if __name__ == "__main__":
    prompts = df[f'{args.type}_prompt'].tolist()
    tokens = df[f'{args.type}_tokens'].tolist()
    data = list(zip(prompts, tokens))
    results = asyncio.run(process_all(data, max_concurrent=30))
    df[f'{args.type}_score'] = [parse_response(r) for r in results]
    results_df = df[['sub_id', f'{args.type}_score']]
    results_df.to_json(OUTPUT, orient='records', lines=True)
    print(f"Saved {args.type} scores to {OUTPUT}")
