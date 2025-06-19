from openai import AsyncOpenAI
import os
import json
import glob
import asyncio
from tqdm import tqdm
import time

# 並列数を制限（APIのレート制限に応じて調整）
MAX_CONCURRENT_CALLS = 3
MAX_RETRIES = 3
RETRY_DELAY = 10  # 秒

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# 並列数を制限するためのセマフォ
semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)

async def translate_content(content, retry_count=0):
    try:
        async with semaphore:  # セマフォを使用して並列数を制限
            completion = await client.chat.completions.create(
                extra_body={},
                model="qwen/qwen3-235b-a22b:free",
                messages=[
                    {
                        "role": "user",
                        "content": f"Translate the following English text to Japanese. Keep function names and code-related content (such as description of function) unchanged. Please note that this is a part of BFCL benchmark dataset. Please translate to natural Japanese as much as possible :\n\n{content}"
                    }
                ]
            )
            return completion.choices[0].message.content
    except Exception as e:
        if retry_count < MAX_RETRIES:
            print(f"Error occurred, retrying ({retry_count + 1}/{MAX_RETRIES}): {str(e)}")
            await asyncio.sleep(RETRY_DELAY)
            return await translate_content(content, retry_count + 1)
        else:
            print(f"Failed after {MAX_RETRIES} retries: {str(e)}")
            raise

async def process_item(item):
    try:
        # Translate the question content
        for message in item['question'][0]:
            if message['role'] == 'user':
                message['content'] = await translate_content(message['content'])
        return item
    except Exception as e:
        print(f"Error processing item: {str(e)}")
        return item  # エラーが発生しても元のアイテムを返す

async def process_bfcl_file(input_file, output_file):
    print(f"Processing {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Process items in parallel using asyncio with progress bar
        tasks = [process_item(item) for item in data]
        translated_data = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Translating {os.path.basename(input_file)}"):
            try:
                result = await coro
                translated_data.append(result)
            except Exception as e:
                print(f"Error in task: {str(e)}")
                continue

        # Write the translated data
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in translated_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Completed translation of {input_file} -> {output_file}")
    except Exception as e:
        print(f"Error processing file {input_file}: {str(e)}")

async def main():
    # Get all BFCL JSON files from the data directory
    input_dir = "scripts/evaluator/evaluate_utils/bfcl_pkg/data"
    output_dir = "scripts/evaluator/evaluate_utils/bfcl_pkg/data_jp"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of all files to process
    input_files = glob.glob(os.path.join(input_dir, "BFCL_v3_*.json"))
    total_files = len(input_files)
    
    # Process all JSON files
    for idx, input_file in enumerate(input_files, 1):
        # Generate output filename by replacing the input directory with output directory
        output_file = input_file.replace(input_dir, output_dir)
        await process_bfcl_file(input_file, output_file)

if __name__ == "__main__":
    asyncio.run(main())