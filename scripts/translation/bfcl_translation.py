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
            print(f"翻訳中: {content[:50]}...")
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
            translated = completion.choices[0].message.content
            print(f"翻訳完了: {translated[:50]}...")
            return translated
    except Exception as e:
        if retry_count < MAX_RETRIES:
            print(f"エラーが発生しました。再試行中 ({retry_count + 1}/{MAX_RETRIES}): {str(e)}")
            await asyncio.sleep(RETRY_DELAY)
            return await translate_content(content, retry_count + 1)
        else:
            print(f"{MAX_RETRIES}回の再試行後に失敗しました: {str(e)}")
            # エラーが発生しても元のコンテンツを返す（翻訳されていない状態）
            return content

async def process_item(item):
    try:
        # 質問内容を翻訳 - 全てのターンを処理
        message_count = 0
        for turn_idx, turn in enumerate(item['question']):
            for message_idx, message in enumerate(turn):
                if message['role'] in ['user', 'system']:
                    message_count += 1
                    print(f"  Turn {turn_idx + 1}, Message {message_idx + 1}: {message['role']} message")
                    message['content'] = await translate_content(message['content'])
        print(f"  Total messages translated: {message_count}")
        return item
    except Exception as e:
        print(f"アイテム処理中にエラーが発生しました: {str(e)}")
        return item  # エラーが発生しても元のアイテムを返す

async def process_bfcl_file(input_file, output_file):
    print(f"Processing {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]

        print(f"Loaded {len(data)} items from {input_file}")

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Process items in batches for efficiency
        batch_size = 5  # Process 5 items at a time
        translated_data = []
        
        for batch_start in range(0, len(data), batch_size):
            batch_end = min(batch_start + batch_size, len(data))
            batch = data[batch_start:batch_end]
            
            print(f"Processing batch {batch_start//batch_size + 1}/{(len(data) + batch_size - 1)//batch_size} (items {batch_start + 1}-{batch_end})")
            
            # Create tasks for this batch
            tasks = [process_item(item) for item in batch]
            
            # Process batch with progress bar
            batch_results = []
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Batch {batch_start//batch_size + 1}"):
                try:
                    result = await coro
                    batch_results.append(result)
                except Exception as e:
                    print(f"Error in batch task: {str(e)}")
                    continue
            
            translated_data.extend(batch_results)
            print(f"Completed batch {batch_start//batch_size + 1}")

        # Write the translated data
        print(f"Writing {len(translated_data)} items to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in translated_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Completed translation of {input_file} -> {output_file}")
    except Exception as e:
        print(f"Error processing file {input_file}: {str(e)}")

async def main():
    # Get all BFCL JSON files from the data directory
    input_dir = "/home/olachinkeigpu/Project/llm-leaderboard/artifacts/bfcl:v0/bfcl"
    output_dir = "/home/olachinkeigpu/Project/llm-leaderboard/artifacts/bfcl:v0/bfcl_improved"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of all files to process
    #input_files = glob.glob(os.path.join(input_dir, "BFCL_v3_*.json"))
    input_files = [
        "/home/olachinkeigpu/Project/llm-leaderboard/artifacts/bfcl:v0/bfcl/BFCL_v3_multi_turn_base.json",
    ]

    # Process all JSON files
    for idx, input_file in enumerate(input_files, 1):
        # Generate output filename by replacing the input directory with output directory
        output_file = input_file.replace(input_dir, output_dir)
        await process_bfcl_file(input_file, output_file)

if __name__ == "__main__":
    asyncio.run(main())