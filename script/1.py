import json
import os
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

#Configuration
API_KEY = "sk-nPZVE0SK10LootDQF5F25d2bBeA9489585D2Ba7b90Bd5dCc"  # 请替换为你的API Key
BASE_URL = "https://api.gpt.ge/v1/" # 如果使用中转服务，请修改此处
MODEL_NAME = "gpt-5" # 用户指定的模型名称
INPUT_FILE = "alpacaeval_results.json" # 输入文件名
OUTPUT_FILE = "evaluation_result.json" # 输出结果文件名
MAX_WORKERS = 5 # 并发线程数，根据你的API限速（RPM）调整

# 初始化客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def evaluate_single_item(item):
    """
    调用 GPT 对单个条目进行评测
    """
    instruction = item.get('instruction', '')
    reference = item.get('reference_output', '')
    answer = item.get('stu_answer', '')

    # 构造 Prompt，强制要求 JSON 格式返回以便程序解析
    system_prompt = "你是一个严谨的评测助手。请判断'学生回答'与'参考答案'在给定'指令'下含义是否一致。"
    user_prompt = f"""
    【指令 / Instruction】:
    {instruction}

    【参考答案 / Reference Output】:
    {reference}

    【学生回答 / Student Answer】:
    {answer}

    请判断学生回答的含义是否准确。如果含义一致（即使措辞不同），认为是正确的。如果参考答案包含了多个要点，学生回答只要覆盖了其中的核心要点即可认为是正确的。如果这个问题的答案不唯一（例如开放性问题），请判断学生回答是否合理且与指令相关，若合理且相关则认为是正确的。
    请务必仅返回一个JSON对象，格式为：{{"is_correct": true}} 或 {{"is_correct": false}}
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # 开启 JSON 模式能大大提高解析成功率 (如果模型支持)
            # 如果 gpt-5.2 还不支持 json_object，请去掉这一行或做好异常处理
            response_format={"type": "json_object"}, 
            temperature=0  # 设置为0以获得最确定的结果
        )
        
        content = response.choices[0].message.content
        result_json = json.loads(content)
        
        # 将评测结果写入 item
        item['is_correct'] = result_json.get('is_correct', False)
        item['eval_reason'] = result_json.get('reason', 'No reason provided') # 如果你让模型输出了理由
        return item

    except Exception as e:
        print(f"Error evaluating item: {e}")
        item['is_correct'] = False # 出错默认判错，或者你可以标记为 None
        item['error_log'] = str(e)
        return item

def main():
    # 1. 读取 JSON 文件
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到文件 {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"一共加载了 {len(data)} 条数据，准备开始评测...")
    print(f"使用的模型: {MODEL_NAME}")

    results = []
    correct_count = 0

    # 2. 使用线程池并发请求 (提高速度)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_item = {executor.submit(evaluate_single_item, item): item for item in data}
        
        # 使用 tqdm 显示进度条
        for future in tqdm(as_completed(future_to_item), total=len(data), desc="Evaluating"):
            item = future.result()
            results.append(item)
            if item.get('is_correct'):
                correct_count += 1

    # 3. 计算准确率
    total = len(results)
    accuracy = (correct_count / total * 100) if total > 0 else 0

    # 4. 输出统计结果
    print("\n" + "="*30)
    print(f"评测完成！")
    print(f"总条目数: {total}")
    print(f"正确条目: {correct_count}")
    print(f"准确率: {accuracy:.2f}%")
    print("="*30)

    # 5. 保存带有评测结果的新文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"详细评测结果已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()