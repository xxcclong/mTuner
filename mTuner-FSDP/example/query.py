import requests


server_addr = "http://localhost:21002"


def q(prompt):
    params = {"input": prompt}
    response = requests.post(
        server_addr + "/chat",
        json=params,
        stream=True,
    )
    msg = response.json()["output"]
    print(msg)
    return msg


def change(d):
    params = {
        "ckpt": d,
    }
    response = requests.post(
        server_addr + "/ckpt",
        json=params,
        stream=True,
    )
    msg = response.json()["output"]
    print(msg)


def clear():
    response = requests.post(
        server_addr + "/clear",
        stream=True,
    )
    msg = response.json()["output"]
    print(msg)


if __name__ == "__main__":
    # prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response: Let's think step by step.".format_map(
    #     {"instruction": "what is five times larger than 244?"}
    # )
    p = """
请从以下句子中提取关键词并以JSON格式返回：
"我最喜欢在秋天去纽约中央公园散步，因为那里的景色非常美。"
    """

    output = q(p)
    import re, json

    def extract_json_from_text(text):
        # 使用正则表达式匹配JSON格式的内容
        pattern = r"\{.*?\}"
        matches = re.findall(pattern, text, re.DOTALL)

        # 提取到的所有匹配项
        extracted_jsons = []

        for match in matches:
            try:
                # 解析匹配项为JSON
                extracted_json = json.loads(match)
                extracted_jsons.append(extracted_json)
            except json.JSONDecodeError:
                # 如果解析失败，则忽略该匹配项
                continue

        return extracted_jsons

    extracted_jsons = extract_json_from_text(output)
    for json_obj in extracted_jsons:
        print(json_obj)

    # import os
    # files = os.listdir("dataset/ps/")
    # for file in files:
    #     if not file.endswith(".txt"):
    #         continue
    #     with open(f"dataset/ps/{file}", "r") as f:
    #         prompt = f.read()
    #     q(prompt)

    # target_dir = "/data/siqizhu/ftcode/FineTuneHub/outputs/2024-03-04/21-08-03"
    # for item in os.listdir(target_dir):
    #     if item.endswith(".pt"):
    #         print(item)
    #         change(os.path.join(target_dir, item))
    #         q(prompt)
