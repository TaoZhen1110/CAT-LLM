import json
import requests
from ACC_BLEU_BERT import BLEU, bert_sco
from multiprocessing import Pool
import csv
import os
import pandas as pd


def get_chat_response(query) -> str:
    # url = 'http://11.0.192.3:8000/generate'  # llama2-70b

    # url = 'http://11.0.192.5:8000/generate'  # llama2-13b
    # url = 'http://11.0.192.5:8001/generate'  # qwen-14b
    # url = 'http://11.0.192.5:8002/generate'  # llama2-7b
    # url = 'http://11.0.192.5:8003/generate'  # qwen-7b

    url = 'http://11.0.192.6:8000/generate'  # baichuan2-13b
    # url = 'http://11.0.192.6:8001/generate'  # baichuan2-7b
    # url = 'http://11.0.192.6:8002/generate'  # chatglm3-6b
    # url = 'http://11.0.192.6:8003/generate'  # gptj-6b
    # url = 'http://11.0.192.6:8004/generate'  # phi2-2.7b

    payload = json.dumps({
        "prompt": query,
        "temperature": 0,
        "max_tokens": 2048,
        "n": 1,
        # 可选的参数在这里：https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
    })
    headers = {
        'Content-Type': 'application/json'
    }
    res = requests.request("POST", url, headers=headers, data=payload)
    res = res.json()['text'][0].replace(query, '')
    return res


def create_json_object(id, bleu_1, file2_line, file3_line):
    # 创建 JSON 对象
    json_object = {
        "id": id,
        "bleu-1": bleu_1,
        "Liu_text": file2_line.strip(),
        "target_text": file3_line.strip(),
    }
    return json_object


def transfer_style(line):
    if isinstance(line, str):
        json_data = json.loads(line)
    else:
        json_data = line
    line1 = json_data['Xian_text']
    line2 = json_data['Liu_text']
    id = json_data['id']
    new_key = 'ID'

    for i in range(2):
        origin_text = line1.strip()

        prompt = f"""
        {origin_text}

        请你将上述文本按照刘慈欣的写法风格来转变语言的风格，要求你输出的文本，\
        语言本意保持原文本，而风格符合刘慈欣的写法。注意不允许添加别的句子。最后你只需输出风格转换后的文本即可。

        风格转换后的文本：
        """
        ######## 获得目标文章
        target_text = get_chat_response(prompt)
        if '\n' in target_text:
            target_text1 = target_text.splitlines()[0]
        else:
            target_text1 = target_text
        ####### 计算迁移后的文本与标准文本的相关性，即计算迁移准确率

        truth_text = line2.strip()

        #######  计算风格迁移后的内容保留的 BLEU 值
        BLEU_score = BLEU([truth_text], [target_text1])
        print("该文风迁移方法的准确率BLEU值为：", BLEU_score)

        #######  计算风格迁移后的内容保留的 BERT_Score 值
        BERT_Score = bert_sco([target_text1], [truth_text])
        print("该文风迁移方法的准确率BERT_Score值为：", BERT_Score)

        all_dict = BLEU_score.copy()  # 创建一个副本，以保留原始字典
        all_dict.update(BERT_Score)
        new_dict = {new_key: id}
        all_dict.update(new_dict)

        bleu_1 = all_dict["bleu-1"]

        with open("/home/dinghao/TST_LLM/Evaluation/Baichuan_Style/8/txt/Model_3.txt", "a",
                  encoding='utf-8') as json_file:
            json_object = create_json_object(id, bleu_1, truth_text, target_text1)
            json_file.write(json.dumps(json_object, ensure_ascii=False) + '\n')

        with open("/home/dinghao/TST_LLM/Evaluation/Baichuan_Style/8/csv/Model_3.csv", 'a', newline='',
                  encoding='utf-8') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=all_dict.keys())

            # 如果是第一次写入，写入标题行
            if csv_file.tell() == 0:
                csv_writer.writeheader()

            # 写入当前字典数据
            csv_writer.writerow(all_dict)


if __name__ == "__main__":
    p = Pool(processes=22)
    dir = "/home/dinghao/TST_LLM/Evaluation/data/8/Santi_test.txt"
    csv_path = "/home/dinghao/TST_LLM/Evaluation/Baichuan_Style/8/csv/Model_3.csv"

    remaining_data = []
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        id_list = df['ID'].tolist()
        id_list = list(set(id_list))
        with open(dir, 'r', encoding='utf-8') as input_file:
            for line in input_file:
                json_data = json.loads(line)
                # 判断 id 是否在要删除的列表中
                if json_data.get('id') not in id_list:
                    remaining_data.append(json_data)
        print("123444")
        p.map(transfer_style, remaining_data)
    else:
        print("5678")
        with open(dir, "r", encoding="utf-8") as file1:
            p.map(transfer_style, file1)

    p.close()
    p.join()