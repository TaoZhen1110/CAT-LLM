import time
import requests
import json
from ACC_BLEU_BERT import BLEU,bert_sco
import pandas as pd
from multiprocessing import Pool
import csv
import os

def get_chat_response(input_text):
    url = ""
    payload = json.dumps({
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": input_text
            }
        ],
        "temperature": 0,
        "seed": 42,
    })
    headers = {
        'token': '',
        'User-Agent': '',
        'Content-Type': 'application/json',
        'Accept': '*/*',
        'Host': '',
        'Connection': 'keep-alive'
    }

    response = requests.post(url, headers=headers, data=payload)

    # Navigate through the JSON response to get the content
    try:
        content = response.json()['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError):
        content = "123"  # or raise an error if you prefer

    return content

def create_json_object(id, bleu_1, file2_line, file3_line):
    # 创建 JSON 对象
    json_object = {
        "id": id,
        "bleu-1": bleu_1,
        "Qian_text": file2_line.strip(),
        "target_text": file3_line.strip(),
    }
    return json_object


def transfer_style(line):
    if isinstance(line, str):
        json_data = json.loads(line)
    else:
        json_data = line
    line1 = json_data['Xian_text']
    line2 = json_data['Qian_text']
    id = json_data['id']
    new_key = 'ID'
    for i in range(10):
        origin_text = line1.strip()

        time.sleep(5)

        prompt = f"""
        {origin_text}

        请你将上述文本按照钱钟书的写法风格来转变语言的风格，要求你输出的文本，\
        语言本意保持原文本，而风格符合钱钟书的写法。注意不允许添加别的句子。最后你只需输出风格转换后的文本即可。

        """
        ######## 获得目标文章
        target_text = get_chat_response(prompt)
        if '\n' in target_text:
            target_text = target_text.splitlines()
            target_text = [element for element in target_text if element != ""]
            target_text1 = target_text[-1]
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

        with open("/home/taoz/TST_LLM/Evaluation/result4_1106_Style/1/txt/Model3_text.txt", "a",
                  encoding='utf-8') as json_file:
            json_object = create_json_object(id, bleu_1, truth_text, target_text1)
            json_file.write(json.dumps(json_object, ensure_ascii=False) + '\n')

        with open("/home/taoz/TST_LLM/Evaluation/result4_1106_Style/1/csv/Model_3.csv", 'a', newline='',
                  encoding='utf-8') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=all_dict.keys())

            # 如果是第一次写入，写入标题行
            if csv_file.tell() == 0:
                csv_writer.writeheader()

            # 写入当前字典数据
            csv_writer.writerow(all_dict)


if __name__ == "__main__":

    p = Pool(processes=12)
    dir = "/home/taoz/TST_LLM/Evaluation/data/1/Weicheng_test.txt"
    csv_path = "/home/taoz/TST_LLM/Evaluation/result4_1106_Style/1/csv/Model_3.csv"

    remaining_data = []
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        id_list = df['ID'].tolist()
        with open(dir, 'r', encoding='utf-8') as input_file:
            for line in input_file:
                json_data = json.loads(line)
                # 判断 id 是否在要删除的列表中
                if json_data.get('id') not in id_list:
                    remaining_data.append(json_data)
        print("1234")
        p.map(transfer_style, remaining_data)
    else:
        print("5678")
        with open(dir, "r", encoding="utf-8") as file1:
            p.map(transfer_style, file1)

    p.close()
    p.join()




