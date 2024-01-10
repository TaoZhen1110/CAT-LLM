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


style_text = ("中国，1967年。“红色联合”对“四·二八兵团”总部大楼的攻击已持续了两天，他们的旗帜在大楼周围躁动地飘扬着，"
              "仿佛渴望干柴的火种。“红色联合”的指挥官心急如焚，他并不惧怕大楼的守卫者，那二百多名“四·二八”战士，"
              "与诞生于1966年初、经历过大检阅和大串联的“红色联合”相比要稚嫩许多。他怕的是大楼中那十几个大铁炉子，"
              "里面塞满了烈性炸药，用电雷管串联起来，他看不到它们，但能感觉到它们磁石般的存在，开关一合，玉石俱焚，"
              "而“四·二八”的那些小红卫兵们是有这个精神力量的。比起已经在风雨中成熟了许多的第一代红卫兵，"
              "新生的造反派们像火炭上的狼群，除了疯狂还是疯狂。大楼顶上出现了一个娇小的身影，"
              "那个美丽的女孩子挥动着一面“四·二八”的大旗，她的出现立刻招来了一阵杂乱的枪声，射击的武器五花八门，"
              "有陈旧的美式卡宾枪、捷克式机枪和三八大盖，"
              "也有崭新的制式步枪和冲锋枪——后者是在“八月社论”发表之后从军队中偷抢来的——连同那些梭镖和大刀等冷兵器，"
              "构成了一部浓缩的近现代史……“四·二八”的人在前面多次玩过这个游戏，在楼顶上站出来的人，除了挥舞旗帜外，"
              "有时还用喇叭筒喊口号或向下撒传单，每次他们都能在弹雨中全身而退，为自己挣到崇高的荣誉。")

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

        请你将上述文本按照下面的文本风格来转变语言的风格，要求你输出的文本，\
        语言本意保持原文本，而风格与下面的文本风格相同。注意，你只需输出文本，不要加额外的句子或说明。

        {style_text}

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

        with open("/home/dinghao/TST_LLM/Evaluation/Baichuan_Style/8/txt/Model_2.txt", "a",
                  encoding='utf-8') as json_file:
            json_object = create_json_object(id, bleu_1, truth_text, target_text1)
            json_file.write(json.dumps(json_object, ensure_ascii=False) + '\n')

        with open("/home/dinghao/TST_LLM/Evaluation/Baichuan_Style/8/csv/Model_2.csv", 'a', newline='',
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
    csv_path = "/home/dinghao/TST_LLM/Evaluation/Baichuan_Style/8/csv/Model_2.csv"
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