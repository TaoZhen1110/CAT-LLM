import pandas as pd
import requests
import json
from ACC_BLEU_BERT import BLEU,bert_sco
from multiprocessing import Pool
import time
import csv
import os

def get_chat_response(input_text):
    url = ""
    payload = json.dumps({
        "model": "gpt-3.5-turbo",     #gpt-4-1106-preview
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



style = ("该风格从句子角度来看，文章的平均句长为24，且该作者文章句式长短结合。"
         "长短句结合常常用于表达复杂的情感，且可以营造出文章的节奏感和韵律感。大体看来，该作者作品主题情感基调主要是'好'。"
         "文章可能会描述一种积极向上的情感状态，表达对某人、某事或某物的喜爱和满意。"
         "这种情绪可能通过赞美、推崇或表达满足感来体现。对于整散句这一块，该文本的整句使用更多。"
         "整句结构使得文章更具逻辑性和结构性，各部分之间的连接更为紧密。"
         "此外，文章还使用较多的反讽修辞：通过言辞上的反语或讽刺，对某一事物或观点进行批评或嘲笑，"
         "达到强调作者立场或调侃的目的。"
         "此外，该风格从词语角度来看，使用较多的实词。对于实词来说，该文章中动词及名词使用较多。"
         "对于虚词来说，该文章中限定词及连词使用较多。同时，词语中占比最高的分别为词长为1以及词长为2该文章中，"
         "也经常使用'呢, 哦, 啊, 了, 呀'等语气词。在词语音节这一块，经常使用'但,要,又,这,人,不,里,便,很,都,和,在,说,有,也,却,来,到,去,着,吃'等单音节词语，"
         "同时，经常使用'阿Ｑ,已经,一个,因为,起来,什么,时候,知道,没有,自己,辫子,他们,母亲,而且,我们,七斤'等多音节词语。"
         "在成语使用上，经常使用'青面獠牙, 阿弥陀佛, 不以为然, 怒目而视, 易子而食, 之乎者也, 自言自语, 自作自受'等成语。")



def create_json_object(id, bleu_1, file2_line, file3_line):
    # 创建 JSON 对象
    json_object = {
        "id": id,
        "bleu-1": bleu_1,
        "Lu_text": file2_line.strip(),
        "target_text": file3_line.strip(),
    }
    return json_object


def transfer_style(line):
    if isinstance(line, str):
        json_data = json.loads(line)
    else:
        json_data = line
    line1 = json_data['Xian_text']
    line2 = json_data['Lu_text']
    id = json_data['id']
    new_key = 'ID'


    for i in range(10):
        origin_text = line1.strip()

        time.sleep(5)

        prompt = f"""
        {origin_text}

        请你将上述文本按照下面的目标风格的描述来转变语言的风格，要求你输出的文本，\
        语言本意保持原文本，而语言风格则要遵循下面的风格描述。\
        最后你只需输出风格转换后的文本即可。

        {style}
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

        with open("/home/taoz/TST_LLM/Evaluation/ablation_result/Style/txt/Model_sen_word.txt","a",encoding='utf-8') as json_file:
            json_object = create_json_object(id,bleu_1,truth_text,target_text1)
            json_file.write(json.dumps(json_object,ensure_ascii=False)+'\n')


        with open("/home/taoz/TST_LLM/Evaluation/ablation_result/Style/csv/Model_sen_word.csv", 'a', newline='',
                  encoding='utf-8') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=all_dict.keys())

            # 如果是第一次写入，写入标题行
            if csv_file.tell() == 0:
                csv_writer.writeheader()

            # 写入当前字典数据
            csv_writer.writerow(all_dict)



if __name__ == "__main__":
    p = Pool(processes=12)
    dir = "/home/taoz/TST_LLM/Evaluation/data/2/Nahan_test.txt"
    csv_path = "/home/taoz/TST_LLM/Evaluation/ablation_result/Style/csv/Model_sen_word.csv"

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
































