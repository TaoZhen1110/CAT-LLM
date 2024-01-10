import json
import requests
from ACC_BLEU_BERT import BLEU,bert_sco
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

    # url = 'http://11.0.192.6:8000/generate'  # baichuan2-13b
    # url = 'http://11.0.192.6:8001/generate'  # baichuan2-7b
    url = 'http://11.0.192.6:8002/generate'  # chatglm3-6b
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


style = ("该风格从词语角度来看，使用较多的实词。对于实词来说，该文章中动词及名词使用较多。"
          "对于虚词来说，该文章中限定词及介词使用较多。同时，词语中占比最高的分别为词长为2以及词长为1该文章中，"
          "也经常使用'嗯嗯, 呢, 哦, 嗯, 了, 呀, 吗, 吧, 啊'等语气词。在词语音节这一块，"
          "经常使用'对,在,着,不,那,和,中,但,到,有,像,上,也,从,人,都,这,就,被,后,说,很,把,您'等单音节词语，"
          "同时，经常使用'他们,这个,已经,一个,自己,没有,可能,现在,我们,什么,知道,罗辑'等多音节词语。"
          "在成语使用上，经常使用'不可思议, 微不足道, 小心翼翼, 不知不觉, 不动声色, 黄金时代, 面目全非, 一模一样'等成语。"
          "此外，该风格从句子角度来看，文章的平均句长为32，且该作者文章句式长句较多，内涵丰富，叙事具体，说理周详，"
          "感情充沛。大体看来，该作者作品主题情感基调主要是'好'。"
          "文章可能会描述一种积极向上的情感状态，表达对某人、某事或某物的喜爱和满意。"
          "这种情绪可能通过赞美、推崇或表达满足感来体现。对于整散句这一块，该文本的散句使用更多。"
          "散句结构相对简短，呈现出更为灵活、自由的写作风格，适合表达抒情、富有感情色彩的内容。"
          "此外，文章还使用较多的夸张修辞：夸张通过夸大事物的特征或程度，引起读者的注意，强调某种情感或观点，"
          "有时用于幽默或夸张的效果。")




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


    for i in range(5):
        origin_text = line1.strip()


        prompt = f"""
        {origin_text}

        请你将上述文本按照下面的目标风格的描述来转变语言的风格，要求你输出的文本，\
        语言本意保持原文本，而语言风格则要遵循下面的风格描述。\
        最后你只需输出风格转换后的文本即可。

        {style}
        
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

        with open("/home/dinghao/TST_LLM/Evaluation/Baichuan_Style/8/txt/Model_1.txt","a",encoding='utf-8') as json_file:
            json_object = create_json_object(id,bleu_1,truth_text,target_text1)
            json_file.write(json.dumps(json_object,ensure_ascii=False)+'\n')


        with open("/home/dinghao/TST_LLM/Evaluation/Baichuan_Style/8/csv/Model_1.csv", 'a', newline='',
                  encoding='utf-8') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=all_dict.keys())

            # 如果是第一次写入，写入标题行
            if csv_file.tell() == 0:
                csv_writer.writeheader()

            # 写入当前字典数据
            csv_writer.writerow(all_dict)




if __name__ == "__main__":
    p = Pool(processes=20)
    dir = "/home/dinghao/TST_LLM/Evaluation/data/8/Santi_test.txt"
    csv_path = "/home/dinghao/TST_LLM/Evaluation/Baichuan_Style/8/csv/Model_1.csv"

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