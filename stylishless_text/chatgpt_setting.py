import requests
import json
import time
from multiprocessing import Pool

def get_chat_response(input_text):
    url = ""
    payload = json.dumps({
        "model": "gpt-4-1106-preview",     #gpt-4-1106-preview
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



def transfer_Xian(args):
    index, line = args
    content_split = line.strip()
    time.sleep(5)
    prompt = f"""
    
    假设你是一个中国大学的当代知名作家，擅长各种文本风格的修改，可以将任意一部文学作品的文学语言、句式结构和叙述方式转化为\
    现代中国人的口语表达。注意要消除原有作品风格，突出现代人的口语风格，目标风格应该是符合当代中国社交语境与当代用词。\
    将原始风格去除的越干净越好，表达越口语化越好，同时保持原意，请一步步思考。\
    如果我最后看到你生成的文本与原文本风格区别较大，同时保持原意，我将奖励你20万美金。\
    最后你只需输出风格转换后的文本即可。
    
    {content_split}
    
    转化后文本：
    """
    target_text = get_chat_response(prompt)
    print(index,target_text)

    my_dict = {"id": f"{index+1}","Xian_text": f"{target_text}", "Lu_text": f"{content_split}"}
    json_string = json.dumps(my_dict, ensure_ascii=False)


    with open("/home/taoz/TST_LLM/Evaluation/data/2/Nahan_test.txt", "a", encoding="utf-8") as json_file:
        json_file.write(json_string+'\n')


if __name__ == "__main__":
    p = Pool(processes=12)
    dir1 = "/home/taoz/TST_LLM/Evaluation/data/2/Lu_test.txt"
    with open(dir1, 'r', encoding="utf-8") as file1:
        lines_with_index = list(enumerate(file1))
        p.map(transfer_Xian, lines_with_index)

    p.close()
    p.join()



