from ACC_BLEU_BERT import BLEU,bert_sco
import csv
from multiprocessing import Pool
import json


def content_preserve(line):
    if isinstance(line, str):
        json_data = json.loads(line)
    else:
        json_data = line

    line1 = json_data['Xian_text']
    target_id = json_data['id']
    new_key = 'ID'
    truth_text = line1.strip()


    matching_jsons = []
    with open("/home/taoz/TST_LLM/Evaluation/Baichuan_Style/1/txt/Model_1.txt", 'r') as file:
        for line in file:
            json_data = json.loads(line)
            if json_data.get('id') == target_id:
                matching_jsons.append(json_data)

    for match_json in matching_jsons:
        target_text = match_json["target_text"]


        #######  计算风格迁移后的内容保留的 BLEU 值
        BLEU_score = BLEU([truth_text], [target_text])
        print("该文风内容保存的准确率BLEU值为：", BLEU_score)


        #######  计算风格迁移后的内容保留的 BERT_Score 值
        BERT_Score = bert_sco([target_text], [truth_text])
        print("该文风内容的准确率BERT_Score值为：", BERT_Score)

        all_dict = BLEU_score.copy()  # 创建一个副本，以保留原始字典
        all_dict.update(BERT_Score)
        new_dict = {new_key: target_id}
        all_dict.update(new_dict)

        #4_1106_Content
        with open("/home/taoz/TST_LLM/Evaluation/Baichuan_Content/1/Model_1.csv", 'a', newline='',
                  encoding='utf-8') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=all_dict.keys())

            # 如果是第一次写入，写入标题行
            if csv_file.tell() == 0:
                csv_writer.writeheader()

            # 写入当前字典数据
            csv_writer.writerow(all_dict)


if __name__ == "__main__":
    p = Pool(processes=10)
    dir = "/home/taoz/TST_LLM/Evaluation/data/1/Weicheng_test.txt"
    with open(dir, "r", encoding="utf-8") as file1:
        p.map(content_preserve, file1)

    p.close()
    p.join()

#experiment_result4_1106_
#