from bert_score import score
# from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import jieba




# def tranacc(sentences):
#     ##数据准备
#     tokenizer = BertTokenizer.from_pretrained("/home/taoz/TST_LLM/TST_sentence/pretrained")
#     targetloader = MyDataLoader(dataset=dataacc2(sentences, tokenizer=tokenizer),
#                                   batch_size=1, shuffle=False)
#     ##加载模型
#     model = metricacc_model(checkpoint="/home/taoz/TST_LLM/TST_sentence/pretrained", freeze="3")
#     model = nn.DataParallel(model).cuda()
#     model.load_state_dict(torch.load("/home/taoz/TST_LLM/Evaluation/run_text/run_0/transacc.pth"))
#     ##test
#     ACC = 0.0
#     count = 0
#     model.eval()
#     for step, sample_batched in enumerate(targetloader):
#         input_ids, attention_mask, token_type_ids = (x.cuda() for x in sample_batched)
#         with torch.no_grad():
#             output = model(enc_inputs=input_ids, attention_mask=attention_mask,
#                         token_type_ids=token_type_ids)
#         pre_label = torch.argmax(output, 1)
#         count += 1
#         ACC += torch.sum(pre_label == 1)
#
#     average_acc = ACC/count
#
#     return average_acc

# text = "猴王参访仙道，无缘得遇。在于南赡部洲，串长城，游小县，不觉八九年余。\
# 忽行至西洋大海，他想着海外必有神仙。独自个依前作筏，又飘过西海，直至西牛贺洲地界。\
# 登岸遍访多时，忽见一座高山秀丽，林麓幽深。他也不怕狼虫，不惧虎豹，登山顶上观看。\
# 果是好山：千峰排戟，万仞开屏。日映岚光轻锁翠，雨收黛色冷含青。\
# 瘦藤缠老树，古渡界幽程。奇花瑞草，修竹乔松：修竹乔松，万载常青欺福地；奇花瑞草，四时不谢赛蓬瀛。\
# 幽鸟啼声近，源泉响溜清。重重谷壑芝兰绕，处处崖苔藓生。起伏峦头龙脉好，必有高人隐姓名。\
# 正观看间，忽闻得林深之处，有人言语，急忙趋步，穿入林中，侧耳而听，原来是歌唱之声。\
# 歌曰：“观棋柯烂，伐木丁丁，云边谷口徐行。卖薪沽酒，狂笑自陶情。\
# 苍径秋高，对月枕松根，一觉天明。认旧林，登崖过岭，持斧断枯藤。\
# 收来成一担，行歌市上，易米三升。"



# def read_sentences(content):
#     sentences = re.split(r'([。！？])', content)  # 使用正则捕获组来保留标点符号
#     sentences = ["".join(sentences[i:i+2]) for i in range(0, len(sentences), 2)]
#     return sentences
#
# sentences = read_sentences(text)
# print(sentences)
#
# trans_style_ACC = tranacc(sentences)
# print("该文风迁移方法的准确率ACC为：",trans_style_ACC)


def proline(line):
    return [w for w in jieba.cut("".join(line.strip().split()))]

def BLEU(reference_list, hypothesis_list):
    res = {}
    for i in range(1, 5):
        res["bleu-%d" % i] = []

    for origin_reference, origin_candidate in tqdm(zip(reference_list, hypothesis_list)):
        origin_reference = proline(origin_reference)
        origin_candidate = proline(origin_candidate)
        assert isinstance(origin_candidate, list)  # 确保 candidate 是列表
        for i in range(1, 5):
            res["bleu-%d" % i].append(sentence_bleu(references=[origin_reference],
                                                    hypothesis=origin_candidate,
                                                    weights=tuple([1. / i for j in range(i)]),
                                                    smoothing_function=SmoothingFunction().method1))
    for key in res:
        res[key] = round(np.mean(res[key]),5)

    return res

# text = "我爱你中国，你爱谁啊？啦啦啦啦？"
# text1 = "我爱你中国，你爱谁啊？啦啦？"
#
# print(BLEU([text1], [text]))



def bert_sco(cand_list, refs_list):

    result1 = {}
    (P, R, F), hashname = score(cand_list, refs_list, model_type= "bert-base-chinese",lang="zh", return_hash=True)
    # result = f"{hashname}: Precision={P.mean().item():.6f} Recall={R.mean().item():.6f} F1={F.mean().item():.6f}"

    result1["BERT-Precision"] = round(P.mean().item(),5)
    result1["BERT-Recall"] = round(R.mean().item(),5)
    result1["BERT-F1"] = round(F.mean().item(), 5)



    return result1

# cand_list = ['我们都曾经年轻过']
# refs_list = ['我们都曾经年轻过呀']
#
# print(bert_sco(cand_list,refs_list))
