import re
from cnsenti import Emotion
import jieba.posseg as pseg
import jieba
from collections import defaultdict
from collections import Counter
import os

import torch
import torch.nn as nn
from transformers import BertTokenizer
from TST_sentence.sentence_prepare import MyDataLoader, sentencedata
from TST_sentence.Model import ScatteredClassification, RhetoricClassification

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

#########  读取整篇文章   #########

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    processed_text = re.sub(r'[\s\u3000]+', '', content)
    return processed_text



#######################  语体定义  ###########################

#######################  篇章定义  ###########################




#############################  句子定义  ##################################


#########   将文章中的各个句子分开
def read_sentences(content):
    sentences = re.split(r'([。！？])', content)
    sentences = ["".join(sentences[i:i+2]) for i in range(0, len(sentences), 2) if "".join(sentences[i:i+2])]
    return sentences


#########  统计整篇文章的句子总数
def sum_sentence(sentences):
    sentence_sum = 0
    for sentence in sentences:
        if sentence.strip():
            sentence_sum = sentence_sum + 1
    return sentence_sum

#########  统计整篇文章的字总数
def count_chinese_characters(text):
    # 使用正则表达式移除标点符号和空格
    cleaned_text = re.sub(r'[^\u4e00-\u9fff]+', '', text)
    # 计算字符总数
    total_characters = len(cleaned_text)
    return total_characters

######  统计平均句长
def average_length(char_allcount,sentence_sum):
    average_sentence_length = char_allcount//sentence_sum
    return average_sentence_length

########  句子中字数计数
def item_number(sentence):
    pattern = re.compile(r'[\u4e00-\u9fa5a-zA-Z0-9]+')
    # 使用正则表达式匹配并提取非标点字符
    non_punctuation_chars = pattern.findall(sentence)
    # 计算非标点字符的数量
    char_count = len("".join(non_punctuation_chars))
    return char_count

######## 文章句子长短句判断
def judge_length(sentences,sentence_sum):
    item_0_10 = 0
    item_10_20 = 0
    item_20_30 = 0
    item_more30 = 0
    for sentence in sentences:
        if sentence.strip():
            item_num = item_number(sentence)
            if item_num <= 10:
                item_0_10 = item_0_10 + 1
            elif item_num > 10 and item_num <= 20:
                item_10_20 = item_10_20 + 1
            elif item_num > 20 and item_num <= 30:
                item_20_30 = item_20_30 + 1
            else:
                item_more30 = item_more30 + 1


    if (item_0_10+item_10_20)/sentence_sum > 0.6:
        aaa1 = "该作者文章句式短句较多，具有短小精悍，具有简洁、明快、干净利索的特点。"
        return aaa1
    elif (item_0_10+item_10_20)/sentence_sum >= 0.4 and (item_0_10+item_10_20)/sentence_sum <= 0.6:
        aaa2 = "该作者文章句式长短结合。长短句结合常常用于表达复杂的情感，且可以营造出文章的节奏感和韵律感。"
        return aaa2
    elif (item_0_10+item_10_20)/sentence_sum < 0.4:
        aaa3 = "该作者文章句式长句较多，内涵丰富，叙事具体，说理周详，感情充沛。"
        return aaa3



########  句子中标点计数
def judge_qu_ex(content):
    # 使用正则表达式来匹配标点符号
    punctuation = re.findall(r'[。！？]', content)
    # 计算问号的数量
    question_mark_count = punctuation.count('？')
    # 计算感叹号的数量
    exclamatory_mark_count = punctuation.count('！')

    # 计算标点符号的总数量
    total_punctuation_count = len(punctuation)
    # 计算问号在标点符号中的占比
    if total_punctuation_count > 0:
        question_mark_percentage = question_mark_count / total_punctuation_count
        exclamatory_mark_percentage = exclamatory_mark_count / total_punctuation_count
    else:
        question_mark_percentage = 0
        exclamatory_mark_percentage = 0

    aa=""
    bb=""
    if question_mark_percentage >= 0.15:
        aa = "该作者在写作时，用问号较多。问句通常用于引导读者思考，激发其思考和参与，使文章更具互动性。"
    if exclamatory_mark_percentage >= 0.15:
        bb = "该作者在写作时，用感叹句较多。感叹句通常用于表达强烈的情感、赞美或惊讶，使文章更具情感色彩。"
    cc = aa+bb
    return cc


##############  句子中情绪计数
def dict_slice(adict, start, end):
    keys = adict.keys()
    dict_slice = {}
    for k in list(keys)[start:end]:
        dict_slice[k] = adict[k]
    return dict_slice

def judge_emotion(sentences):
    emotion = Emotion()
    key_counts = {}
    for sentence in sentences:
        if sentence.strip():
            result = emotion.emotion_count(sentence)
            result_emotion = dict_slice(result, 2,9)
            if any(value != 0 for value in result_emotion.values()):
                max_key = max(result_emotion, key=result_emotion.get)
                max_value = result_emotion[max_key]
                if max_key in key_counts:
                    key_counts[max_key] += max_value
                else:
                    key_counts[max_key] = max_value

    max_emotion = max(key_counts, key=key_counts.get)
    if max_emotion == "好":
        all_emotion1 = ("大体看来，该作者作品主题情感基调主要是'好'。文章可能会描述一种积极向上的情感状态，"
                       "表达对某人、某事或某物的喜爱和满意。这种情绪可能通过赞美、推崇或表达满足感来体现。")
        return all_emotion1
    elif max_emotion == "乐":
        all_emotion2 = ("大体看来，该作者作品主题情感基调主要是'乐'。文章可能会传达一种愉悦和幸福的情绪，"
                       "可能包含快乐的事件、幽默的描写或令人感到温馨的故事。快乐的情绪通常会让读者感到轻松和愉快。")
        return all_emotion2
    elif max_emotion == "哀":
        all_emotion3 = ("大体看来，该作者作品主题情感基调主要是'哀'。文章可能会描述失落、悲痛或忧郁的情感体验。"
                        "这种情绪可能通过叙述失去、分离或其他令人伤心的事件来表达。")
        return all_emotion3
    elif max_emotion == "怒":
        all_emotion4 = ("大体看来，该作者作品主题情感基调主要是'怒'。文章可能会表达一种愤慨或愤怒的情绪，"
                        "可能是对不公正、背叛或挫折的反应。愤怒的情绪可能通过尖锐的批评、"
                        "激烈的争论或描述冲突的场景来体现。")
        return all_emotion4
    elif max_emotion == "惧":
        all_emotion5 = ("大体看来，该作者作品主题情感基调主要是'惧'。文章可能会描绘一种恐惧或担忧的情感状态，"
                        "可能是对未知、危险或威胁的反应。恐惧的情绪可能通过紧张的情节、悬疑的氛围或描述角色的内心恐慌来传达。")
        return all_emotion5
    elif max_emotion == "恶":
        all_emotion6 = ("大体看来，该作者作品主题情感基调主要是'恶'。文章可能会表达一种厌恶或反感的情绪，"
                        "可能是对某些行为、现象或物体的不满和排斥。"
                        "厌恶的情绪可能通过描述令人不快的场景或使用贬义词汇来体现。")
        return all_emotion6
    elif max_emotion == "惊":
        all_emotion7 = ("大体看来，该作者作品主题情感基调主要是'惊'。文章可能会描述一种惊奇或震惊的情绪，"
                        "可能是对意外事件或出乎意料的转折的反应。惊讶的情绪可能通过描写突发事件、"
                        "角色的惊异反应或使用戏剧性的揭示来表达。")
        return all_emotion7


########## 句子的整散句
def scattered(sentences):
    ##数据准备
    tokenizer = BertTokenizer.from_pretrained("/home/taoz/TST_LLM/TST_sentence/pretrained")
    scatteredloader = MyDataLoader(dataset=sentencedata(sentences,tokenizer=tokenizer),
                               batch_size=1, shuffle=False)
    ##加载模型
    model = ScatteredClassification(checkpoint="/home/taoz/TST_LLM/TST_sentence/pretrained", freeze="3")
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load("/home/taoz/TST_LLM/TST_sentence/run_text/run_0/whole_scattered.pth"))

    ##test
    pre_scatter = []
    model.eval()
    for step, sample_batched in enumerate(scatteredloader):
        input_ids, attention_mask, token_type_ids = (x.cuda() for x in sample_batched)
        with torch.no_grad():
            outputs = model(enc_inputs=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        pre_label = torch.argmax(outputs, 1)
        pre_scatter.append(pre_label)

    scatter_counts = Counter(pre_scatter)
    sorted_counts = sorted(scatter_counts.items(), key=lambda x: x[1], reverse=True)   # 按个数降序排序
    if sorted_counts[0][0] == 0:
        scatter_result1 = ("对于整散句这一块，该文本的整句使用更多。"
                           "整句结构使得文章更具逻辑性和结构性，各部分之间的连接更为紧密。")
        return scatter_result1
    elif sorted_counts[0][0] == 1:
        scatter_result2 = ("对于整散句这一块，该文本的散句使用更多。"
                           "散句结构相对简短，呈现出更为灵活、自由的写作风格，适合表达抒情、富有感情色彩的内容。")
        return scatter_result2


def rhetoric(sentences):
    ##数据准备
    tokenizer = BertTokenizer.from_pretrained("/home/taoz/TST_LLM/TST_sentence/pretrained")
    rhetoricloader = MyDataLoader(dataset=sentencedata(sentences, tokenizer=tokenizer),
                                   batch_size=1, shuffle=False)
    ##加载模型
    model = RhetoricClassification(checkpoint="/home/taoz/TST_LLM/TST_sentence/pretrained", freeze="3")
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load("/home/taoz/TST_LLM/TST_sentence/run_text/run_1/rhetoric.pth"))

    ##test
    pre_rhetoric = []
    model.eval()
    for step, sample_batched in enumerate(rhetoricloader):
        input_ids, attention_mask, token_type_ids = (x.cuda() for x in sample_batched)
        with torch.no_grad():
            outputs = model(enc_inputs=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        pre_label = torch.argmax(outputs, 1)
        pre_rhetoric.append(pre_label)

    scatter_counts = Counter(pre_rhetoric)
    sorted_counts = sorted(scatter_counts.items(), key=lambda x: x[1], reverse=True)  # 按个数降序排序
    rhetoric_result1=""
    rhetoric_result2=""
    if sorted_counts[0][0] == 0:
        rhetoric_result1 = ("对于整篇文章句子的修辞手法，使用最多的是比喻修辞："
                           "比喻增强了文章表达的形象感和生动性，使读者更容易理解抽象的概念。")
    elif sorted_counts[0][0] == 1:
        rhetoric_result1 = ("对于整篇文章句子的修辞手法，使用最多的是拟人修辞："
                           "拟人赋予非人的事物或抽象概念人类的特征，使其更具生命力和可感性，"
                           "有助于读者更好地理解和产生共鸣。")
    elif sorted_counts[0][0] == 2:
        rhetoric_result1 = ("对于整篇文章句子的修辞手法，使用最多的是夸张修辞："
                           "夸张通过夸大事物的特征或程度，引起读者的注意，强调某种情感或观点，"
                           "有时用于幽默或夸张的效果。")
    elif sorted_counts[0][0] == 3:
        rhetoric_result1 = ("对于整篇文章句子的修辞手法，使用最多的是引用修辞："
                           "引用他人的言论或观点，以加强作者的论证，提供权威性或引发读者对比，从而强调文章主题。")
    elif sorted_counts[0][0] == 4:
        rhetoric_result1 = ("对于整篇文章句子的修辞手法，使用最多的是排比修辞："
                           "排比将一系列并列的成分或句子以相同的语法结构呈现，产生韵律感，增强修辞效果，"
                           "使文章更加生动和引人注目。")
    elif sorted_counts[0][0] == 5:
        rhetoric_result1 = ("对于整篇文章句子的修辞手法，使用最多的是反讽修辞："
                           "通过言辞上的反语或讽刺，对某一事物或观点进行批评或嘲笑，达到强调作者立场或调侃的目的。")
    elif sorted_counts[0][0] == 6:
        rhetoric_result1 = ("对于整篇文章句子的修辞手法，使用最多的是反问修辞："
                           "通过提出问题，引导读者思考，加深对某一观点的理解或者制造一种戏剧性的效果。")
    elif sorted_counts[0][0] == 7:
        rhetoric_result1 = ("对于整篇文章句子的修辞手法，使用最多的是设问修辞："
                           "设问通过提出问题引起读者思考，产生一种引导性的效果，有时用于表达作者的疑虑或不确定。")
    elif sorted_counts[0][0] == 8:
        rhetoric_result1 = ("对于整篇文章句子的修辞手法，使用最多的是对仗修辞："
                           "对仗将语言或句子按照一定的对称结构进行安排，增强修辞效果，"
                           "使文章更富有音韵感，适用于诗歌和韵文。")
    elif sorted_counts[0][0] == 9:
        rhetoric_result1 = ""


    if sorted_counts[1][0] == 0:
        rhetoric_result2 = ("此外，文章还使用较多的比喻修辞:"
                           "比喻增强了文章表达的形象感和生动性，使读者更容易理解抽象的概念。")
    elif sorted_counts[1][0] == 1:
        rhetoric_result2 = ("此外，文章还使用较多的拟人修辞："
                           "拟人赋予非人的事物或抽象概念人类的特征，使其更具生命力和可感性，"
                           "有助于读者更好地理解和产生共鸣。")
    elif sorted_counts[1][0] == 2:
        rhetoric_result2 = ("此外，文章还使用较多的夸张修辞："
                           "夸张通过夸大事物的特征或程度，引起读者的注意，强调某种情感或观点，"
                           "有时用于幽默或夸张的效果。")
    elif sorted_counts[1][0] == 3:
        rhetoric_result2 = ("此外，文章还使用较多的引用修辞："
                           "引用他人的言论或观点，以加强作者的论证，提供权威性或引发读者对比，从而强调文章主题。")
    elif sorted_counts[1][0] == 4:
        rhetoric_result2 = ("此外，文章还使用较多的排比修辞："
                           "排比将一系列并列的成分或句子以相同的语法结构呈现，产生韵律感，增强修辞效果，"
                           "使文章更加生动和引人注目。")
    elif sorted_counts[1][0] == 5:
        rhetoric_result2 = ("此外，文章还使用较多的反讽修辞："
                           "通过言辞上的反语或讽刺，对某一事物或观点进行批评或嘲笑，达到强调作者立场或调侃的目的。")
    elif sorted_counts[1][0] == 6:
        rhetoric_result2 = ("此外，文章还使用较多的反问修辞："
                           "通过提出问题，引导读者思考，加深对某一观点的理解或者制造一种戏剧性的效果。")
    elif sorted_counts[1][0] == 7:
        rhetoric_result2 = ("此外，文章还使用较多的设问修辞："
                           "设问通过提出问题引起读者思考，产生一种引导性的效果，有时用于表达作者的疑虑或不确定。")
    elif sorted_counts[1][0] == 8:
        rhetoric_result2 = ("此外，文章还使用较多的对仗修辞："
                           "对仗将语言或句子按照一定的对称结构进行安排，增强修辞效果，"
                           "使文章更富有音韵感，适用于诗歌和韵文。")
    elif sorted_counts[1][0] == 9:
        rhetoric_result2 = ""

    rhetoric_result = rhetoric_result1 + rhetoric_result2
    return rhetoric_result


#############################  词语定义  ##################################
##############   判断文本的实词与虚词   ###################

def real_virtual(content):

    words = pseg.cut(content)
    real_word_counts = {'n': 0, 'v': 0, 'a': 0, 'r': 0, 'm': 0, 'q': 0}
    virtual_word_counts = {'d': 0, 'p': 0, 'c': 0, 'u': 0, 'o': 0}
    total_real_words = 0
    total_virtual_words = 0

    for word, flag in words:
        if flag in real_word_counts:
            real_word_counts[flag] += 1
            total_real_words += 1
        elif flag in virtual_word_counts:
            virtual_word_counts[flag] += 1
            total_virtual_words += 1

    # 定义一个映射字典，用于存储键名的映射关系
    real_mapping = {
        "n": "名词",
        "v": "动词",
        "a": "形容词",
        "r": "副词",
        "m": "数词",
        "q": "量词"
    }

    # 创建一个新字典，用于存储更改后的键名
    real_new_dict = {}

    # 遍历原始字典并更改键名
    for key, value in real_word_counts.items():
        new_key = real_mapping.get(key, key)  # 如果映射中存在映射关系，则使用映射的键名，否则保持不变
        real_new_dict[new_key] = value


    # 定义一个映射字典，用于存储键名的映射关系
    virtual_mapping = {
        "d": "限定词",
        "p": "介词",
        "c": "连词",
        "u": "助词",
        "o": "其他",
    }

    # 创建一个新字典，用于存储更改后的键名
    virtual_new_dict = {}

# 遍历原始字典并更改键名
    for key, value in virtual_word_counts.items():
        new_key = virtual_mapping.get(key, key)  # 如果映射中存在映射关系，则使用映射的键名，否则保持不变
        virtual_new_dict[new_key] = value


    if total_real_words > total_virtual_words:
        aa = "该作者用词方面，使用较多的实词。"
        return aa, real_new_dict, virtual_new_dict
    elif total_real_words < total_virtual_words:
        bb = "该作者用词方面，使用较多的虚词。"
        return bb, real_new_dict, virtual_new_dict
    elif total_real_words == total_virtual_words:
        cc = "该作者用词方面，实词与虚词使用次数相差不大。"
        return cc, real_new_dict, virtual_new_dict


######## 判断哪些实词
def judge_real(real_new_dict):
    sorted_items1 = sorted(real_new_dict.items(), key=lambda x: x[1], reverse=True)

    first_real_key = sorted_items1[0][0]
    second_real_key = sorted_items1[1][0]

    real_word = f"对于实词来说，该文章中{first_real_key}及{second_real_key}使用较多。"
    return real_word

######## 判断哪些虚词
def judge_virtual(virtual_new_dict):
    sorted_items2 = sorted(virtual_new_dict.items(), key=lambda x: x[1], reverse=True)

    first_virtual_key = sorted_items2[0][0]
    second_virtual_key = sorted_items2[1][0]

    virtual_word = f"对于虚词来说，该文章中{first_virtual_key}及{second_virtual_key}使用较多。"
    return virtual_word

####################  判断词长  #####################
def judge_wordlength(content):
    # 使用jieba分词
    words = list(jieba.cut(content))

    # 初始化词长统计字典
    word_length_count = defaultdict(int)

    # 统计词长，排除标点符号
    for word in words:
        if word.isalnum():  # 只统计由字母或数字组成的词语
            word_length = len(word)
            word_length_count[word_length] += 1

    sorted_word_length_count = sorted(word_length_count.items(), key=lambda x: x[1], reverse=True)

    max_wordlength = (f"同时，词语中占比最高的分别为词长为{sorted_word_length_count[:2][0][0]}"
                      f"以及词长为{sorted_word_length_count[:2][1][0]}")
    return max_wordlength


###################   统计语气词    #################
def count_modal_particles(text):
    # 定义中文语气词的正则表达式模式
    modal_particle_pattern = r'[吗呢吧啊呀]|(?:哦{1,2}|嗯{1,2}|了{1,2}|吧{1,2})' # 此处列出一些常见的语气词

    # 使用正则表达式查找语气词
    # modal_particles = re.findall(modal_particle_pattern, text)
    modal_particles = set(re.findall(modal_particle_pattern, text))

    # 使用Counter统计语气词的出现次数
    # modal_particles_count = Counter(modal_particles)
    a = ', '.join(modal_particles)
    modal_words = f"该文章中，也经常使用'{a}'等语气词。"

    return modal_words


#####################  统计单音节词语  #########################
def count_frequent_monosyllabic_words(text):
    # 常见的单音节虚词
    common_monosyllabic_words = {"你", "我", "它", "她", "他", "是", "的", "了", "地"}

    # 使用jieba进行分词
    words = jieba.cut(text)

    # 筛选出单音节词并排除标点和常见虚词
    monosyllabic_words = [word for word in words if
                          len(word) == 1 and word.isalpha() and word not in common_monosyllabic_words]

    # 使用Counter统计所有单音节词的出现次数
    monosyllabic_word_counts = Counter(monosyllabic_words)

    # 计算总单音节词个数
    total_monosyllabic_words = len(monosyllabic_words)

    # 计算在所有单音节词中占比超过10%的单音节词
    frequent_monosyllabic_words = {word: count / total_monosyllabic_words for word, count in
                                   monosyllabic_word_counts.items() if count / total_monosyllabic_words > 0.01}

    a = ",".join(frequent_monosyllabic_words)

    if frequent_monosyllabic_words:
        monosyllabic_words = f"在词语音节这一块，经常使用'{a}'等单音节词语，"
    else:
        monosyllabic_words = "在词语音节这一块，文本中较少使用单音节词。"

    return monosyllabic_words


########################  统计多音节词语  #########################
def count_frequent_multi_syllable_words(text):
    # 使用jieba进行分词
    words = jieba.cut(text)

    # 筛选出多音节词并排除标点
    multi_syllable_words = [word for word in words if len(word) > 1 and word.isalpha()]

    # 使用Counter统计所有多音节词的出现次数
    multi_syllable_word_counts = Counter(multi_syllable_words)

    # 计算总多音节词个数
    total_multi_syllable_words = len(multi_syllable_words)

    # 计算在所有多音节词中占比超过10%的多音节词及比例
    frequent_multi_syllable_words = {word: count / total_multi_syllable_words for word, count in
                                     multi_syllable_word_counts.items() if count / total_multi_syllable_words > 0.004}

    b = ",".join(frequent_multi_syllable_words)

    if frequent_multi_syllable_words:
        syllable_words = f"同时，经常使用'{b}'等多音节词语。"
    else:
        syllable_words = "同时，文本中较少使用多音节词。"

    return syllable_words


####################  判断成语  ########################

def find_top_idioms(text, top_n=8):
    with open('/home/taoz/TST_LLM/sentence_word_define_dataset/idiom.txt', 'r', encoding='utf-8') as file:
        idiom_list = [line.strip() for line in file]

    idiom_counts = Counter()

    # 遍历成语词典，检查是否出现在文本中
    for idiom in idiom_list:
        count = text.count(idiom)
        if count > 0:
            idiom_counts[idiom] = count

    # 获取出现次数最多的前 top_n 个成语
    top_idioms = [idiom for idiom, _ in idiom_counts.most_common(top_n)]

    c = ", ".join(top_idioms)

    if top_idioms:
        frequent_idioms = f"在成语使用上，经常使用'{c}'等成语。"
    else:
        frequent_idioms = "该文本较少或者不使用成语。"
    return frequent_idioms




def main_textstyle(dir):
    ##### 读取整个文本
    content = read_file(dir)

    ##### 句子风格定义
    sentences = read_sentences(content)       # 将文章中的各个句子分开
    sentence_sum = sum_sentence(sentences)    # 统计整篇文章的句子总数
    char_allcount = count_chinese_characters(content)      # 统计整篇文章的字总数
    ave_length = average_length(char_allcount, sentence_sum)     # 统计平均句长
    long_short = judge_length(sentences,sentence_sum)      # 文章句子长短句判断
    question_exclamatory = judge_qu_ex(content)            # 句子中标点计数
    all_emotion = judge_emotion(sentences)                 # 情绪判断
    scatter_result = scattered(sentences)                  # 整散句判断
    rhetoric_result = rhetoric(sentences)                  # 句子修辞判断

    ##### 词语风格定义
    real_virtual_words, real_new_dict, virtual_new_dict = real_virtual(content)    # 实词与虚词判断
    real_word = judge_real(real_new_dict)                  # 判断哪些实词占比最高
    virtual_word = judge_virtual(virtual_new_dict)         # 判断哪些虚词占比最高
    max_wordlength = judge_wordlength(content)             # 判断占比最高的两种词长
    modal_words = count_modal_particles(content)           # 判断经常使用哪些语气词
    monosyllabic_words = count_frequent_monosyllabic_words(content)   # 判断使用哪些单音节词语
    syllable_words = count_frequent_multi_syllable_words(content)     # 判断使用哪些多音节词语
    frequent_idioms = find_top_idioms(content)                        # 判断使用了哪些成语

    style_define = (f"这篇文章的风格从句子角度来看，文章的平均句长为{ave_length}，且{long_short}"
                    f"{question_exclamatory}{all_emotion}{scatter_result}{rhetoric_result}"
                    f"此外，从文章的词语角度来看，{real_virtual_words}{real_word}{virtual_word}"
                    f"{max_wordlength}{modal_words}{monosyllabic_words}{syllable_words}"
                    f"{frequent_idioms}")
    return style_define


if __name__ == "__main__":
    dir = "/home/taoz/TST_LLM/Evaluation/data/1/Qian_Other.txt"
    # dir = "/home/taoz/TST_LLM/示例.txt"
    print(main_textstyle(dir))
























