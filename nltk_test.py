import nltk
from nltk.tag.util import tuple2str
from nltk.corpus import treebank
from nltk.tag import untag
import os, os.path
import nltk.data

# nltk.download('universal_tagset')
# # 词性标注
text1 = nltk.word_tokenize("It is a pleasant day today")
print(nltk.pos_tag(text1))
# CC  - 并列连词
# CD  -基数
# DT  - 限定词
# EX  - 存在的there
# FW  - 外来词
# IN  - 介词或从属连词
# JJ  - 形容词
# JJR  - 形容词，比较级
# JJS  - 形容词，最高级
# LS  - 列表项标记
# MD  - 情态动词
# NN  - 名词，单数或不可数
# NNS  - 名词，复数
# NNP  - 专有名词，单数
# NNPS - 专有名词，复数
# PDT  -  前置限定词
# POS  - 所有格结尾
# PRP  - 人称代词
# PRP$  - 所有格代词（prolog版本为PRP-S）
# RB  - 副词
# RBR  - 副词，比较级
# RBS  - 副词，最高级
# RP  - 小品词
# SYM  - 符号
# TO - to
# UH  - 叹词
# VB  - 动词
# 基本形式VBD  - 动词，过去式
# VBG  - 动词，动名词或现在分词
# VBN  - 动词，过去分词
# VBP  - 动词，现在时非第三人称单数
# VBZ  - 动词，现在时第三人称单数
# WDT  -  WH-限定词
# WP  -  WH-代词
# WP $  - 所有格WH-代词（prolog版本为WP-S）
# WRB  -  WH-副词

# 通过POS标注实现单词意思消歧的示例
text = nltk.word_tokenize("I cannot bear the pain of bear")
print(nltk.pos_tag(text))
# [('I', 'PRP'), ('can', 'MD'), ('not', 'RB'), ('bear', 'VB'), ('the', 'DT'),
# ('pain', 'NN'), ('of', 'IN'), ('bear', 'NN')]

taggedword = nltk.tag.str2tuple('bear/NN')
print(taggedword)

# 可以从给定文本中生成元组序列。
sentence = '''The/DT sacred/VBN Ganga/NNP flows/VBZ in/IN this/DT
region/NN ./. This/DT is/VBZ a/DT pilgrimage/NN ./. People/NNP from/IN
all/DT over/IN the/DT country/NN visit/NN this/DT place/NN ./. '''
print([nltk.tag.str2tuple(t) for t in sentence.split()])
# [('The', 'DT'), ('sacred', 'VBN'), ('Ganga', 'NNP'), ('flows', 'VBZ'),
# ('in', 'IN'), ('this', 'DT'), ('region', 'NN'), ('.', '.'), ('This', 'DT'),
# ('is', 'VBZ'), ('a', 'DT'), ('pilgrimage', 'NN'), ('.', '.'), ('People', 'NNP'),
# ('from', 'IN'), ('all', 'DT'), ('over', 'IN'), ('the', 'DT'), ('country', 'NN'), ('visit', 'NN'),
# ('this', 'DT'), ('place', 'NN'), ('.', '.')]

# 将元组（word和pos标签）转换为单词和标签
taggedtok = ('bear', 'NN')
print(tuple2str(taggedtok))
# bear/NN

# 查看在treebank语料中出现的一些常见标签
treebank_tagged = treebank.tagged_words(tagset='universal')
tag = nltk.FreqDist(tag for (word, tag) in treebank_tagged)
print(tag.most_common())
# [('NOUN', 28867), ('VERB', 13564), ('.', 11715), ('ADP', 9857), ('DET', 8725), ('X', 6613), ('ADJ', 6397),
# ('NUM', 3546), ('PRT', 3219), ('ADV', 3171), ('PRON', 2737), ('CONJ', 2265)]

# 去除句子的标注。
print(untag([('beautiful', 'NN'), ('morning', 'NN')]))
# ['beautiful', 'morning']

# 创建POS标注的语料库

# create = os.path.expanduser('～/nltkdoc')
# if not os.path.exists(create):
#     os.mkdir(create)
#
# print(os.path.exists(create))
# print(create in nltk.data.path)
#
# # 系统找不到指定的路径。: '～/nltkdoc'

# 选择某个机器学习算法

# 使用POS标注的语料库开发组块器
sent = [("A", "DT"), ("wise", "JJ"), ("small", "JJ"), ("girl", "NN"),
        ("of", "IN"), ("village", "NN"), ("became", "VBD"), ("leader", "NN")]
# 通过可选的DT、任意数目的JJ，以及的后面NN、可选的IN和任意数目的NN，定义名词短语（Noun Phrase）的组块规则
grammar = "NP: {<DT>?<JJ>*<NN><IN>?<NN>*}"
find = nltk.RegexpParser(grammar)
res = find.parse(sent)
print(res)
# (S
#   (NP A/DT wise/JJ small/JJ girl/NN of/IN village/NN)
#   became/VBD
#   (NP leader/NN))
res.draw()

# 使用任意数量的名词创建名词短语组块规则
noun1 = [("financial", "NN"), ("year", "NN"), ("account", "NN"), ("summary", "NN")]
gram = "NP:{<NN>+}"
find = nltk.RegexpParser(gram)
print(find.parse(noun1))