#coding:utf-8
import sys
import linecache

lines = linecache.getlines('./corpus/corpus')
res = open(file='./corpus/train.de', mode='w', encoding='utf-8')
res_tgt = open(file='./corpus/train.en', mode='w', encoding='utf-8')

def clean_sentence(sent):
	sent = sent.replace('&quot;', '"')
	sent = sent.replace(' @-@ ', '')
	sent = sent.replace(' �', '')
	sent = sent.replace('�', '')
	sent = sent.replace('&amp;', '&')
	sent = sent.replace('&lt;', '<')
	sent = sent.replace('&gt;', '>')
	sent = sent.replace('&nbsp;', '')
	sent = sent.strip()
	return sent

for line in lines:
	line = line.strip()
	items = line.split('\t')
	if len(items) != 3:
		continue
	res.write(clean_sentence(items[0]) + '\r')
	res_tgt.write(clean_sentence(items[1]) + '\r')
	
	