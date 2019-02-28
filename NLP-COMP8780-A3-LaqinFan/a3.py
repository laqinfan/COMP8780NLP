import a2
import nltk
from collections import Counter
from nltk.tokenize import RegexpTokenizer

############################################################################
# (i) [10 points] Use the assignment#2's hash of hashes to train a
# baseline lexicalized statistical tagger on the entire BROWN corpus.
############################################################################
def p1_baselineTagger(word_list, word_tag_hash):
	new_tags = []
	#use the most frequent tag of each word to tag the word
	for w in word_list:
		tag = max(word_tag_hash[w], key=lambda i: word_tag_hash[w][i])
		new_tags.append(tag)
	tagger = {}
	tagger = {k:v for k, v in zip(word_list,new_tags )}
	# print(tagger)
	# print(word_list)
	return tagger

############################################################################
# (ii) [20 points] Use the baseline lexicalized statistical tagger to tag 
# all the words in the SnapshotBROWN.pos.all.txt file. Evaluate and report the
# performance of this baseline tagger on the Snapshot file.
############################################################################
def p2_evaluate(tagger, word_tag_hash):
	#read file SnapshotBROWN.pos.all.txt for later evaluation of tagger
	with open("SnapshotBROWN.pos.all",'r') as f:
		# read file content
		content = "".join(line for line in f if not line.isspace())

	#use key word(TOP) to split content into sentence list
	sentence_list = content.split(a2.key_word)

	word_list = [] #store all the word in SnapshotBROWN.pos.all
	value_list = [] # store all the tags in SnapshotBROWN.pos.all

	for s in sentence_list:
		new_s = ""
		for w in s.split('('):
			if ')' in w:
				tag = w.split(')')[0]
				t = "".join(tag)
				if t[0] != '-' and t != a2.key:
					new_s = new_s + t + " "
					t = tag.split(' ')
					word_list.append(t[1].lower())
					value_list.append(t[0])

	new_tags = []
	#use the most frequent tag of each word to tag the word
	for w in word_list:
		if w in tagger:
			tag = max(word_tag_hash[w], key=lambda i: word_tag_hash[w][i])
			new_tags.append(tag)
		else:
			new_tags.append("NA")

	count = 0
	#calculate the accuracy of this new tagger
	if len(new_tags) != len(value_list):
	    raise ValueError("Lists must have the same length.") 
	else:
		for tag1, tag2 in zip(new_tags,value_list):
			if tag1 == tag2:
				count += 1
	accuracy = float(count)/float(len(value_list))

	print("Performance of this tagger by accuracy: {0:.0%}".format(accuracy))

############################################################################
# (iii) [20 points] add few rules to handle unknown words for the tagger
#    in (ii). The rules can be morphological, contextual, or of other
#    nature. Use 25 new sentences to evaluate this tagger (the (ii) tagger +
#    unknown word rules). You can pick 25 sentences from a news article
#    from the web and report the performance on those.
############################################################################

def p3_sentenceEval(tagger, word_tag_hash):
	# read 25 sentences
	with open("sentences") as f:
		text = "".join(line for line in f if not line.isspace())

	#get the word token in the sentences
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(text)

	new_tag = []

	#add a few rules to handle unknown words
	for w in tokens:
		if w in tagger:
			tag = tagger[w]
			new_tag.append(tag)
		else:
			if w.endswith("ly"):
				new_tag.append("RB")
			elif w.endswith("ed"):
				new_tag.append("VBN")
			elif w.endswith("ing"):
				new_tag.append("VBG")
			elif w.endswith("ness"):
				new_tag.append("NN")
			elif w.isnumeric():
				new_tag.append("CD")
			elif w.istitle():
				new_tag.append("NNP")
			else:
				new_tag.append("NA")
	# Apply part-of-speach tagger to the words on 25 sentences
	nltk_tag = [t[1] for t in nltk.pos_tag(tokens)]

	c = 0
	for t1, t2 in zip(new_tag, nltk_tag):
		if t1 == t2:
			c += 1
	accuracy = float(c)/float(len(nltk_tag))

	print("Accuracy on 25 sentences: {0:.0%}".format(accuracy))



if __name__=="__main__":

	baseline_tagger = p1_baselineTagger(a2.word_list, a2.word_tag_hash)
	p2_evaluate(baseline_tagger, a2.word_tag_hash)
	p3_sentenceEval(baseline_tagger, a2.word_tag_hash)


