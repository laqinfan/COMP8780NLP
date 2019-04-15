Input file: SnapshotBROWN.pos.all
Output:

distinct rules: (part of them)
NP -> DT NNP NNP NNP NNP 
NP -> NNP 
NP -> DT NNS 
NP -> NN 
NP -> DT NN 
ADVP -> RB 
NP -> JJ NNS 
NP -> DT NNP NNP NNP 
WHNP -> WDT 
ADJP -> JJ 
AUX -> VBD 
VP -> VBN 
NP -> DT NNP NN NN 
NP -> NNP NNP 
AUX -> TO 
NP -> NNP NNP NNP 
VP -> VBD 
NP -> NNS 
NP -> PRP 
ADJP -> JJ CC JJ 
NP -> NNP NNS 
NP -> DT JJ NN 
NP -> DT NNP CC NNP NNP VBG NNS 
NP -> DT CD NNS 
AUX -> MD 
NP -> JJR NN 
ADJP -> VBN 
NP -> NN NNS NNS 
NP -> DT JJ NNP 
NP -> VBG NNS 
NP -> DT 
NP -> DT NNP NNP NN NN NN 
......


P1.
1. The number of distinct rules :  15346

2. Top 10 most frequent rules:
	PP -> IN NP   73890
	NP -> PRP   45614
	NP -> DT NN   31487
	ADVP -> RB   27822
	NP -> NN   20988
	NP -> NNP   18715
	S -> NP VP   16900
	AUX -> TO   15005
	AUX -> MD   14006
	ADJP -> JJ   12481

3. The non-terminal with the most alternative rules is NP:7148


P2. Head word is the word that carries the main semantic information, and is a noun in noun phrases, the main verb in verb phrases, the adjective in adjective phrases, preposition in prepositional phrases. If we lexicalize it by adding head words into the grammar, it will become pretty huge. Since we extract the grammar from Brown file, the number of distinct rules is about more than 10000, and the grammar is composed of different rules (NP -> NP NNP, VP -> VBN PP, NP -> DT NNP PP, etc), if we add any noun (room, class, teacher,...) in NP, add any verb(eat, play, teach,...) in VP, add any preposition (in, on, about,...) to PP, etc, the grammar will grow exponentially.  







