class Viterbi:
    def __init__(self):
        self.trainPath = "./WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_02-21.pos"
        self.A = dict()  # A[(i, j)] = P(state j | state i), trans; But in training it stores Count(i, j)
        self.B = dict()  # B[(o_i, j)] = P(word o_i | state j), emit But in training it stores Count(oi, j)
        self.sumTag = dict()   # sum[j] = Count(j)



    def train(self):
        # WSJ_02 - 21. pos: words and tags for training corpus
        file = open(self.trainPath)
        tagSeq = []
        wordSeq = []
        while True:
            line = file.readline()
            if not line:
                break
            for i in range(0, 32):
                line = line.replace(chr(i), " ")
            # print(line)
            if line == " ":
                # print(tagSeq)
                # print(wordSeq)
                self.execSentence(tagSeq, wordSeq)
                tagSeq = []
                wordSeq = []
                break
            else:
                line = line.split(" ")
                tagSeq.append(line[1])
                wordSeq.append(line[0])
        file.close()

    def execSentence(self, tagSeq, wordSeq):
        for i, elem in enumerate(tagSeq):
            if elem in self.sumTag:
                self.sumTag[elem] += 1
            else:
                self.sumTag[elem] = 1

            if (wordSeq[i], tagSeq[i]) in self.B:
                self.B[wordSeq[i], tagSeq[i]] += 1
            else:
                self.B[wordSeq[i], tagSeq[i]] = 1

            if i == 0:
                if (tagSeq[i], "start") in self.A:
                    self.A[tagSeq[i], "start"] += 1
                else:
                    self.A[tagSeq[i], "start"] = 1

            if i == len(tagSeq) - 1:
                if (tagSeq[i], "end") in self.A:
                    self.A[tagSeq[i], "end"] += 1
                else:
                    self.A[tagSeq[i], "end"] = 1
            else:
                if(tagSeq[i], tagSeq[i+1]) in self.A:
                    self.A[tagSeq[i], tagSeq[i+1]] += 1
                else:
                    self.A[tagSeq[i], tagSeq[i + 1]] = 1

    def forward(self):
        pass


viterbi = Viterbi()
viterbi.train()