import math
import numpy as np

class Viterbi:
    def __init__(self):
        self.trainPath = "./WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_02-21.pos"
        self.A_dict = dict()  # A[(i, j)] = P(state j | state i), trans; But in training it stores Count(i, j)
        self.B_dict = dict()  # B[(o_i, j)] = P(word o_i | state j), emit But in training it stores Count(oi, j)

        self.sumstate = {"start": 0, "end": 0}   # sum[j] = Count(j)
        self.states = []

    def train(self):
        # WSJ_02 - 21. pos: words and states for training corpus
        file = open(self.trainPath)
        stateSeq = []
        wordSeq = []
        while True:
            line = file.readline()
            if not line:
                break
            for i in range(0, 32):
                line = line.replace(chr(i), " ")
            # print(line)
            if line == " ":
                # print(stateSeq)
                # print(wordSeq)
                self.execSentence(stateSeq, wordSeq)
                stateSeq = []
                wordSeq = []
            else:
                line = line.split(" ")
                stateSeq.append(line[1])
                wordSeq.append(line[0])

        for key in self.A_dict:
            self.A_dict[key] = self.A_dict[key] / self.sumstate[key[0]]
        for key in self.B_dict:
            self.B_dict[key] = self.B_dict[key] / self.sumstate[key[1]]

        self.states = [key for key in self.sumstate if key != "start" and key != "end"]
        file.close()

    def execSentence(self, stateSeq, wordSeq):
        self.sumstate["start"] += 1
        self.sumstate["end"] += 1
        for i, elem in enumerate(stateSeq):
            if elem in self.sumstate:
                self.sumstate[elem] += 1
            else:
                self.sumstate[elem] = 1

            if (wordSeq[i], stateSeq[i]) in self.B_dict:
                self.B_dict[wordSeq[i], stateSeq[i]] += 1
            else:
                self.B_dict[wordSeq[i], stateSeq[i]] = 1

            if i == 0:
                if (stateSeq[i], "start") in self.A_dict:
                    self.A_dict["start", stateSeq[i]] += 1
                else:
                    self.A_dict["start", stateSeq[i]] = 1

            if i == len(stateSeq) - 1:
                if (stateSeq[i], "end") in self.A_dict:
                    self.A_dict[stateSeq[i], "end"] += 1
                else:
                    self.A_dict[stateSeq[i], "end"] = 1
            else:
                if(stateSeq[i], stateSeq[i+1]) in self.A_dict:
                    self.A_dict[stateSeq[i], stateSeq[i+1]] += 1
                else:
                    self.A_dict[stateSeq[i], stateSeq[i+1]] = 1




    def forward(self, input):
        for i in range(0, 32):
            input = input.replace(chr(i), " ")
        input = input.split(" ")
        print(input)
        print(len(input))

        v = np.zeros((len(self.states), len(input)+1))
        v.fill(-math.inf)
        trace = np.zeros((len(self.states), len(input)+1))
        trace.fill(-1)
        T = len(input)
        final_state = -1

        #   First column is not start, but the last column is the "end"
        for i, state in enumerate(self.states):
            if self.A("start", state) > 0 and self.B(state, input[0]) > 0:
                v[i, 0] = math.log(self.A("start", state)) + math.log(self.B(state, input[0]))
            else:
                v[i, 0] = -math.inf
        for j, word in enumerate(input):
            if j == 0:
                continue
            # j >= 1
            for i, state in enumerate(self.states):  # calc the v[i, j] (v[state, word])
                max = -math.inf
                lastState = -1
                for k, pre_state in enumerate(self.states):
                    if v[k, j-1] > -math.inf:
                        prob = v[k, j-1] + math.log(self.A(pre_state, state)) + math.log(self.B(word, state))
                        if prob > max:
                            max = prob
                            lastState = k
                v[i, j] = max
                trace[i, j] = k
        # end state
        for k, pre_state in enumerate(self.states):
            max = -math.inf
            if v[k, T-1] > -math.inf:
                prob = v[k, T-1] + math.log(self.A(pre_state, "end"))
                if prob > max:
                    max = prob
                    final_state = k

    def A(self, pre_state, state):
        if (pre_state, state) in self.A_dict:
            return self.A_dict[pre_state, state]
        return 0

    def A_log(self, pre_state, state, base=math.e):
        if (pre_state, state) in self.A_dict:
            return math.log(self.A_dict[pre_state, state], base)
        return -math.inf

    def B(self, word, j):
        if (word, j) in self.B_dict:
            return self.B_dict[word, j]
        return 0

    def B_log(self, word, j, base = math.e):
        if (word, j) in self.B_dict:
            return math.log(self.B_dict[word, j], base)
        return -math.inf




viterbi = Viterbi()
viterbi.train()
# print(len(viterbi.states), viterbi.states)
viterbi.forward("I have a dream.")
