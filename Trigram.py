import math
import numpy as np
import time


class Trigram:
    def __init__(self):
        self.trainPath = "./WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_02-21.pos"
        self.testPath = "./WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_24.words"

        self.A_dict = dict()  # A[(i, j)] = Count(state j -> state i), trans; But in training it stores Count(i, j)
        self.B_dict = dict()  # B[(o_i, j)] = P(word o_i | state j), emit But in training it stores Count(oi, j)

        self.countNotAppear = 0
        self.sumstate = {"start": 0, "end": 0}   # sum[j] = Count(j)
        self.count_suffix = dict()
        self.count_state_suffix = dict()
        self.states = []
        self.state2num = dict()
        self.num2state = dict()

        self.prob_states = dict()
        self.prob_suffix = dict()
        self.sigma_state = 0
        self.sigma_suffix = 0

    def train(self):
        # WSJ_02 - 21. pos: words and states for training corpus
        file = open(self.trainPath, "r")
        stateSeq = []
        wordSeq = []
        while True:
            line = file.readline()
            if not line:
                break
            for i in range(0, 32):
                line = line.replace(chr(i), " ")
            if line == " ":
                self.execSentence(stateSeq, wordSeq)
                stateSeq = []
                wordSeq = []
            else:
                line = line.split(" ")
                stateSeq.append(line[1])
                wordSeq.append(line[0])


        for key in self.B_dict:
            self.B_dict[key] = self.B_dict[key] / self.sumstate[key[1]]

        self.states = [key for key in self.sumstate if key != "start" and key != "end"]

        for i, state in enumerate(self.states):
            self.state2num[state] = i
            self.num2state[i] = state
            self.prob_states[state] = self.sumstate[state] / self.sigma_state

        for key in self.count_state_suffix:
            self.count_state_suffix[key] = self.count_state_suffix[key] / self.count_suffix[key[1]]
        for key in self.count_suffix:
            self.prob_suffix[key] = self.count_suffix[key] / self.sigma_suffix
        file.close()

    def execSentence(self, stateSeq, wordSeq):
        self.sumstate["start"] += 1
        self.sumstate["end"] += 1
        for i, elem in enumerate(stateSeq):
            # suffix count
            length = min(10, len(wordSeq[i]))
            for j in range(1, length+1):
                suffix = wordSeq[i][len(wordSeq[i]) - j:]
                self.sigma_suffix += 1
                if suffix in self.count_suffix:
                    self.count_suffix[suffix] += 1
                else:
                    self.count_suffix[suffix] = 1

                if (elem, suffix) in self.count_state_suffix:
                    self.count_state_suffix[elem, suffix] += 1
                else:
                    self.count_state_suffix[elem, suffix] = 1
            #
            self.sigma_state += 1
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
                if i > 0:
                    if (stateSeq[i - 1], stateSeq[i], "end") in self.A_dict:
                        self.A_dict[stateSeq[i - 1], stateSeq[i], "end"] += 1
                    else:
                        self.A_dict[stateSeq[i - 1], stateSeq[i], "end"] = 1
            else:
                if(stateSeq[i], stateSeq[i+1]) in self.A_dict:
                    self.A_dict[stateSeq[i], stateSeq[i+1]] += 1
                else:
                    self.A_dict[stateSeq[i], stateSeq[i+1]] = 1

                if i > 0:
                    if (stateSeq[i - 1], stateSeq[i], stateSeq[i+1]) in self.A_dict:
                        self.A_dict[stateSeq[i - 1], stateSeq[i], stateSeq[i+1]] += 1
                    else:
                        self.A_dict[stateSeq[i - 1], stateSeq[i], stateSeq[i+1]] = 1

    def forward(self, input):
        pass

    def test(self, testPath = "./WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_24.words"):
        pass


start = time.time()

trigram = Trigram()
trigram.train()

trainend = time.time()

trigram.test("./WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_24.words")

end = time.time()
print("Running for: ", end-start, "s.")
print("Training time: ", trainend - start, "s.")
print("Test time: ", end - trainend, "s.")