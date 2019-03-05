import math
import numpy as np
import time


class Viterbi:
    def __init__(self):
        self.trainPath = "./WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_02-21.pos"
        self.testPath = "./WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_24.words"
        self.A_dict = dict()  # A[(i, j)] = P(state j | state i), trans; But in training it stores Count(i, j)
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
            else:
                if(stateSeq[i], stateSeq[i+1]) in self.A_dict:
                    self.A_dict[stateSeq[i], stateSeq[i+1]] += 1
                else:
                    self.A_dict[stateSeq[i], stateSeq[i+1]] = 1

    def test(self, testPath = "./WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_24.words"):
        outPath = testPath
        outPath = outPath[outPath.rfind("/")+1: outPath.rfind(".")] + ".pos"
        testfile = open(testPath, "r")
        testout = open(outPath, "w")

        wordSeq = []

        while True:
            line = testfile.readline()
            if not line:
                break
            for i in range(0, 32):
                line = line.replace(chr(i), " ")
            if line == " ":
                # print(wordSeq)
                stateSeq = self.forward(wordSeq)

                self.writeFile(testout, wordSeq, stateSeq)
                wordSeq = []
            else:
                line = line.split(" ")
                wordSeq.append(line[0])
        print("Count(Unknown words) = ", self.countNotAppear)
        testfile.close()
        testout.close()

    def forward(self, input):  # input is a list like ["I", "like", "her", "."]
        v = np.zeros((len(self.states), len(input)))
        v.fill(-math.inf)
        trace = np.zeros((len(self.states), len(input)))
        trace.fill(-1)
        T = len(input)
        showup = False
        for i, state in enumerate(self.states):
            if self.A("start", state) > 0 and self.B(input[0], state) > 0:
                showup = True
                v[i, 0] = self.A_log("start", state) + self.B_log(input[0], state)
            else:
                v[i, 0] = -math.inf

        if not showup:
            self.countNotAppear += 1
            for i, state in enumerate(self.states):
                v[i, 0] = self.A_log("start", state) + self.B_log_not_appear(input[0], state)

        for j, word in enumerate(input):
            if j == 0:
                continue
            # j >= 1

            showup = False

            for i, state in enumerate(self.states):  # calc the v[i, j] (v[state, word])
                max = -math.inf
                lastState = -1
                for k, pre_state in enumerate(self.states):
                    if not math.isinf(v[k, j-1]) and not math.isnan(v[k, j-1]):
                        prob = v[k, j-1] + self.A_log(pre_state, state) + self.B_log(word, state)
                        if prob > max:
                            showup = True
                            max = prob
                            lastState = k

                v[i, j] = max
                trace[i, j] = lastState

            if not showup:
                self.countNotAppear += 1
                for i, state in enumerate(self.states):
                    max = -math.inf
                    lastState = -1
                    for k, pre_state in enumerate(self.states):
                        if not math.isinf(v[k, j - 1]) and not math.isnan(v[k, j - 1]):
                            prob = v[k, j - 1] + self.A_log(pre_state, state) + self.B_log_not_appear(word, state)
                            if prob > max:
                                max = prob
                                lastState = k
                    v[i, j] = max
                    trace[i, j] = lastState
        # end state

        max = -math.inf
        final_state = -1
        for k, pre_state in enumerate(self.states):
            if v[k, T-1] > -math.inf:
                prob = v[k, T-1] + self.A_log(pre_state, "end")
                if prob > max:
                    max = prob
                    final_state = k
        path = []
        self.back_path(final_state, T-1, trace, path)
        return path

    def back_path(self, state_int, t, trace, path):
        if state_int == -1:
            path = path.reverse()
            return

        state = self.num2state[state_int]
        path.append(state)
        pre_state_int = int(trace[state_int, t])
        self.back_path(pre_state_int, t-1, trace, path)

    def A(self, pre_state, state):
        if (pre_state, state) in self.A_dict:
            return self.A_dict[pre_state, state]
        return 0

    def A_log(self, pre_state, state, base=math.e):
        if (pre_state, state) in self.A_dict:
            return math.log(self.A_dict[pre_state, state], base)
        return -math.inf

    def B(self, word, state):
        if (word, state) in self.B_dict:
            return self.B_dict[word, state]
        return 0

    def B_log(self, word, state, base = math.e):
        if (word, state) in self.B_dict:
            return math.log(self.B_dict[word, state], base)
        return -math.inf

    def B_log_not_appear(self, word, state, base = math.e):
        find_suffix = False
        length = min(len(word), 10)
        max = 0
        for i in range(1, length+1):
            suffix = word[len(word)-i:]
            if suffix in self.prob_suffix and (state, suffix) in self.count_state_suffix:
                find_suffix = True
                prob = self.count_state_suffix[state, suffix] * self.prob_suffix[suffix] / self.prob_states[state]
                if prob > max:
                    max = prob
        if find_suffix:
            return math.log(max, base)
        else:
            return math.log(1. / self.sumstate[state], base)

    def print_A(self):
        for pre_state in self.states:
            for state in self.states:
                print(self.A(pre_state, state), end=" ")
            print()

    def stcs2lst(self, input):
        for i in range(0, 32):
            input = input.replace(chr(i), " ")
        input = input.split(" ")
        return input

    def writeFile(self, outfile, wordSeq, stateSeq):
        for i, word in enumerate(wordSeq):
            outfile.write(wordSeq[i] + "\t" + stateSeq[i] + "\n")
        outfile.write("\n")


start = time.time()

viterbi = Viterbi()
viterbi.train()

trainend = time.time()

viterbi.test("./WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_23.words")

end = time.time()
print("Running for: ", end-start, "s.")
print("Training time: ", trainend - start, "s.")
print("Test time: ", end - trainend, "s.")
