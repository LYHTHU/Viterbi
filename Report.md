# NLP: Assignment 4

##### Yunhao Li, NetID: yl6220
##### Mar/04/2019

## Basic Implementation
1. Create a class `Viterbi` to process the problem.

2. Implement the training function.

    Using `dict` to store `A` and `B`

3. Implement the forward(predict) function. The input is a list of word sequence, and it will return a tag sequence.

4. Implement the `test(testPath)` function. The default value is `./WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_24.words`. When it finishes, a `.pos`file will be created in the working directory.

5. Because the probability of a sequence can be extremely small and may overflow, so I use `log` function to calculate the probability. If the probability is `0`, the `log(prob)=-inf`, expressed by `math.inf`.

6. For unknown words:

    In the basic version, if an unknown word occur, I assume it show up only 1 time for every state. 
    So

     ```P(word | state) = 1 / Count(state)```

7. The **accuracy** of basic version running on `WSJ_24.word` is `94.05%`.

## Additional Work

+ Unknown words. 

  + In additional works, I implement the morphology algorithm. 
    1. **Train:** For every word `W` occurs in the training set labeled as state `S`, the suffixes of it is counted. For every word, its suffixes include the last 1~`L` chars (last 1 char, last 2 chars,  ... last L chars), where `L = min{word.length, 10}`. After statistics, I get 3 matrices. 

       + The first one is `count_state_suffix`, which is used to store `Count(state, suffix)`. 

       + The second one is `count_suffix` , which is used to store `Count(suffix)`

       + A integer `sigma_suffix` is used to store the sigma of the `Count(suffix)` of all suffixes.

       + So `P(suffix) = count_suffix[suffix] / sigma_suffix`

         ​    `P(state | suffix) = count_state_suffix[state, suffix] / count_suffix[suffix]`

    2. **Test:** When a unknown word `W` occurs, we calculate `P(W | state)` as follows:

       + Iterate all the suffixes of `W`
         + For each suffix `suffix`, calculate `P(suffix|state)=P(S|suffix)*P(suffix)/P(state)`
       + Then we find the `suffix` that makes the` P(suffix|state)` maximal. And use that value as`P(word|state)`.

  + Using morphology, the **accuracy** is `95.41%`.