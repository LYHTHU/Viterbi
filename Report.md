# NLP: Assignment 4

##### Yunhao Li, NetID: yl6220
##### Mar/04/2019

## Basic Implementation
1. Implement the training function

2. Implement the forward(predict) function

3. For unknown words:<br>
    In the basic version, if an unknown word occur, I assume it show up only 1 time for every state. 
    So
    $$
    P(word | state) = \frac{1}{Count(state)}
    $$

4. The accuracy of basic version running on WSJ_24.word is 94.05%.

## Additional Work

+ Unknown words. <br>

  In additional works, I implement the morphology algorithm. 

  Using morphology, the accuracy is 95.4