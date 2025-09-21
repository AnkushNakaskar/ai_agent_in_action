# Building Autonomous Agent :
###### Ref : https://www.manning.com/books/ai-agents-in-action
### This chapter explain the building agent 
### [first_btree.py](first_btree.py) : 
 * This file explain the behavioural tree. It has concept of sequence and selector
 * Sequence is the one where condition is true, it will execute all the siblings , like HasApple, will execute Eat Apple
 * Selector : if the one sequence is not true, it will move to next available sequence until it find the success. 
 * You can try it out by making the hasApple return false.
