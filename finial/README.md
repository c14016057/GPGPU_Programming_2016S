Implementing Random Forest on the GPU
=========================================
## Instroduction
This is a finial project from a course, GPGPU. We implement random forests on the GPU.

We use dfs to train  decision trees on gpu and create predict tree on cpu.

## Algorithm
### Greedy scan all data points to find best cut
-----------------------
![decisiontree](https://github.com/c14016057/GPGPU_Programming_2016S/blob/master/finial/figure/decisiontree.png)
- Decision Tree
	- RR scan each feature
	- Greedy scan all data nodes
	- Use Gini as metrix
### Impurity metrix
- Gini
	- Easy to implementation
- Information gain
### BFS & DFS
- BFS
	- Good parallism for depth prediction tree
- DfS
	- Good GPU utilization for balance prediction tree
## Optimization 
### Greedy scan one feature v.s Greedy scan all features
------------------------
![scanmethod](https://github.com/c14016057/GPGPU_Programming_2016S/blob/master/finial/figure/scanmethod.png)
- Scan one feature
	- faster on create one predict node
	- tree in not balance
- Scan all feature
	- slower on create one predict node
	- tree is balance
### Sorting each feature before compute impurity
---------------------------
- Reduce time complex from O(n<sup>2</sup>) to O(nlog(n))
### Prefix sum label before counting each label proportion
--------------------------
- Reduce time complex from O(n<sup>2</sup>) to O(n/p)
### Create prediction tree on CPU instead of GPU
-------------------------
- Reduce the time for copy memory from gpu to cpu

## Reference
----------------------------
- [演算法筆記](http://www.csie.ntnu.edu.tw/~u91029/Classification.html)
- [Optimizing Random Forests on GPU](http://digitalassets.lib.berkeley.edu/techreports/ucb/text/EECS-2014-205.pdf)



