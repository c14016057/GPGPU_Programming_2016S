Implementing Random Forest on the GPU
=========================================
## Instruction
This is a finial project from a course, GPGPU. We implement random forests on the GPU.

We use dfs to train  decision trees on gpu and create predict tree on cpu.

## Algorithm
### Greedy scan all data points to find best cut
-----------------------
![decisiontree](/decisiontree.png)
- Decision Tree
	- RR scan each feature
	- Greedy scan all data nodes
	- Use Gini as metrix
## Optimization 
### Greedy scan one feature v.s Greedy scan all features
------------------------
![scanmethod](/scanmethod.png)
- Scan one feature
	-faster on create one predict node
	-tree in not balance
- Scan all feature
	-slower on create one predict node
	-tree is balance
### Sorting each feature before compute impurity
---------------------------
- Reduce time complex from O(n<sup>2) to O(nlog(n))
### Prefix sum label before counting each label proportion
--------------------------
- Reduce time complex from O(n<sup>2) to O(n/p)
### Create prediction tree on CPU instead of GPU
-------------------------
- Reduce the time for copy memory from gpu to cpu





