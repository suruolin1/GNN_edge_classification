# GNN_edge_classification
## 代码组织架构：

	--- check_pkl.py 用于数据预处理

	--- train.py用于模型训练

	--- model.py是sage模型（实际使用的模型）

	--- gat_model.py是GAT模型

	--- 其余代码是RPLAN-Toolbox-master1的代码

	--- RPLAN-Toolbox-master1\RPLAN-Toolbox-master\data下放训练数据

	--- 代码执行顺序：batch_tf.py数据预处理
		             check_pkl.py数据预处理及打包
		             train.py模型训练和测试
		             

## 环境配置：
	--- python>=3.7

	--- matlab (for alignment)

	--- numpy, scipy, scikit-image, matplotlib

	--- shapely (for visualization)

	---faiss (Linux only, for clustering)

	---dgl

## 数据预处理（padding）：边的最大数量为15
	方案1：
		对边数量不足15的图（graph）补全，填充内容是已有的边。
		eg.   old:   src = [0123456542]
			 dst = [5236102153]
			 edge = [0123203102]
		         new :   src = [0123456542      01234]
			     dst = [5236102153     52361]
			     edge = [0123203102  01232]
	方案2：
		对边数量不足15的图（graph）补全，填充内容是4(表示相同的节点的重叠关系，此时有5种边)。
		eg.   old:   src = [0123456542]
			 dst = [5236102153]
			 edge = [0123203102]
		         new :   src = [0123456542      00000]
			     dst = [5236102153     00000]
			     edge = [0123203102  44444]
模型：见model.py

## 训练：
	batchsize = 64  (太大内存不够用)
	enpochs = 2000(实际训练20-80轮左右就变化不大了)
	loss：交叉熵损失
		在loss上增加L2（L1）正则化（实验中两者好像没有太大区别）
	优化器：adam    学习率0.00001
		使用StepLR动态调整学习率，每200个batch，学习率变为原来的0.9倍


 
train loss: 每个batch记录一次

ver loss（验证集）:每个batch记录一次

train accurancy: 统计每个batch里的所有图的所有边数，作为分母，
	          统计对这个batch预测的结果中正确的边的个数，作为分子
	          两者的比值作为accurancy

ver accurancy(验证集):统计每个batch里的所有图的所有边数，作为分母，
	                   统计对这个batch预测的结果中正确的边的个数，作为分子
	                   两者的比值作为accurancy

test accurancy（测试集）:统计所有batch里的所有图的所有边数，作为分母，
	                   统计对所有batch预测的结果中正确的边的个数，作为分子
	                   两者的比值作为accurancy
	
