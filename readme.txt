代码组织架构：

	--- check_pkl.py 用于数据预处理

	--- train.py用于模型训练

	--- model.py是sage模型（实际使用的模型）

	--- gat_model.py是GAT模型

	--- 其余代码是RPLAN-Toolbox-master1的代码

	--- RPLAN-Toolbox-master1\RPLAN-Toolbox-master\data下放训练数据

	--- 代码执行顺序：batch_tf.py数据预处理
		             check_pkl.py数据预处理及打包
		             train.py模型训练和测试
		             

环境配置：
	--- python>=3.7

	--- matlab (for alignment)

	--- numpy, scipy, scikit-image, matplotlib

	--- shapely (for visualization)

	---faiss (Linux only, for clustering)

	---dgl

