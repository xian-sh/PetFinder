# 数据集
kaggle比赛数据下载链接：https://www.kaggle.com/c/petfinder-pawpularity-score/
## 子文件/目录


	test需要进行最终进行预测的图片存储路径
	
	**train官方给的训练数据集（文件太大，需要自行下载数据放在里面）**
	
	dataset.py数据集类，返回img,feature,label
	
	rename_dataset.py对train对应的train.csv包含的数据进行打乱分割，一部分训练，一部分验证（可手动处理excel实现同样效果）
	
	test.csv 官方给的需要预测的图片对应的其他特征数据
	
	train.csv官方给的训练集的图片对应的其他特征数据
	
	train_data.csv自己分割得到的2000张训练图片对应的其他特征文件（可手动随意添加删改）
	
	val_data.csv 自己分割得到的500张验证图片对应的其他特征文件（可手动随意添加删改）
