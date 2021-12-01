# checkpoint文件用来保存模型训练的中间结果
##子文件：

	
	resnet_train_acc.txt         保存目前验证集对应的模型的训练精度
	
	resnet_val_acc.txt           保存当前验证集的最优精度（.4f%）
	
	ResNet50_ Cats_Dogs_val.pth  保存当前最优验证精度对应的训练模型
	
	vgg_train_acc.txt            保存目前验证集对应的模型的训练精度
	
	vgg_val_acc.txt              保存当前验证集的最优精度（.4f%）
	
	VGG16_Cats_Dogs_train.pth    该权值文件是在只有训练集的条件下训练出来的训练集精度为54%的模型用于权值初始化
	
	VGG16_Cats_Dogs_val.pth      保存当前最优验证精度对应的训练模型