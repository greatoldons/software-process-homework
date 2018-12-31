import numpy as np 
import operator
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from datetime import datetime
import scipy.io as sio
from sklearn import preprocessing
data=sio.loadmat('E://pythontest/light/data2_train.mat')
array=data['data2_train']
data2=sio.loadmat('E://pythontest/light/data3_train.mat')
array2=data2['data3_train']
data3=sio.loadmat('E://pythontest/light/data5_train.mat')
array3=data3['data5_train']
data4=sio.loadmat('E://pythontest/light/data6_train.mat')
array4=data4['data6_train']
data5=sio.loadmat('E://pythontest/light/data8_train.mat')
array5=data5['data8_train']
data6=sio.loadmat('E://pythontest/light/data10_train.mat')
array6=data6['data10_train']
data7=sio.loadmat('E://pythontest/light/data11_train.mat')
array7=data7['data11_train']
data8=sio.loadmat('E://pythontest/light/data12_train.mat')
array8=data8['data12_train']
data9=sio.loadmat('E://pythontest/light/data14_train.mat')
array9=data9['data14_train']
def test_plus(array,y):
	#建立对应的标签数组，为方便计算暂时用1-9来代表九个类别
    n=len(array)
    tp=[y for i in range(n)]
    return tp
def change(lebal,y,num):
	'''
	将数据标签转换为one-hot 形式
	lebal：为输入的数据标签集
	y：当前i对应的类别
	'''
	new_lebal=[]
	for i in range(num):
		if lebal[i]==y:
			new_lebal.append(1)
		else:
			new_lebal.append(0)
	return new_lebal


def get_predict(predict,predict_lebal,lebal):
	'''
	获得九类别预测结果的函数，
	predict：为第i类别的一对多svc预测结果
	predict——lebal： 总预测结果数组
	lebal： 当前svc为第几类别
	'''
	total=len(predict)
	for i in range(total):
		if predict[i]==1:
			predict_lebal[i]=lebal


def get_acc(predict,y_test_hot,lebal):
	'''
	打印正确率
	'''
	total=len(predict)
	acc=0
	for i in range(total):
		if predict[i]==y_test_hot[i]:
			acc+=1
	print('第',lebal,'类ova样本分类，准确率为：',acc/total,'%')


y1=test_plus(array,1)
y2=test_plus(array2,2)
y3=test_plus(array3,3)
y4=test_plus(array4,4)
y5=test_plus(array5,5)
y6=test_plus(array6,6)
y7=test_plus(array7,7)
y8=test_plus(array8,8)
y9=test_plus(array9,9)
#样本合并
#合并成一个大的train集，和一个对应的lebal集
train=np.concatenate((array,array2,array3,array4,array5,array6,array7,array8,array9))
y=np.concatenate((y1,y2,y3,y4,y5,y6,y7,y8,y9))
train=preprocessing.StandardScaler().fit_transform(train)
# 运用pca方法，显示支持特征
# 并确定最终降维维数
# pca=PCA(n_components=200)
# pca.fit(train)
# value=pca.explained_variance_ratio_
# value.sort()
# featrue=0
# i=199
# max=value.mean()
# while(True):
# 	if value[i]>max:
# 		featrue+=1
# 		i-=1
# 	else:
# 		break
# print(featrue)
pca=PCA(n_components=6)
pca.fit(train)
new_train=pca.fit_transform(train)
# 根据7：3的比率分割数据集，随机种子设为1
from sklearn.cross_validation import train_test_split
x_train,x_test, y_train, y_test =train_test_split(new_train,y,test_size=0.3, random_state=1)
num_train=len(x_train)
num_test=len(x_test)


# 运用测试集合求取相应的正确率进行调参，使用训练集样本
#建立9*测试集数量的矩阵，行为九次svc的类别计数器
predict_lebal=np.zeros([num_test])
for i in range(9):
	lebal=i+1
	#对于第i类的样本标签one-hot化
	y_train_hot=change(y_train,lebal,num_train)
	y_test_hot=change(y_test,lebal,num_test)
	#建立svc模型,并且对当前类别i进行模型学习
	svc=SVC(kernel='rbf',gamma=0.125,C=15)
	svc.fit(x_train,y_train_hot)
	predict=svc.predict(x_test)

	get_acc(predict,y_test_hot,lebal)
	get_predict(predict,predict_lebal,lebal)
#输出最终的预测结果
print(predict_lebal[1:40:1])
print(get_acc(predict_lebal,y_test,10))


# #最终测试集导入
# test_data=sio.loadmat('E://pythontest/light/data_test_final.mat')
# #读取对应的特征矩阵
# final_test_data=test_data['data_test_final']
# # 进行预处理和降维
# # 降维数据和训练时一致
# final_train=preprocessing.StandardScaler().fit_transform(final_test_data)
# pca2=PCA(n_components=6)
# x_train=pca2.fit_transform(x_train)
# pca2.fit(final_train)
# new_final_train=pca2.fit_transform(final_train)
# # num_test 一直为相对应的测试数据集
# num_test=len(new_final_train)
# #建立预测标签数组
# predict_lebal=np.zeros([num_test])
# for i in range(9):
# 	lebal=i+1
# 	#对于第i类的样本标签one-hot化
# 	y_train_hot=change(y_train,lebal,num_train)
# 	#建立svc模型,并且对当前类别i进行模型学习
# 	svc=SVC(kernel='rbf',gamma=0.125,C=30)
# 	svc.fit(x_train,y_train_hot)
# 	predict=svc.predict(new_final_train)
# 	get_predict(predict,predict_lebal,lebal)
# #输出最终的预测结果
# #并且将祺转化为真正的标签
# for i in range(num_test):
# 	if predict_lebal[i]==1:
# 		predict_lebal[i]=2
# 	elif predict_lebal[i]==2:
# 		predict_lebal[i]=3
# 	elif predict_lebal[i]==3:
# 		predict_lebal[i]=5
# 	elif predict_lebal[i]==4:
# 		predict_lebal[i]=6
# 	elif predict_lebal[i]==5:
# 		predict_lebal[i]=8
# 	elif predict_lebal[i]==6:
# 		predict_lebal[i]=10
# 	elif predict_lebal[i]==7:
# 		predict_lebal[i]=11
# 	elif predict_lebal[i]==8:
# 		predict_lebal[i]=12
# 	elif predict_lebal[i]==9:
# 		predict_lebal[i]=14
		