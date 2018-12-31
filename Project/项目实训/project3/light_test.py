import scipy.io as sio
import random
from functools import reduce
from numpy import exp
from numpy import *
import numpy
from datetime import datetime
from sklearn.externals import joblib
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

#从本地读取样本
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
    n=len(array)
    tp=[y for i in range(n)]
    return tp
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
train=concatenate((array,array2,array3,array4,array5,array6,array7,array8,array9))
y=concatenate((y1,y2,y3,y4,y5,y6,y7,y8,y9))


def hotchange(y):
    '''
    one-hot化标签集，y为以为一维向量
    '''
    n=len(y)
    label=zeros((n,9))
    for i in range(n):
        for j in range(9):
            if y[i]==(j+1):
                label[i,j]=0.9
            else:
                label[i,j]=0.1
    return label

def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

# 节点类，负责记录和维护节点自身信息以及与这个节点相关的上下游连接，实现输出值和误差项的计算。
class Node():
    def __init__(self, layer_index, node_index):
        '''
        构造节点对象。
        layer_index: 节点所属的层的编号
        node_index: 节点的编号
        '''
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.upstream = []
        self.output = 0
        self.delta = 0

    def set_output(self, output):
        '''
        设置节点的输出值。如果节点属于输入层会用到这个函数。
        '''
        self.output = output

    def append_downstream_connection(self, conn):
        '''
        添加一个到下游节点的连接
        '''
        self.downstream.append(conn)

    def append_upstream_connection(self, conn):
        '''
        添加一个到上游节点的连接
        '''
        self.upstream.append(conn)

    def calc_output(self):
        '''
        根据式1计算节点的输出
        '''
        # 每个节点的输出算法，N元一次方程求和
        output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, self.upstream, 0)
        # 结果放入激活函数
        self.output = sigmoid(output)

    def calc_hidden_layer_delta(self):
        '''
        节点属于隐藏层时，根据式4计算delta
        '''
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def calc_output_layer_delta(self, label):
        '''
        节点属于输出层时，根据式3计算delta
        '''
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def __str__(self):
        '''
        打印节点的信息
        '''
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str

class ConstNode():
    def __init__(self, layer_index, node_index):
        '''
        构造节点对象。
        layer_index: 节点所属的层的编号
        node_index: 节点的编号
        '''    
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1
    def append_downstream_connection(self, conn):
        '''
        添加一个到下游节点的连接
        '''       
        self.downstream.append(conn)
    def calc_hidden_layer_delta(self):
        '''
        节点属于隐藏层时，根据式4计算delta
        '''
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta
    def calc_output_layer_delta(self, label):
        '''
        节点属于输出层时，根据式3计算delta
        '''
        self.delta = self.output * (1 - self.output) * (label - self.output)
    def __str__(self):
        '''
        打印节点的信息
        '''
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str

class Layer():
    def __init__(self, layer_index, node_count):
        '''
        初始化一层
        layer_index: 层编号
        node_count: 层所包含的节点个数
        '''
        self.layer_index = layer_index
        self.nodes = []
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))
        self.nodes.append(ConstNode(layer_index, node_count))
    def set_output(self, data):
        '''
        设置层的输出。当层是输入层时会用到。
        '''
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])
    def calc_output(self):
        '''
        计算层的输出向量
        '''
        for node in self.nodes[:-1]:
            node.calc_output()
    def dump(self):
        '''
        打印层的信息
        '''
        for node in self.nodes:
            print(node)

class Connection():
    def __init__(self, upstream_node, downstream_node):
        '''
        初始化连接，权重初始化为是一个很小的随机数
        upstream_node: 连接的上游节点
        downstream_node: 连接的下游节点
        '''
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.weight = random.uniform(-0.1, 0.1)
        self.gradient = 0.0
    def calc_gradient(self):
        '''
        计算梯度
        '''
        self.gradient = self.downstream_node.delta * self.upstream_node.output
    def get_gradient(self):
        '''
        获取当前的梯度
        '''
        return self.gradient
    def update_weight(self, rate):
        '''
        根据梯度下降算法更新权重
        '''
        self.calc_gradient()
        self.weight += rate * self.gradient
    def __str__(self):
        '''
        打印连接信息
        '''
        return '(%u-%u) -> (%u-%u) = %f' % (
            self.upstream_node.layer_index, 
            self.upstream_node.node_index,
            self.downstream_node.layer_index, 
            self.downstream_node.node_index, 
            self.weight)

class Connections():
    def __init__(self):
        self.connections = []
    def add_connection(self, connection):
        self.connections.append(connection)
    def dump(self):
        for conn in self.connections:
            print(conn)

class Network():
    def __init__(self, layers):
        '''
        初始化一个全连接神经网络
        layers: 二维数组，描述神经网络每层节点数
        '''
        self.connections = Connections()
        self.layers = []
        layer_count = len(layers)
        node_count = 0;
        for i in range(layer_count):
            self.layers.append(Layer(i, layers[i]))
        for layer in range(layer_count - 1):
            connections = [Connection(upstream_node, downstream_node) 
                           for upstream_node in self.layers[layer].nodes
                           for downstream_node in self.layers[layer + 1].nodes[:-1]]
            for conn in connections:
                self.connections.add_connection(conn)
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)
    def train(self, labels, data_set, rate, iteration):
        '''
        训练神经网络
        labels: 数组，训练样本标签。每个元素是一个样本的标签。
        data_set: 二维数组，训练样本特征。每个元素是一个样本的特征。
        '''
        for i in range(iteration):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)
    def train_one_sample(self, label, sample, rate):
        '''
        内部函数，用一个样本训练网络
        '''
        self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rate)
    def calc_delta(self, label):
        '''
        内部函数，计算每个节点的delta
        '''
        output_nodes = self.layers[-1].nodes
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()
    def update_weight(self, rate):
        '''
        内部函数，更新每个连接权重
        '''
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)
    def calc_gradient(self):
        '''
        内部函数，计算每个连接的梯度
        '''
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()
    def get_gradient(self, label, sample):
        '''
        获得网络在一个样本下，每个连接上的梯度
        label: 样本标签
        sample: 样本输入
        '''
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()
    def predict(self, sample):
        '''
        根据输入的样本预测输出值
        sample: 数组，样本的特征，也就是网络的输入向量
        '''
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        return list(map(lambda node: node.output, self.layers[-1].nodes[:-1]))
    def dump(self):
        '''
        打印网络信息
        '''
        for layer in self.layers:
            layer.dump()

def gradient_check(network, sample_feature, sample_label):
    '''
    梯度检查
    network: 神经网络对象
    sample_feature: 样本的特征
    sample_label: 样本的标签
    '''
    # 计算网络误差
    network_error = lambda vec1, vec2: \
            0.5 * reduce(lambda a, b: a + b, 
                      map(lambda v: (v[0] - v[1]) * (v[0] - v[1]),
                          zip(vec1, vec2)))
    # 获取网络在当前样本下每个连接的梯度
    network.get_gradient(sample_label,sample_feature)
    # 对每个权重做梯度检查    
    for conn in network.connections.connections: 
        # 获取指定连接的梯度
        actual_gradient = conn.get_gradient()
        # 增加一个很小的值，计算网络的误差
        epsilon = 0.0001
        conn.weight += epsilon
        error1 = network_error(network.predict(sample_feature), sample_label)
        # 减去一个很小的值，计算网络的误差
        conn.weight -= 2 * epsilon # 刚才加过了一次，因此这里需要减去2倍
        error2 = network_error(network.predict(sample_feature), sample_label)
        # 根据式6计算期望的梯度值
        expected_gradient = (error2 - error1) / (2 * epsilon)
        # 打印
        print('expected gradient: \t%f\nactual gradient: \t%f' % (
            expected_gradient, actual_gradient))


def get_result(vec):
    max_value_index=0
    max_value=0
    for i in range(len(vec)):
        if vec[i]>max_value:
            max_value=vec[i]
            max_value_index=i
    return max_value_index

def evaluate(network_light,test_data,test_labels):
    #返回神经网络测试结果的正确率 
    #error为错误样本数，total为测试总样本数
    
    error=0
    total=len(test_data)
    for i in range(total):
        label=test_labels[i]
        predict=get_result(network_light.predict(test_data[i]))
        if label!=predict:
            error+=1
    return float(error)/float(total)



def train_evaluate(network_light):
    '''
    
    训练神经网络
    error_ratio为错误率
    last_error_ratio最近错误率
    epoch为循环次数，即神经网络学习次数
    '''

    last_error_ratio =1.0
    epoch=0
    while True:
        epoch+=1
        network_light.train(y_train,x_train,0.3,1)
        print('%s epoch %d finshied' %(datetime.now(),epoch))
        #输出第几次训练神经网络
        if epoch%5==0:
            #每训练一次神经网络后进行检测 ，若错误率开始上升则终止学习。
            error_ratio=evaluate(network_light,x_test,y_test)
            print('%s after epoch %d, error ratio is %f' % (datetime.now(), epoch, error_ratio))
            
            if error_ratio>last_error_ratio:
                break
            else:
                last_error_ratio=error_ratio

def get_predict(network_light,data):
    num=len(data)
    predict_result=[]
    for i in range(num):
        predict_result[i]=get_result(network_light.predict(data[i]))
    return predict_result





#根据7：3的比率分割数据集，随机种子设为1
from sklearn.cross_validation import train_test_split
x_train,x_test, y_train, y_test =train_test_split(train,y,test_size=0.3, random_state=1)
#运用归一化方法进行数据标准化
x_train=normalize(x_train)
x_test=normalize(x_test)
print(x_train.shape)
# print(y_train[0:5])
#相应的pca降维
pca=PCA(n_components=6)
x_train=pca.fit_transform(x_train)
x_test=pca.fit_transform(x_test)
print(x_train.shape)
# # value=sort(pca.explained_variance_)
# print(value)
#标签集的one-hot化
y_train=hotchange(y_train)
# y_test=change(y_test)
# 验证集的标签集合不需要one-hot化


network_light=Network([6,8,9])
print('finsh the creating network,start to train_evaluate the network')


    # 训练神经网络
    # error_ratio为错误率
    # last_error_ratio最近错误率
    # epoch为循环次数，即神经网络学习次数
  

last_error_ratio =1.0
epoch=0
while True:
    epoch+=1
    network_light.train(y_train,x_train,0.003,1)
    print('%s epoch %d finshied' %(datetime.now(),epoch))
    #输出第几次训练神经网络
    if epoch%5==0:
        #每训练一次神经网络后进行检测 ，若错误率开始上升则终止学习。
        error_ratio=evaluate(network_light,x_test,y_test)
        print('%s after epoch %d, error ratio is %f' % (datetime.now(), epoch, error_ratio))
        if error_ratio>last_error_ratio:
            break
        else:
            last_error_ratio=error_ratio

