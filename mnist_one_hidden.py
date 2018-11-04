import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#mnist数据集的相关参数
INPUT_NODE=784
OUTPUT_NODE=10
#配置神经网络参数
LAYER1_NODE=500 #隐藏层节点数
BATCH_SIZE =100
LEARNING_RATE_BASE=0.8#基础学习率
LEARNING_RATE_DECAY=0.99#学习率的衰减率
REGULARIZATION_RATE=0.0001#描述模型复杂度的正则化向在损失函数中的系数
TRAINING_STEPS=30000 #训练论述
MOVING_AVERGE_DECAY=0.99 #滑动平均衰减率

#一个辅助函数 给定神经网络的输入和所有参数 计算网络的前向传播
def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):
    #当没有提供滑动平均类时 直接使用参数当前的取值
    if avg_class == None:
        #计算隐藏层的前向传播结果
        layer1=tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
        return tf.matmul(layer1,weights2)+biases2
    else:
        #首先使用avg_class.average函数来计算的出变量的滑动平均值
        #然后计算前向传播结果
        layer1=tf.nn.relu(
            tf.matmul(input_tensor,avg_class.average(weights1))+avg_class.average(biases1)
        )
        return tf.matmul(layer1,avg_class.average(weights2))+avg_class.average(biases2)
#训练模型的过程
def train(mnist):
    x=tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
    y_=tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')

    #s生成隐藏层的参数
    weight1=tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    biases1=tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))
    #生成输出层的参数
    weight2=tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2=tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))
    #计算当前参数神经网络前向传播的结果
    y=inference(x,None,weight1,biases1,weight2,biases2)
    #定义存储训练轮数的变量 这个变量是不可训练的变量
    global_step=tf.Variable(0,trainable=False)
    #给定滑动平均衰减率 和训练轮数的变量 初始化 滑动平均类
    #给定训练轮数的变量可以加快早起变量的更新速度
    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERGE_DECAY,global_step)

    #在所有代表神经网络参数的变量上使用滑动平均 其他辅助变量 就不需要了
    #tf.trainable_variable返回的就是图上集合GraphKeys.TRAINABLE_VARIABLE中的元素
    #这个集合的元素是所有没有指定trainable=False的参数
    variable_averages_op=variable_averages.apply(tf.trainable_variables())

    #计算使用滑动平均后的前向传播结果
    #滑动平均不会被改变变量本身 而是会维护一个影子来记录其滑动平均值
    average_y=inference(x,variable_averages,weight1,biases1,weight2,biases2)

    #计算损失函数
    #cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(y,tf.argmax(y_,1))
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    #计算在当前batch下的所有样例的交叉熵平均值
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    #计算L2正则化损失函数
    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    #计算模型的正则化损失 一般只计算神经网络权重的正则化损失 不计算偏置项
    regularization=regularizer(weight1)+regularizer(weight2)
    #总损失等于交叉熵损失加上正则化损失
    loss=cross_entropy_mean+regularization
    #设置指数衰减的学习率
    learning_rate=tf.train.exponential_decay(
        LEARNING_RATE_BASE,#基础学习率
        global_step,
        mnist.train.num_examples/BATCH_SIZE ,#迭代完所有数据需要的迭代次数
        LEARNING_RATE_DECAY #学习率衰减速度
    )
    #优化算法
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    #训练神经网络时 既要反向传播更新神经网络权重 又要更新每一个参数的滑动平均值
    train_op=tf.group(train_step,variable_averages_op)

    #检验使用滑动平均模型的神经网路前向传播结果是否正确
    correct_prediction=tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    #这个运算首先讲讲将一个布尔型的数值转换为实数星 然后计算平均值 这个平均值就是模型在这个一组数据上的正确率
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    #初始化绘画 并开始训练过程
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        #准备验证数据 用来判断是否停止训练
        validate_feed={
            x:mnist.validation.images,
            y_:mnist.validation.labels
        }
        #准备测试数据 实际应用 这部分 训练不可见 这个只是作为模型优劣的最后判断标准
        test_feed={x:mnist.test.images,y_:mnist.test.labels}
        #迭代的训练神经网络
        for i in range(TRAINING_STEPS):
            #每1000轮输出一次在验证数据及上的测试结果
            if i%1000==0:
                #计算滑动平均模型在验证数据上的结果
                validate_acc=sess.run(accuracy,feed_dict=validate_feed)
                print("after %d traning steps,validate acc is %g"%(i,validate_acc))
            #产生一轮使用使用的一个batch的数据 并运行训练过程
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            test_acc=sess.run(train_op,feed_dict={x:xs,y_:ys})
        #训练结束后 在测试数据上的最终正确率
        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print("after %d train steps ,test acc is %g"%(TRAINING_STEPS,test_acc))
def main(argv=None):
    #声明处理mnist数据集的类 这个类自动加载数据
    mnist=input_data.read_data_sets('mnist/',one_hot=True)
    train(mnist)

if __name__=='__main__':
    tf.app.run()
