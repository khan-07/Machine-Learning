import tensorflow as tf

def inceptionmodule(data, LayerDepths):

    tf_inception_filter1 = tf.Variable(tf.truncated_normal([1,1, data.shape[3], LayerDepths[0]], stddev=0.1))
    tf_inception_filter2 = tf.Variable(tf.truncated_normal([1, 1,data.shape[3], LayerDepths[1]], stddev=0.1))
    tf_inception_filter3 = tf.Variable(tf.truncated_normal([1, 1,data.shape[3], LayerDepths[2]], stddev=0.1))
    tf_inception_filter4 = tf.Variable(tf.truncated_normal([3, 3,LayerDepths[0], LayerDepths[3]], stddev=0.1))
    tf_inception_filter5 = tf.Variable(tf.truncated_normal([5, 5,LayerDepths[1], LayerDepths[4]], stddev=0.1))
    tf_inception_filter6 = tf.Variable(tf.truncated_normal([1, 1,data.shape[3], LayerDepths[5]], stddev=0.1))

    conv1 = tf.nn.conv2d(data, tf_inception_filter1,strides=[1,1,1,1],padding="SAME")

    conv2 = tf.nn.conv2d(data, tf_inception_filter2, strides=[1, 1, 1, 1], padding="SAME")

    max1 = tf.nn.max_pool(data,ksize=[1,3,3,1],strides=[1,1,1,1], padding="SAME")

    conv3 = tf.nn.conv2d(data, tf_inception_filter3, strides=[1, 1, 1, 1], padding="SAME")

    conv4 = tf.nn.conv2d(conv1, tf_inception_filter4, strides=[1, 1, 1, 1], padding="SAME")

    conv5 = tf.nn.conv2d(conv2, tf_inception_filter5, strides=[1, 1, 1, 1], padding="SAME")

    conv6 = tf.nn.conv2d(max1, tf_inception_filter6, strides=[1, 1, 1, 1], padding="SAME")

    return tf.concat([conv3,conv4,conv5,conv6],axis=3)


def google_net(data):

    #depths of each convolution in incepiton module

    inceptionlayer1 = [96,16,64,128,32,32]
    inceptionlayer2 = [128, 32, 128, 192, 96, 64]
    inceptionlayer3 = [96, 16, 192, 208, 48, 64]
    inceptionlayer4 = [112, 24, 160, 224, 64, 64]
    inceptionlayer5 = [128, 24, 128, 256, 64, 64]
    inceptionlayer6 = [144, 32, 112, 288, 64, 64]
    inceptionlayer7 = [160, 32, 256, 320, 128, 128]
    inceptionlayer8 = [160, 32, 256, 320, 128, 128]
    inceptionlayer9 = [192, 48, 384, 384, 128, 128]

    #initalize weigths for all conv layers that are outside inception module

    layer1_weights =  tf.Variable(tf.truncated_normal([7,7,3,64], stddev=0.1))
    layer2_weights = tf.Variable(tf.truncated_normal([1, 1, 64, 64], stddev=0.1))
    layer3_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 192], stddev=0.1))
    output_weights = tf.Variable(tf.truncated_normal([]))

    bias1 = tf.Variable(tf.constant(1.0,shape=[64]))
    bias2 = tf.Variable(tf.constant(1.0,shape=[64]))
    bias3 = tf.Variable(tf.constant(1.0,shape=[192]))

    #build the network

    conv1 = tf.nn.conv2d(data, layer1_weights,strides=[1,2,2,1],padding='SAME')
    conv1 = tf.nn.bias_add(conv1,bias1)
    conv1 = tf.nn.relu(conv1)

    max1  = tf.nn.max_pool(conv1,ksize= [1,3,3,1],strides= [1,2,2,1],padding="SAME")

    norm1 =  tf.nn.local_response_normalization(max1, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)

    conv2 = tf.nn.conv2d(norm1,layer2_weights,strides=[1,1,1,1],padding="SAME")

    conv3 = tf.nn.conv2d(conv2,layer3_weights,strides=[1,1,1,1],padding="SAME")

    norm2 = tf.nn.local_response_normalization(conv3, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)

    max2  = tf.nn.max_pool(norm2,ksize= [1,3,3,1],strides= [1,2,2,1],padding="SAME")

    inception1 = inceptionmodule(max2, inceptionlayer1)

    inception2 = inceptionmodule(inception1, inceptionlayer2)

    max3 = tf.nn.max_pool(inception2,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")

    inception3 = inceptionmodule(max3,inceptionlayer3)

    inception4 = inceptionmodule(inception3, inceptionlayer4)

    inception5 = inceptionmodule(inception4, inceptionlayer5)

    inception6 = inceptionmodule(inception5, inceptionlayer6)

    inception7 = inceptionmodule(inception6, inceptionlayer7)

    max4 = tf.nn.max_pool(inception7,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")

    inception8 = inceptionmodule(max4, inceptionlayer8)

    inception9 = inceptionmodule(inception8, inceptionlayer9)

    avg1 = tf.nn.pool(inception9,window_shape=[1,7,7,1],pooling_type="AVG",padding="VALID")

    avg1 = avg1.get_shape().as_list()

    dropout1 = tf.nn.dropout(avg1, 0.4)

    return tf.layers.dense(dropout1,units=1000)

