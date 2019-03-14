import tensorflow as tf

def alexnet(data):

    #create Variables

    #Filters
    layer1_Weights = tf.Variable(tf.truncated_normal([11,11,3,96], stddev=0.1))
    layer2_Weights = tf.Variable(tf.truncated_normal([5, 5, 96, 256], stddev=0.1))
    layer3_Weights = tf.Variable(tf.truncated_normal([3, 3, 256, 384], stddev=0.1))
    layer4_Weights = tf.Variable(tf.truncated_normal([3, 3, 384, 384], stddev=0.1))
    layer5_Weights = tf.Variable(tf.truncated_normal([3, 3, 384, 256], stddev=0.1))
    output_Weigts =  tf.Variable(tf.truncated_normal([4096, 1000], stddev=0.1))

    #Biases
    bias1 = tf.Variable(tf.constant(1.0, shape=[96]))
    bias2 = tf.Variable(tf.constant(1.0, shape=[256]))
    bias3 = tf.Variable(tf.constant(1.0, shape=[384]))
    bias4 = tf.Variable(tf.constant(1.0, shape=[384]))
    bias5 = tf.Variable(tf.constant(1.0, shape=[256]))
    output_biases = tf.Variable(tf.constant(1.0, shape=[1000]))



    #1st Convolution Layer

    conv1 = tf.nn.conv2d(data, layer1_Weights,[1,4,4,1],"VALID")
    conv1 = tf.nn.bias_add(conv1, bias1)
    conv1 = tf.nn.relu(conv1)

    #1st Max Pooling Layer

    max1 = tf.nn.max_pool(conv1,[1,3,3,1],[1,2,2,1],"VALID")

    #1st normalization layer

    norm1 =  tf.nn.local_response_normalization(max1, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)

    #2nd Convolution Layer

    conv2 = tf.nn.conv2d(norm1, layer2_Weights, [1, 1, 1, 1], "SAME" )
    conv2 = tf.nn.bias_add(conv2, bias2)
    conv2 = tf.nn.relu(conv2)

    #2nd Max Pooling Layer

    max2 = tf.nn.max_pool(conv2, [1, 3, 3, 1], [1, 2, 2, 1], "VALID")

    #3rd Convolution Layer

    conv3 = tf.nn.conv2d(max2, layer3_Weights, [1, 1, 1, 1], "SAME")
    conv3 = tf.nn.bias_add(conv3, bias3)
    conv3 = tf.nn.relu(conv3)

    #4th Convolution Layer

    conv4 = tf.nn.conv2d(conv3,layer4_Weights, [1, 1, 1, 1], "SAME")
    conv4 = tf.nn.bias_add(conv4, bias4)
    conv4 = tf.nn.relu(conv4)

    #5th Convolution Layer

    conv5 = tf.nn.conv2d(conv4, layer5_Weights, [1, 1, 1, 1], "SAME")
    conv5 = tf.nn.bias_add(conv5, bias5)
    conv5 = tf.nn.relu(conv5)

    #3rd Max Pooling Layer

    max3 = tf.nn.max_pool(conv5, [1, 3, 3, 1], [1, 2, 2, 1], "VALID")

    #flatten the ndarray to a list before feeding to fcn

    max3 = max3.get_shape().as_list()

    #1st dropout Layer

    dropout1 = tf.nn.dropout(max3, 0.5)

    #1st fully connected layer
    fcn1 = tf.contrib.layers.fully_connected(dropout1, 4096)

    #2nd dropout Layer
    dropout2 = tf.nn.dropout(fcn1, 0.5)

    #2nd  fully connected layer

    fcn2 = tf.contrib.layers.fully_connected(dropout2, 4096)

    #3rd  fully connected layer
    #since tf cross entropy function automatically performs softmax
    #we need to give it unscaled logits

    output = tf.matmul(fcn2, output_Weigts) + output_biases

    return output








