import tensorflow as tf
from tensorflow import keras

class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        self.fc1 = keras.layers.Dense(1024)
        self.bn1 = keras.layers.BatchNormalization()

        self.fc2 = keras.layers.Dense(7 * 7 * 128)
        self.bn2 = keras.layers.BatchNormalization()

        # [b,7,7,128] --> [b,14,14,64]
        self.conv1 = keras.layers.Conv2DTranspose(64,4,2,'same')
        self.bn3 = keras.layers.BatchNormalization()

        # [b,14,14,64] --> [b,28,28,1]
        self.conv2 = keras.layers.Conv2DTranspose(1,4,2,'same')

    def call(self, input, training=None, mask=None):
        inputz = input[0]
        inputc = input[1]
        inputs = tf.concat([inputz,inputc],axis=1)

        x = self.bn1(self.fc1(inputs),training=training)
        x = self.bn2(self.fc2(x),training=training)

        x = tf.reshape(x,[-1,7,7,128])

        x = tf.nn.relu(self.bn3(self.conv1(x),training=training))
        x = self.conv2(x)

        # tanh激活函数使得输出范围在[-1,1]之间
        x = tf.tanh(x)

        return x

class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        # [b,28,28,1] --> [b,13,13,64]
        self.conv1 = keras.layers.Conv2D(64,4,2,'valid')

        # [b,13,13,64] --> [b,5,5,128]
        self.conv2 = keras.layers.Conv2D(128,5,2,'valid')
        self.bn1 = keras.layers.BatchNormalization()

        # [b,4,4,128] --> [b,1,1,256]
        self.conv3 = keras.layers.Conv2D(256,4, 2, 'valid')
        self.bn2 = keras.layers.BatchNormalization()

        self.fc1 = keras.layers.Dense(1024)

        self.fc2 = keras.layers.Dense(1)

        self.flatten = keras.layers.Flatten()

        self.fc3 = keras.layers.Dense(128)

        self.fc4 = keras.layers.Dense(10)

    def call(self,input,training=None,mask=None):
        inputx = input[0]
        inputc = input[1]

        x = tf.nn.leaky_relu(self.conv1(inputx))
        x = tf.nn.leaky_relu(self.bn1(self.conv2(x),training=training))
        x = tf.nn.leaky_relu(self.bn2(self.conv3(x),training=training))

        x = self.flatten(x)
        x = self.fc1(x)
        logits = self.fc2(x)

        if inputc is None:
            return logits

        x = tf.nn.leaky_relu(self.fc3(x))
        x = self.fc4(x)

        # 注意tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred, name=None)！
        # softmax_cross_entropy_with_logits先对logits做softmax，再计算和labels的交叉熵
        # 因此在全连接层中不需要再添加activation='softmax'！
        # tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred, name=None)等价于：
        # y_ = tf.nn.softmax(pred)
        # a = y_ * y 这里的乘法是点乘
        # a = tf.reduce_sum(a,axis=1)
        # a = -tf.math.log(a)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=inputc,logits=x)
        return logits,loss

if __name__ == "__main__":
    g = Generator()
    d = Discriminator()

    x = tf.random.normal([2, 28, 28, 1])
    z = tf.random.normal([2, 64])
    c = tf.one_hot([0,1],10)

    x_hat = g([z,c])
    z_hat ,c_hat = d([x,c])

    print(x_hat.shape)
    print(z_hat.shape)
    print(c_hat.shape)
