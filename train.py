import tensorflow as tf
import numpy as np
import os
from gan import Generator, Discriminator
import dataset
import saver

z_dim = 64
learning_rate = 1e-3
batch_size = 64
epochs = 1000
is_training = True
epochs_d = 3
epochs_g = 1
alpha = 1
beta = 0.05

def asc_epochs_g(epoch):
    return int(min(epochs_g + beta * epoch,10))

def dsc_epoch_d(epoch):
    return int(max(epochs_d - beta * epoch,1))

def celoss_one(logits):
    # 由于sigmoid_cross_entropy_with_logits先对logits做sigmoid激活
    # 所以在gan.py中self.fc2 = keras.layers.Dense(1)
    # 不需要写成self.fc2 = keras.layers.Dense(1,activation='sigmoid')
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                   labels=tf.ones_like(logits))
    return tf.reduce_mean(loss)

def celoss_zero(logits):
    # 由于sigmoid_cross_entropy_with_logits先对logits做sigmoid激活
    # 所以在gan.py中self.fc2 = keras.layers.Dense(1)
    # 不需要写成self.fc2 = keras.layers.Dense(1,activation='sigmoid')
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                   labels=tf.zeros_like(logits))
    return tf.reduce_mean(loss)

def d_loss_fn(generator,discriminator,batch_z,batch_c,batch_x,batch_y,is_training):
    fake_image = generator([batch_z,batch_c],is_training)
    d_fake_logits,d_fake_loss = discriminator([fake_image,batch_c],is_training)
    d_fake_loss1 = celoss_zero(d_fake_logits)
    d_fake_loss2 = tf.reduce_mean(d_fake_loss)

    d_real_logits,d_real_loss = discriminator([batch_x,batch_y],training=is_training)
    d_real_loss1 = celoss_one(d_real_logits)
    d_real_loss2 = tf.reduce_mean(d_real_loss)

    return d_fake_loss1 + alpha * d_fake_loss2 + d_real_loss1 + alpha * d_real_loss2

def g_loss_fn(generator,discriminator,batch_z,batch_c,is_training):
    fake_image = generator([batch_z, batch_c], is_training)
    d_fake_logits, d_fake_loss = discriminator([fake_image, batch_c], is_training)
    d_fake_loss1 = celoss_one(d_fake_logits)
    d_fake_loss2 = tf.reduce_mean(d_fake_loss)

    return d_fake_loss1 + alpha * d_fake_loss2

def train():
    tf.random.set_seed(22)
    np.random.seed(22)
    data_iter = dataset.load_dataset()

    # 利用数组形式实现多输入模型
    generator = Generator()
    generator.build(input_shape=[(None, z_dim),(None, 10)])
    discriminator = Discriminator()
    discriminator.build(input_shape=[(None, 28, 28, 1),(None, 10)])

    g_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    d_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

    for epoch in range(epochs):
        for i in range(int(60000 / batch_size / epochs_d)):

            batch_z = tf.random.uniform([batch_size, z_dim], minval=0., maxval=1.)
            batch_c = []
            for k in range(batch_size):
                batch_c.append(np.random.randint(0,10))
            batch_c = tf.one_hot(tf.convert_to_tensor(batch_c),10)


            # train D
            for epoch_d in range(epochs_d):
                batch_data = next(data_iter)
                batch_x = batch_data[0]
                batch_y = batch_data[1]
                with tf.GradientTape() as tape:
                    d_loss = d_loss_fn(generator, discriminator, batch_z, batch_c, batch_x, batch_y, is_training)
                grads = tape.gradient(d_loss, discriminator.trainable_variables)
                d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

            # train G
            with tf.GradientTape() as tape:
                g_loss = g_loss_fn(generator, discriminator, batch_z, batch_c, is_training)
            grads = tape.gradient(g_loss, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        print('epoch : {epoch} d-loss : {d_loss} g-loss : {g_loss}'
                .format(epoch=epoch, d_loss=d_loss, g_loss=g_loss))

        z = tf.random.uniform([100, z_dim], minval=0., maxval=1.)
        c = []
        for i in range(10):
            for j in range(10):
                c.append(i)
        c = tf.one_hot(tf.convert_to_tensor(c),10)
        fake_image = generator([z, c], training=False)
        img_path = os.path.join('images-2', 'infogan-%d-final.png' % epoch)
        saver.save_image(fake_image.numpy(), img_path, 10)

if __name__ == "__main__":
    train()