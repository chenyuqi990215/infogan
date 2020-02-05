import tensorflow as tf
from tensorflow.keras import datasets
import saver


def preprocess(x, y):
    x = tf.reshape(tf.cast(x, dtype=tf.float32) / 127.5 - 1,[-1,28,28,1])
    y = tf.one_hot(tf.cast(y, dtype=tf.int32),10)
    return x, y

def load_dataset(batch=32,shuffle=1024,epoch=1000):
    (x, y),(tx, ty) = datasets.mnist.load_data()

    # 构建dataset对象，方便对数据的打乱，批处理等操作
    train_db = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(shuffle).batch(batch).repeat(epoch)
    train_db = train_db.map(preprocess)
    train_iter = iter(train_db)
    return train_iter

if __name__ == "__main__":
    train_iter = load_dataset()
    batch_data = next(train_iter)
    x = batch_data[0]
    y = batch_data[1]
    saver.save_image(x.numpy(),"images/test.png",10)
    print(x.shape)
    print(x)
    print(y.shape)

