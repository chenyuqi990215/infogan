实验1：（epoch=200）

说明：images中生成的mnist数据没有考虑到latent！（非监督学习）
实验参数设置：epochs_d=3 epochs_g=1
出现的问题：训练到115 epochs 图像开始模糊，且g-loss越来越大，d-loss一直很小

def d_loss_fn(generator,discriminator,batch_z,batch_c,batch_x,batch_y,is_training):
    fake_image = generator([batch_z,batch_c],is_training)
    d_fake_logits,d_fake_loss = discriminator([fake_image,batch_c],is_training)
    d_fake_loss1 = celoss_zero(d_fake_logits)
    d_fake_loss2 = tf.reduce_mean(d_fake_loss)

    d_real_logits,d_real_loss = discriminator([batch_x,batch_y],training=is_training)
    d_real_loss1 = celoss_one(d_real_logits)

    return d_fake_loss1 + alpha * d_fake_loss2 + d_real_loss1 

实验2：（epoch=1000）

说明：images中生成的mnist数据没有考虑到latent！（监督学习）
实验参数设置：epochs_d=3 epochs_g=1
出现的问题：训练到115 epochs 图像开始模糊，且g-loss越来越大，d-loss一直很小

def d_loss_fn(generator,discriminator,batch_z,batch_c,batch_x,batch_y,is_training):
    fake_image = generator([batch_z,batch_c],is_training)
    d_fake_logits,d_fake_loss = discriminator([fake_image,batch_c],is_training)
    d_fake_loss1 = celoss_zero(d_fake_logits)
    d_fake_loss2 = tf.reduce_mean(d_fake_loss)

    d_real_logits,d_real_loss = discriminator([batch_x,batch_y],training=is_training)
    d_real_loss1 = celoss_one(d_real_logits)
    d_real_loss2 = tf.reduce_mean(d_real_loss)

    return d_fake_loss1 + alpha * d_fake_loss2 + d_real_loss1 + alpha * d_real_loss2