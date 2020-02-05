实验1：（epoch=200）

说明：images中生成的mnist数据没有考虑到latent！（非监督学习）
出现的问题：训练到115 epochs 图像开始模糊，且g-loss越来越大，d-loss一直很小

实验2：（epoch=1000）

说明：images中生成的mnist数据没有考虑到latent！（监督学习）
出现的问题：训练到220 epochs 图像开始模糊，且g-loss越来越大，d-loss一直很小

实验3：（epoch=200）

说明：images中生成的mnist数据考虑到latent！（非监督学习）
实验结果：从epoch=99时生成的图片可以看出从左到右数字越来越细，但是从上到下的变化不明显！
出现的问题：训练到100 epochs 图像开始模糊，且g-loss越来越大，d-loss一直很小
