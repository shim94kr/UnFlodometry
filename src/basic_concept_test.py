import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

# Range of disparity/inverse depth values
DISP_SCALING = 10
MIN_DISP = 0.01

tgt_image = tf.Variable(tf.ones((16, 416, 128, 3)))
cnv1  = slim.conv2d(tgt_image, 32,  [7, 7], stride=2, scope='cnv1')
cnv1b = slim.conv2d(cnv1,  32,  [7, 7], stride=1, scope='cnv1b')
cnv2  = slim.conv2d(cnv1b, 64,  [5, 5], stride=2, scope='cnv2')
cnv2b = slim.conv2d(cnv2,  64,  [5, 5], stride=1, scope='cnv2b')
cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
cnv3b = slim.conv2d(cnv3,  128, [3, 3], stride=1, scope='cnv3b')
cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
cnv4b = slim.conv2d(cnv4,  256, [3, 3], stride=1, scope='cnv4b')
cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b')
cnv6  = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6')
cnv6b = slim.conv2d(cnv6,  512, [3, 3], stride=1, scope='cnv6b')
cnv7  = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
cnv7b = slim.conv2d(cnv7,  512, [3, 3], stride=1, scope='cnv7b')
upcnv7 = slim.conv2d_transpose(cnv7b, 512, [3, 3], stride=2, scope='upcnv7')
# There might be dimension mismatch due to uneven down/up-sampling
i7_in  = tf.concat([upcnv7, cnv6b], axis=3)
icnv7  = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')

upcnv6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6')
i6_in  = tf.concat([upcnv6, cnv5b], axis=3)
icnv6  = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')

upcnv5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
i5_in  = tf.concat([upcnv5, cnv4b], axis=3)
icnv5  = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')

upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')
i4_in  = tf.concat([upcnv4, cnv3b], axis=3)
icnv4  = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
disp4  = DISP_SCALING * slim.conv2d(icnv4, 1,   [3, 3], stride=1,
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4') + MIN_DISP
disp4_up = tf.image.resize_bilinear(disp4, [np.int(416/4), np.int(128/4)])

cnv1  = slim.conv2d(tgt_image,16,  [7, 7], stride=2, scope='cnv1')
cnv2  = slim.conv2d(cnv1, 32,  [5, 5], stride=2, scope='cnv2')
cnv3  = slim.conv2d(cnv2, 64,  [3, 3], stride=2, scope='cnv3')
cnv4  = slim.conv2d(cnv3, 128, [3, 3], stride=2, scope='cnv4')
cnv5  = slim.conv2d(cnv4, 256, [3, 3], stride=2, scope='cnv5')
cnv6  = slim.conv2d(cnv5, 256, [3, 3], stride=2, scope='cnv6')
cnv7  = slim.conv2d(cnv6, 256, [3, 3], stride=2, scope='cnv7')

upcnv5 = slim.conv2d_transpose(cnv5, 256, [3, 3], stride=2, scope='upcnv5')

upcnv4 = slim.conv2d_transpose(upcnv5, 128, [3, 3], stride=2, scope='upcnv4')
mask4 = slim.conv2d(upcnv4, 2 * 2, [3, 3], stride=1, scope='mask4',
                    normalizer_fn=None, activation_fn=None)

upcnv3 = slim.conv2d_transpose(upcnv4, 64, [3, 3], stride=2, scope='upcnv3')
mask3 = slim.conv2d(upcnv3, 2 * 2, [3, 3], stride=1, scope='mask3',
                    normalizer_fn=None, activation_fn=None)

upcnv2 = slim.conv2d_transpose(upcnv3, 32, [5, 5], stride=2, scope='upcnv2')
mask2 = slim.conv2d(upcnv2,2 * 2, [5, 5], stride=1, scope='mask2',
                    normalizer_fn=None, activation_fn=None)

upcnv1 = slim.conv2d_transpose(upcnv2, 16, [7, 7], stride=2, scope='upcnv1')
mask1 = slim.conv2d(upcnv1, 2 * 2, [7, 7], stride=1, scope='mask1',
                    normalizer_fn=None, activation_fn=None)

tmp = np.array([0, 1])
ref_exp_mask = np.tile(tmp,
                       (16,
                        int(416 / (2 ** 1)),
                        int(128 / (2 ** 1)),
                        1))
ref_exp_mask = tf.constant(ref_exp_mask, dtype=tf.float32)
print (cnv1.get_shape())