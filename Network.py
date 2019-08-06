import tensorflow as tf


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def BN(x, phase_BN, scope):
    return tf.layers.batch_normalization(x, momentum=0.9, training=phase_BN)

def cnnLayer(scope_name, inputs, outChannels, kernel_size, is_training, relu=True, maxpool=True):
    with tf.variable_scope(scope_name) as scope:
        inChannels = inputs.shape[-1].value
        W_conv = tf.get_variable('W_conv', [kernel_size, kernel_size, inChannels, outChannels])
        b_conv = tf.get_variable('b_conv', [outChannels])
        x_conv = conv2d(inputs, W_conv) + b_conv
        out = BN(x_conv, is_training, scope)
        if relu:
            out = tf.nn.relu(out)
        if maxpool:
            out = max_pool_2x2(out)

        return out


def cnn3x3(scope_name, inputs, outChannels, is_training, relu=True):
    with tf.variable_scope(scope_name) as scope:
        inChannels = inputs.shape[-1].value
        W_conv = tf.get_variable('W_conv', [3, 3, inChannels, outChannels])
        b_conv = tf.get_variable('b_conv', [outChannels])
        x_conv = conv2d(inputs, W_conv) + b_conv
        out = BN(x_conv, is_training, scope)
        if relu:
            out = tf.nn.relu(out)

        return out



def conv_transpose(scope_name, inputs, outChannels):
    with tf.variable_scope(scope_name) as scope:
        inChannels = inputs.shape[-1].value
        W = tf.get_variable('W_deconv', [3, 3, outChannels, inChannels])
        output_shape = [inputs.shape[0].value, 2 * inputs.shape[1].value, 2 * inputs.shape[2].value, outChannels]
        output = tf.nn.conv2d_transpose(inputs, W, output_shape=output_shape, strides=[1, 2, 2, 1],
                                        padding='SAME')
        return output



def UNet_2(inputs, phase_BN):
    down1_1 = cnn3x3('down1_1', inputs, 32, phase_BN)
    down1_2 = cnn3x3('down1_2', down1_1, 32, phase_BN)

    down2_0 = max_pool_2x2(down1_2)
    down2_1 = cnn3x3('down2_1', down2_0, 64, phase_BN)
    down2_2 = cnn3x3('down2_2', down2_1, 64, phase_BN)

    down3_0 = max_pool_2x2(down2_2)
    down3_1 = cnn3x3('down3_1', down3_0, 128, phase_BN)
    down3_2 = cnn3x3('down3_2', down3_1, 128, phase_BN)

    down4_0 = max_pool_2x2(down3_2)
    down4_1 = cnn3x3('down4_1', down4_0, 256, phase_BN)
    down4_2 = cnn3x3('down4_2', down4_1, 256, phase_BN)

    down5_0 = max_pool_2x2(down4_2)
    down5_1 = cnn3x3('down5_1', down5_0, 512, phase_BN)
    down5_2 = cnn3x3('down5_2', down5_1, 512, phase_BN)

    up4_0 = tf.concat([conv_transpose('up4_0', down5_2, 256), down4_2], axis=-1)
    up4_1 = cnn3x3('up4_1', up4_0, 256, phase_BN)
    up4_2 = cnn3x3('up4_2', up4_1, 256, phase_BN)

    up3_0 = tf.concat([conv_transpose('up3_0', up4_2, 128), down3_2], axis=-1)
    up3_1 = cnn3x3('up3_1', up3_0, 128, phase_BN)
    up3_2 = cnn3x3('up3_2', up3_1, 128, phase_BN)

    up2_0 = tf.concat([conv_transpose('up2_0', up3_2, 64), down2_2], axis=-1)
    up2_1 = cnn3x3('up2_1', up2_0, 64, phase_BN)
    up2_2 = cnn3x3('up2_2', up2_1, 64, phase_BN)

    up1_0 = tf.concat([conv_transpose('up1_0', up2_2, 32), down1_2], axis=-1)
    up1_1 = cnn3x3('up1_1', up1_0, 32, phase_BN)
    up1_2 = cnn3x3('up1_2', up1_1, 32, phase_BN)

    up1_3 = cnnLayer('up1_3', up1_2, 1, 1, phase_BN, relu=False, maxpool=False)
    out = tf.sigmoid(up1_3)

    return out
