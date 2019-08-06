import tensorflow as tf
import scipy.io as sio
import numpy as np
import os
import Network

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

results_dir = "./new_model/"
DATA_PATH = './Data/'
TFRECORD_TRAIN_PATH = [DATA_PATH + 'train_A.tfrecord', DATA_PATH + 'train_B.tfrecord', DATA_PATH + 'train_C.tfrecord']
TFRECORD_VALID_PATH = [DATA_PATH + 'valid_A.tfrecord', DATA_PATH + 'valid_B.tfrecord']
lr_optical = 1e-8
lr_digital = 1e-4
print('lr_optical:' + str(lr_optical))
print('lr_digital:' + str(lr_digital))

##########################################   Functions  #############################################

def parse_element(example):
    # from tfrecord file to data
    N = 278
    N_Phi = 21

    features = tf.parse_single_example(example,
                                       features={
                                           'RGB': tf.FixedLenFeature([], tf.string),
                                           'DPPhi': tf.FixedLenFeature([], tf.string),
                                           'DP': tf.FixedLenFeature([], tf.string),
                                       })

    RGB_flat = tf.decode_raw(features['RGB'], tf.uint8)
    RGB = tf.reshape(RGB_flat, [N, N, 3])

    DPPhi_flat = tf.decode_raw(features['DPPhi'], tf.uint8)
    DPPhi = tf.reshape(DPPhi_flat, [N, N, N_Phi])

    DP_flat = tf.decode_raw(features['DP'], tf.uint8)
    DP = tf.reshape(DP_flat, [N, N])

    return RGB, DPPhi, DP


def data_augment(RGB_batch_float, DPPhi_float, Phi_batch_scaled):
    data0 = tf.concat([RGB_batch_float, DPPhi_float, tf.expand_dims(Phi_batch_scaled, -1)], axis=3)

    # flip both images and labels
    data1 = tf.map_fn(lambda img: tf.image.random_flip_up_down(tf.image.random_flip_left_right(img)), data0)

    # only adjust the RGB value of the image
    r1 = tf.random_uniform([]) * 0.3 + 0.8
    RGB_out = data1[:, :, :, 0:3] * r1

    return RGB_out, data1[:, :, :, 3:-1], data1[:, :, :, -1]


def read2batch(TFRECORD_PATH, batchsize):
    # load tfrecord and make them to be usable data
    dataset = tf.data.TFRecordDataset(TFRECORD_PATH)
    dataset = dataset.map(parse_element).repeat().shuffle(buffer_size=5000)
    dataset = dataset.batch(batchsize, drop_remainder=True)
    iterator = dataset.make_one_shot_iterator()
    RGB_batch, DPPhi_batch, DP_batch = iterator.get_next()

    RGB_batch_float = tf.image.convert_image_dtype(RGB_batch, tf.float32)
    DPPhi_float = tf.cast(DPPhi_batch, tf.float32)
    Phi_batch_scaled = (tf.cast(DP_batch, tf.float32) - 10) / 210

    RGB_batch_float, DPPhi_float, Phi_batch_scaled = data_augment(RGB_batch_float, DPPhi_float, Phi_batch_scaled)

    return RGB_batch_float, DPPhi_float, Phi_batch_scaled


def add_gaussian_noise(images, std):
    noise = tf.random_normal(shape=tf.shape(images), mean=0.0, stddev=std, dtype=tf.float32)
    return tf.nn.relu(images + noise)


def fft2dshift(input):
    dim = int(input.shape[1].value)  # dimension of the data
    if dim % 2 == 0:
        print('Please make the size of kernel odd')
    channel1 = int(input.shape[0].value)  # channels for the first dimension
    # shift up and down
    u = tf.slice(input, [0, 0, 0], [channel1, int((dim + 1) / 2), dim])
    d = tf.slice(input, [0, int((dim + 1) / 2), 0], [channel1, int((dim - 1) / 2), dim])
    du = tf.concat([d, u], axis=1)
    # shift left and right
    l = tf.slice(du, [0, 0, 0], [channel1, dim, int((dim + 1) / 2)])
    r = tf.slice(du, [0, 0, int((dim + 1) / 2)], [channel1, dim, int((dim - 1) / 2)])
    output = tf.concat([r, l], axis=2)
    return output


def gen_OOFphase(Phi_list, N, wvls):
    # return (Phi_list,pixel,pixel,color)
    x0 = np.linspace(-1.1, 1.1, N)
    xx, yy = np.meshgrid(x0, x0)
    OOFphase = np.empty([len(Phi_list), N, N, len(wvls)], dtype=np.float32)
    for j in range(len(Phi_list)):
        Phi = Phi_list[j]
        for k in range(len(wvls)):
            OOFphase[j, :, :, k] = Phi * (xx ** 2 + yy ** 2) * wvls[1] / wvls[k];
    return OOFphase


def gen_PSFs(h, OOFphase, wvls, idx, N_R, N_G, N_B):
    n = 1.5  # diffractive index

    with tf.variable_scope("Red"):
        OOFphase_R = OOFphase[:, :, :, 0]
        phase_R = tf.add(2 * np.pi / wvls[0] * (n - 1) * h, OOFphase_R)
        Pupil_R = tf.pad(tf.multiply(tf.complex(idx, 0.0), tf.exp(tf.complex(0.0, phase_R))),
                         [[0, 0], [(N_R - N_B) // 2, (N_R - N_B) // 2], [(N_R - N_B) // 2, (N_R - N_B) // 2]],
                         name='Pupil_R')
        Norm_R = tf.cast(N_R * N_R * np.sum(idx ** 2), tf.float32)
        PSF_R = tf.divide(tf.square(tf.abs(fft2dshift(tf.fft2d(Pupil_R)))), Norm_R, name='PSF_R')

    with tf.variable_scope("Green"):
        OOFphase_G = OOFphase[:, :, :, 1]
        phase_G = tf.add(2 * np.pi / wvls[1] * (n - 1) * h, OOFphase_G)
        Pupil_G = tf.pad(tf.multiply(tf.complex(idx, 0.0), tf.exp(tf.complex(0.0, phase_G))),
                         [[0, 0], [(N_G - N_B) // 2, (N_G - N_B) // 2], [(N_G - N_B) // 2, (N_G - N_B) // 2]],
                         name='Pupil_G')
        Norm_G = tf.cast(N_G * N_G * np.sum(idx ** 2), tf.float32)
        PSF_G = tf.divide(tf.square(tf.abs(fft2dshift(tf.fft2d(Pupil_G)))), Norm_G, name='PSF_G')

    with tf.variable_scope("Blue"):
        OOFphase_B = OOFphase[:, :, :, 2]
        phase_B = tf.add(2 * np.pi / wvls[2] * (n - 1) * h, OOFphase_B)
        Pupil_B = tf.multiply(tf.complex(idx, 0.0), tf.exp(tf.complex(0.0, phase_B)), name='Pupil_B')
        Norm_B = tf.cast(N_B * N_B * np.sum(idx ** 2), tf.float32)
        PSF_B = tf.divide(tf.square(tf.abs(fft2dshift(tf.fft2d(Pupil_B)))), Norm_B, name='PSF_B')

    N_crop_R = int((N_R - N_B) / 2)  # Num of pixel need to cropped at each side for R
    N_crop_G = int((N_G - N_B) / 2)  # Num of pixel need to cropped at each side for G

    PSFs = tf.stack(
        [PSF_R[:, N_crop_R:-N_crop_R, N_crop_R:-N_crop_R], PSF_G[:, N_crop_G:-N_crop_G, N_crop_G:-N_crop_G], PSF_B],
        axis=3)
    return PSFs


def blurImage(RGBPhi, DPPhi, PSFs):
    N_B = PSFs.shape[1].value
    N_crop = np.int32((N_B - 1) / 2)
    N_Phi = PSFs.shape[0].value

    with tf.variable_scope("Red"):
        sharp_R = RGBPhi[:, :, :, 0:1]
        PSFs_R = tf.reshape(tf.transpose(PSFs[:, :, :, 0], perm=[1, 2, 0]), [N_B, N_B, 1, N_Phi])
        blurAll_R = tf.nn.conv2d(sharp_R, PSFs_R, strides=[1, 1, 1, 1], padding='VALID')
        blur_R = tf.reduce_sum(tf.multiply(blurAll_R, DPPhi[:, N_crop:-N_crop, N_crop:-N_crop, :]), axis=-1)

    with tf.variable_scope("Green"):
        sharp_G = RGBPhi[:, :, :, 1:2]
        PSFs_G = tf.reshape(tf.transpose(PSFs[:, :, :, 1], perm=[1, 2, 0]), [N_B, N_B, 1, N_Phi])
        blurAll_G = tf.nn.conv2d(sharp_G, PSFs_G, strides=[1, 1, 1, 1], padding='VALID')
        blur_G = tf.reduce_sum(tf.multiply(blurAll_G, DPPhi[:, N_crop:-N_crop, N_crop:-N_crop, :]), axis=-1)

    with tf.variable_scope("Green"):
        sharp_B = RGBPhi[:, :, :, 2:3]
        PSFs_B = tf.reshape(tf.transpose(PSFs[:, :, :, 2], perm=[1, 2, 0]), [N_B, N_B, 1, N_Phi])
        blurAll_B = tf.nn.conv2d(sharp_B, PSFs_B, strides=[1, 1, 1, 1], padding='VALID')
        blur_B = tf.reduce_sum(tf.multiply(blurAll_B, DPPhi[:, N_crop:-N_crop, N_crop:-N_crop, :]), axis=-1)

    blur = tf.stack([blur_R, blur_G, blur_B], axis=3)

    return blur


def system(PSFs, RGB_batch_float, DPPhi_float, phase_BN=True):
    with tf.variable_scope("system", reuse=tf.AUTO_REUSE):
        blur = blurImage(RGB_batch_float, DPPhi_float, PSFs)

        # noise
        sigma = 0.01
        blur_noisy = add_gaussian_noise(blur, sigma)

        # estimate depth
        Phi_hat = Network.UNet_2(blur_noisy, phase_BN)

        return blur_noisy, Phi_hat


def cost_rms(Phi_batch_scaled, Phi_hat, N_B):
    Phi_GT = tf.expand_dims(
        Phi_batch_scaled[:, int((N_B - 1) / 2):-int((N_B - 1) / 2), int((N_B - 1) / 2):-int((N_B - 1) / 2)], -1)

    cost = tf.sqrt(tf.reduce_mean(tf.square(Phi_GT - Phi_hat)))
    return cost, Phi_GT


def cost_rms(GT, hat):
    cost = 20 * tf.sqrt(tf.reduce_mean(tf.square(GT - hat))) # RMS, scale from 0-1 to (-10,10)
    return cost

def cost_grad(phi_GT, phi_hat):
    [phi_GTy, phi_GTx] = tf.image.image_gradients(phi_GT)
    [phi_haty, phi_hatx] = tf.image.image_gradients(phi_hat)

    costx = cost_rms(phi_GTx, phi_hatx)
    costy = cost_rms(phi_GTy, phi_haty)

    return costx + costy


######################################### Set parameters   ###############################################

# def main():

zernike = sio.loadmat('zernike_basis.mat')
u2 = zernike['u2']  # basis of zernike poly
idx = zernike['idx']
idx = idx.astype(np.float32)

N_R = 31
N_G = 27
N_B = 23  # size of the blur kernel
wvls = np.array([610, 530, 470]) * 1e-9

N_modes = u2.shape[1]  # load zernike modes

# generate the defocus phase
Phi_list = np.linspace(-10, 10, 21, np.float32)
OOFphase = gen_OOFphase(Phi_list, N_B, wvls)  # return (N_Phi,N_B,N_B,N_color)    

####################################   Build the architecture  #####################################################

with tf.variable_scope("PSFs"):
    a_zernike = tf.get_variable("a_zernike", [N_modes, 1], initializer=tf.zeros_initializer(),
                                constraint=lambda x: tf.clip_by_value(x, -wvls[1] / 2, wvls[1] / 2))
    g = tf.matmul(u2, a_zernike)
    h = tf.nn.relu(tf.reshape(g, [N_B, N_B]) + wvls[1],
                   name='heightMap')  # height map of the phase mask, should be all positive
    PSFs = gen_PSFs(h, OOFphase, wvls, idx, N_R, N_G, N_B)  # return (N_Phi, N_B, N_B, N_color)

batch_size = 20
RGB_batch_float, DPPhi_float, Phi_batch_scaled = read2batch(TFRECORD_TRAIN_PATH, batch_size)
RGB_batch_float_valid, DPPhi_float_valid, Phi_batch_scaled_valid = read2batch(TFRECORD_VALID_PATH, batch_size)

N_crop = int((N_B - 1) / 2)
Phi_GT_train = tf.expand_dims(Phi_batch_scaled[:, N_crop:-N_crop, N_crop:-N_crop], -1)
Phi_GT_valid = tf.expand_dims(Phi_batch_scaled_valid[:, N_crop:-N_crop, N_crop:-N_crop], -1)

[blur_train, Phi_hat_train] = system(PSFs, RGB_batch_float, DPPhi_float)
[blur_valid, Phi_hat_valid] = system(PSFs, RGB_batch_float_valid, DPPhi_float_valid, phase_BN=False)


# cost function

with tf.name_scope("cost"):

    cost_rms_train = cost_rms(Phi_GT_train, Phi_hat_train)
    cost_rms_valid = cost_rms(Phi_GT_valid, Phi_hat_valid)

    cost_grad_train = cost_grad(Phi_GT_train, Phi_hat_train)
    cost_grad_valid = cost_grad(Phi_GT_valid, Phi_hat_valid)

    weight_grad = 1

    cost_train = cost_rms_train + weight_grad * cost_grad_train
    cost_valid = cost_rms_valid + weight_grad * cost_grad_valid


# train ditial and optical part saparetely
vars_optical = tf.trainable_variables("PSFs")
vars_digital = tf.trainable_variables("system")

opt_optical = tf.train.AdamOptimizer(lr_optical)
opt_digital = tf.train.AdamOptimizer(lr_digital)

global_step = tf.Variable(0, trainable=False)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    grads = tf.gradients(cost_train, vars_optical + vars_digital)
    grads_optical = grads[:len(vars_optical)]
    grads_digital = grads[len(vars_optical):]
    train_op_optical = opt_optical.apply_gradients(zip(grads_optical, vars_optical))
    train_op_digital = opt_digital.apply_gradients(zip(grads_digital, vars_digital))
    train_op = tf.group(train_op_optical, train_op_digital)

# tensorboard
tf.summary.scalar('cost_train', cost_train)
tf.summary.scalar('cost_valid', cost_valid)
tf.summary.scalar('cost_rms_train', cost_rms_train)
tf.summary.scalar('cost_rms_valid', cost_rms_valid)
tf.summary.scalar('cost_grad_train', cost_grad_train)
tf.summary.scalar('cost_grad_valid', cost_grad_valid)

tf.summary.histogram('a_zernike', a_zernike)

tf.summary.image('HeightMap', tf.expand_dims(tf.expand_dims(h,0),-1))
tf.summary.image('sharp_valid', RGB_batch_float_valid[0:1, int((N_B - 1) / 2):-int((N_B - 1) / 2), int((N_B - 1) / 2):-int((N_B - 1) / 2), :])
tf.summary.image('blur_valid', blur_valid[0:1, :, :, :])
tf.summary.image('Phi_hat_valid', Phi_hat_valid[0:1, :, :, :])
tf.summary.image('Phi_GT_valid', Phi_GT_valid[0:1, :, :,:])

merged = tf.summary.merge_all()

##########################################   Train  #############################################

variables_to_restore  = [v for v in tf.global_variables() if v.name.startswith('system')]
saver = tf.train.Saver(variables_to_restore)
saver_all = tf.train.Saver(max_to_keep =1)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if not tf.train.checkpoint_exists(results_dir + 'checkpoint'):
        # option1: run a new one
        print('Start to save at: ', results_dir)
    else:
        model_path = tf.train.latest_checkpoint(results_dir)
        load_path = saver_all.restore(sess, model_path)
        print('Continue to save at: ', results_dir)

    train_writer = tf.summary.FileWriter(results_dir + '/summary/', sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(100000):
        ## load the batch
        train_op.run()

        if i % 50 == 0:
            [train_summary, loss_train, loss_valid] = sess.run([merged, cost_train, cost_valid])
            train_writer.add_summary(train_summary, i)

            print("Iter " + str(i) + ", Train Loss = " + \
                  "{:.6f}".format(loss_train) + ", Valid Loss = " + \
                  "{:.6f}".format(loss_valid))

            # save them
            saver_all.save(sess, results_dir + "model.ckpt", global_step = i)

            [ht, at, PSFst] = sess.run([h, a_zernike, PSFs])
            np.savetxt(results_dir + 'HeightMap.txt', ht)
            np.savetxt(results_dir + 'a_zernike.txt', at)
            np.save(results_dir + 'PSFs.npy', PSFst)

    train_writer.close()
    coord.request_stop()
    coord.join(threads)
