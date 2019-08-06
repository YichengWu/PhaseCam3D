

import tensorflow as tf
import scipy.io as sio
import numpy as np
import matplotlib.image
import os
import Network
import imageio

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

results_dir = "./trained_framework/"
DATA_PATH = './Data/'
TFRECORD_TEST_PATH = [DATA_PATH + 'test.tfrecord']


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


def read2batch(TFRECORD_PATH, batchsize):
    # load tfrecord and make them to be usable data
    dataset = tf.data.TFRecordDataset(TFRECORD_PATH)
    dataset = dataset.map(parse_element).repeat()
    dataset = dataset.batch(batchsize, drop_remainder=True)
    iterator = dataset.make_one_shot_iterator()
    RGB_batch, DPPhi_batch, DP_batch = iterator.get_next()

    RGB_batch_float = tf.image.convert_image_dtype(RGB_batch, tf.float32)
    DPPhi_float = tf.cast(DPPhi_batch, tf.float32)
    Phi_batch_scaled = (tf.cast(DP_batch, tf.float32) - 10) / 210

    return RGB_batch_float, DPPhi_float, Phi_batch_scaled

def add_gaussian_noise(images, std):
    noise = tf.random_normal(shape=tf.shape(images), mean=0.0, stddev=std, dtype=tf.float32)
    return tf.nn.relu(images + noise)

def add_SDGN(images, std):
    noise0 = tf.random_normal(shape = tf.shape(images), mean = 0.0, stddev = std, dtype = tf.float32)
    noise1 = tf.multiply(tf.sqrt(images),noise0)     # Noise = N(0,std)*sqrt(I)
    return tf.nn.relu(images+noise1)  
    
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


def gen_OOFphase(Phi_list, N_B, wvls):
    # return (Phi_list,pixel,pixel,color)
    N = N_B
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


def system(PSFs, RGB_batch_float, DPPhi_float, Phi_batch_scaled, phase_BN=True):
    with tf.variable_scope("system", reuse=tf.AUTO_REUSE):
        blur = blurImage(RGB_batch_float, DPPhi_float, PSFs)

        # noise
        sigma = 0.01
        blur_noisy = add_gaussian_noise(blur, sigma)

        Phi_hat = Network.UNet_2(blur_noisy, phase_BN)

        N_B = PSFs.shape[1].value
        Phi_GT = tf.expand_dims(
            Phi_batch_scaled[:, int((N_B - 1) / 2):-int((N_B - 1) / 2), int((N_B - 1) / 2):-int((N_B - 1) / 2)], -1)

        cost = 20 * tf.sqrt(tf.reduce_mean(tf.square(Phi_GT - Phi_hat)))   #RMS, scale from 0-1 to (-10,10)

        return cost, blur_noisy, Phi_hat, Phi_GT


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

hh = np.loadtxt(results_dir + 'HeightMap.txt')

with tf.variable_scope("PSFs"):
    h = tf.constant(hh,tf.float32, name='heightMap')  # height map of the phase mask, should be all positive
    PSFs = gen_PSFs(h, OOFphase, wvls, idx, N_R, N_G, N_B)  # return (N_Phi, N_B, N_B, N_color)

batch_size = 40
RGB_batch_float_test, DPPhi_float_test, Phi_batch_scaled_test = read2batch(TFRECORD_TEST_PATH, batch_size)

[test_cost, blur, Phi_hat, Phi_GT] = system(PSFs, RGB_batch_float_test, DPPhi_float_test, Phi_batch_scaled_test,
                                             phase_BN=False)

saver = tf.train.Saver()

##########################################   Test  #############################################
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    model_path = tf.train.latest_checkpoint(results_dir)
    load_path = saver.restore(sess, model_path)
    print('Testing model from: ', results_dir)

    out_dir = 'test_all/'
    if not os.path.exists(results_dir + out_dir):
        os.makedirs(results_dir + out_dir)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    test_loss_all = []
    for i in range(10):
        [test_loss, Phi_hatt, Phi_GTt, blurt, sharpt] = sess.run([test_cost, Phi_hat, Phi_GT, blur, RGB_batch_float_test[:, int((N_B - 1) / 2):-int((N_B - 1) / 2), int((N_B - 1) / 2):-int((N_B - 1) / 2),:]])
        print("Batch " + str(i) + ", Test Loss = " + "{:.6f}".format(test_loss))
        test_loss_all.append(test_loss)

        for j in range(batch_size):
            matplotlib.image.imsave(results_dir+out_dir+'%d_%d_phiHat.png' %(i,j), Phi_hatt[j,:,:,0],vmin = 0.0, vmax = 1.0, cmap='jet')
            matplotlib.image.imsave(results_dir+out_dir+'%d_%d_phiGT.png' %(i,j), Phi_GTt[j,:,:,0],vmin = 0.0, vmax = 1.0, cmap='jet')
            imageio.imwrite(results_dir+out_dir+'%d_%d_blur.png' %(i,j),np.uint8(blurt[j,:,:,:]*255))
            imageio.imwrite(results_dir+out_dir+'%d_%d_sharp.png' %(i,j),np.uint8(sharpt[j,:,:,:]*255))
    test_loss_avg = np.mean(test_loss_all)
    print("Average Loss = " + "{:.6f}".format(test_loss_avg))
    np.savetxt(results_dir + out_dir + 'test_loss_' + "{:.6f}".format(test_loss_avg) + '.txt', test_loss_all)
    coord.request_stop()
    coord.join(threads)
