import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os

def sparisty_regularization(ss_epi_im, mask, thresholds, alpha, dec_fft, rec_fft, w_st):
    # Initialization
    f0 = tf.cast(ss_epi_im, tf.complex64, "epi_cast")
    mask = tf.cast(mask, tf.complex64, "mask_cast")
    niter = thresholds.shape[0]

    def condition(i, fi, fi_1, fi_2):
        return tf.less(i, niter)

    def body(i, fi, fi_1, fi_2):
        with tf.name_scope("Analysis_Trans"):
            x = fi + alpha * tf.multiply(mask, (f0-fi))
            coeffs = tf.ifft2d(tf.multiply(tf.fft2d(x), dec_fft) )
        with tf.name_scope("Hard_Thresholding"):
            comp = tf.greater(tf.abs(coeffs), tf.multiply(thresholds[i], w_st) )
            coeffs = tf.multiply(tf.cast(comp, tf.complex64), coeffs)
        with tf.name_scope("Synthesis_Trans"):
            coeffs_fft = tf.multiply(tf.fft2d(coeffs), rec_fft )
            f_hat = tf.ifft2d(tf.reduce_sum(coeffs_fft, 1, keepdims = True))
        # two-step overrelaxation
        with tf.name_scope("Double_Overrelaxation"):
            beta1 = tf.reduce_sum((f0 - f_hat) * mask * (f_hat - fi_1), axis=[1, 2, 3], keepdims=True) / tf.reduce_sum((f_hat - fi_1) * mask * (f_hat - fi_1), axis=[1, 2, 3], keepdims=True)
            beta1 = tf.clip_by_value(tf.cast(beta1, tf.float32), tf.constant(0, tf.float32), tf.constant(1, tf.float32))
            f_tilde = f_hat + tf.cast(beta1, tf.complex64) * (f_hat - fi_1)

            beta2 = tf.reduce_sum((f0 - f_tilde) * mask * (f_tilde - fi_2), axis=[1, 2, 3], keepdims=True) / tf.reduce_sum((f_tilde - fi_2) * mask * (f_tilde - fi_2), axis=[1, 2, 3], keepdims=True)
            beta2 = tf.clip_by_value(tf.cast(beta2, tf.float32), tf.constant(0, tf.float32), tf.constant(1, tf.float32))
            f_i_new = f_tilde + tf.cast(beta2, tf.complex64) * (f_tilde - fi_2)

        return tf.add(i, 1), f_i_new, fi, fi_1

    _, fi, _, _ = tf.while_loop(condition, body, [tf.constant(0), f0, f0, f0], name="Sparsity_Regularization")

    return tf.cast(fi, tf.float32)


def create_model(features, labels, mode, params={}):
    tensorboard_dir = params["tensorboard_dir"]
    shearlet_system_dir = params["shearlet_system_dir"]
    ch = params["num_output_channels"]
    batch_size = params["batch_size"]
    height = params["height"]
    width = params["width"]
    alpha = params["alpha"]
    niter = params["niter"]
    thmax = params["thmax"]
    thmin = params["thmin"]

    fmat = sio.loadmat(os.path.join(shearlet_system_dir, 'st_{}_{}.mat'.format(height, width) ) )
    dec = fmat['dec'].astype(np.float32)
    rec = fmat['rec'].astype(np.float32)
    w = fmat['w'].astype(np.float32)
    dec = np.transpose(dec, (2, 0, 1))
    rec = np.transpose(rec, (2, 0, 1))
    w = np.transpose(w, (2, 0, 1))
    dec = np.fft.fft2(dec)
    rec = np.fft.fft2(rec)
    dec_fft = tf.constant(dec, tf.complex64)
    rec_fft = tf.constant(rec, tf.complex64)
    w_st = tf.constant(w, tf.float32)
    thresholds = tf.constant(np.linspace(thmax, thmin, niter), tf.float32)

    im = features["im_epi"]
    mask = features["im_mask"]
    save_name = features["name_save"]

    with tf.name_scope("Normalization"):
        ss_epi_im = tf.reshape(im, [-1, 1, height, width])
        ss_epi_mask = tf.reshape(mask, [-1, 1, height, width])
        val_max = tf.math.reduce_max(ss_epi_im, axis=[2, 3], keepdims=True)
        ss_epi_im_reversed = 255.*(1.-ss_epi_mask) + ss_epi_im
        val_min = tf.math.reduce_min(ss_epi_im_reversed, axis=[2, 3], keepdims=True)

        ss_epi_im = (ss_epi_im - val_min) / (val_max - val_min)
        ss_epi_im = ss_epi_im * ss_epi_mask

    ds_epi_im = sparisty_regularization(ss_epi_im, ss_epi_mask, thresholds, alpha, dec_fft, rec_fft, w_st)

    with tf.name_scope("Reverse_Normalization"):
        ds_epi_im = ds_epi_im * (val_max - val_min) + val_min
        ds_epi_im = tf.clip_by_value(ds_epi_im, tf.constant(0, tf.float32), tf.constant(255, tf.float32))
        ds_epi_im = tf.reshape(ds_epi_im, [-1, ch, height, width])
        ds_epi_im = tf.cast(ds_epi_im, tf.uint8)
        ds_epi_im = tf.transpose(ds_epi_im, [0, 2, 3, 1])

    with tf.name_scope('prediction_samples'):
        tf.summary.image('output', ds_epi_im, batch_size)
    prediction_hooks = []
    pred_summary_hook = tf.train.SummarySaverHook(
                    save_secs=10,
                    output_dir=os.path.join(tensorboard_dir, "prediction"),
                    summary_op=tf.summary.merge_all()
                )
    # Add it to the evaluation_hook list
    prediction_hooks.append(pred_summary_hook)


    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = { 
                        'image': ds_epi_im,
                        'save_name': save_name
                    }
        return tf.estimator.EstimatorSpec(mode, 
                                        predictions=predictions,
                                        prediction_hooks=prediction_hooks)
