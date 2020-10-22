import tensorflow as tf
import numpy as np
import scipy.io as sio
import os, math, sys

def sparisty_regularization(ss_epi_im, mask, thresholds, alpha, dec_fft, rec_fft, w_st):
    # Initialization
    f0 = tf.cast(ss_epi_im, tf.complex64, "epi_cast")
    mask = tf.cast(mask, tf.complex64, "mask_cast")
    g = f0 * mask
    with tf.name_scope("EPI_Initialization"):
        f0 = tf.ifft2d(tf.fft2d(f0) * dec_fft[-1] * rec_fft[-1]) # pre-filtering only using the low-pass filter
    niter = thresholds.shape[0]
    num_tiny = tf.constant(1e-6, tf.complex64)

    def condition(i, fi, fi_1, fi_2):
        return tf.less(i, niter)

    def body(i, fi, fi_1, fi_2):
        with tf.name_scope("Analysis_Trans"):
            x = fi + alpha * tf.multiply(mask, (g-fi))
            coeffs = tf.ifft2d(tf.multiply(tf.fft2d(x), dec_fft) )
        with tf.name_scope("Hard_Thresholding"):
            comp = tf.greater(tf.abs(coeffs), tf.multiply(thresholds[i], w_st) )
            coeffs = tf.multiply(tf.cast(comp, tf.complex64), coeffs)
        with tf.name_scope("Synthesis_Trans"):
            coeffs_fft = tf.multiply(tf.fft2d(coeffs), rec_fft )
            f_hat = tf.ifft2d(tf.reduce_sum(coeffs_fft, 1, keepdims = True))
        # two-step overrelaxation
        with tf.name_scope("Double_Overrelaxation"):
            beta1 = tf.divide( tf.reduce_sum((g - f_hat) * mask * (f_hat - fi_1), axis=[1, 2, 3], keepdims=True),
                                tf.reduce_sum((f_hat - fi_1) * mask * (f_hat - fi_1), axis=[1, 2, 3], keepdims=True) + num_tiny )
            beta1 = tf.clip_by_value(tf.cast(beta1, tf.float32), tf.constant(0, tf.float32), tf.constant(1, tf.float32))
            f_tilde = f_hat + tf.cast(beta1, tf.complex64) * (f_hat - fi_1)

            beta2 = tf.divide( tf.reduce_sum((g - f_tilde) * mask * (f_tilde - fi_2), axis=[1, 2, 3], keepdims=True), 
                                tf.reduce_sum((f_tilde - fi_2) * mask * (f_tilde - fi_2), axis=[1, 2, 3], keepdims=True) + num_tiny )
            beta2 = tf.clip_by_value(tf.cast(beta2, tf.float32), tf.constant(0, tf.float32), tf.constant(1, tf.float32))
            f_i_new = f_tilde + tf.cast(beta2, tf.complex64) * (f_tilde - fi_2)

        return tf.add(i, 1), f_i_new, fi, fi_1

    _, fi, _, _ = tf.while_loop(condition, body, [tf.constant(0), f0, f0, f0], name="Sparsity_Regularization")

    return tf.cast(fi, tf.float32)

def load_shearlet_system(path, height, width):
    try:
        fmat = sio.loadmat(path)
    except FileNotFoundError:
        print(f"Could not find file: {path}")
        sys.exit()
        
    dec = fmat["dec"].astype(np.float32)
    rec = fmat["rec"].astype(np.float32)
    ksize, _, nfilter = dec.shape 
    assert ksize <= height
    assert ksize <= width
    w = fmat["w"].astype(np.float32) * ksize / math.sqrt(height * width)
    
    row_begin = int(math.ceil((height-ksize)/2.))
    col_begin = int(math.ceil((width-ksize)/2.))
    
    # Tensorflow code
    dec = tf.transpose(dec, (2, 0, 1))
    rec = tf.transpose(rec, (2, 0, 1))
    w = tf.transpose(w, (2, 0, 1))
    
    dec = tf.signal.fftshift(dec, (1, 2))
    rec = tf.signal.fftshift(rec, (1, 2))
    
    paddings = tf.constant([[0, 0], [row_begin, height-ksize-row_begin], [col_begin, width-ksize-col_begin]])
    dec_padd = tf.pad(dec, paddings, "CONSTANT")
    rec_padd = tf.pad(rec, paddings, "CONSTANT")

    dec_padd = tf.signal.ifftshift(dec_padd, (1, 2))
    rec_padd = tf.signal.ifftshift(rec_padd, (1, 2))
    
    dec_fft, rec_fft = tf.signal.fft2d(tf.cast(dec_padd, tf.complex64)), tf.signal.fft2d(tf.cast(rec_padd, tf.complex64))

    return dec_fft, rec_fft, w

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

    with tf.name_scope("Load_Shearlet_System"):
        dec_fft, rec_fft, w_st = load_shearlet_system(os.path.join(shearlet_system_dir, "st_127_127_4.mat"), height, width)
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
