from __future__ import print_function

import time
import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from PIL import Image
import scipy.io as scio
from model import Model

MODEL_SAVE_PATH = './model/model.ckpt'
output_path = './results/custom/'
path = './test_custom/'

def rgb2ycbcr(img):
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128 / 255.0
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128 / 255.0
    return Y, Cb, Cr

def ycbcr2rgb(Y, Cb, Cr):
    R = Y + 1.402 * (Cr - 128 / 255.0)
    G = Y - 0.34414 * (Cb - 128 / 255.0) - 0.71414 * (Cr - 128 / 255.0)
    B = Y + 1.772 * (Cb - 128 / 255.0)
    R = np.expand_dims(R, axis=-1)
    G = np.expand_dims(G, axis=-1)
    B = np.expand_dims(B, axis=-1)
    return np.concatenate([R, G, B], axis=-1)

def main():
    print('\nBegin to process custom test images...\n')
    
    # Get all frame files sorted by number
    files = sorted([f for f in os.listdir(path) if f.startswith('frame_') and f.endswith('.png')])
    time_cost = np.ones([len(files)-1], dtype=float)  # -1 because we process pairs

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with tf.Graph().as_default(), tf.Session() as sess:
        M = Model(BATCH_SIZE=1, INPUT_H=None, INPUT_W=None, is_training=False)
        t_list = tf.trainable_variables()
        saver = tf.train.Saver(var_list=t_list)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, MODEL_SAVE_PATH)

        # Process consecutive pairs of frames
        for i in range(len(files)-1):
            frame1_name = files[i]
            frame2_name = files[i+1]
            print(f"\033[0;33;40m[{i+1}/{len(files)-1}]: Processing {frame1_name} and {frame2_name}\033[0m")

            img1 = np.array(Image.open(os.path.join(path, frame1_name))) / 255.0
            img2 = np.array(Image.open(os.path.join(path, frame2_name))) / 255.0

            Shape1 = img1.shape
            Shape2 = img2.shape
            print("shape1:", Shape1, "shape2:", Shape2)
            
            # Convert RGB to YCbCr
            if len(Shape1) > 2:
                img1, img1_cb, img1_cr = rgb2ycbcr(img1)
            if len(Shape2) > 2:
                img2, img2_cb, img2_cr = rgb2ycbcr(img2)

            h1 = Shape1[0]
            w1 = Shape1[1]
            h2 = Shape2[0]
            w2 = Shape2[1]
            assert (h1 == h2 and w1 == w2), 'Two images must have the same shape!'
            
            img1 = img1.reshape([1, h1, w1, 1])
            img2 = img2.reshape([1, h1, w1, 1])

            start = time.time()
            outputs = sess.run(M.generated_img, feed_dict={M.SOURCE1: img1, M.SOURCE2: img2})
            output = outputs[0, :, :, 0]
            
            # Convert back to RGB if input was RGB
            if len(Shape1) > 2 and len(Shape2) == 2:
                output = ycbcr2rgb(output, img1_cb, img1_cr)
            if len(Shape2) > 2 and len(Shape1) == 2:
                output = ycbcr2rgb(output, img2_cb, img2_cr)
                
            end = time.time()
            time_cost[i] = end - start
            print(f"Testing [{i+1}] success, Testing time is [{end-start}]\n")
            
            # Save the fused result
            output_name = f'fused_{i:04d}.png'
            Image.fromarray((output * 255).astype(np.uint8)).save(os.path.join(output_path, output_name))

    # Save timing information
    scio.savemat(os.path.join(output_path, 'time.mat'), {'T': time_cost})

if __name__ == '__main__':
    main()