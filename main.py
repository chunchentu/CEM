## main.py -- sample code to test attack procedure
##
## Copyright (C) 2018, PaiShun Ting <paishun@umich.edu>
##                     Chun-Chen Tu <timtu@umich.edu>
##                     Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
## Copyright (C) 2017, Huan Zhang <ecezhang@ucdavis.edu>.
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import os
import sys
import tensorflow as tf
import numpy as np
import random
import time
from keras.layers import Lambda
from setup_mnist import MNIST, MNISTModel

import Utils as util
from aen_attack import AEADEN


def main(args):
    with tf.Session() as sess:
        random.seed(121)
        np.random.seed(1211)

        image_id = args['img_id']
        target_label = args['target_label']
        arg_max_iter = args['maxiter']
        arg_b = args['binary_steps']
        arg_init_const = args['init_const']
        arg_mode = args['mode']
        arg_kappa = args['kappa']
        arg_beta = args['beta']
        arg_gamma =args['gamma']
        
        AE_model = util.load_AE("mnist_AE_1")
        data, model =  MNIST(), MNISTModel("models/mnist", sess, False)

        orig_img, target = util.generate_data(data, image_id, target_label)

        attack = AEADEN(sess, model, mode = arg_mode, AE = AE_model, batch_size=1, kappa=arg_kappa, init_learning_rate=1e-2,
            binary_search_steps=arg_b, max_iterations=arg_max_iter, initial_const=arg_init_const, beta=arg_beta, gamma=arg_gamma)

        adv_img = attack.attack(orig_img, target)

        # if type(adv_img) is list:
        #     adv_img = adv_img[0]
        # if len(adv_img.shape) == 3:
        #     adv_img = adv_img.reshape((1,) + adv_img.shape)

        orig_prob, orig_class, orig_prob_str = util.model_prediction(model, orig_img)
        adv_prob, adv_class, adv_prob_str = util.model_prediction(model, adv_img)
        delta_prob, delta_class, delta_prob_str = util.model_prediction(model, orig_img-adv_img)

        INFO = "[INFO]id:{}, kappa:{}, Orig class:{}, Adv class:{}, Delta class: {}, Orig prob:{}, Adv prob:{}, Delta prob:{}".format(image_id, arg_kappa, orig_class, adv_class, delta_class, orig_prob_str, adv_prob_str, delta_prob_str)
        print(INFO)

        suffix = "id{}_kappa{}_Orig{}_Adv{}_Delta{}".format(image_id, arg_kappa, orig_class, adv_class, delta_class)
        arg_save_dir = "{}_ID{}_Gamma_{}".format(arg_mode, image_id, arg_gamma)
        os.system("mkdir -p Results/{}".format(arg_save_dir))
        util.save_img(orig_img, "Results/{}/Orig_original{}.png".format(arg_save_dir, orig_class))
        util.save_img(adv_img, "Results/{}/Adv_{}.png".format(arg_save_dir, suffix))
        util.save_img(np.absolute(orig_img-adv_img)-0.5, "Results/{}/Delta_{}.png".format(arg_save_dir, suffix))

        sys.stdout.flush()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--img_id", type=int)
    parser.add_argument("-t", "--target_label", type=int)
    parser.add_argument("-m", "--maxiter", type=int, default=1000)
    parser.add_argument("-b", "--binary_steps", type=int, default=9)
    parser.add_argument("-c", "--init_const", type=float, default=10.0)
    parser.add_argument("--mode", choices=["PN", "PP"], default="PN")
    parser.add_argument("--kappa", type=float, default=0)
    parser.add_argument("--beta", type=float, default=1e-1)
    parser.add_argument("--gamma", type=float, default=0)

    args = vars(parser.parse_args())
    main(args)
