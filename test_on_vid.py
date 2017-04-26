from __future__ import print_function

# import torch
import tensorflow as tf
from os import listdir
import os
from os.path import isfile, join
import numpy as np
from scipy import misc
from scipy import ndimage

import sys
import numpy as np
import time
import matplotlib.pyplot as plt

# Get all the custom helper util headers
import utils.data_prep
import utils.add_summary
import utils.train_utils
import utils.eval_utils
import utils.get_flags
import models.hg_graph_builder
import utils.test_utils



# Read up and set up all the flag variables
FLAG = utils.get_flags.get_flags()
import imageio


def main(_):
    if not FLAG.dataset_dir:
        raise ValueError(
            'You must supply the dataset directory with --dataset_dir')
    if not FLAG.dataset_dir:
        raise ValueError('You must supply the model_path with --load_ckpt_path')
    # DataHolder = utils.test_utils.TestDataHolder(FLAG)
    
    print('data loaded... phhhh')
    
    # Set up formatting for the movie files
    # Writer = anima    tion.writers['ffmpeg']
    # writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    
    with tf.Graph().as_default():
        
        # builder = include.hg_graph_builder.HGgraphBuilder(FLAG)
        
        builder = models.hg_graph_builder.HGgraphBuilder_MultiGPU(FLAG)
        print("build finished, There it stands, tall and strong...")
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        saver = tf.train.Saver()
        
        with tf.Session(config=config) as sess:
            
            # All the variable initialiezed in MoFoking RunTime
            # Confusing the world gets when yoda asks initializer operator before
            
            
            print(FLAG.load_ckpt_path)
            
            saver.restore(sess, FLAG.load_ckpt_path)
            print('model Initialized...')
            
            print("Let the Testing Begin...")
            
            # Train the model, and also write summaries.
            # Every 10th step, measure test-set accuracy, and write test summaries
            # All other steps, run train_step on training data, & add training summaries
            yo = []
            filename = '/home/sudeep/projects/AmazonLab126/deep-3DHyPE' \
                       '/Dataset/S1/Videos/_ALL_1.60457274.mp4'
            #filename = '/home/sudeep/friends_indoor_clip.mp4'
            filename = '/home/sudeep/dance.mp4'
            filename = '/home/sudeep/100P.mp4'
            filename = '/home/sudeep/sit.mp4'
            filename = '/home/sudeep/0CVZN.mp4'
            #filename = '/home/sudeep/0J471.mp4'

            vid = imageio.get_reader(filename, 'ffmpeg')
            for step in range(50, 2400, 5):
                image_ = vid.get_data(step)
                #image_ = image_[100:600, 200:700, :]
                #image_ = image_[:, 180:1000, :]
                #image_ = image_[200:800, 200:800, :]
                image_l = misc.imresize(image_, (256, 256)).astype(
                    np.float32)
                _x = []
                gt = []
                _x.append(np.reshape(image_l, (1, 256, 256, 3)))
                feed_dict_x = {i: d for i, d in zip(builder._x, _x)}
                
                time_ = time.clock()
                output_ = sess.run([builder.output], feed_dict_x)
                print("Time to feed and run the network", time.clock() - time_)
                steps = map(int, FLAG.structure_string.split('-'))
                ypy = 0
                figs = []
                ouu = output_[0][-1][-1][-1]
                ouu[ouu<0] = 0
                ouu_ = np.reshape(ouu[0], [64, 64, 64, 14])
                ouu = ndimage.median_filter(ouu_, size=[2,2,2,1])
                ouu = np.reshape(ouu,[1,64,64,64*14])
                for idh in xrange(len(map(int, FLAG.gpu_string.split('-')))):
                    heat_map = np.sum(ouu[0], axis=-1)
                    pred_cords = utils.eval_utils.get_coordinate(
                        ouu, steps, 14)
                    ouu_ = np.reshape(ouu[0],[64,64,64,14])
                    #utils.data_prep.plot_3d(np.sum(ouu_,axis=-1),threshold=0.7)
                    # print (pred_cords)
                    
                    utils.test_utils.visualize_stickman(pred_cords[0],
                                                        image_, heat_map, step)


if __name__ == '__main__':
    tf.app.run()
