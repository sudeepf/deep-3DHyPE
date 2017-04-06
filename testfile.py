from __future__ import print_function

# import torch
import tensorflow as tf
from os import listdir
import os
from os.path import isfile, join
import numpy as np
from scipy import misc

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

# Read up and set up all the flag variables
FLAG = utils.get_flags.get_flags()


def main(_):
    if not FLAG.dataset_dir:
        raise ValueError(
            'You must supply the dataset directory with --dataset_dir')
    
    DataHolder = utils.train_utils.DataHolder(FLAG)
    
    print('data loaded... phhhh')
    
    with tf.Graph().as_default():
        
        # builder = models.hg_graph_builder.HGgraphBuilder_MultiGPU(FLAG)
        builder = None
        # builder = models.hg_graph_builder.HGgraphBuilder(FLAG)
        print("build finished, There it stands, tall and strong...")
        
        graph_def = tf.get_default_graph().as_graph_def()
        graphpb_txt = str(graph_def)
        with open('graphpb.txt', 'w') as f:
            f.write(graphpb_txt)
        
        # lol = lol2
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        with tf.Session(config=config) as sess:
            
            # Train the model, and also write summaries.
            # Every 10th step, measure test-set accuracy, and write test summaries
            # All other steps, run train_step on training data, & add training summaries
            structure = map(int, FLAG.structure_string.split('-'))
            
            for step in range(DataHolder.train_data_size):
                
                _x = []
                vec_64 = []
                vec_32 = []
                vec_16 = []
                vec_8 = []
                gt = []
                time_ = time.clock()
                for i in map(int, FLAG.gpu_string.split('-')):
                    fd = DataHolder.get_next_train_batch()
                    _x.append(fd[0])
                    vec_64.append(fd[1])
                    vec_32.append(fd[2])
                    vec_16.append(fd[3])
                    vec_8.append(fd[4])
                    # gt.append(fd[5])
                
                print(_x[0][0].shape)
                #plt.imshow(_x[0][0])
                #plt.show()
                # print ("PreProcessing Time - incd reading", time.clock()-time_)
                time_ = time.clock()
                
                
                #plt.imshow(np.sum(data_out[-1][-1][0], axis=-1))
                #plt.show()
                # print("Time to feed and run the network", time.clock()-time_)
                

if __name__ == '__main__':
    tf.app.run()
