import tensorflow as tf
import numpy as np

class stacked_hourglass():
    def __init__(self, steps, FLAG, name='stacked_hourglass'):
        self.nb_stack = len(steps)
        self.steps = steps
        self.name = name
        self.module_supervisions = [[], [], [], []]
        self.FLAG = FLAG
        
    def __call__(self, x):
        with tf.name_scope(self.name) as scope:
            padding = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]],
                             name='padding')
            
            # First Thing first: get encoder wts
            
            #self.refiner_wts = self._get_refiner_wts(self.FLAG)
            
            with tf.variable_scope("preprocessing") as sc:
                conv1 = self._conv(padding, 64, 7, 2, 'VALID', 'conv1')
                norm1 = tf.contrib.layers.batch_norm(conv1, 0.9, epsilon=1e-5,
                                                     activation_fn=tf.nn.relu,
                                                     scope=sc)
                r1 = self._residual_block(norm1, 128, 'r1')
                pool = tf.contrib.layers.max_pool2d(r1, [2, 2], [2, 2], 'VALID',
                                                    scope=scope)
                r2 = self._residual_block(pool, 128, 'r2')
                r3_h = self._residual_block(r2, 256, 'r3_h')
                r3 = tf.contrib.layers.max_pool2d(r3_h, [2, 2], [2, 2], 'VALID',
                                                  scope=scope)
                r3 = self._residual_block(r3, 1024, 'r3')
                
            
            hg = [None] * self.nb_stack
            ll = [None] * self.nb_stack
            ll_ = [None] * self.nb_stack
            out = [None] * self.nb_stack
            out_ = [None] * self.nb_stack
            sum_ = [None] * self.nb_stack
            with tf.variable_scope('_hourglass_0_with_supervision') as sc:
                hg[0] = self._hourglass(r3, 3, 32 * 14, '_hourglass', 0)
                ll[0] = self._conv_bn_relu(hg[0], 1024, name='conv_1')
                sum_[0] = tf.add(ll[0], r3)
            for i in range(1, self.nb_stack - 1):
                with tf.variable_scope(
                            '_hourglass_' + str(i) + '_with_supervision') as sc:
                    hg[i] = self._hourglass(sum_[i - 1], 3, 32 * 14,
                                            '_hourglass', i)
                    ll_[i] = self._conv(hg[i], 1024, 1, 1, 'VALID', 'll')
                    sum_[i] = tf.add(ll_[i], sum_[i - 1])
            with tf.variable_scope(
                        '_hourglass_' + str(
                            self.nb_stack - 1) + '_with_supervision') as sc:
                hg[self.nb_stack - 1] = self._hourglass(sum_[self.nb_stack - 2],
                                                        3,
                                                        32 * 14,
                                                        '_hourglass',
                                                        self.nb_stack - 1)

                self.module_supervisions[-1].append(
                    self._conv(hg[self.nb_stack - 1], 32*14, 1, 1, 'VALID',
                                                                         'finalOut'))
                
                
            return self.module_supervisions
        
    def _get_refiner_wts(self, FLAG):
        wts = {}
        with tf.variable_scope('RefinerNet/'):
            init1 = tf.contrib.layers.xavier_initializer_conv2d(uniform=True)
            flatten_size = FLAG.volume_res * FLAG.volume_res * \
                           FLAG.volume_res
        
            wts['w_p'] = tf.get_variable('Matrix1', [flatten_size, FLAG.ref_1],
                                     tf.float32, init1)
            # tf.random_normal_initializer(stddev=0.02))
            wts['b_p'] = tf.get_variable('bias1', [FLAG.ref_1],
                                     initializer=tf.constant_initializer(0.0))
        
            wts['w_p1'] = tf.get_variable('Matrix2', [FLAG.num_joints *
                                                  FLAG.ref_1,
                                                  FLAG.ref_2],
                                      tf.float32, init1)
            # tf.random_normal_initializer(stddev=0.02))
            wts['b_p1'] = tf.get_variable('bias2', [FLAG.ref_2],
                                      initializer=tf.constant_initializer(0.0))
        
            wts['w_p2'] = tf.get_variable('Matrix3', [FLAG.ref_2,
                                                  FLAG.ref_3],
                                      tf.float32, init1)
            # tf.random_normal_initializer(stddev=0.02))
            wts['b_p2'] = tf.get_variable('bias3', [FLAG.ref_3],
                                      initializer=tf.constant_initializer(0.0))
        
            wts['w_p3'] = tf.get_variable('Matrix4', [FLAG.ref_3,
                                                  FLAG.ref_2],
                                      tf.float32, init1)
            # tf.random_normal_initializer(stddev=0.02))
            wts['b_p3'] = tf.get_variable('bias4', [FLAG.ref_2],
                                      initializer=tf.constant_initializer(0.0))
        
            wts['w_p4'] = tf.get_variable('Matrix5', [FLAG.ref_2,
                                                  FLAG.num_joints * FLAG.ref_1],
                                      tf.float32, init1)
            # tf.random_normal_initializer(stddev=0.02))
            wts['b_p4'] = tf.get_variable('bias5', [FLAG.num_joints * FLAG.ref_1],
                                      initializer=tf.constant_initializer(0.0))
        
            wts['w_p5'] = tf.get_variable('Matrix6', [FLAG.ref_1, flatten_size],
                                      tf.float32, init1)
            # tf.random_normal_initializer(stddev=0.02))
            wts['b_p5'] = tf.get_variable('bias6', [flatten_size],
                                      initializer=tf.constant_initializer(0.0))
    
        return wts
    
    
    def _refiner(self, input, FLAG, wts):
    
        with tf.name_scope('RefinerNet/'):
            
            shape_ = input.get_shape().as_list()
            input_ = tf.reshape(input, [FLAG.batch_size] + shape_[1:-1]
                               + [shape_[-1] / FLAG.num_joints,
                                  FLAG.num_joints])
            
            
            joint_features = []
            
            for v in range(14):
                in_ = input_[:,:,:,:,v]
                shape = in_.get_shape().as_list()
                dim = np.prod(shape[1:])
                in_f = tf.reshape(in_, [-1, dim])
                joint_features.append(tf.nn.bias_add(tf.matmul(in_f,
                                                               wts['w_p']),
                    wts['b_p']))
            
            joint_vector = tf.concat(joint_features,axis=1)

            
            rep_1 = tf.nn.bias_add(tf.matmul(joint_vector, wts['w_p1']), wts['b_p1'])

            rep_2 = tf.nn.bias_add(tf.matmul(rep_1, wts['w_p2']), wts['b_p2'])

            rep_3 = tf.nn.bias_add(tf.matmul(rep_2, wts['w_p3']), wts['b_p3'])

            rep_4 = tf.nn.bias_add(tf.matmul(rep_3, wts['w_p4']), wts['b_p4'])

            outputs = []
            for v in range(0,FLAG.num_joints * FLAG.ref_1, FLAG.ref_1):
                vin = rep_4[:,v:v+FLAG.ref_1]
                
                vout = tf.nn.bias_add(tf.matmul(vin, wts['w_p5']), wts['b_p5'])
                outputs.append(tf.reshape(vout, [FLAG.batch_size] + shape_[1:-1]
                               + [shape_[-1] / FLAG.num_joints, 1]))
            output_ = tf.concat(outputs, axis=-1)
            
            out_r = tf.reshape(output_, [FLAG.batch_size, FLAG.volume_res,
                                         FLAG.volume_res,
                                         FLAG.volume_res*FLAG.num_joints])
            
            return out_r
            

    def _conv(self, inputs, nb_filter, kernel_size=1, strides=1, pad='VALID',
              name='conv'):
        with tf.variable_scope(name) as scope:
            with tf.device('/cpu:0'):
                shape = [kernel_size, kernel_size,
                         inputs.get_shape().as_list()[3],
                         nb_filter]
                kernel = tf.get_variable(name, shape, initializer= \
                    tf.contrib.layers.xavier_initializer(
                        uniform=False))
            conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1],
                                padding=pad,
                                data_format='NHWC')
            return conv
    
    def _conv_bn_relu(self, inputs, nb_filter, kernel_size=1, strides=1,
                      name='weights'):
        with tf.variable_scope(name) as scope:
            with tf.device('/cpu:0'):
                shape = [kernel_size, kernel_size,
                         inputs.get_shape().as_list()[3],
                         nb_filter]
                kernel = tf.get_variable(name, shape, initializer= \
                    tf.contrib.layers.xavier_initializer(uniform=False))
            conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1],
                                padding='SAME', data_format='NHWC')
            norm = tf.contrib.layers.batch_norm(conv, 0.9, epsilon=1e-5,
                                                activation_fn=tf.nn.relu,
                                                scope=scope)
            return norm
    
    def _conv_block(self, inputs, nb_filter_out, name='_conv_block'):
        with tf.variable_scope(name) as scope:
            with tf.variable_scope('norm_conv1') as sc:
                norm1 = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5,
                                                     activation_fn=tf.nn.relu,
                                                     scope=sc,
                                                     fused=True)
                conv1 = self._conv(norm1, nb_filter_out / 2, 1, 1, 'SAME',
                                   name='conv1')
            with tf.variable_scope('norm_conv2') as sc:
                norm2 = tf.contrib.layers.batch_norm(conv1, 0.9, epsilon=1e-5,
                                                     activation_fn=tf.nn.relu,
                                                     scope=sc,
                                                     fused=True)
                conv2 = self._conv(norm2, nb_filter_out / 2, 3, 1, 'SAME',
                                   name='conv2')
            with tf.variable_scope('norm_conv3') as sc:
                norm3 = tf.contrib.layers.batch_norm(conv2, 0.9, epsilon=1e-5,
                                                     activation_fn=tf.nn.relu,
                                                     scope=sc,
                                                     fused=True)
                conv3 = self._conv(norm3, nb_filter_out, 1, 1, 'SAME',
                                   name='conv3')
            return conv3
    
    def _skip_layer(self, inputs, nb_filter_out, name='_skip_layer'):
        if inputs.get_shape()[3].__eq__(tf.Dimension(nb_filter_out)):
            return inputs
        else:
            with tf.name_scope(name) as scope:
                conv = self._conv(inputs, nb_filter_out, 1, 1, 'SAME',
                                  name='conv')
                return conv
    
    def _residual_block(self, inputs, nb_filter_out, name='_residual_block'):
        with tf.variable_scope(name) as scope:
            _conv_block = self._conv_block(inputs, nb_filter_out)
            _skip_layer = self._skip_layer(inputs, nb_filter_out)
            return tf.add(_skip_layer, _conv_block)
    
    def _hourglass(self, inputs, n, nb_filter_res, name='_hourglass', rank_=0):
        with tf.variable_scope(name) as scope:
            
            if n > 1:
                # Upper branch
                up1 = self._residual_block(inputs, (128*64*14) /
                                           nb_filter_res, \
                      'up1')
                # Lower branch
                pool = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2],
                                                    'VALID')
                low1 = self._residual_block(pool, (128*64*2*14) / nb_filter_res,
                                            'low1')
                
                low2 = self._hourglass(low1, n - 1, nb_filter_res / 2, 'low2',
                                       rank_)
                low2 = tf.concat([low2, up1], axis=-1, name = 'merged1')
            else:
                low1 = self._residual_block(inputs, (128*64*14) / nb_filter_res, 'low1_1')
                low2 = self._residual_block(low1, (128*64*14) / nb_filter_res, 'low2_1')
            
            low3 = self._residual_block(low2, (128*64*14) / nb_filter_res, 'low3')
            low3 = self._conv_bn_relu(low3, (128*64*14) / nb_filter_res,
                                      name='low3_1')
            
            lower1 = self._conv(low3, nb_filter_res, 1, 1, 'VALID', 'lower1')
            
            self.module_supervisions[n - 1].append(lower1)
            
            #if n == 3:
                #lower1 = self._refiner(lower1, self.FLAG, self.refiner_wts)
                #print(lower1.get_shape().as_list())
                #self.module_supervisions[n].append(lower1)
            
            if n < 3:
                lower2 = tf.image.resize_nearest_neighbor(lower1,
                                                          tf.shape(lower1)[
                                                          1:3] * 2,
                                                          name='upsampling_2')
                lower2 = self._conv_bn_relu(lower2, ((128*64*14) / (nb_filter_res*2)),
                                             name = 'lower2')
                low4 = tf.image.resize_nearest_neighbor(low3,
                                                        tf.shape(low3)[1:3] *2,
                                                        name='upsampling_1')
                low4 = self._conv_block(low4, ((128*64*14) / (nb_filter_res*2))
                                            , name = 'low4_1')
            else:
                lower2 = self._conv_bn_relu(lower1, (128*64*14) / (nb_filter_res*2),
                                            name='lower2')
                low4 = self._residual_block(low3, (128*64*14) / (nb_filter_res*2), 'low4')
                low4 = self._conv_block(low4, (128*64*14) / (nb_filter_res*2),
                                          name='low4_1')
            
            if n < 5:
                return tf.add(lower2, low4, name='merge')
