# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 15:33:12 2020

@author: satheshkm
"""
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import os
import sys

#%%
loc = os.path.dirname(os.path.realpath(__file__))
os.chdir(loc)

#%%
def make_boundary(data, top, bottom, left, right):
    data[0,:] = top
    data[-1,:] = bottom
    data[:,0] = left
    data[:,-1] = right
    
    data[0,0] = 0.5*(top + left)
    data[0,-1] = 0.5*(top + right)
    data[-1,0] = 0.5*(bottom + left)
    data[-1,-1] = 0.5*(bottom + right)
    
    
def fd_solver(geometry, boundary, tol=1e-4):
    temp_boundary = geometry[boundary]
    kernel = np.array([[0., 0.25, 0.], [0.25, 0., 0.25], [0., 0.25, 0.]])
    k = 1
    while True:
        mod_array = convolve2d(geometry, kernel, mode='same', boundary='symm')
        mod_array[boundary] = temp_boundary
        change = np.sum(np.abs(mod_array-geometry))
        if change <= tol or k>10000000:
            print(change, k)
            return geometry
        geometry = mod_array
        k += 1


def make_geometry(grid_size, top, bottom, left, right):
    
    avg = 0.25*(top + bottom + left + right)
    data = np.zeros((grid_size, grid_size))
    boundary = np.zeros((grid_size, grid_size), dtype=np.bool)
    make_boundary(data, top, bottom, left, right)
    make_boundary(boundary, 1, 1, 1, 1)
    return data, boundary

def set_boundary(image_size, batch_size, max_temp=100):
    data = np.zeros((batch_size, image_size, image_size, 1)).astype('float32')
    for i in range(batch_size):
        data[i,:,0,:] = np.random.uniform(max_temp)
        data[i,0,:,:] = np.random.uniform(max_temp)
        data[i,:,-1,:] = np.random.uniform(max_temp)
        data[i,-1,:,:] = np.random.uniform(max_temp)
    return data
#%%

geometry, boundary = make_geometry(256, 0,100, 0, 100)

steady_state = fd_solver(geometry, boundary)
plt.contourf(steady_state)
plt.colorbar()
plt.show()

#%%
res_path = r'results/fd_results/'
#%%
res_name = res_path + 'data_3_256.npz'
np.savez(res_name, boundary=geometry, result=steady_state)

#%%
import tensorflow as tf

#%%
class UNet():
    
    def __init__(self, image_size=256, max_temp=100):
        self.image_size = image_size
        self.max_temp = max_temp
        self.center_mask = np.zeros([1, self.image_size, self.image_size, 1]).astype('float32')
        self.center_mask[:,:,0,:] = 1
        self.center_mask[:,:,-1,:] = 1
        self.center_mask[:,0,:,:] = 1
        self.center_mask[:,-1,:,:] = 1
        self.boundary_mask = 1 - self.center_mask
        self.center_mask = tf.constant(self.center_mask)
        self.boundary_mask = tf.constant(self.boundary_mask)
        
    
    def __call__(self, input_layer):
        
        a = []
        
        with tf.variable_scope('model'):
    
            
            value = tf.layers.conv2d(input_layer, filters=16, kernel_size=(3, 3), padding='same')
            value = tf.nn.relu(value)
            value = tf.layers.conv2d(value, filters=16,  kernel_size=(3, 3), padding='same')
            value = tf.nn.relu(value)
            a.append(value)
            value = tf.layers.max_pooling2d(value, pool_size=2, strides=2)
        
            
            value = tf.layers.conv2d(value, filters=32, kernel_size=(3, 3), padding='same')
            value = tf.nn.relu(value)
            value = tf.layers.conv2d(value, filters=32,  kernel_size=(3, 3), padding='same')
            value = tf.nn.relu(value)   
            a.append(value)
            value = tf.layers.max_pooling2d(value, pool_size=2, strides=2)
            
            value = tf.layers.conv2d(value, filters=64, kernel_size=(3, 3), padding='same')
            value = tf.nn.relu(value)
            value = tf.layers.conv2d(value, filters=64,  kernel_size=(3, 3), padding='same')
            value = tf.nn.relu(value)
            a.append(value)
            value = tf.layers.max_pooling2d(value, pool_size=2, strides=2) 
            
            value = tf.layers.conv2d(value, filters=128, kernel_size=(3, 3), padding='same')
            value = tf.nn.relu(value)
            value = tf.layers.conv2d(value, filters=128,  kernel_size=(3, 3), padding='same')
            value = tf.nn.relu(value)
            a.append(value)
            value = tf.layers.max_pooling2d(value, pool_size=2, strides=2)  
            
            value = tf.layers.conv2d(value, filters=256, kernel_size=(3, 3), padding='same')
            value = tf.nn.relu(value)
            value = tf.layers.conv2d(value, filters=256,  kernel_size=(3, 3), padding='same')
            value = tf.nn.relu(value)
            
            value = tf.layers.conv2d_transpose(value, filters=128, kernel_size=(2, 2), strides=2, padding='same')
            value = tf.concat([value, a[-1]], axis=3)
            value = tf.layers.conv2d(value, filters=128, kernel_size=(3, 3), padding='same')
            value = tf.nn.relu(value)
            value = tf.layers.conv2d(value, filters=128,  kernel_size=(3, 3), padding='same')
            value = tf.nn.relu(value)
            
            value = tf.layers.conv2d_transpose(value, filters=64, kernel_size=(2, 2), strides=2, padding='same')
            value = tf.concat([value, a[-2]], axis=3)
            value = tf.layers.conv2d(value, filters=64, kernel_size=(3, 3), padding='same')
            value = tf.nn.relu(value)
            value = tf.layers.conv2d(value, filters=64,  kernel_size=(3, 3), padding='same')
            value = tf.nn.relu(value)
            
            value = tf.layers.conv2d_transpose(value, filters=32, kernel_size=(2, 2), strides=2, padding='same')
            value = tf.concat([value, a[-3]], axis=3)
            value = tf.layers.conv2d(value, filters=32, kernel_size=(3, 3), padding='same')
            value = tf.nn.relu(value)
            value = tf.layers.conv2d(value, filters=32,  kernel_size=(3, 3), padding='same')
            value = tf.nn.relu(value)
        
            value = tf.layers.conv2d_transpose(value, filters=16, kernel_size=(2, 2), strides=2, padding='same')
            value = tf.concat([value, a[-4]], axis=3)
            value = tf.layers.conv2d(value, filters=16, kernel_size=(3, 3), padding='same')
            value = tf.nn.relu(value)
            value = tf.layers.conv2d(value, filters=16,  kernel_size=(3, 3), padding='same')
            value = tf.nn.relu(value)    
            
            value = tf.layers.conv2d(value, filters=1, kernel_size=(1, 1), strides=1, padding='same')
            value = tf.nn.tanh(value)
            
            value = (value + 1) * self.max_temp / 2.
            out = (value * self.boundary_mask) + (input_layer * self.center_mask)
        
        
        return out
#%%
def phy_loss(image_size=256, batch_size=8):
    
    kernel = tf.Variable([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], trainable=False, dtype=tf.float32)
    kernel = tf.reshape(kernel, [3, 3, 1, 1])
    w = [4., 16.]
    in_shape = image_size
    loss_shape = image_size
    down_shape = []
    mask_shape = []
    while loss_shape > 32:
        loss_shape = loss_shape // 4
        mask_shape.append(loss_shape)
        idx = np.round(np.linspace(0, in_shape-1, loss_shape)).astype(np.int32)
        down_shape.append(np.ix_(idx, idx))
    mask_arr = []
    for row, col in down_shape:
        temp = np.zeros((batch_size, in_shape, in_shape, 1), dtype='bool')
        temp[:,row, col, :]=True
        mask_arr.append(temp)

    def loss(image):
        with tf.variable_scope('loss'):
            loss_ = tf.reduce_mean(tf.abs(tf.nn.conv2d(image, kernel, strides=1, padding='SAME')))
            i = 0
            for mask in mask_arr:
                #print(rows, cols)
                img = tf.boolean_mask(image, mask)
                img = tf.reshape(img,(batch_size, mask_shape[i], mask_shape[i], 1))
                loss_ += w[i]*tf.reduce_mean(tf.square(tf.nn.conv2d(img, kernel, strides=1, padding='VALID')))
                i += 1
        return loss_
    return loss

def optimizer(loss, global_step, learning_rate=0.001):
    with tf.variable_scope('optimizer'):
        opti = tf.train.AdamOptimizer(learning_rate=learning_rate)
        opti = opti.minimize(loss, global_step=global_step)
    return opti

#%%


def train(max_temp=100, image_size=256, batch_size=8, epochs=10, steps_per_epoch=500, learning_rate=0.001):
    mpath = r'model'
    img_path = r'results/train/'
    v = np.linspace(0, 100, 11)    
    with tf.Graph().as_default():
        model = UNet(image_size=image_size, max_temp=max_temp)
        loss = phy_loss(image_size, batch_size=batch_size)
        
        with tf.variable_scope('main'):
            
            in_layer = tf.placeholder(dtype=tf.float32, shape=[batch_size, image_size, image_size, 1])
            global_step = tf.Variable(1, trainable=False)
            temp_pred = model(in_layer)
            loss_val = loss(temp_pred)
            opti = optimizer(loss_val, global_step, learning_rate)
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            
            stream_loss = 0
            with tf.Session() as sess:
                
                sess.run(init) 
                ckpt = tf.train.get_checkpoint_state(mpath)
                if ckpt and ckpt.model_checkpoint_path:
                    print(f'Restoring model from {mpath}')
                    saver.restore(sess, ckpt.model_checkpoint_path)
                for i in range(1, epochs+1):
                    for j in range(1, steps_per_epoch+1):
                        x_batch = set_boundary(image_size, batch_size, max_temp)
                        _, temp, tot_loss = sess.run([opti, temp_pred, loss_val], feed_dict={in_layer:x_batch})
                        stream_loss += tot_loss
                        
                        if j%100==0:
                            i_name = img_path + 'train_' + str((i-1)*steps_per_epoch + j) + '.jpg'
                            plt.imshow(temp[0,:,:,0], cmap='rainbow',vmin=0.0, vmax=100.0)
                            plt.colorbar(ticks=v)
                            plt.savefig(i_name)
                            plt.show()

                            print(f'Completed {j} steps in {i}th epoch with loss: {stream_loss/(((i-1)*steps_per_epoch) + j):.5f}')
                    print(f'Finished {i} epochs with loss: {stream_loss/(i*steps_per_epoch) :.5f}')
                    saver.save(sess, save_path = mpath+'/PINN', global_step=i+38)
                        
#%%
train(100,256,8,3,500,0.00001)
#%%

def evaluate(geometry, temp_act):
    img_size = geometry.shape[1]
    mpath_1 = r'model'
    img_path = r'results/'
    with tf.Graph().as_default():
         model = UNet(img_size)
         loss = phy_loss(img_size, batch_size=1)
         with tf.variable_scope('main'):
             
             in_layer = tf.placeholder(dtype=tf.float32, shape=[1, img_size, img_size, 1])
             temp_pred = model(in_layer)
             test_loss = loss(temp_pred)
             #var_list = tf.trainable_variables()

             saver = tf.train.Saver()
             
             with tf.Session() as sess:
                 ckpt = tf.train.get_checkpoint_state(mpath_1)
                 saver.restore(sess, ckpt.model_checkpoint_path)
                 tf.get_default_graph().as_graph_def()

                 x_geo = np.expand_dims(geometry, axis=(0,-1))
                 temp_act = np.expand_dims(temp_act, axis=(0,-1))

                 pred, loss_ = sess.run([temp_pred, test_loss], feed_dict={in_layer:x_geo})
 
                 temp_plt = np.concatenate([temp_act, pred, np.abs(temp_act-pred)], axis=2)
                #temp_plt = pred
                 plt.imshow(temp_plt[0,:,:,0], cmap='jet', interpolation='nearest')
                 plt.colorbar()
                 plt.show()
                 print('MSE: ', loss_)
#%%
res_name = res_path + 'data_3_256.npz'
test_arr = np.load(res_name)
test_geo = test_arr['boundary']
test_res = test_arr['result']

#x_test = set_boundary(256, 1, 100)
evaluate(test_geo, test_res)

#%%


                     
                 
             

