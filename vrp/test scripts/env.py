import tensorflow as tf
import numpy as np
import sys
from configs import ParseParams
import collections
from DataGenerator import DataGenerator
from sklearn.preprocessing import normalize

class State(collections.namedtuple("State",
                                        ("load",
                                         "demand",
                                         'd_sat',
                                         "mask"))):
    pass

class Env(object):
    def __init__(self,
                 args):
        '''
        This is the environment for VRP.
        Inputs:
            args: the parameter dictionary. It should include:
                args['n_nodes']: number of nodes in VRP
                args['n_custs']: number of customers in VRP
                args['input_dim']: dimension of the problem which is 3
        '''
        self.capacity = args['capacity']
        self.n_nodes = args['n_nodes']
        self.n_cust = args['n_cust']
        self.input_dim = args['input_dim']
        
        self.data_Gen = DataGenerator(args)
 
        self.args = args
        self.initialize_train_step()
    
    def initialize_train_step( self ):
        self.input_data = self.data_Gen.get_train_next()
        self.input_data = tf.cast( self.input_data, dtype=tf.float32)
        input_concat = np.concatenate( self.input_data )
        transposed_input_concat = np.transpose(input_concat)
        before_norm_by_feature = np.reshape( transposed_input_concat,(self.args['input_dim'],-1))
        norm_by_feature = normalize(before_norm_by_feature, axis=1)
        #before_norm_by_feature will be array of shape (3, batch_size * n_nodes)
        # for example batch_size is 2, n_nodes is 3
        #  [ [x0, x1, x2, x3, x4, x5 ],
        #    [y0, y1, y2, y3, y4, y5 ],
        #    [d0, d1, d2, d3, d4, d5 ]]
        # nomolized would be  x_norm = sqrt( x0^2 + x1^2 + ...+ x5^2)
        # [ [ x0/x_norm, x1/x_norm, ...x5/x_norm],
        #   [ y0/y_norm, ............. y5/y_norm],
        #   [ d0/d_norm, ............. d5/d_norm]]
        data = self.input_data
        self.input_data_norm = np.reshape(np.transpose(norm_by_feature),(data.shape[0],data.shape[1],data.shape[2]))
        self.embeded_data = np.zeros(shape=(self.args['batch_size'],self.args['n_nodes'],self.args['embedding_dim']))
        self.input_pnt = self.input_data[:,:,:(self.input_dim -1)]      # all but demand
        self.demand = self.input_data[:,:,-1]
        self.batch_size = tf.shape(input=self.input_pnt)[0]
        
    def reset(self,beam_width=1):
        '''
        Resets the environment. This environment might be used with different decoders.
        In case of using with beam-search decoder, we need to have to increase
        the rows of the mask by a factor of beam_width.
        '''

        # dimensions
        self.beam_width = beam_width
        self.batch_beam = self.batch_size * beam_width

        self.input_pnt = self.input_data[:,:,:2]        # corresponds to all x,y
        self.demand = self.input_data[:,:,-1]           # corresponds to all the demand, sixe[batch,nb_nodes]

        # modify the self.input_pnt and self.demand for beam search decoder
#         self.input_pnt = tf.tile(self.input_pnt, [self.beam_width,1,1])

        # demand: [batch_size * beam_width, max_time]
        # demand[i] = demand[i+batchsize]
        self.demand = tf.tile(self.demand, [self.beam_width,1])

        # load: [batch_size * beam_width]
        self.load = tf.ones([self.batch_beam])*self.capacity

        # create mask
        self.mask = tf.zeros([self.batch_size*beam_width,self.n_nodes],
                dtype=tf.float32)

        # update mask -- mask if customer demand is 0 and depot
        self.mask = tf.concat([tf.cast(tf.equal(self.demand,0), tf.float32)[:,:-1],
            tf.ones([self.batch_beam,1])],1)

        state = State(load=self.load,
                    demand = self.demand,
                    d_sat = tf.zeros([self.batch_beam,self.n_nodes]),
                    mask = self.mask )

        return state

    def step(self,
             idx,
             beam_parent=None):
        '''
        runs one step of the environment and updates demands, loads and masks
        '''

        # if the environment is used in beam search decoder
        if beam_parent is not None:
            # BatchBeamSeq: [batch_size*beam_width x 1]
            # [0,1,2,3,...,127,0,1,...],
            batchBeamSeq = tf.expand_dims(tf.tile(tf.cast(tf.range(self.batch_size), tf.int64),
                                                 [self.beam_width]),1)
            # batchedBeamIdx:[batch_size*beam_width]
            batchedBeamIdx= batchBeamSeq + tf.cast(self.batch_size,tf.int64)*beam_parent
            # demand:[batch_size*beam_width x sourceL]
            self.demand= tf.gather_nd(self.demand,batchedBeamIdx)
            #load:[batch_size*beam_width]
            self.load = tf.gather_nd(self.load,batchedBeamIdx)
            #MASK:[batch_size*beam_width x sourceL]
            self.mask = tf.gather_nd(self.mask,batchedBeamIdx)


        BatchSequence = tf.expand_dims(tf.cast(tf.range(self.batch_beam), tf.int64), 1)
        batched_idx = tf.concat([BatchSequence,idx],1)

        # how much the demand is satisfied
        temp = tf.gather_nd( self.demand, batched_idx )
        d_sat = tf.minimum( temp, self.load)

        # update the demand
        t1 = tf.cast(tf.shape(input=self.demand),tf.int64)
        print( t1.numpy())
        d_scatter = tf.scatter_nd(batched_idx, d_sat, tf.cast(tf.shape(input=self.demand),tf.int64))      # sparse tensor containing d_sat for the interesting idx
        print( d_scatter.numpy())
        self.demand = tf.subtract(self.demand, d_scatter)
        print( self.demand.numpy())
        # update load
        self.load -= d_sat

        # refill the truck -- idx: [10,9,10] -> load_flag: [1 0 1]
        t1 = tf.equal( idx, self.n_cust)
        t2 = tf.cast(t1, tf.float32)
        t3 = tf.squeeze( t2, 1)
        #check any of the selected idx is referring to the depot
        load_flag = tf.squeeze(tf.cast(tf.equal(idx,self.n_cust),tf.float32),1)
        self.load = tf.multiply(self.load,1-load_flag) + load_flag *self.capacity
        #for any batch, if it's refilled, then the load reset to capacity
        # mask for customers with zero demand
        t1 =  tf.equal(self.demand,0)
        t2 = tf.cast( t1, tf.float32 )
        t3 = t2[:,:-1]
        t4 = tf.zeros([self.batch_beam, 1])
        t5 = [t3, t4]
        t6 = tf.concat(t5, 1)
        self.mask = tf.concat([tf.cast(tf.equal(self.demand,0), tf.float32)[:,:-1],
                                          tf.zeros([self.batch_beam,1])],1)

        # mask if load= 0
        # mask if in depot and there is still a demand
        t1 = tf.cast(tf.equal(self.load,0),    
            tf.float32)     #any load empty?
        t2 = tf.expand_dims( t1, 1)   #add back batch dimension
        t3 = tf.tile( t2, [1,self.n_cust])
        
        t4 = tf.reduce_sum(input_tensor=self.demand,axis=1)  #remaining sum  of demand for each element of the batch
        t5 = tf.greater( t4, 0)
        t6 = tf.cast( t5, tf.float32)   # 1 if there are still demands
        
        t7 = tf.equal(idx,self.n_cust)
        t8 = tf.cast( t7, tf.float32)
        t9 = tf.squeeze( t8 )         #refil mask
        
        t10 = tf.multiply( t6, t9 )
        t11 = tf.expand_dims( t10, 1 )
        
        t12 = tf.concat( [t3, t11],1) #t3 is tensor of [batch, customer], element value 0 means load is not empty yet, 1 mean load is empty, can't satisfy
                                      # any more  customer demands, need to go back depot to refill
                                      # t11 is tensor of shape [batch, 1], element value 1 means there is still demand for that batch and the truck just goes 
                                      # back to depot to refil,  next step the truck should go out again next step
        print( t12.numpy() )
        #in test data, first step, t12 is a [4,11] tensor with all four 0 values,  means there some customers demand satsified, but there is still demand for each case and no tack to depot to refill happened
        #  and no demand with 0 amount 
        self.mask += tf.concat( [tf.tile(tf.expand_dims(tf.cast(tf.equal(self.load,0),
            tf.float32),1), [1,self.n_cust]),
            tf.expand_dims(tf.multiply(tf.cast(tf.greater(tf.reduce_sum(input_tensor=self.demand,axis=1),0),tf.float32),
                             tf.squeeze( tf.cast(tf.equal(idx,self.n_cust),tf.float32))),1)],1)

        state = State(load=self.load,
                    demand = self.demand,
                    d_sat = d_sat,
                    mask = self.mask )

        return state