import collections
import sys
sys.path.append("../")
from VRP.vrp_utils import create_VRP_dataset
import numpy as np

class DataGenerator(object):
    def __init__(self,
                 args):

        '''
        This class generates VRP problems for training and test
        Inputs:
            args: the parameter dictionary. It should include:
                args['random_seed']: random seed
                args['test_size']: number of problems to test
                args['n_nodes']: number of nodes
                args['n_cust']: number of customers
                args['batch_size']: batchsize for training
        '''
        self.args = args
        assert not self.args['ups']
        self.rnd = np.random.RandomState(seed= args['random_seed'])
        print('Created train iterator.')

        # create test data
        self.n_problems = args['test_size']
        self.test_data = create_VRP_dataset(self.n_problems,args['n_cust'],args['data_dir'],
            seed = args['random_seed']+1,data_type='test')

        self.reset()

    def reset(self):
        self.count = 0

    def get_train_next(self):
        '''
        Get next batch of problems for training
        Retuens:
            input_data: data with shape [batch_size x max_time x 3]
        '''

        input_pnt = self.rnd.uniform(0,1,
            size=(self.args['batch_size'],self.args['n_nodes'],2))

        demand = self.rnd.randint(1,10,[self.args['batch_size'],self.args['n_nodes']])
        demand[:,-1]=0 # demand of depot

        input_data = np.concatenate([input_pnt,np.expand_dims(demand,2)],2)

        return input_data


    def get_test_next(self):
        '''
        Get next batch of problems for testing
        '''
        if self.count<self.args['test_size']:
            input_pnt = self.test_data[self.count:self.count+1]
            self.count +=1
        else:
            warnings.warn("The test iterator reset.")
            self.count = 0
            input_pnt = self.test_data[self.count:self.count+1]
            self.count +=1

        return input_pnt

    def get_test_all(self):
        '''
        Get all test problems
        '''
        return self.test_data