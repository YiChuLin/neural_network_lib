import numpy as np

class Layer(object):
    def __init__(self,input_shape,param_shapes,output_shape):

        self.input_shape=input_shape
        self.output_shape=output_shape
        self.param_shapes=param_shapes

        self.input_batch=None
        self.output_batch=None
        self.params=[]

        self.input_batch_grad=None
        self.output_batch_grad=None
        self.params_grad=[]

        self.is_initialized=False

    def initializer(self,param_shape):
        raise NotImplementedError

    def _initialize_params(self):
        for param_shape in self.param_shapes:
            self.params.append(self.initializer(param_shape))
            self.params_grad.append(np.zeros(param_shape))
        self.is_initialized=True

    def compute_forward(self,input):
        raise NotImplementedError

    def _compute_batch_forward(self):
        self.output_batch=np.zeros((self.batch_size,)+self.output_shape)
        for i in range(self.batch_size):
            self.output_batch[i]=self.compute_forward(self.input_batch[i].copy())

    def compute_params_grad(self,output_grad,input,output):
        raise NotImplementedError

    def _compute_params_batch_grad(self):
        for param_grad in self.params_grad:
            param_grad[:]=0
        for i in range(self.batch_size):
            params_grad_single=self.compute_params_grad(self.output_batch_grad[i].copy(),self.input_batch[i].copy(),self.output_batch[i].copy())
            for j in range(len(self.params_grad)):
                self.params_grad[j]+=params_grad_single[j]

    def compute_input_grad(self,output_grad,input):
        raise NotImplementedError

    def _compute_input_batch_grad(self):
        self.input_batch_grad=np.zeros((self.batch_size,)+self.input_shape)
        for i in range(self.batch_size):
            self.input_batch_grad[i]=self.compute_input_grad(self.output_batch_grad[i].copy(),self.input_batch[i].copy())

    def forward(self,input_batch):
        if(not self.is_initialized):
            self._initialize_params()

        assert(input_batch.shape[1:]==self.input_shape)
        self.input_batch=input_batch

        self.batch_size=input_batch.shape[0]

        self._compute_batch_forward()
                
        return self.output_batch

    def backward(self,output_batch_grad,learner,regulariser):
        if(not self.is_initialized):
            self._initialize_params()

        assert(callable(learner))
        assert(output_batch_grad.shape[1:]==self.output_shape)
        assert(self.batch_size==output_batch_grad.shape[0])

        self.output_batch_grad=output_batch_grad

        self._compute_params_batch_grad()

        if callable(regulariser):
            for i in range(len(self.params)):
                self.params_grad[i]+=regulariser(self.params[i].copy())

        for i in range(len(self.params)):
            self.params[i][:]=learner(self.params[i].copy(),self.params_grad[i].copy())
        
        self._compute_input_batch_grad()

        return self.input_batch_grad

