from keras.layers.core import Layer
from keras.models import Sequential,Model
from keras import regularizers
from keras.layers import Input,Dense,Multiply,Concatenate,LSTM,TimeDistributed,Permute,Dropout,Activation,CuDNNLSTM,Dropout,Conv1D,MaxPooling1D,GlobalAveragePooling1D
from keras import backend as K
import keras
from keras.layers.normalization import BatchNormalization
regul=0.001 
class RelationalModel(Layer):
    def __init__(self, input_size,n_of_features,filters,rm=None,reuse_model=False,**kwargs):
        self.input_size=input_size
        self.n_of_features=n_of_features
        n_of_filters = len(filters)
        if(reuse_model):
            relnet = rm    
        else:
            input1 = Input(shape=(n_of_features,))
            x=input1
            for i in range(n_of_filters-1):
                x = Dense(filters[i],kernel_regularizer=regularizers.l2(regul),activity_regularizer=regularizers.l2(regul),
                          bias_regularizer=regularizers.l2(regul),activation='relu')(x)
                
            x = Dense(filters[-1],kernel_regularizer=regularizers.l2(regul),
              bias_regularizer=regularizers.l2(regul),activation='linear')(x)    
            relnet = Model(inputs=[input1],outputs=[x])
        self.relnet = relnet
        self.output_size = filters[-1]
        
        super(RelationalModel, self).__init__(**kwargs)

    def build(self, input_shape):
        self.relnet.build((None,self.n_of_features)) #,self.input_size
        self.trainable_weights = self.relnet.trainable_weights

    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        input_size = self.input_size
        return (None,)+input_size+(int(output_size),)

    def call(self, X):
        X = K.reshape(X,(-1,self.n_of_features))
        output = self.relnet.call(X)
        output= K.reshape(output,((-1,)+self.input_size+(self.output_size,)))
        return output
    
    def getRelnet(self):
        return self.relnet
class ObjectModel(Layer):
    def __init__(self, input_size,n_of_features,filters,om=None,reuse_model=False,**kwargs):
        self.input_size=input_size 
        self.n_of_features=n_of_features
        n_of_filters = len(filters)

        if (reuse_model):
            objnet = om
        else:
            input1 = Input(shape=(n_of_features,))
            x=input1
            for i in range(n_of_filters-1):
                x = Dense(filters[i],kernel_regularizer=regularizers.l2(regul),activity_regularizer=regularizers.l2(regul),
                          bias_regularizer=regularizers.l2(regul),activation='relu')(x)
            x = Dense(filters[-1],kernel_regularizer=regularizers.l2(regul),
                  bias_regularizer=regularizers.l2(regul),activation='linear')(x)
                
            objnet = Model(inputs=[input1],outputs=[x])
        self.objnet = objnet

        self.output_size = filters[-1]
        super(ObjectModel, self).__init__(**kwargs)

    def build(self, input_shape):
        self.objnet.build((None,self.input_size,self.n_of_features))
        self.trainable_weights = self.objnet.trainable_weights

    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        input_size = self.input_size

        return (None,)+input_size+(int(output_size),)

    def call(self, X):
        X = K.reshape(X,(-1,self.n_of_features))
        output = self.objnet.call(X)
        output= K.reshape(output,((-1,)+self.input_size+(self.output_size,)))

        return output
    def getObjnet(self):
        return self.objnet
class RecurrentRelationalModel(Layer):
    def __init__(self, input_size,n_of_features,filters,rm=None,reuse_model=False,timestep_diff=5,**kwargs):
        self.input_size=input_size
        self.n_of_features=n_of_features
        self.timestep_diff=timestep_diff
        n_of_filters = len(filters)
        if(reuse_model):
            relnet = rm    
        else:
            input1 = Input(shape=(timestep_diff,n_of_features)) 
            x = CuDNNLSTM(filters[0],kernel_regularizer=regularizers.l2(regul),
                      bias_regularizer=regularizers.l2(regul))(input1)
            for i in range(n_of_filters-2):
                x = Dense(filters[1+i],kernel_regularizer=regularizers.l2(regul),
                          bias_regularizer=regularizers.l2(regul),activation='relu')(x)             
            if filters[-1]==1:
                x = Dense(filters[-1],kernel_regularizer=regularizers.l2(regul),
                          bias_regularizer=regularizers.l2(regul),activation='sigmoid')(x)
            else:
                x = Dense(filters[-1],kernel_regularizer=regularizers.l2(regul),
                              bias_regularizer=regularizers.l2(regul),activation='softmax')(x)
            relnet = Model(inputs=[input1],outputs=[x])
        self.relnet = relnet
        self.output_size = filters[-1]
        
        super(RecurrentRelationalModel, self).__init__(**kwargs)

    def build(self, input_shape):
        self.relnet.build((None,self.n_of_features)) #,self.input_size
        self.trainable_weights = self.relnet.trainable_weights

    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        input_size = self.input_size
        return (None,input_size,int(output_size))

    def call(self, X):              
        x = K.reshape(X,(-1,self.timestep_diff,self.n_of_features))
        output = self.relnet.call(x)
        output= K.reshape(output,((-1,)+ self.input_size + (self.output_size,)))
        return output
    
    def getRelnet(self):
        return self.relnet
class RecurrentRelationalModel_many(Layer):
    def __init__(self, input_size,n_of_features,filters,relnetpart1_model=None,relnetpart2_model=None,reuse_model=False,timestep_diff=5,lookstart=50,**kwargs):
        self.input_size=input_size
        self.n_of_features=n_of_features
        self.timestep_diff=timestep_diff
        n_of_filters = len(filters)
        input1 = Input(shape=(timestep_diff,n_of_features)) 
        x = CuDNNLSTM(filters[0],kernel_regularizer=regularizers.l2(regul),
                  bias_regularizer=regularizers.l2(regul),return_sequences=True)(input1)
        for i in range(n_of_filters-2):
            x = TimeDistributed(Dense(filters[1+i],kernel_regularizer=regularizers.l2(regul),
                      bias_regularizer=regularizers.l2(regul),activation='relu'))(x)
        relnetpart1 = Model(inputs=[input1],outputs=[x])

        input2 = Input(shape=(100,)) 
        x=input2
        if filters[-1]==1:
            x = Dense(filters[-1],kernel_regularizer=regularizers.l2(regul),
                      bias_regularizer=regularizers.l2(regul),activation='sigmoid')(x)
        else:
            x = Dense(filters[-1],kernel_regularizer=regularizers.l2(regul),
                          bias_regularizer=regularizers.l2(regul),activation='softmax')(x)
        relnetpart2 = Model(inputs=[input2],outputs=[x])
        if(reuse_model):
            relnetpart1.set_weights(relnetpart1_model.get_weights()) 
            relnetpart2.set_weights(relnetpart2_model.get_weights()) 
        relnet = Model(inputs=[input1,input2],outputs=[x])
        
        self.relnet = relnet
        self.relnetpart1 = relnetpart1
        self.relnetpart2 = relnetpart2
        self.lookstart=lookstart
        self.output_size = filters[-1]
        self._count=1
        super(RecurrentRelationalModel_many, self).__init__(**kwargs)

    def build(self, input_shape):
        self.relnetpart1.build((None,self.n_of_features))
        self.relnetpart2.build((None,100))
        self.trainable_weights=[]
        for w in self.relnetpart1.trainable_weights:
                self.trainable_weights.append(w)
        for w in self.relnetpart2.trainable_weights:
                self.trainable_weights.append(w)
        
    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        input_size = self.input_size
        return (None,input_size,int(output_size))

    def call(self, X ):              
        x = K.reshape(X,(-1,self.timestep_diff,self.n_of_features))
        output = self.relnetpart1.call(x)
        output = self.relnetpart2.call(output[:,self.lookstart:,:])        
        output= K.reshape(output,(-1,self.input_size,self.timestep_diff-self.lookstart,self.output_size))
        return output
    
    def getRelnet(self):
        return self.relnetpart1,self.relnetpart2