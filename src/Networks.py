from keras.layers import Permute,Subtract,Add,Lambda,Input,Concatenate,TimeDistributed,Activation,Dropout,dot,Reshape
import tensorflow as tf
from keras.activations import tanh,relu
from keras import optimizers
from keras import regularizers
from Blocks import *
class PropagationNetwork:
    def __init__(self):
        self.Nets={}
        self.set_weights=False
    def setModel(self,n_objects,PATH):
        self.Nets[n_objects].load_weights(PATH)
    def getModel(self,n_objects,object_dim=6,relation_dim=1):
        if n_objects in self.Nets.keys():
            return self.Nets[n_objects]
        n_relations  = n_objects * (n_objects - 1)
        #Inputs
        objects= Input(shape=(n_objects,object_dim),name='objects')

        sender_relations= Input(shape=(n_objects,n_relations),name='sender_relations')
        receiver_relations= Input(shape=(n_objects,n_relations),name='receiver_relations')
        permuted_senders_rel=Permute((2,1))(sender_relations)
        permuted_receiver_rel=Permute((2,1))(receiver_relations)
        relation_info= Input(shape=(n_relations,relation_dim),name='relation_info')
        propagation= Input(shape=(n_objects,100),name='propagation')

        # Getting sender and receiver objects
        senders=dot([permuted_senders_rel,objects],axes=(2,1))
        receivers=dot([permuted_receiver_rel,objects],axes=(2,1))
        
        # Getting specific features of objects for relationNetwork
        get_attributes=Lambda(lambda x: x[:,:,0:2], output_shape=(n_relations,2))
        get_pos=Lambda(lambda x: x[:,:,2:4], output_shape=(n_relations,2))
        get_vel=Lambda(lambda x: x[:,:,4:6], output_shape=(n_relations,2))
        # Getting specific features of objects for objectNetwork        
        get_attributes2=Lambda(lambda x: x[:,:,0:2], output_shape=(n_objects,2))
        get_velocities=Lambda(lambda x: x[:,:,4:6], output_shape=(n_objects,2))
        
        if(self.set_weights):
            rm = RelationalModel((n_relations,),8+relation_dim,[150,150,150,150],self.relnet,True)
            om = ObjectModel((n_objects,),4,[100,100],self.objnet,True)
            rmp = RelationalModel((n_relations,),350,[150,150,100],self.relnetp,True)
            omp = ObjectModel((n_objects,),300,[100,102],self.objnetp,True)
        else:
            rm=RelationalModel((n_relations,),8+relation_dim,[150,150,150,150])
            om = ObjectModel((n_objects,),4,[100,100])
            
            rmp=RelationalModel((n_relations,),350,[150,150,100])
            omp = ObjectModel((n_objects,),300,[100,102])
            
            self.set_weights=True
            self.relnet=rm.getRelnet()
            self.objnet=om.getObjnet()
            self.relnetp=rmp.getRelnet()
            self.objnetp=omp.getObjnet()
            
        r_att=get_attributes(receivers)
        s_att=get_attributes(senders)

        r_pos=get_pos(receivers)
        s_pos=get_pos(senders)
        r_vel=get_vel(receivers)
        s_vel=get_vel(senders)

        r_posvel = Concatenate()([r_pos,r_vel])
        s_posvel = Concatenate()([s_pos,s_vel])
        
        # Getting dynamic state differences.
        dif_rs=Subtract()([r_posvel,s_posvel])
        
        # Creating Input of Relation Network
        rel_vector_wo_prop= Concatenate()([relation_info,dif_rs,r_att,s_att])
        obj_vector_wo_er=Concatenate()([get_velocities(objects),get_attributes2(objects)])   
        
        rel_encoding=Activation('relu')(rm(rel_vector_wo_prop))
        obj_encoding=Activation('relu')(om(obj_vector_wo_er))
        rel_encoding=Dropout(0.1)(rel_encoding)
        obj_encoding=Dropout(0.1)(obj_encoding)
        prop=propagation
        prop_layer=Lambda(lambda x: x[:,:,2:], output_shape=(n_objects,100))

        for _ in range(5):
            senders_prop=dot([permuted_senders_rel,prop],axes=(2,1))
            receivers_prop=dot([permuted_receiver_rel,prop],axes=(2,1))
            rmp_vector=Concatenate()([rel_encoding,senders_prop,receivers_prop])
            x = rmp(rmp_vector)
            effect_receivers = Activation('tanh')(dot([receiver_relations,x],axes=(2,1)))
            omp_vector=Concatenate()([obj_encoding,effect_receivers,prop])#
            x = omp(omp_vector)
            prop=Activation('tanh')(Add()([prop_layer(x),prop]))
        no_hand=Lambda(lambda x: x[:,1:,:2], output_shape=(n_objects-1,2),name='target')
        predicted=no_hand(x)
        model = Model(inputs=[objects,sender_relations,receiver_relations,relation_info,propagation],outputs=[predicted])
        
        adam = optimizers.Adam(lr=0.0001, decay=0.0)
        model.compile(optimizer=adam, loss='mse')
        self.Nets[n_objects]=model
        return model

class TemporalPropagationNetwork:
    def __init__(self):
        self.Nets={}
        self.set_weights=False
                    
    def setModel(self,n_objects,PATH):
        self.Nets[n_objects].load_weights(PATH)
    def getModel(self,n_objects,object_dim=6,relation_dim=1,timestep_diff=100,lookstart=50):
        n_relations  = n_objects * (n_objects - 1)
        #Inputs
        objects= Input(shape=(timestep_diff, n_objects,object_dim),name='objects')
        relation_data= Input(shape=(timestep_diff,n_relations,8),name='diff_objects')

        sender_relations_in= Input(shape=(timestep_diff,n_objects,n_relations),name='sender_relations')
        receiver_relations_in= Input(shape=(timestep_diff,n_objects,n_relations),name='receiver_relations')
        sender_relations=sender_relations_in
        receiver_relations=receiver_relations_in
        known_relations= Input(shape=(1,n_relations),name='known_relations')
        unknown_relations= Input(shape=(1,n_relations),name='unknown_relations')
        
        relation_dummy= Input(shape=(n_relations,relation_dim),name='relation_dummy')
        permuted_senders_rel=Permute((1,3,2))(sender_relations)
        permuted_receiver_rel=Permute((1,3,2))(receiver_relations)
        
        propagation= Input(shape=(timestep_diff,n_objects,100),name='propagation')
        
        # Lambda Matmul Layers
        prop_objects=Lambda(lambda x:tf.matmul(receiver_relations,x),
                                            output_shape =(timestep_diff,n_objects,100))#,
        prop_senders=Lambda(lambda x:tf.matmul(permuted_senders_rel,x),
                                            output_shape =(timestep_diff,n_relations,100))#,
        prop_receivers=Lambda(lambda x:tf.matmul(permuted_receiver_rel,x),
                                            output_shape =(timestep_diff,n_relations,100))
        get_weights=Lambda(lambda x: x[:,:,:,0:2], output_shape=(timestep_diff,n_objects,2))
        get_velo=Lambda(lambda x: x[:,:,:,4:6], output_shape=(timestep_diff,n_objects,2))

        if(self.set_weights):    
            rm = RelationalModel((timestep_diff,n_relations),8,[150,150,150,150],self.relnet,True)
            om = ObjectModel((timestep_diff,n_objects),4,[100,100],self.objnet,True)
            rmp = RelationalModel((timestep_diff,n_relations),350,[150,100],self.relnetp,True)
            omp = ObjectModel((timestep_diff,n_objects),300,[150,100],self.objnetp,True)
            rme = RecurrentRelationalModel_many(n_relations,100,[100,relation_dim],self.rmepart1,self.rmepart2,True,timestep_diff,lookstart)
        else:
            rm=RelationalModel((timestep_diff,n_relations),8,[150,150,150,150])
            om = ObjectModel((timestep_diff,n_objects),4,[100,100])
            rmp=RelationalModel((timestep_diff,n_relations),350,[150,100])
            omp = ObjectModel((timestep_diff,n_objects),300,[150,100])
            rme = RecurrentRelationalModel_many(n_relations,100,[100,relation_dim],None,None,False,timestep_diff,lookstart)
            self.set_weights=True
            self.relnet=rm.getRelnet()
            self.objnet=om.getObjnet()
            self.relnetp=rmp.getRelnet()
            self.objnetp=omp.getObjnet()
            self.rmepart1,self.rmepart2=rme.getRelnet()
        
        # Creating Input of Relation Network
        rel_vector_wo_prop= relation_data
        obj_vector_wo_er= Concatenate()([get_weights(objects),get_velo(objects)])
        rel_encoding=rm(rel_vector_wo_prop)
        obj_encoding=om(obj_vector_wo_er)
        rel_encoding=Activation('relu')(rm(rel_vector_wo_prop))
        obj_encoding=Activation('relu')(om(obj_vector_wo_er))
        rel_encoding=Dropout(0.5)(rel_encoding)
        obj_encoding=Dropout(0.5)(obj_encoding)
        prop=propagation  
        for _ in range(7):
            senders_prop=prop_senders(prop)
            receivers_prop=prop_receivers(prop)
            rmp_vector=Concatenate()([rel_encoding,senders_prop,receivers_prop])
            x = Activation('relu')(rmp(rmp_vector))
            effect_receivers = prop_objects(x)
            omp_vector=Concatenate()([obj_encoding,effect_receivers,prop])
            prop = Activation('relu')(omp(omp_vector))
        x=Permute((2,1,3))(x)
        predicted=rme(x)
        start_step=lookstart
        predicted=Lambda(lambda x:x[:,:,:],
                        output_shape =(n_relations,timestep_diff-start_step,relation_dim),name='target')(predicted)
        model = Model(inputs=[objects,relation_data,sender_relations_in,receiver_relations_in,propagation,known_relations,unknown_relations,relation_dummy],outputs=[predicted])
        adam = optimizers.Adam(lr=0.0001, decay=0.0)
        if relation_dim==1:
            los_func='binary_crossentropy'
        else:
            los_func='categorical_crossentropy'
        model.compile(optimizer=adam, loss=los_func,metrics=['accuracy'])#categorical_crossentropy
        self.Nets[n_objects]=model
        return model