import keras
import copy
from sklearn.metrics import mean_squared_error
from IPython.display import clear_output
from DatasetLoader import *
import matplotlib.pyplot as plt
import numpy as np
class Change_Noise_Callback(keras.callbacks.Callback):
    def __init__(self,generator,gauss_noise=0.20):
        self.gauss_noise=gauss_noise
        self.generator=generator
 
    def on_epoch_begin(self,epoch, logs={}):
        if epoch==250:
            self.generator.changeGauss(0)
        elif epoch>50 and epoch<250:
            self.generator.changeGauss(self.gauss_noise-(epoch-50)*self.gauss_noise/200.0)
        return
class Test_My_Metrics_Callback(keras.callbacks.Callback):
    def __init__(self,PN,n_of_dataset,n_of_rel,scaler,**kwargs):
        self.valTrajs=dict()
        self.val_origin_pos=dict()
        self.val_origin_vel=dict()
        self.val_receiver_relations=dict()
        self.val_sender_relations=dict()
        self.val_relation_info=dict()
        self.bestEpochOnPosError=dict()
        self.lowestPosError=dict()
        self.bestEpochOnVelError=dict()
        self.lowestVelError=dict()
        self.num_of_objects=dict()
        self.num_of_trajs=dict()
        self.n_of_rel=n_of_rel
        self.scaler=scaler
        self.n_of_dataset=n_of_dataset
        self.PN=PN
        for i in range(n_of_dataset):
            dataset = kwargs['dataset_'+str(i)]
            startIndex=dataset.val_traj_start
            endIndex=dataset.test_traj_start
            self.num_of_trajs[i]=endIndex-startIndex
            self.valTrajs[i]=dataset.data[dataset.indexes[startIndex:endIndex]].copy()
            self.val_relation_info[i]=dataset.r_i[dataset.indexes[startIndex:endIndex]].copy()
            self.num_of_objects[i]=dataset.n_of_obj+1
            self.num_of_relation=self.num_of_objects[i]*(self.num_of_objects[i]-1)

            self.val_origin_pos[i]=self.valTrajs[i][:,:100,:,2:4];
            self.val_origin_vel[i]=self.valTrajs[i][:,:100,:,4:6];

            self.bestEpochOnPosError[i]=-1
            self.lowestPosError[i]=10000
            self.bestEpochOnVelError[i]=-1
            self.lowestVelError[i]=10000
    def on_epoch_end(self,epoch, logs={}):
        for k in range(self.n_of_dataset): 
            num_of_traj=self.num_of_trajs[k]
            num_of_object=self.num_of_objects[k]
            Pn=self.PN.getModel(num_of_object,6,self.n_of_rel)
            num_of_relation=num_of_object*(num_of_object-1)
                        
            dataToModel= np.zeros([num_of_traj,100,num_of_object,6])
            dataToModel[:,0,:,:]= copy.deepcopy(self.valTrajs[k][:,0,:,:])
            dataToModel[:,:,0,:]= copy.deepcopy(self.valTrajs[k][:,:100,0,:])
            dataToModel[:,:,:,:2]= copy.deepcopy(self.valTrajs[k][:,:100,:,:2])
            dataToModel[:,0,1:,4:6]= 0
            
            # Normalization
            
            dataToModel = self.scaler.transform(dataToModel)
            
            val_receiver_relations = np.zeros((num_of_traj, num_of_object, num_of_relation), dtype=float);
            val_sender_relations   = np.zeros((num_of_traj, num_of_object, num_of_relation), dtype=float);
            val_relation_info = np.zeros((num_of_traj, num_of_relation, self.n_of_rel))
            propagation = np.zeros((num_of_traj, num_of_object,100), dtype=float);

            cnt = 0
            for m in range(num_of_object):
                for j in range(num_of_object):
                    if(m != j):
                        inzz=np.linalg.norm(dataToModel[:,0,m,2:4]-dataToModel[:,0,j,2:4],axis=1)<self.scaler.relation_threshold
                        val_receiver_relations[inzz, j, cnt] = 1.0
                        val_sender_relations[inzz, m, cnt]   = 1.0
                        
                        if self.n_of_rel>1:
                            val_relation_info[:,cnt,1:]=self.val_relation_info[k][:,0,m*num_of_object+j,:]
                            val_relation_info[np.sum(val_relation_info[:,cnt,1:self.n_of_rel],axis=1)==0,cnt,0]=1
                        else:
                            val_relation_info[:,cnt,:]=self.val_relation_info[k][:,0,m*num_of_object+j,:]
                        cnt += 1
            for i in range(1,100):
                velocities=Pn.predict({'objects': dataToModel[:,i-1,:,:],'sender_relations': val_sender_relations,'receiver_relations': val_receiver_relations,'relation_info': val_relation_info,'propagation':propagation})
                dataToModel[:,i,1:,2:4]=dataToModel[:,i-1,1:,2:4]
                dataToModel[:,i,1:,4:6]=velocities[:,:,:]
                dataToModel[:,i,1:,:]=PositionCalculateNext(dataToModel[:,i,1:,:],self.scaler)
                cnt = 0
                propagation = np.zeros((num_of_traj, num_of_object,100), dtype=float);
                val_receiver_relations = np.zeros((num_of_traj, num_of_object, num_of_relation), dtype=float);
                val_sender_relations   = np.zeros((num_of_traj, num_of_object, num_of_relation), dtype=float);
                for m in range(num_of_object):
                    for j in range(num_of_object):
                        if(m != j):
                            inzz=np.linalg.norm(dataToModel[:,i,m,2:4]-dataToModel[:,i,j,2:4],axis=1)<self.scaler.relation_threshold
                            val_receiver_relations[inzz, j, cnt] = 1.0
                            val_sender_relations[inzz, m, cnt]   = 1.0
                            cnt += 1
            val_pred=self.scaler.inv_transform(dataToModel)
            pos_error=mean_squared_error(val_pred[:,:,1:,2:4].reshape(-1,2),self.val_origin_pos[k][:,:,1:,:].reshape(-1,2))
            vel_error=mean_squared_error(val_pred[:,:,1:,4:6].reshape(-1,2),self.val_origin_vel[k][:,:,1:,:].reshape(-1,2))
            print 'val'+str(self.num_of_objects[k])+'_pos_loss' 
            logs['val'+str(self.num_of_objects[k])+'_pos_loss'] = pos_error
            logs['val'+str(self.num_of_objects[k])+'_vel_loss'] = vel_error
            if pos_error<self.lowestPosError[k]:
                self.lowestPosError[k]=pos_error
                self.bestEpochOnPosError[k]=epoch
            if vel_error<self.lowestVelError[k]:
                self.lowestVelError[k]=vel_error
                self.bestEpochOnVelError[k]=epoch        
        return 
class PlotLosses(keras.callbacks.Callback):
    def __init__(self,CSV_PATH,n_of_dataset,num_of_objects):
        self.CSV_PATH=CSV_PATH
        self.n_of_dataset=n_of_dataset
        self.num_of_objects=num_of_objects
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = list()
        self.losses = list()
        self.val_losses = list()
        
        self.pos_losses=list()
        self.vel_losses=list()
        
        for i in range(self.n_of_dataset):
            self.pos_losses.append(list())
            self.vel_losses.append(list())
        with open(self.CSV_PATH,'w') as f:  
            f.write('loss,val_loss')
            for i in range(self.n_of_dataset):
                f.write(',val'+str(self.num_of_objects[i])+'_pos_loss')
            for i in range(self.n_of_dataset):
                f.write(',val'+str(self.num_of_objects[i])+'_vel_loss')
            f.write('\n')
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        f = open(self.CSV_PATH,'a') 
        f.write(str(logs.get('loss'))+','+str(logs.get('val_loss')))
        
        self.i += 1
        clear_output(wait=True)
        self.fig = plt.figure(figsize=(20,6))
        plt.subplot(1,3,1)
        plt.plot(self.x, np.log(self.losses), label="loss")
        plt.plot(self.x, np.log(self.val_losses), label="val_loss")

        plt.xlim([-0.1, self.i+10])   
        plt.legend(loc=1)
        plt.subplot(1,3,2)
        for z in range(self.n_of_dataset):
            print 'val'+str(self.num_of_objects[z])+'_pos_loss'
            self.pos_losses[z].append(logs.get('val'+str(self.num_of_objects[z])+'_pos_loss'))
            plt.plot(self.x, np.log(self.pos_losses[z]), label='val'+str(self.num_of_objects[z])+'_pos_loss')
            print str(self.num_of_objects[z])+' objects poss loss:',logs.get('val'+str(self.num_of_objects[z])+'_pos_loss')
            f.write(',' + str(logs.get('val'+str(self.num_of_objects[z])+'_pos_loss')))
        plt.xlim([-0.1, self.i+10])  
        plt.legend(loc=1)
        plt.subplot(1,3,3)
        for z in range(self.n_of_dataset):
            self.vel_losses[z].append(logs.get('val'+str(self.num_of_objects[z])+'_vel_loss'))
            plt.plot(self.x, np.log(self.vel_losses[z]), label='val'+str(self.num_of_objects[z])+'_vel_loss')
            print str(self.num_of_objects[z])+' objects vel loss:',logs.get('val'+str(self.num_of_objects[z])+'_vel_loss')
            f.write(',' + str(logs.get('val'+str(self.num_of_objects[z])+'_vel_loss')))
        f.write('\n')
        f.close()
        plt.xlim([-0.1, self.i+10])    
        plt.legend(loc=1)
        plt.show();