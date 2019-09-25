import keras
import random
import numpy as np
import copy
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,n_objects,n_of_rel_type,number_of_frames,num_of_traj,dataset,dataRelation,relation_threshold,isTrain=True,batch_size=100,shuffle=True):
        'Initialization'
        self.n_objects = n_objects
        self.relation_threshold=relation_threshold
        self.batch_size = batch_size
        self.n_of_features = 6
        self.num_of_traj=num_of_traj
        self.n_of_rel_type=n_of_rel_type
        self.n_relations  = n_objects * (n_objects - 1) # number of edges in fully connected graph
        self.shuffle = shuffle
        self.number_of_frames=number_of_frames
        self.currEpoch=0
        self.data=dataset
        self.dataRelation=dataRelation
        self.indexes = 1 + np.arange(self.number_of_frames-1)
        for i in range(1,self.num_of_traj):
            self.indexes=np.concatenate([self.indexes,(i*self.number_of_frames +1 + np.arange(self.number_of_frames-1))])
        self.std_dev_pos=0.05*np.std(self.data[:,:,2:4])
        self.std_dev_vel=0.05*np.std(self.data[:,:,4:6])
        self.add_gaus =0.20
        self.propagation = np.zeros((self.batch_size, self.n_objects,100), dtype=float);
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num_of_traj*(self.number_of_frames-1) / self.batch_size))/2

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        
        data_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        # Generate data
        X, y = self.__data_generation(data_indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.currEpoch=self.currEpoch+1
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def changeGauss(self,addGauss):
        self.add_gaus=addGauss
    def __data_generation(self, data_indexes):
        """

        """
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        temp_data=self.data[data_indexes,:,:].copy()
        data_x_vel_indexes=[idx - 1 for idx in data_indexes]
        # In here we know velocity of objects on previous timesteps, but not this one. So velocity input it robot hands velocity and last velocities of objects.
        temp_data[:,1:,4:6]=self.data[data_x_vel_indexes,1:,4:6]
        if self.add_gaus>0:
            for i in range(self.batch_size):
                for j in range(self.n_objects):
                    if (random.random()<self.add_gaus):
                        temp_data[i,j,2]=temp_data[i,j,2]+np.random.normal(0, self.std_dev_pos)
                    if (random.random()<self.add_gaus):
                        temp_data[i,j,3]=temp_data[i,j,3]+np.random.normal(0, self.std_dev_pos)
                    if (random.random()<self.add_gaus):
                        temp_data[i,j,4]=temp_data[i,j,4]+np.random.normal(0, self.std_dev_vel)
                    if (random.random()<self.add_gaus):
                        temp_data[i,j,5]=temp_data[i,j,5]+np.random.normal(0, self.std_dev_vel)
                        
        cnt = 0
        x_receiver_relations = np.zeros((self.batch_size, self.n_objects, self.n_relations), dtype=float);
        x_sender_relations   = np.zeros((self.batch_size, self.n_objects, self.n_relations), dtype=float);

        x_relation_info = np.zeros((self.batch_size, self.n_relations, self.n_of_rel_type))
        for i in range(self.n_objects):
            for j in range(self.n_objects):
                if(i != j):
                    
                    inzz=np.linalg.norm(temp_data[:,i,2:4]-temp_data[:,j,2:4],axis=1)< self.relation_threshold
                    if self.n_of_rel_type==1:
                        x_relation_info[:,cnt,:] = self.dataRelation[data_indexes,i*self.n_objects+j,:]
                    else:
                        x_relation_info[:,cnt,1:] = self.dataRelation[data_indexes,i*self.n_objects+j,:]
                        x_relation_info[np.sum(x_relation_info[:,cnt,1:self.n_of_rel_type],axis=1)==0,cnt,0]=1
                    x_receiver_relations[inzz, j, cnt] = 1.0
                    x_sender_relations[inzz, i, cnt]   = 1.0
                    cnt += 1
        x_object = temp_data 
        y = self.data[data_indexes,1:,4:6]
        return {'objects': x_object,'sender_relations': x_sender_relations,\
                'receiver_relations': x_receiver_relations,'relation_info': x_relation_info,\
                'propagation': self.propagation},{'target': y}
    
class RelationDataGeneratorMany(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,n_objects,n_of_rel_type,number_of_frames,num_of_traj,dataset,dataRelation,relation_threshold,isTrain=True,batch_size=100,timestep_diff=5,lookout=50,shuffle=True):
        'Initialization'
        self.n_objects = n_objects
        self.relation_threshold=relation_threshold
        self.n_of_rel_type=n_of_rel_type
        self.batch_size = batch_size
        self.n_of_features = 6
        self.num_of_traj=num_of_traj
        self.n_relations  = n_objects * (n_objects - 1) # number of edges in fully connected graph
        self.shuffle = shuffle
        self.number_of_frames=number_of_frames
        self.currEpoch=0
        self.data=dataset
        self.relationalData=np.zeros((number_of_frames*num_of_traj,self.n_relations,8))
        cnt=0
        for i in range(self.n_objects):
            for j in range(self.n_objects):
                if(i != j):
                    self.relationalData[:,cnt,:4]=self.data[:,i,2:6]-self.data[:,j,2:6]
                    self.relationalData[:,cnt,4]=self.data[:,i,0]
                    self.relationalData[:,cnt,5]=self.data[:,j,0]
                    self.relationalData[:,cnt,6]=self.data[:,i,1]
                    self.relationalData[:,cnt,7]=self.data[:,j,1]
                    cnt=cnt+1
        self.dataRelation=dataRelation
        self.lookout=lookout
        self.timestep_diff=timestep_diff
        self.indexes = timestep_diff+ 1 + np.arange(self.number_of_frames-timestep_diff-1) 
        for i in range(1,self.num_of_traj):
            self.indexes=np.concatenate([self.indexes,(i*self.number_of_frames +timestep_diff+ 1 + np.arange(self.number_of_frames-timestep_diff-1))])
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num_of_traj*(self.number_of_frames-self.timestep_diff) / self.batch_size))/2

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        
        data_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(data_indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.currEpoch=self.currEpoch+1
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def extract_traj_data(self,n_of_frame,traj_index):
        data_indexes = self.indexes[traj_index*(self.number_of_frames-self.timestep_diff-1):traj_index*(self.number_of_frames-self.timestep_diff-1)+n_of_frame]
        X, y = self.__data_generation(data_indexes,n_of_frame)
        return X,y
        
    def __data_generation(self, data_indexes,batch_size=0):
        """

        """
        relation_threshold=self.relation_threshold
        if batch_size==0:
            batch_size=self.batch_size
        self.propagation = np.zeros((batch_size,self.timestep_diff, self.n_objects,100), dtype=float);
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        data_indexes2=[idx - self.timestep_diff for idx in data_indexes]
        
        temp_data=np.zeros((batch_size,self.timestep_diff,self.n_objects,self.n_of_features))
        temp_data_rel=np.zeros((batch_size,self.timestep_diff,self.n_relations,8))

        for i in range(self.timestep_diff):
            data_indexes3 = [idx +i for idx in data_indexes2]
            temp_data[:,i,:,:]=self.data[data_indexes3,:,:].copy()            
            temp_data_rel[:,i,:,:]=self.relationalData[data_indexes3,:,:].copy()
            
        self.x_receiver_relations = np.zeros((batch_size,self.timestep_diff, self.n_objects, self.n_relations), dtype=float);
        self.x_sender_relations   = np.zeros((batch_size,self.timestep_diff, self.n_objects, self.n_relations), dtype=float);
        knownObjects=np.zeros((batch_size,1,self.n_objects))
        knownObjects[:,0,0]=1
        knownRelations=np.zeros((batch_size,1,self.n_relations))
        self.relation_info = np.zeros((batch_size, self.n_relations, self.n_of_rel_type))
        cnt = 0
        for i in range(self.n_objects):
            for j in range(self.n_objects):
                if(i != j):                        
                    if self.n_of_rel_type==1:
                        self.relation_info[:,cnt,:] = self.dataRelation[data_indexes,i*self.n_objects+j,:]
                    else:
                        self.relation_info[:,cnt,1:] = self.dataRelation[data_indexes,i*self.n_objects+j,:]
                        self.relation_info[np.sum(self.relation_info[:,cnt,1:],axis=1)==0,cnt,0]=1
                    cnt +=1
        cnt=0
        for timestep in range(self.timestep_diff):
            cnt = 0
            for i in range(self.n_objects):
                for j in range(self.n_objects):
                    if(i != j):                        
                        inzz=np.linalg.norm(temp_data[:,timestep,i,2:4]-temp_data[:,timestep,j,2:4],axis=1)< self.relation_threshold
                        self.x_receiver_relations[inzz,timestep, j, cnt] = 1.0
                        self.x_sender_relations[inzz,timestep, i, cnt]   = 1.0
                        cnt += 1
        relation_dummy=self.relation_info
        self.relation_info2=self.relation_info.reshape((batch_size, self.n_relations,1, self.n_of_rel_type))
        self.relation_info2=np.tile(self.relation_info2,[1,1,(self.timestep_diff-self.lookout),1])
        
        unknownRelations=1-knownRelations
        x_object = temp_data        
        
        x_diff_object = temp_data_rel
        
        return {'objects': x_object,'diff_objects': x_diff_object,'sender_relations': self.x_sender_relations,\
                'receiver_relations': self.x_receiver_relations,\
                'propagation': self.propagation,'known_relations': knownRelations,'unknown_relations': unknownRelations,'relation_dummy':relation_dummy},{'target': self.relation_info2}    