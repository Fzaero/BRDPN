from tqdm import tqdm
import numpy as np
import os.path

class MyScaler:
    """
    Scale parameters for data.
    """
    def scale(self,data,position_threshold):
        
        multiplier = np.zeros((6))
        multiplier[0]=6 # Put max radius at about 1.
        multiplier[1]=1 # Categorical value, stays same.
        pos_vals= data[:,:,2:4]
        multiplier[2:4]=1/np.std(pos_vals[pos_vals<position_threshold]) 
        self.relation_threshold=multiplier[2]*position_threshold
        vel_vals = data[:,0,4:6]
        multiplier[4:6]=1/np.std(vel_vals[np.abs(vel_vals)>0.0001])
        self.multiplier = multiplier #
        self.inv_multiplier=np.ones((6,))/self.multiplier
    def transform(self,data):
        data_transformed=data*self.multiplier
        return data_transformed
    def inv_transform(self,data):
        data_transformed=data*self.inv_multiplier
        return data_transformed
class MyDataset:
    def __init__(self,**kwargs):
        self.PATH= kwargs['PATH']
        self.n_of_obj = kwargs['n_of_obj']
        self.n_of_rel = self.n_of_obj*(self.n_of_obj+1)

        self.f_size = 6
        self.num_of_rel_type = kwargs['n_of_rel_type'] # + 2
        self.fr_size = kwargs['fr_size']
        
        self.n_of_scene = kwargs['n_of_scene']
        self.n_of_exp = kwargs['n_of_exp']
        if 'scaler' in kwargs.keys():
            self.scaler= kwargs['scaler']
            self.scalerTrain=False
        else:
            self.scaler= MyScaler()
            self.scalerTrain=True
        self.n_of_traj = self.n_of_scene * self.n_of_exp
        self.indexes=np.arange(self.n_of_traj)
        self.data=np.zeros([self.n_of_traj,self.fr_size,self.n_of_obj+1,self.f_size])

        self.r_i=np.zeros([self.n_of_traj,self.fr_size,(self.n_of_obj+1)**2,self.num_of_rel_type])
        
        startIndex=3
            
        def getScene(scene_ind):
            temp_txt=np.loadtxt(self.PATH + str(scene_ind)+'/info.txt',skiprows=1)
            self.data[scene_ind*self.n_of_exp:(scene_ind+1)*self.n_of_exp,:,1:,0]=temp_txt[0:-1:3]
            self.data[scene_ind*self.n_of_exp:(scene_ind+1)*self.n_of_exp,:,0,0]=0.1
            self.data[scene_ind*self.n_of_exp:(scene_ind+1)*self.n_of_exp,:,0,1]=1

            if os.path.isfile(self.PATH + str(scene_ind)+'/info2.txt'):
                temp_txt2=np.loadtxt(self.PATH + str(scene_ind)+'/info2.txt',skiprows=1)
                for obj_ind in range(self.n_of_obj-3):
                    if temp_txt2[obj_ind,0]>1:
                        obj1=obj_ind+startIndex+1
                        obj2=int(temp_txt2[obj_ind,1])
                        self.r_i[scene_ind*self.n_of_exp:(scene_ind+1)*self.n_of_exp,:,obj1*(self.n_of_obj+1)+obj2,int(temp_txt2[obj_ind,0]-2)]=1
                        self.r_i[scene_ind*self.n_of_exp:(scene_ind+1)*self.n_of_exp,:,obj2*(self.n_of_obj+1)+obj1,int(temp_txt2[obj_ind,0]-2)]=1
            for exp_ind in range(self.n_of_exp):
                for obj_ind in range(self.n_of_obj):
                    temp_txt=np.loadtxt(self.PATH + str(scene_ind)+'/'+str(exp_ind+1)+'/Cylinder'+str(obj_ind+1)+'.txt')
                    self.data[scene_ind*self.n_of_exp+exp_ind,:,obj_ind+1,2:4]=temp_txt[:,:2]

                temp_txt=np.loadtxt(self.PATH + str(scene_ind)+'/'+str(exp_ind+1)+'/hand.txt')
                self.data[scene_ind*self.n_of_exp+exp_ind,:,0,2:4]= temp_txt[:,:2]
        map(getScene,tqdm(range(self.n_of_scene)))
        self.data[:,:self.fr_size-1,:,4:]=self.data[:,1:self.fr_size,:,2:4]-self.data[:,:self.fr_size-1,:,2:4] 
    def divideDataset(self,tr_size,val_size):
        tr_set_size=int(self.n_of_traj*tr_size)
        val_set_size=int(self.n_of_traj*val_size)
        test_set_size=self.n_of_traj-tr_set_size-val_set_size
        self.val_traj_start=tr_set_size 
        self.test_traj_start=tr_set_size+val_set_size
        
        ## Creating Data Split
        self.data_tr   =np.zeros([tr_set_size*self.fr_size,self.n_of_obj+1,self.f_size])
        self.data_val  =np.zeros([val_set_size*self.fr_size  ,self.n_of_obj+1,self.f_size])
        self.data_test =np.zeros([test_set_size*self.fr_size ,self.n_of_obj+1,self.f_size])

        ## Creating Relation Split
        self.r_i_tr   =np.zeros([tr_set_size*self.fr_size  ,(self.n_of_obj+1)**2,self.num_of_rel_type])
        self.r_i_val  =np.zeros([val_set_size*self.fr_size ,(self.n_of_obj+1)**2,self.num_of_rel_type])
        self.r_i_test =np.zeros([test_set_size*self.fr_size,(self.n_of_obj+1)**2,self.num_of_rel_type])

        self.r_i_tr[:,:]=self.r_i[:tr_set_size].reshape(-1,(self.n_of_obj+1)**2,self.num_of_rel_type)
        self.r_i_val[:,:]=self.r_i[tr_set_size:(tr_set_size+val_set_size)].reshape(-1,(self.n_of_obj+1)**2,self.num_of_rel_type)
        self.r_i_test[:,:]=self.r_i[self.n_of_traj-test_set_size:].reshape(-1,(self.n_of_obj+1)**2,self.num_of_rel_type)

        self.data_tr[:,:,:]=self.data[:tr_set_size].reshape(-1,(self.n_of_obj+1),self.f_size)
        self.data_val[:,:,:] =self.data[tr_set_size:(tr_set_size+val_set_size)].reshape(-1,(self.n_of_obj+1),self.f_size)
        self.data_test[:,:,:]=self.data[(tr_set_size+val_set_size):].reshape(-1,(self.n_of_obj+1),self.f_size)
        
        if self.scalerTrain:
            self.scaler.scale(self.data_tr,0.35)
        self.data_tr=self.scaler.transform(self.data_tr)
        self.data_val=self.scaler.transform(self.data_val)
        self.data_test=self.scaler.transform(self.data_test)
        
def PositionCalculateNext(data,scaler):
        data_shape=data.shape
        data_inv_transformed = scaler.inv_transform(data)
        data_inv_transformed[:,:,2:4]=data_inv_transformed[:,:,2:4]+data_inv_transformed[:,:,4:]
        return scaler.transform(data_inv_transformed)
class MyDataset2:
    """
    Usage Example:
    """
    def __init__(self,**kwargs):
                
        self.PATH= kwargs['PATH']
        self.n_of_obj = kwargs['n_of_obj']
        self.n_of_rel = self.n_of_obj*(self.n_of_obj+1)

        self.f_size = 6
        self.num_of_rel_type = kwargs['n_of_rel_type'] # + 2
        self.fr_size = kwargs['fr_size']
        
        self.n_of_scene = kwargs['n_of_scene']
        self.n_of_exp = kwargs['n_of_exp']
        self.n_of_traj = self.n_of_scene * self.n_of_exp
        self.n_of_groups = kwargs['n_of_groups']
        self.data=np.zeros([self.n_of_traj,self.fr_size,self.n_of_obj+1,self.f_size])
        self.scaler= kwargs['scaler']
        self.r_i=np.zeros([self.n_of_traj,self.fr_size,(self.n_of_obj+1)**2,self.num_of_rel_type])            
        def getScene(scene_ind):
            temp_txt=np.loadtxt(self.PATH + str(scene_ind)+'/info.txt',skiprows=1)
            self.data[scene_ind*self.n_of_exp:(scene_ind+1)*self.n_of_exp,:,1:,0]=temp_txt[0:-1:3]
            self.data[scene_ind*self.n_of_exp:(scene_ind+1)*self.n_of_exp,:,0,0]=0.1
            self.data[scene_ind*self.n_of_exp:(scene_ind+1)*self.n_of_exp,:,0,1]=1
            if os.path.isfile(self.PATH + str(scene_ind)+'/info2.txt'):
                temp_txt2=np.loadtxt(self.PATH + str(scene_ind)+'/info2.txt',skiprows=1)
                for rel_ind in range(self.n_of_groups):
                    if temp_txt2[rel_ind,0]>1:
                        obj1=int(temp_txt2[rel_ind,1])
                        obj2=int(temp_txt2[rel_ind,2])
                        self.r_i[scene_ind*self.n_of_exp:(scene_ind+1)*self.n_of_exp,:,obj1*(self.n_of_obj+1)+obj2,int(temp_txt2[rel_ind,0]-2)]=1
                        self.r_i[scene_ind*self.n_of_exp:(scene_ind+1)*self.n_of_exp,:,obj2*(self.n_of_obj+1)+obj1,int(temp_txt2[rel_ind,0]-2)]=1                        

            for exp_ind in range(self.n_of_exp):
                for obj_ind in range(self.n_of_obj):
                    temp_txt=np.loadtxt(self.PATH + str(scene_ind)+'/'+str(exp_ind+1)+'/Cylinder'+str(obj_ind+1)+'.txt')
                    self.data[scene_ind*self.n_of_exp+exp_ind,:,obj_ind+1,2:4]=temp_txt[:,:2]

                temp_txt=np.loadtxt(self.PATH + str(scene_ind)+'/'+str(exp_ind+1)+'/hand.txt')
                self.data[scene_ind*self.n_of_exp+exp_ind,:,0,2:4]= temp_txt[:,:2]
        map(getScene,tqdm(range(self.n_of_scene)))
        self.data[:,:self.fr_size-1,:,4:]=self.data[:,1:self.fr_size,:,2:4]-self.data[:,:self.fr_size-1,:,2:4] 
    def divideDataset(self,tr_size,val_size):
        tr_set_size=int(self.n_of_traj*tr_size)
        val_set_size=int(self.n_of_traj*val_size)
        test_set_size=self.n_of_traj-tr_set_size-val_set_size
        self.val_traj_start=tr_set_size 
        self.test_traj_start=tr_set_size+val_set_size
        
        ## Creating Data Split
        self.data_tr   =np.zeros([tr_set_size*self.fr_size,self.n_of_obj+1,self.f_size])
        self.data_val  =np.zeros([val_set_size*self.fr_size  ,self.n_of_obj+1,self.f_size])
        self.data_test =np.zeros([test_set_size*self.fr_size ,self.n_of_obj+1,self.f_size])

        ## Creating Relation Split
        self.r_i_tr   =np.zeros([tr_set_size*self.fr_size  ,(self.n_of_obj+1)**2,self.num_of_rel_type])
        self.r_i_val  =np.zeros([val_set_size*self.fr_size ,(self.n_of_obj+1)**2,self.num_of_rel_type])
        self.r_i_test =np.zeros([test_set_size*self.fr_size,(self.n_of_obj+1)**2,self.num_of_rel_type])

        self.r_i_tr[:,:]=self.r_i[:tr_set_size].reshape(-1,(self.n_of_obj+1)**2,self.num_of_rel_type)
        self.r_i_val[:,:]=self.r_i[tr_set_size:(tr_set_size+val_set_size)].reshape(-1,(self.n_of_obj+1)**2,self.num_of_rel_type)
        self.r_i_test[:,:]=self.r_i[self.n_of_traj-test_set_size:].reshape(-1,(self.n_of_obj+1)**2,self.num_of_rel_type)

        self.data_tr[:,:,:]=self.data[:tr_set_size].reshape(-1,(self.n_of_obj+1),self.f_size)
        self.data_val[:,:,:] =self.data[tr_set_size:(tr_set_size+val_set_size)].reshape(-1,(self.n_of_obj+1),self.f_size)
        self.data_test[:,:,:]=self.data[(tr_set_size+val_set_size):].reshape(-1,(self.n_of_obj+1),self.f_size)
        
        self.data_tr=self.scaler.transform(self.data_tr)
        self.data_val=self.scaler.transform(self.data_val)
        self.data_test=self.scaler.transform(self.data_test)