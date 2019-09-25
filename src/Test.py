import copy
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
from DatasetLoader import *
from tqdm import tqdm
def Test(dataset,Ins,frame_len,relation_threshold):
    n_objects = dataset.n_of_obj+1
    num_of_rel_type=dataset.num_of_rel_type
    if num_of_rel_type>1:
        num_of_rel_type=num_of_rel_type+1
    n_relations=n_objects*(n_objects-1)
    In=Ins.getModel(n_objects)
    GroundData=dataset.data[dataset.test_traj_start:]
    n_of_traj=GroundData.shape[0]
    
    xy_origin_pos=copy.deepcopy(GroundData[:,:frame_len,:,2:4]);
    xy_origin_vel=copy.deepcopy(GroundData[:,:frame_len,:,4:6]);
    
    dataToModel= np.zeros([n_of_traj,frame_len,n_objects,6])
    dataToModel[:,0,:,:]= copy.deepcopy(GroundData[:,0,:,:])
    dataToModel[:,:,0,:]= copy.deepcopy(GroundData[:,:frame_len,0,:])
    dataToModel[:,:,:,:2]= copy.deepcopy(GroundData[:,:frame_len,:,:2])
    dataToModel[:,0,1:,4:6]= 0
    r= dataToModel[:,0,:,0]
    dataToModel=dataset.scaler.transform(dataToModel)
    val_receiver_relations = np.zeros((n_of_traj, n_objects, n_relations), dtype=float);
    val_sender_relations   = np.zeros((n_of_traj, n_objects, n_relations), dtype=float);
    val_relation_info = np.zeros((n_of_traj, n_relations, num_of_rel_type))
    propagation = np.zeros((n_of_traj, n_objects, 100))
    cnt = 0
    for m in range(n_objects):
        for j in range(n_objects):
            if(m != j):
                inzz=np.linalg.norm(dataToModel[:,0,m,2:4]-dataToModel[:,0,j,2:4],axis=1)<relation_threshold
                val_receiver_relations[inzz, j, cnt] = 1.0
                val_sender_relations[inzz, m, cnt]   = 1.0
                if num_of_rel_type>1:
                    val_relation_info[:,cnt,1:]=dataset.r_i[dataset.test_traj_start:,0,m*n_objects+j,:]
                    val_relation_info[np.sum(val_relation_info[:,cnt,1:],axis=1)==0,cnt,0]=1
                else:
                    val_relation_info[:,cnt,:]=dataset.r_i[dataset.test_traj_start:,0,m*n_objects+j,:]
                cnt += 1
    edges=val_relation_info
    for i in range(1,frame_len):
        velocities=In.predict({'objects': dataToModel[:,i-1,:,:],'sender_relations': val_sender_relations,'receiver_relations': val_receiver_relations,'relation_info': val_relation_info,'propagation':propagation})
        dataToModel[:,i,1:,2:4]=dataToModel[:,i-1,1:,2:4]
        dataToModel[:,i,1:,4:6]=velocities[:,:,:]; 
        dataToModel[:,i,1:,:]=PositionCalculateNext(dataToModel[:,i,1:,:],dataset.scaler)
        val_receiver_relations = np.zeros((n_of_traj, n_objects, n_relations), dtype=float);
        val_sender_relations   = np.zeros((n_of_traj, n_objects, n_relations), dtype=float);
        cnt = 0
        for m in range(n_objects):
            for j in range(n_objects):
                if(m != j):
                    inzz=np.linalg.norm(dataToModel[:,i,m,2:4]-dataToModel[:,i,j,2:4],axis=1)<relation_threshold
                    val_receiver_relations[inzz, j, cnt] = 1.0
                    val_sender_relations[inzz, m, cnt]   = 1.0                    
                    cnt += 1
    pred_xy = dataset.scaler.inv_transform(dataToModel)
    
    xy_calculated_pos=pred_xy[:,:,:,2:4]
    xy_calculated_vel=pred_xy[:,:,:,4:6]
    print 'mse-pos:',np.log(mean_squared_error(xy_calculated_pos[:,:,1:,:].reshape(-1,2),xy_origin_pos[:,:,1:,:].reshape(-1,2)))
    print 'mse-vel:',np.log(mean_squared_error(xy_calculated_vel[:,:,1:,:].reshape(-1,2),xy_origin_vel[:,:,1:,:].reshape(-1,2)))
    return xy_origin_pos,xy_calculated_pos,r,edges

def make_video_Fixed(xy,r,edge,filename):
    os.system("rm -rf pics/*");
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                  comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    fig = plt.figure(figsize=(8,8))
    ax = plt.gca()
    ax.cla() 
    ax.set_xlim((-1.00, 1.00))
    ax.set_ylim((-1.00, 1.00))
    fig_num=len(xy);
    color=['blue','red','green','yellow','purple','cyan','orange','black','pink'];
    with writer.saving(fig, filename, len(xy)):
        for i in range(len(xy)):
            cnt=0
            for j in range(len(xy[0])):
                circle = plt.Circle((xy[i,j,1],xy[i,j,0]), r[j]/2,color=color[j%9], fill=True)
                ax.add_artist(circle)
                for k in range(len(xy[0])):
                    if j!=k:
                        if edge[cnt,0]==1:
                            plt.plot([xy[i,j,1],xy[i,k,1]], [xy[i,j,0],xy[i,k,0]], color='black', linestyle='solid', lw=5)
                        cnt=cnt+1
            plt.axis('off')
            writer.grab_frame();
            ax.cla()
            ax.set_xlim((-1.0, 1.0))
            ax.set_ylim((-1.0, 1.0))
def make_video_Mixed(xy,r,edge,filename):
    os.system("rm -rf pics/*");
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                  comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    fig = plt.figure(figsize=(8,8))
    ax = plt.gca()
    ax.cla() 
    ax.set_xlim((-1.00, 1.00))
    ax.set_ylim((-1.00, 1.00))
    fig_num=len(xy);
    color=['blue','red','green','yellow','purple','cyan','orange','black','pink'];
    with writer.saving(fig, filename, len(xy)):
        for i in range(len(xy)):
            cnt=0
            for j in range(len(xy[0])):
                circle = plt.Circle((xy[i,j,1],xy[i,j,0]), r[j]/2,color=color[j%9], fill=True)
                ax.add_artist(circle)
                for k in range(len(xy[0])):
                    if j!=k:
                        if edge[cnt,1]==1:
                            plt.plot([xy[i,j,1],xy[i,k,1]], [xy[i,j,0],xy[i,k,0]], color='black', linestyle='solid', lw=5)
                        elif edge[cnt,2]==1:
                            plt.plot([xy[i,j,1],xy[i,k,1]], [xy[i,j,0],xy[i,k,0]], color='gray', linestyle='solid', lw=5)
                        elif edge[cnt,3]==1:
                            plt.plot([xy[i,j,1],xy[i,k,1]], [xy[i,j,0],xy[i,k,0]], color='purple', linestyle='solid', lw=5)

                        cnt=cnt+1
            plt.axis('off')
            writer.grab_frame();
            ax.cla()
            ax.set_xlim((-1.0, 1.0))
            ax.set_ylim((-1.0, 1.0))

def Test_TPN(dataset,Fn,frame_len,relation_threshold):
    n_objects = dataset.n_of_obj+1
    n_of_rel=dataset.num_of_rel_type
    if n_of_rel>1:
        n_of_rel=n_of_rel+1
    n_relations=n_objects*(n_objects-1)
    GroundData=copy.deepcopy(dataset.data[dataset.test_traj_start:])
    n_of_traj=GroundData.shape[0]

    GroundData2=np.zeros((n_of_traj,frame_len,n_objects,6))
    GroundData2[:,:,:,:]=GroundData[:,:frame_len,:,:]
    xy_origin_pos=copy.deepcopy(GroundData[:,:frame_len,:,2:4]);
    r= GroundData[:,0,:,0]
    relationalData=np.zeros((n_of_traj,frame_len,n_relations,8))
    edges=np.zeros((n_of_traj,n_relations,n_of_rel))
    cnt=0
    GroundData2=dataset.scaler.transform(GroundData2)
    for i in range(n_objects):
        for j in range(n_objects):
            if(i != j):
                relationalData[:,:,cnt,:4]=GroundData2[:,:,i,2:6]-GroundData2[:,:,j,2:6]
                relationalData[:,:,cnt,4]=GroundData2[:,:,i,0]
                relationalData[:,:,cnt,5]=GroundData2[:,:,j,0]
                relationalData[:,:,cnt,6]=GroundData2[:,:,i,1]
                relationalData[:,:,cnt,7]=GroundData2[:,:,j,1]
                if n_of_rel>1:
                    edges[:,cnt,1:]=dataset.r_i[dataset.test_traj_start:,0,i*n_objects+j,:]
                    edges[np.sum(edges[:,cnt,1:],axis=1)==0,cnt,0]=1
                else:
                    edges[:,cnt,:]=dataset.r_i[dataset.test_traj_start:,0,i*n_objects+j,:]
                cnt=cnt+1
                
    val_receiver_relations = np.zeros((n_of_traj,frame_len ,n_objects, n_relations), dtype=float);
    val_sender_relations   = np.zeros((n_of_traj,frame_len ,n_objects, n_relations), dtype=float);
    knownObjects=np.zeros((n_of_traj,frame_len,1,n_objects))
    knownObjects[:,:,0,0]=1
    knownRelations=np.zeros((n_of_traj,frame_len,1,n_relations))
    unknownRelations=np.zeros((n_of_traj,frame_len,1,n_relations))

    for timestep in range(frame_len):
        cnt = 0
        for m in range(n_objects):
            for j in range(n_objects):
                if(m != j):
                    inzz=np.linalg.norm(GroundData2[:,timestep,m,2:4]-GroundData2[:,timestep,j,2:4],axis=1)<relation_threshold
                    val_receiver_relations[inzz,timestep, j, cnt] = 1.0
                    val_sender_relations[inzz,timestep, m, cnt]   = 1.0
                    cnt += 1
    relation_dummy=np.zeros((n_of_traj,n_relations, n_of_rel))
    relation_dummy[:,:,0]=1
    propagation = np.zeros((n_of_traj,frame_len, n_objects,100), dtype=float);

    FN_input=dict()
    FN_input['objects'] = GroundData2
    FN_input['diff_objects'] = relationalData
    FN_input['sender_relations'] = val_sender_relations
    FN_input['receiver_relations'] = val_receiver_relations
    FN_input['propagation']=propagation
    FN_input['known_relations']=knownRelations[:,0,:,:]
    FN_input['unknown_relations']=unknownRelations[:,0,:,:]
    FN_input['relation_dummy']=relation_dummy
    prediction=Fn.predict(FN_input)
        
    return xy_origin_pos,edges,r,prediction

def Test_PN_relation(dataset,Pn,frame_len,relation_threshold):
    n_objects = dataset.n_of_obj+1
    n_of_rel=dataset.num_of_rel_type
    if n_of_rel>1:
        n_of_rel=n_of_rel+1
    n_relations=n_objects*(n_objects-1)
    GroundData=copy.deepcopy(dataset.data[dataset.test_traj_start:])
    n_of_traj=GroundData.shape[0]
    GroundData2=np.zeros((n_of_traj,frame_len,n_objects,6))
    GroundData2[:,:,:,:]=GroundData[:,:frame_len,:,:]
    xy_origin_pos=copy.deepcopy(GroundData[:,:frame_len,:,2:4]);
    r= GroundData[:,0,:,0]
    edges=np.zeros((n_of_traj,n_relations,n_of_rel))
    cnt=0
    GroundData2=dataset.scaler.transform(GroundData2)
    for i in range(n_objects):
        for j in range(n_objects):
            if(i != j):
                if n_of_rel>1:
                    edges[:,cnt,1:]=dataset.r_i[dataset.test_traj_start:,0,i*n_objects+j,:]
                    edges[np.sum(edges[:,cnt,1:],axis=1)==0,cnt,0]=1
                else:
                    edges[:,cnt,:]=dataset.r_i[dataset.test_traj_start:,0,i*n_objects+j,:]
                cnt=cnt+1
    val_receiver_relations = np.zeros((n_of_traj,frame_len ,n_objects, n_relations), dtype=float);
    val_sender_relations   = np.zeros((n_of_traj,frame_len ,n_objects, n_relations), dtype=float);
    for timestep in range(frame_len):
        cnt = 0
        for m in range(n_objects):
            for j in range(n_objects):
                if(m != j):
                    inzz=np.linalg.norm(GroundData2[:,timestep,m,2:4]-GroundData2[:,timestep,j,2:4],axis=1)<relation_threshold
                    val_receiver_relations[inzz,timestep, j, cnt] = 1.0
                    val_sender_relations[inzz,timestep, m, cnt]   = 1.0
                    cnt += 1
    prediction = np.zeros((n_of_traj,frame_len,n_relations,n_of_rel))

    for i in range(frame_len):
        propagation = np.zeros((n_of_traj, n_objects,100), dtype=float);
        FN_input=dict()
        FN_input['objects'] = GroundData2[:,i,:,:]
        FN_input['sender_relations'] = val_sender_relations[:,i,:,:]
        FN_input['receiver_relations'] = val_receiver_relations[:,i,:,:]
        FN_input['propagation']=propagation
        prediction[:,i,:,:]=Pn.predict(FN_input)
    return xy_origin_pos,edges,r,prediction

def make_video_TPN_fixed(xy,r,edge,filename):
    os.system("rm -rf pics/*");
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                  comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    fig = plt.figure(figsize=(16,16))
    ax = plt.gca()
    ax.cla() # clear things for fresh plot

    # change default range so that new circles will work
    ax.set_xlim((-1.00, 1.00))
    ax.set_ylim((-1.00, 1.00))
    fig_num=len(xy);
    color=['blue','red','green','yellow','purple','cyan','orange','black','pink'];
    with writer.saving(fig, filename, len(xy)-5):
        for i in range(5,len(xy)):
            cnt=0
            for j in range(len(xy[0])):
                circle = plt.Circle((xy[i,j,1],xy[i,j,0]), r[j]/2,color=color[j%9], fill=True)
                ax.add_artist(circle)
                for k in range(len(xy[0])):
                    if j!=k:
                        if edge[cnt,i,0]>0.5:
                            plt.plot([xy[i,j,1],xy[i,k,1]], [xy[i,j,0],xy[i,k,0]], color='black', linestyle='solid',alpha=edge[cnt,i,0], lw=5)
                        cnt=cnt+1
            plt.axis('off')
            writer.grab_frame();
            ax.cla()
            ax.set_xlim((-1.0, 1.0))
            ax.set_ylim((-1.0, 1.0))
            
def make_video_TPN_mixed(xy,r,edge,filename):
    os.system("rm -rf pics/*");
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                  comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    fig = plt.figure(figsize=(16,16))
    ax = plt.gca()
    ax.cla() # clear things for fresh plot

    # change default range so that new circles will work
    ax.set_xlim((-1.00, 1.00))
    ax.set_ylim((-1.00, 1.00))
    fig_num=len(xy);
    color=['blue','red','green','yellow','purple','cyan','orange','black','pink'];
    with writer.saving(fig, filename, len(xy)):
        for i in range(len(xy)):
            cnt=0
            for j in range(len(xy[0])):
                circle = plt.Circle((xy[i,j,1],xy[i,j,0]), r[j]/2,color=color[j%9], fill=True)
                ax.add_artist(circle)
                for k in range(len(xy[0])):
                    if j!=k:
                        edge_type=np.argmax(edge[cnt,i,:])
                        if edge_type==1:
                            plt.plot([xy[i,j,1],xy[i,k,1]], [xy[i,j,0],xy[i,k,0]], color='black', linestyle='solid',alpha=edge[cnt,i,1], lw=5)
                        elif edge_type==2:
                            plt.plot([xy[i,j,1],xy[i,k,1]], [xy[i,j,0],xy[i,k,0]], color='gray', linestyle='solid',alpha=edge[cnt,i,2], lw=5)
                        elif edge_type==3:
                            plt.plot([xy[i,j,1],xy[i,k,1]], [xy[i,j,0],xy[i,k,0]], color='purple', linestyle='solid',alpha=edge[cnt,i,3], lw=5)
                        cnt=cnt+1
            plt.axis('off')
            writer.grab_frame();
            ax.cla()
            ax.set_xlim((-1.0, 1.0))
            ax.set_ylim((-1.0, 1.0))