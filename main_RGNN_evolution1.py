
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import model_RGNN_evolution1 as model
from utils_RGNN_evolution1 import*
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.tri as tri

wDr = os.getcwd()

field_path = wDr + r'\Field_evolution' 

data = load_graph(path=field_path, data_types = 6,
                                   steps=101, split = None,
                                   shuffle = True)

train_data, test_data = train_test_split(data, train_size = 0.9, random_state = 42, shuffle=True)

mean, std = mean_std(train_data)
train_data = transform_stdScaler(mean, std, train_data)
test_data = transform_stdScaler(mean, std, test_data)

os.chdir(wDr)

# Model
learned_model = model.EncodeProcessDecode(
                    node_feat_size = 4,                    
                    edge_feat_size = 3,
                    output_size=6,
                    latent_size=128,                  
                    message_passing_steps=10,
                    window = 1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = learned_model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
scheduler = ExponentialLR(optimizer, gamma=0.9)
print('Learnable parameters',sum(p.numel() for p in model.parameters()))

# My normalization
def normalizer(features):
    # Calculate global mean and std (across all samples and features)
    mean = features.mean()
    std = features.std()
    # Avoid division by zero in case std is 0 (which means constant column)
    if std == 0:
        std = 1.0  # Prevent division by zero by setting std to 1    
    # Normalize using global mean and std
    features = (features - mean) / std    
    # Return the normalized tensor
    return features

#%% BCs
E11=0.08; # finite strain component of Green-Lagrangian strain tensor
e11 = np.sqrt(2*E11 + 1) - 1
steps = train_data[0].ground_truth.shape[0]
bc_strainPBC = np.linspace(0,e11,num = steps)

#%% Data-set creation for training
'''Steps'''
start_step = 1
delta_step = 1
steps = train_data[0].ground_truth.shape[0]
''''''''''''''''''''''''

from torch_geometric.data import InMemoryDataset

class data_set(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def processed_file_names(self):
        return ['data_set.pt']

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        for geom in range(len(train_data)):                                 
            
            edge_index = train_data[geom].edge_index
            node_features_tmp = train_data[geom].node_features            
            edge_features = train_data[geom].edge_features       
            fields = train_data[geom].ground_truth                                      
            
            for step in range(start_step,steps,delta_step):                 
                
                # Assign the macroscopic strain to each node
                node_features_tmp2 = torch.cat([node_features_tmp, bc_strainPBC[step] * torch.ones(node_features_tmp.shape[0],1)], dim=-1)                   
                node_features = normalizer(node_features_tmp2) 
                
                edge_features = normalizer(edge_features)                
                ground_truth = fields[step,:,:]                          
                       
                data = Data(edge_index = edge_index -1, x=node_features, edge_attr=edge_features, y = ground_truth)         
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

dataset = data_set(root = wDr + r'\Datasets')

# Training
from torch_geometric.loader import DataLoader
dataloader = DataLoader(dataset, batch_size = 1, shuffle = True)
import time

def train(model, dataloader):
    # set the model in training mode
    model.train()           
    start_time = time.time()
    loss_all = 0 
       
    for data in dataloader:                                                
        graph = data.to(device)      
        out = model(graph).to(device) # probably I have to change this to process series           
        ground_truth = graph.y
        loss = F.l1_loss(out, ground_truth).to(device) 
        loss.backward()
        optimizer.step()              
        optimizer.zero_grad()                                               
        loss_all += loss.item()       
    return loss_all/len(dataloader)

loss_hist = []
epochs = 1

for epoch in range(epochs):    
    start_time = time.time()       
    print('Epoch:', epoch)    
    loss_avg = train(model, dataloader)    
    print('Loss epoch:', loss_avg)        
    loss_hist.append(loss_avg)       
    print('Time per epoch', (time.time() - start_time))    
    scheduler.step()

# End training


import matplotlib.pyplot as plt


#ps  Testing on different steps than training
'''Steps'''
start_step = 5
delta_step = 10
steps = train_data[0].ground_truth.shape[0]
# 20 predictions per geometry
''''''''''''''''''''''''
# step_list = list(range(start_step,steps,delta_step))

def test_f(model, test_data, geom):   
    model.eval()
    
    pred = []
    loss_all = 0       
    # for geom in range(test_data_len):
        
    # print('Sample:', geom)
    
    edge_index = test_data[geom].edge_index
    node_features = test_data[geom].node_features            
    edge_features = test_data[geom].edge_features        
    fields = test_data[geom].ground_truth
    elements = test_data[geom].connectivity        
        
    loss_tmp = 0        
    out_step = []        
    for step in range(start_step,steps,delta_step):      
        # print('Step:', step)                                
        
        node_features_tmp = torch.cat([node_features, bc_strainPBC[step] * torch.ones(node_features.shape[0],1)], dim=-1)                    
        node_features_tmp = normalizer(node_features_tmp)            
        
        edge_features = normalizer(edge_features)                  
               
        data = Data(edge_index = edge_index -1, x=node_features_tmp, edge_attr=edge_features)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        graph = data.to(device)
                                
        start_time = time.time() 
        out = model(graph)
        print('Inference', (time.time() - start_time))
        
        device = torch.device('cpu')
        out_step.append(out.to(device))
        
        # Loss (average) on steps
        ground_truth = fields[step,:,:]
        
        # loss = F.mse_loss(out, ground_truth)
        loss = F.l1_loss(out.to(device), ground_truth)
        loss_tmp += loss.item()                                               
   
    loss_all += loss_tmp / ((steps-start_step)/delta_step)          
    pred.append(out_step)              
        
    # return pred, loss_all/test_data_len
    return pred, loss_all

# test_data_len = len(test_data)
# test_data_len = 1
geom = 0
pred, mae_test = test_f(model, test_data, geom)    
print('MAE on test data',mae_test)
#%% Save model and load model
# Saving
path = wDr + r'\Models'
model_name = r'\fiber_evolution_latent64_L10.pt'

save_model = False
# save_model = True

if save_model == True:
    torch.save(model.state_dict(), path + model_name)

# Loading
load_model = False

if load_model == True:  
    learned_model = model.EncodeProcessDecode(
                        node_feat_size = 4,                    
                        edge_feat_size = 3,
                        output_size=6,
                        latent_size=64,                  
                        message_passing_steps=10,
                        window = 1)
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model = learned_model.to(device)    
    model.load_state_dict(torch.load(path + model_name))
#%% Stress-strain curves
# Finite strain component of Green-Lagrangian strain tensor
E11=0.08;  
lam_minus1 = np.sqrt(2*E11 + 1) - 1
eff_strain = np.linspace(0, lam_minus1, num = steps)

# you should choose a geoemtry and steps not chosen for tuning 
geom = 0 
compon = 2

elements = test_data[geom].connectivity.detach().numpy()    
node_features = test_data[geom].node_features        
node_features = node_features.detach().numpy()
fields = test_data[geom].ground_truth

step_list = list(range(start_step,steps,delta_step))
avg_stress_truth = []
avg_stress_pred = []
eff_strain_d = []
for step_pred in range(len(step_list)):   
    step = step_list[step_pred]   
    
    eff_strain_d.append(eff_strain[step])
    '''Ground truth'''
    field_truth = fields[step,:,:]
    # field_truth = inverse_transform_minmax(maxs[2][step,:,:], mins[2][step,:,:], field_truth)
    field_truth = inverse_transform_stdScal(mean[2][step,:,:], std[2][step,:,:], field_truth)
    field_truth = field_truth[:,compon].detach().numpy()     
    
    avg_stress_truth.append(np.mean(field_truth, axis = 0))
    
    '''Prediction'''
    field = pred[geom][step_pred]        
    # field = inverse_transform_minmax(maxs[2][step,:,:], mins[2][step,:,:], field)
    field = inverse_transform_stdScal(mean[2][step,:,:], std[2][step,:,:], field)
    field = field[:,compon].detach().numpy()    
    
    avg_stress_pred.append(np.mean(field, axis = 0))

a = np.array(eff_strain_d, dtype=float)
b = np.array(avg_stress_pred, dtype=float)
c = np.array(avg_stress_truth, dtype=float)

# Plot the data
plt.figure()
plt.plot(a, b, label='ML')         # Plot b
plt.plot(a, c, label='Truth')      # Plot c on the same axes

# Add labels and legend
plt.xlabel(r'$\overline{\varepsilon}$ (%)')  # Add x-axis label
plt.ylabel(r'$\overline{\sigma}$ (-)')       # Add y-axis label
plt.legend(loc='best')                       # Add legend

# Show the plot
plt.show()

##########################################

# converts quad elements into tri elements for plotting
def quads_to_tris(quads):
    tris = [[None for j in range(3)] for i in range(2*len(quads))]
    for i in range(len(quads)):
        j = 2*i
        n0 = quads[i][0]
        n1 = quads[i][1]
        n2 = quads[i][2]
        n3 = quads[i][3]
        tris[j][0] = n0
        tris[j][1] = n1
        tris[j][2] = n2
        tris[j + 1][0] = n2
        tris[j + 1][1] = n3
        tris[j + 1][2] = n0
    return tris

# plots finite element mesh
def plot_fem_mesh(nodes_x, nodes_y, elements, meshcolor = 'black'):    
    
    for element in elements:
        x = [nodes_x[element[i]] for i in range(len(element))]
        y = [nodes_y[element[i]] for i in range(len(element))]
        plt.fill(x, y, edgecolor=meshcolor, fill=False)

def plotting(scale_fac = None, def_shape = True, compon = None, coords = None, field = None, elements = None, levels_contour = None, cmap = None):
    
    if def_shape == True:
        nodes_x = coords[:,0] + scale_fac * field[:,0]
        nodes_y = coords[:,1] + scale_fac * field[:,1]
    else:        
        nodes_x = coords[:,0]
        nodes_y = coords[:,1]
    elements = elements - 1
    
    if compon != None:
        nodal_values = field[:,compon]
    else:
        nodal_values = field    
    
    # convert all elements into triangles
    elements_all_tris = quads_to_tris(elements)
    
    # create an unstructured triangular grid instance
    triangulation = tri.Triangulation(nodes_x, nodes_y, elements_all_tris)
    
    # fig, ax = plt.subplots(figsize=(4, 4))  
    # fig.patch.set_visible(False)
    # ax.axis('off')
    fig = plt.figure()
    # plot FE mesh
    meshcolor = 'black'
    #ps plot_fem_mesh(nodes_x, nodes_y, elements, meshcolor = meshcolor)
    
    # plot the contours
    plt.tricontourf(triangulation, nodal_values,levels = levels_contour, cmap=cmap)
    
    # show    
    plt.colorbar()
    plt.axis('equal')
    # plt.show()        
    return fig

def fig_to_array(fig):
    fig.canvas.draw()
    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return data
def plot_test(scale_fac, def_shape, test_data, pred, geom, compon,
              levels_contour, cmap, step, video):
    
    elements = test_data[geom].connectivity.detach().numpy()    
    node_features = test_data[geom].node_features        
    node_features = node_features.detach().numpy()
    
    fields = test_data[geom].ground_truth
   
    img_pred = []
    img_true = []
    step_list = list(range(start_step,steps,delta_step))
    for step_pred in range(len(step_list)):        
        step = step_list[step_pred]
       
        coords = node_features[:,:2]  #ps coords = node_features[:,:2,step]        
        '''Ground truth'''
        field_truth = fields[step,:,:]
        # field_truth = inverse_transform_minmax(maxs[2][step,:,:], mins[2][step,:,:], field_truth)
        field_truth = inverse_transform_stdScal(mean[2][step,:,:], std[2][step,:,:], field_truth)
        field_truth = field_truth.detach().numpy()     
        
        fig = plotting(scale_fac, def_shape, compon, coords,
                       field_truth, elements, levels_contour, cmap)
           
        if video == True:
            img_true.append(fig_to_array(fig))        
            
        '''Prediction'''
        field = pred[geom][step_pred]        
        # field = inverse_transform_minmax(maxs[2][step,:,:], mins[2][step,:,:], field)
        field = inverse_transform_stdScal(mean[2][step,:,:], std[2][step,:,:], field)
        field = field.detach().numpy()    
        
        fig = plotting(scale_fac, def_shape, compon, coords,
                       field, elements, levels_contour, cmap)
        
        if video == True:
                img_pred.append(fig_to_array(fig))        
                   
    return img_true, img_pred 

# Choose geometry
geom = 0

# Levels in the contour field to plot
levels_contour = np.linspace(0,51,100)
# levels_contour = 100

# Color map fields
cmap = 'viridis'

# Choose field to plot
compon = 2

def_shape = True
scale_fac = 1.0
video = True
img_true, img_pred = plot_test(scale_fac, def_shape, test_data, pred, geom, compon,
          levels_contour, cmap, step, video)

plt.close('all')
#%% Deformation video with field
import matplotlib.cm as cm
import matplotlib.animation as animation

# Active interactive mode for plotting automatically
plt.ion()

def video_maker(img1, img2, save = True, name = None, fps = 1, dpi = 300):

    frames = [] # for storing the generated images
    # fig = plt.figure()
    fig = plt.figure()
    ax1=fig.add_subplot(1,2,1)
    ax2=fig.add_subplot(1,2,2)
    for i in range(len(img1)):                
        
        ax1.axis('off')
        ax2.axis('off')
        im1 = ax1.imshow(img1[i],cmap=cm.Greys_r,animated=True)
        im2 = ax2.imshow(img2[i],cmap=cm.Greys_r,animated=True)
        
        frames.append([im1, im2])
        # frames.append([plt.imshow(img[i], cmap=cm.Greys_r,animated=True)])
    
    ani = animation.ArtistAnimation(fig, frames, interval=1000, blit=True,
                                    repeat_delay=1000)
    
    # FFwriter = animation.FFMpegWriter()
    if save == True:
        writergif = animation.PillowWriter(fps=fps) 
        # writergif.setup(fig, name + '.gif' , dpi = 1200) 

        ani.save(name + '.gif',writer =writergif, dpi = dpi)
    plt.show()
    
    return ani

name_file = 'testgeom0_s11_test3_further_steps'
saveFlag = True
video = video_maker(img_true, img_pred, save = saveFlag,
                    name = name_file, fps = 1, dpi = 500)
# video_maker(img_true, save = True, name = 's11_truth', fps = 1)
# video_maker(img_pred, save = True, name = 's11_pred', fps = 1)

#%% Error on the fields prediction (metrics)