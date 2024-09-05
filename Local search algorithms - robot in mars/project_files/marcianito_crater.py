import numpy as np
import math
import plotly.graph_objects as px
import random
import time
import copy

mars_map = np.load('crater_map.npy')
nr, nc = mars_map.shape

scale = 10.045

def convert(x, y):
    r = nr - int(np.round(y/scale))
    c = int(np.round(x/scale))
    return r, c

class Marcianito(object):
    def __init__(self, initial_pos, mapa = mars_map):
        self.map = mapa
        self.pos = initial_pos

    def cost(self):
        row_mono, col_mono = self.pos
        c = self.map[row_mono][col_mono]
        return c
    
    def neighbor(self, pos):
        new_pos = copy.deepcopy(pos)
        row_mono_i, col_mono_i = new_pos
        row_mono_n, col_mono_n = new_pos
        pasos = [-1, 0, 1]
        valid_position = False

        while not valid_position:
            row_mono_n = row_mono_i + int(random.choice(pasos))
            col_mono_n = col_mono_i + int(random.choice(pasos))

            if (row_mono_n != 0 
                and row_mono_n != nr - 1 
                and col_mono_n != 0 
                and col_mono_n != nc- 1 
                and self.map[row_mono_n][col_mono_n] != -1
                and abs(self.map[row_mono_i][col_mono_i] - self.map[row_mono_n][col_mono_n]) <= 2):

                valid_position = True
        
        return Marcianito((row_mono_n, col_mono_n))

#------------------------------------------------------------------------------------------------------------------
#   Program
#------------------------------------------------------------------------------------------------------------------
random.seed(time.time()*1000)

marciano = Marcianito(convert(3350,5800))      # Initialize board

cost = marciano.cost()         # Initial cost    
step = 0                    # Step count

alpha = 0.99998              # Coefficient of the exponential temperature schedule        
t0 = 1                      # Initial temperature
t = t0

path_x, path_y, path_z = [], [], []


#Greedy
'''
for i in range (200000):
    
    print(scale*marciano.pos[1],' ' ,(nr - marciano.pos[0])*scale)

    step +=1
    # Get random neighbor
    neighbor = marciano.neighbor(marciano.pos)
    new_cost = neighbor.cost()

    # Test neighbor
    if new_cost < cost:
        marciano = neighbor
        cost = new_cost

    path_x.append(scale*marciano.pos[1])
    path_y.append((nr - marciano.pos[0])*scale)
    path_z.append(cost)

    print("Iteration: ", step, "    Cost: ", cost)

'''
#   Recocido simulado

while t > 0.005 and cost > 0:

    print(scale*marciano.pos[1],' ' ,(nr - marciano.pos[0])*scale)

    # Calculate temperature
    t = t0 * math.pow(alpha, step)
    step += 1
        
    # Get random neighbor
    neighbor = marciano.neighbor(marciano.pos)
    new_cost = neighbor.cost()

    # Test neighbor
    if new_cost < cost:
        marciano = neighbor
        cost = new_cost
    else:
        # Calculate probability of accepting the neighbor
        p = math.exp(-(new_cost - cost)/t)
        if p >= random.random():
            marciano = neighbor
            cost = new_cost
    
    path_x.append(scale*marciano.pos[1])
    path_y.append((nr - marciano.pos[0])*scale)
    path_z.append(cost)

    print("Iteration: ", step, "    Cost: ", cost, "    Temperature: ", t)

x = scale*np.arange(mars_map.shape[1])
y = scale*np.arange(mars_map.shape[0])
X, Y = np.meshgrid(x, y)

fig = px.Figure(data = [px.Surface(x=X, y=Y, z=np.flipud(mars_map), colorscale='hot', cmin = 0, 
                                    lighting = dict(ambient = 0.0, diffuse = 0.8, fresnel = 0.02, roughness = 0.4, specular = 0.2),
                                    lightposition=dict(x=0, y=nr/2, z=2*mars_map.max())),
                    
                        px.Scatter3d(x = path_x, y = path_y, z = path_z, name='path', mode='markers',
                                        marker=dict(color=np.linspace(0, 1, len(path_x)), colorscale="Bluered", size=4))],
            
                layout = px.Layout(scene_aspectmode='manual', 
                                    scene_aspectratio=dict(x=1, y=nr/nc, z=max(mars_map.max()/x.max(), 0.2)), 
                                    scene_zaxis_range = [0,mars_map.max()])
                )
fig.show()
