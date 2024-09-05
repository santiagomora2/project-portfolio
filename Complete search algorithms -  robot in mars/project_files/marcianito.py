from simpleai.search import SearchProblem, astar, breadth_first, depth_first, iterative_limited_depth_first, greedy
import numpy as np
import math
import plotly.graph_objects as px

mars_map = np.load('mars_map.npy')
nr, nc = mars_map.shape

scale = 10.0174

def convert(x, y):
    r = nr - int(np.round(y/scale))
    c = int(np.round(x/scale))
    return r, c
# original
#goal = convert(3150, 6800)
# 500 m
# goal = convert(2100, 9300)
# goal = convert(4600, 1050)
# 1000-5000 m
# goal = convert(2900, 8000)
# 10 000 m
goal = convert(4517, 2504)

class Marcianito(SearchProblem):

    def __init__(self, initial_state):
        SearchProblem.__init__(self, initial_state)

    def actions(self, state):
        row_mono, col_mono = state
        act = []
        #laterales
        if row_mono - 1 != 0 and mars_map[row_mono - 1][col_mono] != -1 and abs(mars_map[row_mono - 1][col_mono] - mars_map[row_mono][col_mono]) <= 0.25:
            act.append((row_mono - 1, col_mono))
        if row_mono + 1 != nr - 1 and mars_map[row_mono + 1][col_mono] != -1 and abs(mars_map[row_mono + 1][col_mono] - mars_map[row_mono][col_mono]) <= 0.25:
            act.append((row_mono + 1, col_mono))
        #arriba abajo
        if col_mono - 1 != 0 and mars_map[row_mono][col_mono - 1] != -1 and abs(mars_map[row_mono][col_mono - 1] - mars_map[row_mono][col_mono]) <= 0.25:
            act.append((row_mono, col_mono - 1))
        if col_mono + 1 != nc - 1 and mars_map[row_mono][col_mono + 1] != -1 and abs(mars_map[row_mono][col_mono + 1] - mars_map[row_mono][col_mono]) <= 0.25:
            act.append((row_mono, col_mono + 1))
        #diagonales
        if row_mono - 1 != 0 and col_mono - 1 != 0 and mars_map[row_mono - 1][col_mono - 1] != -1 and abs(mars_map[row_mono - 1][col_mono - 1] - mars_map[row_mono][col_mono]) <= 0.25:
            act.append((row_mono - 1, col_mono - 1))
        if row_mono - 1 != 0 and col_mono + 1 != nc - 1 and mars_map[row_mono - 1][col_mono + 1] != -1 and abs(mars_map[row_mono - 1][col_mono + 1] - mars_map[row_mono][col_mono]) <= 0.25:
            act.append((row_mono - 1, col_mono + 1))
        if row_mono + 1 != nr - 1 and col_mono - 1 != 0 and mars_map[row_mono + 1][col_mono - 1] != -1 and abs(mars_map[row_mono + 1][col_mono - 1] - mars_map[row_mono][col_mono]) <= 0.25:
            act.append((row_mono + 1, col_mono - 1))
        if row_mono + 1 != nr - 1 and col_mono + 1 != nc - 1 and mars_map[row_mono + 1][col_mono + 1] != -1 and abs(mars_map[row_mono + 1][col_mono + 1] - mars_map[row_mono][col_mono]) <= 0.25:
            act.append((row_mono + 1, col_mono + 1))
        return act

    def result(self, state, action):
        return action

    def is_goal(self, state):
        return state == goal

    def cost(self, state, action, state2):
        return 1

    def heuristic(self, state):
        row_mono, col_mono = state
        row_exit, col_exit = goal

        return abs(row_exit - row_mono) + abs(col_exit - col_mono)

# original
# initial_state = convert(2850, 6400)
# 500 m
# initial_state = convert(1800, 9500)
# initial_state = convert(4500, 500)
# 1000-5000 m
# initial_state = convert(4480, 4400)
# 10 000 m
initial_state = convert(5409, 13974)

problem = Marcianito(initial_state)
result = astar(problem, graph_search=True)

print(result)

#----------------------------------------------------------------

n_rows, n_columns = nr, nc
x_ini, y_ini= 5409, 13974
x_goal, y_goal = 4517, 2504

row_ini = n_rows-round(y_ini/scale)
col_ini = round(x_ini/scale)

row_goal = n_rows-round(y_goal/scale)
col_goal = round(x_goal/scale)

if result != None:
    path_x = []
    path_y = []
    path_z = []
    prev_state = []
    distance = 0
    for i, (action, state) in enumerate(result.path()):    
        path_x.append( state[1] * scale  )            
        path_y.append(  (n_rows - state[0])*scale  )
        path_z.append(mars_map[state[0]][state[1]]+1)
        
        if len(prev_state) > 0:
            distance +=  math.sqrt(
            scale*scale*(state[0] - prev_state[0])**2 + scale*scale*(state[1] - prev_state[1])**2 + (
                mars_map[state[0], state[1]] - mars_map[prev_state[0], prev_state[1]])**2)

        prev_state = state

    print("Total distance", distance)

else:
    print("Unable to find a path between that connect the specified points")

## Plot results
if result != None: 

    x = scale*np.arange(mars_map.shape[1])
    y = scale*np.arange(mars_map.shape[0])
    X, Y = np.meshgrid(x, y)

    fig = px.Figure(data = [px.Surface(x=X, y=Y, z=np.flipud(mars_map), colorscale='hot', cmin = 0, 
                                        lighting = dict(ambient = 0.0, diffuse = 0.8, fresnel = 0.02, roughness = 0.4, specular = 0.2),
                                        lightposition=dict(x=0, y=n_rows/2, z=2*mars_map.max())),
                        
                            px.Scatter3d(x = path_x, y = path_y, z = path_z, name='path', mode='markers',
                                            marker=dict(color=np.linspace(0, 1, len(path_x)), colorscale="Bluered", size=4))],
                
                    layout = px.Layout(scene_aspectmode='manual', 
                                        scene_aspectratio=dict(x=1, y=n_rows/n_columns, z=max(mars_map.max()/x.max(), 0.2)), 
                                        scene_zaxis_range = [0,mars_map.max()])
                    )
    fig.show()
