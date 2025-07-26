import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import pandas as pd
from OpenGL.GL import *
from OpenGL_2D_class_GLFW import gl2D, gl2DCircle
from HersheyFont import HersheyFont
hf = HersheyFont()


class Node:
    def __init__(self, x, y, ix, iy, number, type):
        self.Type = type
        self.number = number
        self.x = x
        self.y = y
        self.ix = ix
        self.iy = iy
        self.Temp = None
        self.up = None
        self.down = None
        self.left = None
        self.right = None

    def get_pipe_wall_resistance(self, system, dz=None):
        if dz is None:
            dz = system.pipeNodeAxialSpacing
        r_i = system.innerPipeDiameter / 2.0
        r_o = system.outerPipeDiameter / 2.0
        return np.log(r_o / r_i) / (2.0 * np.pi * system.k_pipe * dz)

    def addToMatrix(self, system):
        i = self.number

        # 1) Fluid/fluid‐flow nodes: conduction to water + upwind advection
        if self.Type in ('flowUp', 'flowDown', 'fluid'):
            # 1a) Conduction through pipe wall (fluid → water)
            water = self.right
            if water and water.Type == 'water':
                dz = system.pipeNodeAxialSpacing
                R_wall = self.get_pipe_wall_resistance(system, dz)
                G = 1.0 / R_wall
                j = water.number

                system.K[i, j] -= G
                system.K[i, i] += G

            # 1b) Upwind advection (only for flowing legs)
            if self.Type in ('flowUp', 'flowDown'):
                F = system.m_dot_per_leg * system.cp_fluid
                neighbor = self.up if self.Type == 'flowUp' else self.down
                if neighbor:
                    j = neighbor.number
                    system.K[i, i] += F
                    system.K[i, j] -= F

        # 2) Water nodes: back‑link conduction to all fluids + conduction to ground
        elif self.Type == 'water':
            # 2a) Back‑link to every fluid at this layer via pipe wall
            for fluid in getattr(self, 'radial_neighbors', []):
                dz = system.pipeNodeAxialSpacing
                R_wall = self.get_pipe_wall_resistance(system, dz)
                G = 1.0 / R_wall
                j = fluid.number

                system.K[i, j] -= G
                system.K[i, i] += G

            # 2b) Radial conduction to the first ground node
            ground = self.right
            if ground and ground.Type == 'ground':
                dr = ground.x - self.x
                r_face = 0.5 * (self.x + ground.x)
                if self.up and self.down:
                    dz = self.down.y - self.up.y
                elif self.up:
                    dz = self.y - self.up.y
                elif self.down:
                    dz = self.down.y - self.y
                else:
                    # fallback if isolated—use the pipe spacing as a guess
                    dz = system.pipeNodeAxialSpacing
                k_w = system.get_conductivity('water')
                A_r = 2.0 * np.pi * r_face * dz
                G = k_w * A_r / dr
                j = ground.number

                # water equation
                system.K[i, j] -= G
                system.K[i, i] += G
                # ground equation (mirror)
                system.K[j, i] -= G
                system.K[j, j] += G

            if self.Type == 'water':
                k_w = system.k_water
                # cross‑sectional area for axial conduction: approximate
                # A = 2π r * dr, with dr from radial spacing to neighbor
                if self.left and self.right:
                    dr = 0.5 * (self.right.x - self.left.x)
                elif self.left:
                    dr = self.x - self.left.x
                elif self.right:
                    dr = self.right.x - self.x
                else:
                    dr = system.innerGroundRadius  # fallback
                A_ax = 2.0 * np.pi * self.x * dr

                for neigh in (self.up, self.down):
                    if neigh and neigh.Type == 'water':
                        dz = abs(neigh.y - self.y)
                        G = k_w * A_ax / dz
                        j = neigh.number
                        system.K[i, j] -= G
                        system.K[i, i] += G

        # 3) Ground nodes: radial conduction to other ground nodes
        elif self.Type == 'ground':
            for neigh in (self.left, self.right):
                if neigh and neigh.Type == 'ground':
                    dr = abs(neigh.x - self.x)
                    r_face = 0.5 * (self.x + neigh.x)

                    # axial span of this ring
                    if self.up and self.down:
                        dz = self.down.y - self.up.y
                    elif self.up:
                        dz = self.y - self.up.y
                    elif self.down:
                        dz = self.down.y - self.y
                    else:
                        dz = system.lowerWaterNodeAxialSpacing

                    k_g = system.get_conductivity('ground')
                    A_r = 2.0 * np.pi * r_face * dz
                    G = k_g * A_r / dr
                    j = neigh.number

                    system.K[i, j] -= G
                    system.K[i, i] += G




class System:
    def __init__(self):
        self.filename = "ButteShaftParametersSimplified.txt"
        self.title = None
        self.Nodes = []
        self.solverMatrix = []
        self.RadialNodeLocations = []
        self.RadialNodeTypes = []
        self.AxialNodeLocations = []
        self.AxialNodeTypes = []
        self.temps = []
        self.pipeDepth = None
        self.shaftDepth = None
        self.innerGroundRadius = None
        self.outerGroundRadius = None
        self.firstGroundNodeRadialSpacing = None
        self.pipeNodeAxialSpacing = None
        self.lowerWaterNodeAxialSpacing = None
        self.innerPipeDiameter = None
        self.outerPipeDiameter = None
        self.groundNodeRadialSpacingGrowthRate = None
        self.filedata = None

        # material properties
        self.k_pipe = 0.479
        self.k_water = 0.6
        self.k_ground = 2.77
        self.k_fluid = 0.6
        self.rho_water = 998
        self.cp_water = 4180
        # self.u_axial_water = 1
        self.rho_fluid = 987
        self.cp_fluid = 4337.34
        self.rho_ground = 2800.0
        self.cp_ground = 837.0
        # self.rho_pipe = 1
        # self.cp_pipe = 1
        self.num_legs = 4
        self.m_dot_per_leg = 3
        self.h_conv = 1
        self.T_fluid_in = 10.0
        self.use_axial_ground = True
        self.K = None
        self.b0 = None
        self.C = None
        self.T_initial_fluid = 35.0
        self.T_initial_water = 31.0
        self.T_initial_ground = 30.0
        self.displayed_node_indices = []
        self.current_display_hour = 0
        self.frame = 0

        # self.xmin = -self.pipeNodeAxialSpacing  # a little extra space on the left
        # self.xmax = self.pipeNodeAxialSpacing * 19 # 19 determined by trial and error
        # self.ymin = self.shaftDepth - (self.pipeDepth / 5)  # show 1/5 of the pipe depth
        # self.ymax = self.shaftDepth + self.pipeNodeAxialSpacing
        # self.numberOfAnimationFrames = len(self.temps) - 1
        self.AnimDelayTime = 1
        self.AnimReverse = False
        self.AnimRepeat = False
        self.AnimReset = False
        self.allowDistortion = False





        # After  ProcessFileData() is called, the self.Animator class must have meaningful values in
        # the following drawing size class attributes (data items):
        # self.xmin, self.xmax, self.ymin,self. ymax    - Used to set the window working space
        # self.allowDistortion  - Will circles display as round or elliptical?
        # And the following animation control class attributes:
        # self.numberOfAnimationFrames   - total number of animation frames
        # self.AnimDelayTime  - delay time between frames
        # self.AnimReverse, self.AnimRepeat, self.AnimReset















    def get_conductivity(self, material_type):
        """
        Return the thermal conductivity [W/m·K] for a given node Type.
        """
        if material_type == 'water':
            return self.k_water
        elif material_type == 'pipeWall':
            return self.k_pipe
        elif material_type == 'ground':
            return self.k_ground
        elif material_type in ('fluid', 'flowUp', 'flowDown'):
            return self.k_fluid
        else:
            raise ValueError(f"Unknown material_type '{material_type}'")

    def ReadNetworkData(self):

        f1 = open(self.filename, 'r')  # open the file for reading
        self.filedata = f1.readlines()  # read the entire file as a list of strings
        f1.close()  # close the file  ... very important

        # data is an array of strings, read from a Truss data file
        for line in self.filedata:  # loop over all the lines
            if "#" in line:  # ignore anything after the first #
                line = line.split('#')[0]
            line = line.strip().replace('(', '').replace(')', '')
            line = line.replace('"', '').replace("'", '')
            cells = line.split(',')
            keyword = cells[0].lower().strip()

            if keyword == 'title': self.title = cells[1].replace("'", "").strip()
            if keyword == 'pipedepth': self.pipeDepth = float(cells[1])
            if keyword == 'shaftdepth': self.shaftDepth = float(cells[1])
            if keyword == 'innergroundradius': self.innerGroundRadius = float(cells[1])
            if keyword == 'outergroundradius': self.outerGroundRadius = float(cells[1])
            if keyword == 'firstgroundnoderadialspacing': self.firstGroundNodeRadialSpacing = float(cells[1])
            if keyword == 'pipenodeaxialspacing': self.pipeNodeAxialSpacing = float(cells[1])
            if keyword == 'lowerwaternodeaxialspacing': self.lowerWaterNodeAxialSpacing = float(cells[1])
            if keyword == 'innerpipediameter': self.innerPipeDiameter = float(cells[1])
            if keyword == 'outerpipediameter': self.outerPipeDiameter = float(cells[1])
            if keyword == 'groundnoderadialspacinggrowthrate': self.groundNodeRadialSpacingGrowthRate = float(cells[1])
        pass

    def createRadialNodeLocations(self):
        # clear any old values
        self.RadialNodeLocations = []
        self.RadialNodeTypes = []

        # 1) compute lumped water-node radius
        #    use the outer pipe diameter to get the outer-wall radius
        # outer_pipe_wall_radius = self.outerPipeDiameter / 2.0
        water_r = 0.5 * self.innerGroundRadius

        # 2) place 4 fluid nodes evenly from r=0 to r=water_r
        #    (split [0,water_r] into 5 segments; nodes at i/5*water_r for i=1..4)
        for i in range(1, 5):
            r = (i / 5.0) * water_r
            self.RadialNodeLocations.append(r)
            self.RadialNodeTypes.append('fluid')

        # 3) place the single lumped water node
        self.RadialNodeLocations.append(water_r)
        self.RadialNodeTypes.append('water')

        # 4) now build ground nodes starting at innerGroundRadius
        tol = 1e-8

        # always include first ground node at innerGroundRadius
        r = self.innerGroundRadius
        self.RadialNodeLocations.append(r)
        self.RadialNodeTypes.append('ground')

        dr = self.firstGroundNodeRadialSpacing
        while True:
            r_next = r + dr
            if r_next + tol >= self.outerGroundRadius:
                break
            self.RadialNodeLocations.append(r_next)
            self.RadialNodeTypes.append('ground')
            r = r_next
            dr *= self.groundNodeRadialSpacingGrowthRate

        # ensure the very outer boundary is included
        if abs(self.RadialNodeLocations[-1] - self.outerGroundRadius) > tol:
            self.RadialNodeLocations.append(self.outerGroundRadius)
            self.RadialNodeTypes.append('ground')

        # 5) sort by radius and remove any duplicates
        combined = sorted(
            zip(self.RadialNodeLocations, self.RadialNodeTypes),
            key=lambda x: x[0]
        )
        unique_rs, unique_ts = [], []
        for r, t in combined:
            if not any(abs(r - ur) < tol for ur in unique_rs):
                unique_rs.append(r)
                unique_ts.append(t)

        self.RadialNodeLocations = unique_rs
        self.RadialNodeTypes = unique_ts

    def createAxialNodeLocations(self):

        self.AxialNodeLocations = []
        self.AxialNodeTypes = []

        # 2) fluid axial slices (for all fluid, pipeWall, and ground nodes)
        delta_pipe_axial = self.pipeNodeAxialSpacing
        n_f = int(np.ceil(self.pipeDepth / delta_pipe_axial))
        for i in range(n_f + 1):
            z = min(i * delta_pipe_axial, self.pipeDepth)
            self.AxialNodeLocations.append(z)
            self.AxialNodeTypes.append('fluid')

        dz_w = self.lowerWaterNodeAxialSpacing
        span_w = self.shaftDepth - self.pipeDepth
        n_w = int(np.ceil(span_w / dz_w))
        for i in range(1, n_w + 1):
            z = min(self.pipeDepth + i * dz_w, self.shaftDepth)
            self.AxialNodeLocations.append(z)
            self.AxialNodeTypes.append('water')

        combined = sorted(zip(self.AxialNodeLocations, self.AxialNodeTypes),
                          key=lambda x: x[0])
        self.AxialNodeLocations, self.AxialNodeTypes = map(list, zip(*combined))
        pass

    def createNodes(self):
        try:
            water_idx = self.RadialNodeTypes.index('water')
            water_r = self.RadialNodeLocations[water_idx]
        except ValueError:
            raise RuntimeError("No radial 'water' node found!")

        self.Nodes = []
        count = 0

        for ir, (r, rtype) in enumerate(zip(self.RadialNodeLocations, self.RadialNodeTypes)):
            for iz, (z, ztype) in enumerate(zip(self.AxialNodeLocations, self.AxialNodeTypes)):

                # 1) within pipe‐region slices (z ≤ pipeDepth)
                if ztype == 'fluid':
                    if rtype == 'fluid':
                        # split the 4 fluid pts: inner half → flowDown; outer half → flowUp
                        node_type = 'flowDown' if r < 0.5 * water_r else 'flowUp'
                    elif rtype == 'water':
                        node_type = 'water'  # the single lumped water‐node
                    elif rtype == 'ground':
                        node_type = 'ground'  # explicit ground‐node at fluid depth
                    else:
                        continue

                # 2) below pipe depth (shaft water region)
                elif ztype == 'water':
                    node_type = 'ground' if rtype == 'ground' else 'water'

                else:
                    continue

                node = Node(x=r, y=z,
                            ix=ir, iy=iz,
                            number=count,
                            type=node_type)
                self.Nodes.append(node)
                count += 1

    def linkNodes(self):
        # 1) initialize pointers & new radial_neighbors list
        for node in self.Nodes:
            node.up = node.down = node.left = node.right = None
            # new list to hold back-links for water
            node.radial_neighbors = []

        # 2) vertical linking
        for node in self.Nodes:
            node.up = self.findNodeByIndexes(node.ix, node.iy - 1)
            node.down = self.findNodeByIndexes(node.ix, node.iy + 1)

        # 3) find the water node at each axial index
        water_at_layer = {
            node.iy: node
            for node in self.Nodes
            if node.Type == 'water'
        }

        # 4) radial linking
        for node in self.Nodes:
            if node.Type in ('flowUp', 'flowDown', 'fluid'):
                # all fluid nodes point inward to water
                water = water_at_layer.get(node.iy)
                if water:
                    node.right = water

            elif node.Type == 'water':
                # back-link to *all* fluid nodes at this layer
                fluids = [
                    n for n in self.Nodes
                    if n.iy == node.iy
                       and n.Type in ('flowUp', 'flowDown', 'fluid')
                ]
                node.radial_neighbors = fluids

                # still link outward to the first ground ("the wall")
                grounds = [
                    n for n in self.Nodes
                    if n.iy == node.iy and n.Type == 'ground' and n.x > node.x
                ]
                if grounds:
                    wall = min(grounds, key=lambda g: g.x)
                    node.right = wall

            elif node.Type == 'ground':
                # ground keeps full left/right chain
                node.left = self.findNodeByIndexes(node.ix - 1, node.iy)
                node.right = self.findNodeByIndexes(node.ix + 1, node.iy)

    def findNodeByIndexes(self, ix, iy):
        for node in self.Nodes:
            if node.ix == ix and node.iy == iy:
                return node
        return None
    pass

    def get_volume(self, node):
        if node.Type in ('flowUp', 'flowDown'):
            A_pipe = np.pi * (self.innerPipeDiameter / 2.0) ** 2
            dz = self.pipeNodeAxialSpacing
            return A_pipe * dz

            # --- 2) Ring nodes: concentric‐cylinder volume ---
            # radial thickness dr
        if node.left is not None and node.right is not None:
            dr = (node.right.x - node.left.x) * 0.5
        elif node.left is None and node.right is not None:
            # inner boundary: thickness is distance to the first right node
            dr = node.right.x - node.x
        elif node.left is not None and node.right is None:
            # outer boundary: distance back to the last left node
            dr = node.x - node.left.x
        else:
            raise RuntimeError(f"Cannot compute radial dr for node #{node.number}")

            # axial thickness dz
        if node.up is not None and node.down is not None:
            dz = (node.down.y - node.up.y) * 0.5
        else:
            dz = self.pipeNodeAxialSpacing

        return 2.0 * np.pi * node.x * dr * dz

    def get_rho_cp(self, node):
        if node.Type == 'water': return self.rho_water, self.cp_water
        if node.Type in ('fluid', 'flowUp', 'flowDown'): return self.rho_fluid, self.cp_fluid
        if node.Type == 'ground': return self.rho_ground, self.cp_ground
        if node.Type == 'pipeWall': return self.rho_pipe, self.cp_pipe
        raise ValueError(f"Unknown node type for Cp: {node.Type}")

    def buildCapacitance(self):
        N = len(self.Nodes)
        C = np.zeros(N)
        for node in self.Nodes:
            rho, cp = self.get_rho_cp(node)
            vol = self.get_volume(node)
            C[node.number] = rho * cp * vol
        self.C = C

    def buildSteadyMatrices(self):
            N = len(self.Nodes)
            self.K = lil_matrix((N, N))
            self.b0 = np.zeros(N)
            for node in self.Nodes:
                node.addToMatrix(self)

    def saveTempsToCSV(self, temps, filename, selected_nodes=None):
        df = pd.DataFrame(temps, columns=[f'node_{i}' for i in range(temps.shape[1])])
        df.insert(0, 'time_hr', range(temps.shape[0]))
        if selected_nodes is not None:
            cols = ['time_hr'] + [f'node_{i}' for i in selected_nodes]
            df = df[cols]

        new_row_data = {'Name': 'Charlie', 'Age': 35}
        new_row_df = pd.DataFrame([new_row_data])
        df_with_new_row = pd.concat([new_row_df, df], ignore_index=True)
        print(df_with_new_row)


        self.xmin = -self.pipeNodeAxialSpacing  # a little extra space on the left
        self.xmax = self.pipeNodeAxialSpacing * 19  # 19 determined by trial and error
        self.ymin = self.shaftDepth - (self.pipeDepth / 5)  # show 1/5 of the pipe depth
        self.ymax = self.shaftDepth + self.pipeNodeAxialSpacing

        df.to_csv(filename, index=False)
        df.to_csv(filename, mode='a', index=False )
        print(f"Saved temperatures to {filename}")

    def runTransient(self, hours, dt=3600.0, csv_filename=None, selected_nodes=None):
        self.buildSteadyMatrices()
        self.buildCapacitance()
        N = len(self.Nodes)
        A_eff = self.K.tocsr().copy()
        for i in range(N): A_eff[i, i] += self.C[i] / dt
        T = np.zeros(N)
        for node in self.Nodes:
            if node.Type in ('flowUp', 'flowDown', 'fluid'):
                T[node.number] = self.T_initial_fluid
            elif node.Type == 'water':
                T[node.number] = self.T_initial_water
            elif node.Type == 'ground':
                T[node.number] = self.T_initial_ground
            else:
                T[node.number] = 0.0
        temps = np.zeros((hours + 1, N))
        temps[0] = T.copy()
        for n in range(hours): rhs = (self.C / dt) * T + self.b0; T = spsolve(A_eff, rhs); temps[n + 1] = T
        for node in self.Nodes: node.Temp = temps[-1, node.number]
        if csv_filename: self.saveTempsToCSV(temps, csv_filename, selected_nodes)
        self.temps = temps  # <<< store it on the System
        return temps

    def PrepareNextAnimationFrameData(self, current_frame, number_of_frames):
        self.AnimationCallback(current_frame, number_of_frames)





    def DrawPicture(self):
        self.draw_selected_nodes()



    def draw_selected_nodes(self):
        hour = self.current_display_hour
        T = self.temps
        color_map = {
            'flowDown': (1, 0, 0),
            'flowUp': (0.8, 0.2, 0.2),
            'water': (0, 0, 1),
            'ground': (0.2, 0.8, 0.2),
        }

        glClear(GL_COLOR_BUFFER_BIT)

        delta_x = self.pipeNodeAxialSpacing * 2  # double axial spacing looks good
        xval = 0
        for n in self.Nodes:  # note: we travel down columns first
            yval = self.shaftDepth - n.y  # flip the y direction for drawing the picture
            if n.y < (self.pipeDepth / 5):  # we are only going down to 1/5 the pipe depth
                if n.x < (self.innerGroundRadius * 24):  # only going outward a limited amount
                    # col = color_map.get(n.Type, (1.0, 1.0, 1.0))
                    glColor3f(1, 1, 1)
                    hf.drawText(n.Type, xval, yval, scale=1.5, center=True)
                    temp = T[hour, n.number]
                    col = temperature_to_rgb(temp, 30.9, 31)
                    glColor3f(*col)
                    gl2DCircle(xval, yval, radius=1, fill=True)
                    label = f"{temp:.3f}C"
                    glColor3f(1, 1, 1)
                    hf.drawText(label, xval, yval+1.25, scale=1.25, center=True)
            else:
                if yval == 0:  # we are at the bottom of the well
                    xval += delta_x  # move to the next column
            # endif n.y
            # endif

        glBegin(GL_LINE_STRIP)  # draw the viewing box for reference
        xmin = self.gl2D.glXmin  # these 4 lines are the reason we copied the gl2D into sys
        xmax = self.gl2D.glXmax  # right after we created the window
        ymin = self.gl2D.glYmin  # now we have access to the values sent to gl2D.setViewSize()
        ymax = self.gl2D.glYmax  # right after we created the window
        glVertex2f(xmin, ymin)
        glVertex2f(xmin, ymax)
        glVertex2f(xmax, ymax)
        glVertex2f(xmax, ymin)
        glVertex2f(xmin, ymin)
        glEnd()

    def AnimationCallback(self, frame, nframes):
        # calculations needed to configure the picture
        # these could be done here or by calling a class method

        self.current_display_hour = frame


    def ProcessFileData(self, data):
        # from the array of strings, fill the wing dictionary

        for line in data:  # loop over all the lines
            cells = line.strip().replace('(','').replace(')','').split(',')
            keyword = cells[0].strip().lower()

            if keyword == 'title': self.title = cells[1].replace("'", "")

            if keyword == 'params':
                self.spacing = float(cells[1])
                self.rowSize = float(cells[2])

            if keyword == 'row':
                t = []
                for i in range(2,len(cells)): #loop over all temperatures on the row
                    t.append(float(cells[i]))
                self.temps.append(t)

        self.ConnectData()


    def ConnectData(self):
        # required for the animator
        self.xmin = -self.spacing * 2
        self.xmax = self.rowSize * self.spacing * 1.05
        self.ymin = -self.spacing
        self.ymax = self.spacing * 6


        self.numberOfAnimationFrames = len(self.temps) - 1
        self.frame = 0 #for the animation frame on startup












    def read_nodal_temps_csv(self,csv_filename):



        self.xmin = -self.pipeNodeAxialSpacing  # a little extra space on the left
        # self.xmax = self.pipeNodeAxialSpacing * 19 # 19 determined by trial and error
        # self.ymin = self.shaftDepth - (self.pipeDepth / 5)  # show 1/5 of the pipe depth
        # self.ymax = self.shaftDepth + self.pipeNodeAxialSpacing
        # self.numberOfAnimationFrames = len(self.temps) - 1



        df = pd.read_csv(csv_filename)
        # extract time and temperature columns
        time_hr = df['time_hr'].to_numpy()
        temp_cols = [c for c in df.columns if c.startswith('node_')]
        temps = df[temp_cols].to_numpy()
        return time_hr, temps, temp_cols, df



def temperature_to_rgb(temp, t_min, t_max):
    if t_min >= t_max:
        raise ValueError("t_min must be less than t_max")

    # Normalize temperature to range [0.0, 1.0]
    t_norm = (temp - t_min) / (t_max - t_min)

    if t_norm <= 0.25:
        # Blue (0,0,1) → Cyan (0,1,1)
        ratio = t_norm / 0.25
        r = 0.0
        g = ratio
        b = 1.0

    elif t_norm <= 0.5:
        # Cyan (0,1,1) → Green (0,1,0)
        ratio = (t_norm - 0.25) / 0.25
        r = 0.0
        g = 1.0
        b = 1.0 - ratio

    elif t_norm <= 0.75:
        # Green (0,1,0) → Yellow (1,1,0)
        ratio = (t_norm - 0.5) / 0.25
        r = ratio
        g = 1.0
        b = 0.0

    else:
        # Yellow (1,1,0) → Red (1,0,0)
        ratio = (t_norm - 0.75) / 0.25
        r = 1.0
        g = 1.0 - ratio
        b = 0.0

    return (round(r, 3), round(g, 3), round(b, 3))










if __name__ == "__main__":
    sys = System()
    sys.ReadNetworkData()
    sys.createRadialNodeLocations()
    sys.createAxialNodeLocations()
    sys.createNodes()
    sys.linkNodes()
    temps = sys.runTransient(24,
                             dt=3600.0,
                             csv_filename="nodal_temps.csv",
                             selected_nodes=None)
    sys.current_display_hour = 0

    gl2D1 = gl2D(None, sys.draw_selected_nodes, width=1200, height=600)
    sys.gl2D = gl2D1  # save the gl2D object in "sys" so we have access to the gl2D variables later

    gl2D1.setViewSize(-sys.pipeNodeAxialSpacing,  # a little extra space on the left
                     sys.pipeNodeAxialSpacing * 19,  # 19 determined by trial and error
                     sys.shaftDepth - (sys.pipeDepth / 5),  # show 1/5 of the pipe depth
                     sys.shaftDepth + sys.pipeNodeAxialSpacing,
                     # add a little extra space at the top allowDistortion=False
                     allowDistortion=False
                     )
    gl2D1.glWait()

    gl2D2 = gl2D(None, sys.draw_selected_nodes, width=1200, height=600)
    sys.gl2D = gl2D2
    gl2D2.setViewSize(-sys.pipeNodeAxialSpacing,  # a little extra space on the left
                     sys.pipeNodeAxialSpacing * 19,  # 19 determined by trial and error
                     sys.shaftDepth - (sys.pipeDepth / 5),  # show 1/5 of the pipe depth
                     sys.shaftDepth + sys.pipeNodeAxialSpacing,
                     # add a little extra space at the top allowDistortion=False
                     allowDistortion=False
                     )

    nframes = len(sys.temps)-1
    gl2D2.glStartAnimation(sys.AnimationCallback, nframes, delaytime=1,
                          reverse=True, repeat=False, reset=True)

    gl2D2.glWait()
    print("hello")
