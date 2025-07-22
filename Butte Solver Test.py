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

    def get_pipe_wall_resistance(self, dz=None):
        if dz is None:
            dz = self.pipeNodeAxialSpacing
        r_i = self.innerPipeDiameter / 2.0
        r_o = self.outerPipeDiameter / 2.0
        return np.log(r_o / r_i) / (2 * np.pi * self.k_pipe * dz)

    def addToMatrix(self, system):
        i = self.number

        # --- 1) RADIAL conduction / pipe‐wall ---
        if self.Type in ('flowUp', 'flowDown', 'fluid'):
            # fluid → water (pipe wall)
            water = self.right
            if water and water.Type == 'water':
                dz = system.pipeNodeAxialSpacing
                R_wall = system.get_pipe_wall_resistance(dz)
                G = 1.0 / R_wall
                j = water.number

                system.K[i, j] += -G
                system.K[i, i] += G

        elif self.Type == 'water':
            # water → all fluid nodes (back‐link) via pipe‐wall
            for fluid in getattr(self, 'radial_neighbors', []):
                dz = system.pipeNodeAxialSpacing
                R_wall = system.get_pipe_wall_resistance(dz)
                G = 1.0 / R_wall
                j = fluid.number

                system.K[i, j] += -G
                system.K[i, i] += G

            # water → ground
            ground = self.right
            if ground and ground.Type == 'ground':
                dr = ground.x - self.x
                r_face = 0.5 * (self.x + ground.x)
                dz = system.lowerWaterNodeAxialSpacing
                k_w = system.get_conductivity('water')
                A_r = 2.0 * np.pi * r_face * dz
                G = k_w * A_r / dr
                j = ground.number

                system.K[i, j] += -G
                system.K[i, i] += G

        elif self.Type == 'ground':
            # ground ↔ ground
            for neigh in (self.left, self.right):
                if neigh and neigh.Type == 'ground':
                    dr = abs(neigh.x - self.x)
                    r_face = 0.5 * (self.x + neigh.x)

                    # use actual axial span for this ground ring
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

                    system.K[i, j] += -G
                    system.K[i, i] += G

                if self.Type in ('flowUp', 'flowDown'):
                    # 3a) Upwind advection
                    F = system.m_dot_per_leg * system.cp_fluid
                    neighbor = self.up if self.Type == 'flowUp' else self.down
                    if neighbor:
                        j = neighbor.number
                        system.K[i, i] += F
                        system.K[i, j] -= F




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
        self.k_pipe = 1
        self.k_water = 1
        self.k_ground = 1
        self.k_fluid = 1
        self.rho_water = 1
        self.cp_water = 1
        self.u_axial_water = 1
        self.rho_fluid = 1
        self.cp_fluid = 1
        self.rho_ground = 1
        self.cp_ground = 1
        self.rho_pipe = 1
        self.cp_pipe = 1
        self.num_legs = 4
        self.m_dot_per_leg = 1
        self.h_conv = 1
        self.T_fluid_in = 30.0
        self.use_axial_ground = True
        self.K = None
        self.b0 = None
        self.C = None

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
        df.to_csv(filename, index=False)
        print(f"Saved temperatures to {filename}")

    def runTransient(self, hours, dt=3600.0, csv_filename=None, selected_nodes=None):
        self.buildSteadyMatrices()
        self.buildCapacitance()
        N = len(self.Nodes)
        A_eff = self.K.tocsr().copy()
        for i in range(N): A_eff[i, i] += self.C[i] / dt
        T = np.full(N, getattr(self, 'T_initial', 30.0))
        temps = np.zeros((hours + 1, N))
        temps[0] = T.copy()
        for n in range(hours): rhs = (self.C / dt) * T + self.b0; T = spsolve(A_eff, rhs); temps[n + 1] = T
        for node in self.Nodes: node.Temp = temps[-1, node.number]
        if csv_filename: self.saveTempsToCSV(temps, csv_filename, selected_nodes)
        return temps

    def draw_slice_full(self):
        """
        Draws a full radial slice at the 4th axial layer, showing all node types.
        """
        z_target = self.AxialNodeLocations[3]
        tol = self.pipeNodeAxialSpacing / 2.0
        nodes_slice = [n for n in self.Nodes if abs(n.y - z_target) < tol]

        color_map = {
            'flowDown': (1, 0, 0),
            'flowUp': (0.8, 0.2, 0.2),
            'water': (0, 0, 1),
            'ground': (0.2, 0.8, 0.2),
        }

        glClear(GL_COLOR_BUFFER_BIT)
        glPointSize(10.0)
        for n in nodes_slice:
            glColor3f(*color_map.get(n.Type, (1, 1, 1)))
            gl2DCircle(n.x, n.y, radius=0.1, fill=True)

        # optional labels
        glColor3f(1, 1, 1)
        for n in nodes_slice:
            hf.drawText(n.Type, n.x, n.y, scale=0.1)

    def draw_all_nodes(self):
        """
        Draws every node in the system across the full radial and axial extents.
        """
        color_map = {
            'flowDown': (1, 0, 0),
            'flowUp': (0.8, 0.2, 0.2),
            'water': (0, 0, 1),
            'ground': (0.2, 0.8, 0.2),
        }

        glClear(GL_COLOR_BUFFER_BIT)
        glPointSize(10.0)
        for n in self.Nodes:
            col = color_map.get(n.Type, (1.0, 1.0, 1.0))
            glColor3f(*col)
            gl2DCircle(n.x, self.shaftDepth - n.y, radius=0.1, fill=True)

        # optional labels
        glColor3f(1, 1, 1)
        for n in self.Nodes:
            hf.drawText(n.Type, n.x, self.shaftDepth - n.y, scale=0.25)

    def draw_selected_nodes(self):

        color_map = {
            'flowDown': (1, 0, 0),
            'flowUp': (0.8, 0.2, 0.2),
            'water': (0, 0, 1),
            'ground': (0.2, 0.8, 0.2),
        }

        glClear(GL_COLOR_BUFFER_BIT)
        glPointSize(10.0)

        delta_x = self.innerGroundRadius * 2 / 42.0
        xval = 0
        for n in self.Nodes:
            if n.y < (self.pipeDepth / 5):
                if n.x < (self.innerGroundRadius * 2):
                    col = color_map.get(n.Type, (1.0, 1.0, 1.0))
                    glColor3f(*col)
                    gl2DCircle(xval, self.shaftDepth - n.y, radius=1, fill=True)
                    glColor3f(1, 1, 1)
                    hf.drawText(n.Type, xval, self.shaftDepth - n.y, scale=2.5)


            else:
                xval += delta_x

if __name__ == "__main__":
    sys = System()
    sys.ReadNetworkData()
    sys.createRadialNodeLocations()
    sys.createAxialNodeLocations()
    sys.createNodes()
    sys.linkNodes()
    temps = sys.runTransient(24, 3600.0, 'temps_24h.csv', [0, 5, 10])

    # single slice
    #     gl2d = gl2D(None, sys.draw_slice_full, windowType="glfw")
    #     gl2d.setViewSize(
    #         0,
    #          sys.outerGroundRadius,
    #         14.5,
    #         15.5,
    #         allowDistortion=False
    #     )
    #     gl2d.glWait()
    #
    #     # All-nodes visualization
    #     gl2d = gl2D(None, sys.draw_all_nodes, windowType="glfw")
    #     gl2d.setViewSize( 0, sys.outerGroundRadius, 0,
    #         sys.shaftDepth,
    #         allowDistortion=False
    #     )
    #     gl2d.glWait()

    gl2d = gl2D(None, sys.draw_selected_nodes, windowType="glfw")
    gl2d.setViewSize(0, (sys.innerGroundRadius * 2 / 42.0) * 7,
                     sys.shaftDepth - (sys.pipeDepth / 5), sys.shaftDepth,
                     allowDistortion=False
                     )
    gl2d.glWait()

