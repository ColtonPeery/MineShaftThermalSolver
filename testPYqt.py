import numpy as np
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from math import ceil
from OpenGL.GL import *
from OpenGL_2D_class_GLFW import gl2D, gl2DCircle
from HersheyFont import HersheyFont
hf = HersheyFont()

class Node:
    def __init__(self, x, y, ix, iy, number, type):
        self.Type = type
        self.rowNumber = number
        self.x = x
        self.y = y
        self.ix = ix
        self.iy = iy
        self.Temp = None
        self.up = None
        self.down = None
        self.left = None
        self.right = None

    def addToMatrix(self):
        pass


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
        
        # 1) clear any old values
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
        # find the lumped water‐node radius
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

    # def draw_slice(self):
    #     # pick your slice depth
    #     z_target = self.AxialNodeLocations[3]
    #     tol = self.pipeNodeAxialSpacing / 2.0
    #
    #     # all nodes at that depth
    #     nodes_slice = [n for n in self.Nodes if abs(n.y - z_target) < tol]
    #     # keep only ground
    #     ground_slice = [n for n in nodes_slice if n.Type == 'ground']
    #
    #     color_map = {
    #         'ground': (0.2, 0.8, 0.2),
    #     }
    #
    #     glClear(GL_COLOR_BUFFER_BIT)
    #     glPointSize(10.0)
    #     # draw ground nodes only
    #     for n in ground_slice:
    #         glColor3f(*color_map['ground'])
    #         gl2DCircle(n.x, n.y, radius=1, fill=True)
    #
    #     # optional labels for ground
    #     glColor3f(1, 1, 1)
    #     for n in ground_slice:
    #         hf.drawText(n.Type, n.x + 1, n.y + 1, scale=1)
    #
    # def draw_slice_inner_ground(self):
    #     # 1) pick your slice depth
    #     z_target = self.AxialNodeLocations[3]
    #     tol = self.pipeNodeAxialSpacing / 2.0
    #
    #     # 2) set radial max to exactly innerGroundRadius
    #     r_max = self.innerGroundRadius
    #
    #     # 3) filter nodes at that depth AND inside [0, r_max]
    #     nodes_slice = [
    #         n for n in self.Nodes
    #         if abs(n.y - z_target) < tol
    #            and 0.0 <= n.x <= r_max
    #     ]
    #
    #     # 4) draw them
    #     color_map = {
    #         'flowDown': (1, 0, 0),
    #         'flowUp': (0.8, 0.2, 0.2),
    #         'water': (0, 0, 1),
    #         'pipeWall': (0.5, 0.5, 0.5),
    #         'ground': (0.2, 0.8, 0.2),
    #     }
    #
    #     glClear(GL_COLOR_BUFFER_BIT)
    #     glPointSize(10.0)
    #     for n in nodes_slice:
    #         glColor3f(*color_map.get(n.Type, (1, 1, 1)))
    #         gl2DCircle(n.x, n.y, radius=0.0025, fill=True)
    #
    #     # optional labels
    #     glColor3f(1, 1, 1)
    #     for n in nodes_slice:
    #         hf.drawText(n.Type, n.x, n.y, scale=0.0025)

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


        delta_x = self.innerGroundRadius*2 / 42.0
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
# look for x,y for first and last node
#     gl2d = gl2D(None, sys.draw_slice_inner_ground, windowType="glfw")
#     gl2d.setViewSize(
#         sys.Nodes[0], sys.innerGroundRadius,
#         14.5,
#         15.5,
#         allowDistortion=False
#     )
#     gl2d.glWait()
#
#     gl2d = gl2D(None, sys.draw_slice, windowType="glfw")
#     gl2d.setViewSize(
#         0, sys.outerGroundRadius,
#            14.5,
#            15.5,
#         allowDistortion=False
#     )
#     gl2d.glWait()
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
    gl2d.setViewSize(0, (sys.innerGroundRadius*2/42.0)*7,
                    sys.shaftDepth - (sys.pipeDepth/5), sys.shaftDepth,
                     allowDistortion=False
                     )
    gl2d.glWait()
