# Updated add to matrix function to account for new linking scheme
class Node:
    def addToMatrix(self, system):
        i = self.number

        # --- 1) RADIAL conduction / pipe‐wall ---

        if self.Type in ('flowUp', 'flowDown', 'fluid'):
            # fluid → water (pipe wall)
            water = self.right
            if water and water.Type == 'water':
                dz     = system.pipeNodeAxialSpacing
                R_wall = system.get_pipe_wall_resistance(dz)
                G      = 1.0 / R_wall
                j      = water.number

                system.K[i, j] += -G
                system.K[i, i] +=  G

        elif self.Type == 'water':
            # water → all fluid nodes (back‐link) via pipe‐wall
            for fluid in getattr(self, 'radial_neighbors', []):
                dz     = system.pipeNodeAxialSpacing
                R_wall = system.get_pipe_wall_resistance(dz)
                G      = 1.0 / R_wall
                j      = fluid.number

                system.K[i, j] += -G
                system.K[i, i] +=  G

            # water → ground
            ground = self.right
            if ground and ground.Type == 'ground':
                dr     = ground.x - self.x
                r_face = 0.5 * (self.x + ground.x)
                dz     = system.lowerWaterNodeAxialSpacing
                k_w    = system.get_conductivity('water')
                A_r    = 2.0 * np.pi * r_face * dz
                G      = k_w * A_r / dr
                j      = ground.number

                system.K[i, j] += -G
                system.K[i, i] +=  G

        elif self.Type == 'ground':
            # ground ↔ ground
            for neigh in (self.left, self.right):
                if neigh and neigh.Type == 'ground':
                    dr     = abs(neigh.x - self.x)
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
                    G   = k_g * A_r / dr
                    j   = neigh.number

                    system.K[i, j] += -G
                    system.K[i, i] +=  G

