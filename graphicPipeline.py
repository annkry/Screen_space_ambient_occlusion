import numpy as np

from SSAO import SSAO

class Fragment:
    def __init__(self, x : int, y : int, depth : float, interpolated_data):
        self.x = x
        self.y = y
        self.depth = depth
        self.pos_view_space = []
        self.normal_view_space = []
        self.color = []
        self.interpolated_data = interpolated_data

def edgeSide(p, v0, v1) : 
    return (p[0] - v0[0]) * (v1[1] - v0[1]) - (p[1] - v0[1]) * (v1[0] - v0[0])

def edgeSide3D(p, v0, v1) :
    return np.linalg.norm(np.cross(p[0: 3] - v0[0: 3], v1[0: 3] - v0[0: 3]))

class GraphicPipeline:
    def __init__ (self, width, height):
        self.width = width
        self.height = height
        # two pipelines 
        # first pipeline's output (G-buffer) = self.positions, self.normal, self.color
        # second pipeline's output = self.image
        self.positions = np.zeros((height, width, 3))
        self.normal = np.zeros((height, width, 3))
        self.color = np.zeros((height, width, 3))
        self.image = np.zeros((height, width, 3))
        self.depthBuffer = np.ones((height, width))
        self.ssao = SSAO(64) # the number of samples within the semisphere

    def VertexShader(self, vertices, data) :
        outputVertices = np.zeros((vertices.shape[0], 18))
        for i in range(vertices.shape[0]) :
            x = vertices[i][0]
            y = vertices[i][1]
            z = vertices[i][2]
            w = 1.0

            N = np.array([vertices[i][3], vertices[i][4], vertices[i][5]])
            V = data['cameraPosition'] - [x, y, z]
            L = data['lightPosition'] - [x, y, z]

            vec = np.array([[x], [y], [z], [w]])

            # we calculate the positions in view space
            vec_view_space = np.matmul(data['viewMatrix'], vec)

            vec = np.matmul(data['projMatrix'], np.matmul(data['viewMatrix'], vec))

            # we calculate the normals in view space
            N_view_space = np.array([[N[0]], [N[1]], [N[2]], [0.0]])

            N_view_space = np.matmul(data['viewMatrix'], N_view_space)

            N_view_space = N_view_space / np.linalg.norm(N_view_space)

            outputVertices[i][0] = vec[0] / vec[3]
            outputVertices[i][1] = vec[1] / vec[3]
            outputVertices[i][2] = vec[2] / vec[3]

            outputVertices[i][3] = N[0]
            outputVertices[i][4] = N[1]
            outputVertices[i][5] = N[2]

            outputVertices[i][6] = V[0]
            outputVertices[i][7] = V[1]
            outputVertices[i][8] = V[2]

            outputVertices[i][9] = L[0]
            outputVertices[i][10] = L[1]
            outputVertices[i][11] = L[2]

            outputVertices[i][12] = vec_view_space[0]
            outputVertices[i][13] = vec_view_space[1]
            outputVertices[i][14] = vec_view_space[2]

            outputVertices[i][15] = N_view_space[0]
            outputVertices[i][16] = N_view_space[1]
            outputVertices[i][17] = N_view_space[2]

        return outputVertices

    def Rasterizer(self, v0, v1, v2) :
        fragments = []

        # culling back face
        area = edgeSide(v0, v1, v2)
        area3D = edgeSide3D(v0, v1, v2)
        if area < 0 :
            return fragments

        # AABBox computation
        # compute vertex coordinates in screen space
        v0_image = np.array([0, 0])
        v0_image[0] = (v0[0] + 1.0) / 2.0 * self.width 
        v0_image[1] = ((v0[1] + 1.0) / 2.0) * self.height 

        v1_image = np.array([0, 0])
        v1_image[0] = (v1[0] + 1.0) / 2.0 * self.width 
        v1_image[1] = ((v1[1] + 1.0) / 2.0) * self.height 

        v2_image = np.array([0, 0])
        v2_image[0] = (v2[0] + 1.0) / 2.0 * self.width 
        v2_image[1] = (v2[1] + 1.0) / 2.0 * self.height 

        # compute the two point forming the AABBox
        A = np.min(np.array([v0_image, v1_image, v2_image]), axis = 0)
        B = np.max(np.array([v0_image, v1_image, v2_image]), axis = 0)

        # cliping the bounding box with the borders of the image
        max_image = np.array([self.width - 1, self.height - 1])
        min_image = np.array([0.0, 0.0])

        A  = np.max(np.array([A, min_image]), axis = 0)
        B  = np.min(np.array([B, max_image]), axis = 0)
        
        # cast bounding box to int
        A = A.astype(int)
        B = B.astype(int)
        # compensate rounding of int cast
        B = B + 1

        # for each pixel in the bounding box
        for j in range(A[1], B[1]) : 
           for i in range(A[0], B[0]) :
                x = (i + 0.5) / self.width * 2.0 - 1 
                y = (j + 0.5) / self.height * 2.0 - 1

                p = np.array([x, y])
                
                area0 = edgeSide(p, v0, v1)
                area1 = edgeSide(p, v1, v2)
                area2 = edgeSide(p, v2, v0)

                # test if p is inside the triangle
                if (area0 >= 0 and area1 >= 0 and area2 >= 0) : 
                    
                    # computing 2d barycentric coordinates
                    lambda0 = area1 / area
                    lambda1 = area2 / area
                    lambda2 = area0 / area
                    
                    one_over_z = lambda0 * 1 / v0[2] + lambda1 * 1 / v1[2] + lambda2 * 1 / v2[2]
                    z = 1 / one_over_z
                    
                    p = np.array([x, y, z])
                    
                    area0 = edgeSide3D(p, v0, v1)
                    area1 = edgeSide3D(p, v1, v2)
                    area2 = edgeSide3D(p, v2, v0)

                    lambda0 = area1 / area3D
                    lambda1 = area2 / area3D
                    lambda2 = area0 / area3D

                    # interpolation of the vectors N, V, L as well as position and normal in view space
                    interpolated_data = lambda0 * v0[3: len(v0)] + lambda1 * v1[3: len(v1)] + lambda2 * v2[3: len(v2)]
                    
                    # emiting Fragment
                    fragments.append(Fragment(i, j, z, interpolated_data))

        return fragments
    
    def fragmentShader(self, fragment, data):

        interpolated_data = fragment.interpolated_data
        N = np.array(interpolated_data[0:3])
        V = np.array(interpolated_data[3:6])
        L = np.array(interpolated_data[6:9])
        # deriving the position and the normal in view space
        fragment.pos_view_space = np.array(interpolated_data[9:12])
        fragment.normal_view_space = np.array(interpolated_data[12:15])
       
        # we need to normalize N, V, and L
        N = N / np.linalg.norm(N)
        V = V / np.linalg.norm(V)
        L = L / np.linalg.norm(L)

        fragment.normal = N
       
        modelColor = np.array([1, 1, 1])
        # intensity of the specular effect
        a = 64
        ki = 0.1
        kd = 0.9
        ks = 0.3

        ambient = 1
        diffuse = max(np.dot(L, N), 0)
        # reflected ray
        R = 2 * np.dot(L, N) * N - L
        specular = max(np.dot(R, V)**a, 0.0)

        phong = ambient * ki + diffuse * kd + specular * ks

        # we set the color of the fragment
        fragment.color = modelColor * phong 

    # smoothly interpolates x between 0 and 1 based on the range defined by k1 and k2
    def smoothstep(self, k1, k2, x):
        x = np.clip((x - k1) / (k2 - k1), 0.0, 1.0)
        return x * x * (3 - 2 * x)
    
    # this function ensures that x is limited between 0 and b
    def clamp(self, b, x):
        return max(min(b, x), 0)

    def SSAOShader(self, h, w, pos, color, normal, data): # calculating the ssao factor
        # we set the initial value of the number of occluded samples as 0.0
        ssao = 0.0

        # we create a tangent matrix
        tangent = np.array([normal[1], -normal[0], normal[2]])
        bitangent = np.cross(normal, tangent)
        TBN = np.array([tangent, bitangent, normal])

        for i in range(0, self.ssao.nbSamples):
            # sample in world space
            samplePos = np.matmul(TBN, self.ssao.samples[i])

            # transform sample to view space
            sampleVec =  np.array([samplePos[0], samplePos[1], samplePos[2], 0.0])
            samplePos = np.matmul(data['viewMatrix'], sampleVec)

            # pos is the fragment position in view space
            # sample is also in view space
            samplePos = pos + samplePos[:-1] * self.ssao.radius

            # we need to calculate the projected vector of the sample
            # first, we derive the x, y, and z coordinates of the sample
            [x, y, z] = samplePos

            # now, we obtain the projection vector by multiplying the sample and the projection matrix
            vec = np.array([x, y, z, 1.0])
            vec = np.matmul(data['projMatrix'], vec)
            # we need to divide the coordinates by the last w coordinate to maintain a point, so that we have the last coordinate equal to 1.0
            vec /= vec[3]

            # now, we derive the pixel coordinates of the sample point by performing the right scaling
            j_pix = int((vec[0] + 1.0) / 2.0 * self.width)
            i_pix = int((vec[1] + 1.0) / 2.0 * self.height)
            # we ensure that we stay within bounds
            j_pix = self.clamp(self.width - 1, j_pix)
            i_pix = self.clamp(self.height - 1, i_pix)

            # we derive the distance between the camera for the projected sample pixel
            samplePixelDepth = self.depthBuffer[i_pix][j_pix]
        
            # we also need to define a range check, the idea here is that when a fragment is far away from the fragment that is rendered 
            # in the same pixel as the sample point, then we want the range check to be close to 0.0 and at the same time, we say that there is no occlusion
            rangeCheck = self.smoothstep(0.0, 1.0, self.ssao.radius / abs(pos[2] - samplePixelDepth))
            # we calculate whether the occlusion occurs, in order to do so, we want to check if the current fragment that is rendered
            # in the same pixel is closer (than a sample) to the camera or not, the z-axis is not pointing to the camera, so if an object has a greater z-coordinate
            # than the sample z-coordinate, it is further from the camera (and the sample is closer to the camera) and the occlusion occurs.
            ssao += (1.0 if samplePixelDepth >= vec[2] else 0.0) * rangeCheck


        # here, if a fragment is very much occluded (high ssao), its color then shoud be close to 0.0 -- which is dark (black)
        self.image[h][w] = (1.0 - ssao / self.ssao.nbSamples) * color

    def SSAOBlurrShader(self, fragments, data):
        # we need to add padding for each column and each row (one '1' for both sides: left-hand and right-hand side)
        padding = np.pad(self.image, ((1, 1), (1, 1), (0, 0)), mode='constant')
        # we create a kernel of a size 3 x 3 with the coefficients - we will take the Gaussian filter of 9 points around a pixel (including the pixel itself)
        # the Gaussian kernel:
        #---------------------------
        #  1/48  |  4/48  |  1/48  |
        #---------------------------
        #  4/48  |  28/48 |  4/48 |
        #---------------------------
        #  1/48  |  4/48  |  1/48  |
        #---------------------------
        kernel = np.array([[1/48, 4/48,  1/48],
                           [4/48, 28/48, 4/48],
                           [1/48, 4/48,  1/48]])
    
        for h in range(self.height):
            for w in range(self.width):
                for c in range(0, 3):
                    # here, we would calculate the blurring effect
                    self.image[h][w][c] = np.sum(padding[h:h + 3, w:w + 3, c] * kernel)
    

    def draw(self, vertices, triangles, data): # data contains all the matrices
        # calling vertex shader
        newVertices = self.VertexShader(vertices, data)
        
        fragments = []
        # calling Rasterizer
        for i in triangles :
            fragments.extend(self.Rasterizer(newVertices[i[0]], newVertices[i[1]], newVertices[i[2]]))
        
        for f in fragments:

            self.fragmentShader(f, data)

            # depth test
            if self.depthBuffer[f.y][f.x] > f.depth : 
                self.depthBuffer[f.y][f.x] = f.depth
                # here, we set up G-buffer, we need to store information about the fragment's position in view space and the normal in view space
                # as well as color associated with the (f.y, f.x) pixel
                self.positions[f.y][f.x] = f.pos_view_space
                self.normal[f.y][f.x] = f.normal_view_space
                self.color[f.y][f.x] = f.color
                # self.image[f.y][f.x] = f.color # this line is for generating the images with phong

        # we first perform the calculation of SSAO associated with each pixel present in the image, that is, its z-coordinate is not equal to 1.0
        for h in range(0, self.height):
            for w in range(0, self.width):
                if self.depthBuffer[h][w] != 1.0:
                    self.SSAOShader(h, w, self.positions[h][w], self.color[h][w], self.normal[h][w], data)

        # to mitigate the effect of atrifacts, we would use the blurring effect
        self.SSAOBlurrShader(fragments, data)