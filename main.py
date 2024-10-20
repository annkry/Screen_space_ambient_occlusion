import numpy as np
from readply import readply

from graphicPipeline import GraphicPipeline
from camera import Camera
from projection import Projection

width = 1280 #400
height = 720 #270


pipeline = GraphicPipeline(width, height)

cameraPosition = np.array([1.1, 1.1, 1.1])
lookAt = np.array([-0.577, -0.577, -0.577])
up = np.array([0.33333333, 0.33333333, -0.66666667])
right = np.array([-0.57735027, 0.57735027, 0.])

cam = Camera(cameraPosition, lookAt, up, right)

nearPlane = 0.1
farPlane = 10.0
fov = 1.91986
aspectRatio = width / height

proj = Projection(nearPlane, farPlane, fov, aspectRatio) 


lightPosition = np.array([10, 0, 10])


vertices, triangles = readply('models/Suzanne.ply')

data = dict([
  ('viewMatrix', cam.getMatrix()),
  ('projMatrix', proj.getMatrix()),
  ('cameraPosition', cameraPosition),
  ('lightPosition', lightPosition),
])

pipeline.draw(vertices, triangles, data)

import matplotlib.pyplot as plt
# imgplot = plt.imshow(pipeline.normal)
# plt.show()
# imgplot = plt.imshow(pipeline.positions)
# plt.show()
imgplot = plt.imshow(pipeline.image)
plt.show()