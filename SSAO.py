import numpy as np
import random

class SSAO:

  def __init__(self, nbSamples) :
    self.nbSamples = nbSamples # the number of samples within the semisphere
    self.radius = 0.5
    self.samples = self.samplesInit(nbSamples)

  def lerp(self, a, b, t) :
    return a + t * (b - a)
  
  def samplesInit(self, nbSamples) :
    # generate nbSamples random samples
    # we need to generate random 3d vectors [x, y, z]
    # where x and y are from the set [-1, 1], z is from the set [0, 1] (semisphere)
    s_list = []
    for i in range(0, nbSamples):
      v = np.array([np.random.random(1)[0] * 2 - 1, np.random.random(1)[0] * 2 - 1, np.random.random(1)[0]])
      v = v / np.linalg.norm(v)
      v = v * np.random.random(1)[0]
      scale = (i / self.nbSamples) * (i / self.nbSamples)
      # we want to move possibly point to the [0,0,0] point
      s = self.lerp(0.1, 1.0, scale)
      v = v * s
      s_list.append(v)
      
    return s_list
  