import math
import random
# plane = lambda x,y,z: 5 + 5*x - 8*y + z
#
inputs = list()
results = list()
#
# def generate_random_superplane(dimensions=140):
#     plane = [random.random()*100 for i in range(dimesnsions + 1)]
#     def superplane(x):
#         # x is vector
#         s = plane[0]
#         for xi, pi in zip(x, plane):
#             s += xi*pi
#         return s
#     return superplane
#
# superplane = generate_random_superplane()
#
# for i in range(5000):
#
#     inputs.append((x, y, z))
#     if f >= 0:
#         results.append(True)
#     else:
#         results.append(False)

wobbly_plane = lambda x,y,z: math.sin(x) - 2 * math.cos(y) + z
for i in range(5000):
    x = random.random()*2*math.pi
    y = random.random()*2*math.pi
    z = random.random()*2 - 1
    f = wobbly_plane(x, y, z)
    inputs.append((x,y,z))
    if f >= 0:
        results.append(True)
    else:
        results.append(False)
with open('examples/wobbly.py', 'w') as f:
    f.write('INPUT = ' + str(inputs) + '\n')
    f.write('RESULTS = ' + str(results) + '\n')
