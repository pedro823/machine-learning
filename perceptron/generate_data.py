import random
plane = lambda x,y,z: 5 + 5*x - 8*y + z
inputs = list()
results = list()

for i in range(5000):
    x = random.random() * 50 - 25
    y = random.random() * 50 - 25
    z = random.random() * 50 - 25
    f = plane(x, y, z)
    inputs.append((x, y, z))
    if f >= 0:
        results.append(True)
    else:
        results.append(False)

with open('data.py', 'w') as f:
    f.write('INPUT = ' + str(inputs) + '\n')
    f.write('RESULTS = ' + str(results) + '\n')
