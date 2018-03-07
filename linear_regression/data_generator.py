import random as r
random = r.SystemRandom()

samples = []
results = []

function = lambda x: x**2 + 10 * x + 10

for i in range(10000):
    x = random.random() * 100
    samples.append([x])
    results.append(function(x))

with open('examples/data_2d_quadratic.py', 'w') as f:
    f.write('INPUT = ' + str(samples) + '\n')
    f.write('RESULTS = ' + str(results) + '\n')
