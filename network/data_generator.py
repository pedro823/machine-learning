import sys
import random
import math

def quad(x):
    return ((x[0] > 0 and x[1] < 0) or (x[0] < 0 and x[1] > 0))

def dist(x):
    return x[0]**2 + x[1]**2

def gen_quad():
    with open("quad_data.py", "w") as f:
        v = []
        for i in range(1000):
            p = [[random.random()*2-1, random.random()*2-1]]
            p[0].append(p[0][0]**2)
            p[0].append(p[0][1]**2)
            p[0].append(p[0][0]*p[0][1])
            p.append([1 if quad(p[0]) else -1])
            v.append(p)
        f.write(f"DATA = {v}\n")

def gen_circ():
    with open("circ_data.py", "w") as f:
        v = []
        for i in range(1000):
            theta = random.random()*6.2831
            r = random.random()
            p = [[r*math.cos(theta), r*math.sin(theta)]]
            p[0].append(p[0][0]**2)
            p[0].append(p[0][1]**2)
            p[0].append(p[0][0]*p[0][1])
            p.append([1 if r < 0.5 else -1])
            v.append(p)
        f.write(f"DATA = {v}\n")

if __name__ == '__main__':
    if sys.argv[1] == "quad":
        gen_quad()
    elif sys.argv[1] == "circ":
        gen_circ()
    else:
        print("Usage: python data_generator.py <quad/circ>")
