from color import Color
from itertools import product
from geometry import Vector
import random

class Shape:
    def __init__(self, color=None):
        self.color = color if color is not None else Color()
    def draw(self, image, super_sampling = 6):
        color = self.color
        r = float(image.resolution)
        jitter = [Vector((x + random.random()) / super_sampling / r,
                         (y + random.random()) / super_sampling / r)
                  for (x, y) in product(xrange(super_sampling), repeat=2)]
        lj = len(jitter)
        total_pixels = 0
        tb = self.bound
        if not tb.overlaps(image.bounds()):
            return
        l_x = max(0, int(tb.low.x * r))
        l_y = max(0, int(tb.low.y * r))
        h_x = min(r-1, int(tb.high.x * r))
        h_y = min(r-1, int(tb.high.y * r))
        corners = [(0, 0), (1.0/r, 0), (0, 1.0/r), (1.0/r, 1.0/r)]
        for y in xrange(l_y, int(h_y+1)):
            x = l_x
            while x <= h_x:
                corner = Vector(x / r, y / r)
                b = self.signed_distance_bound(corner) * r
                sqrt2 = 2 ** 0.5
                if b > sqrt2:
                    steps = int(b - (sqrt2 - 1))
                    for x_ in xrange(x, min(x + steps, int(h_x+1))):
                        image.pixels[y][x_].draw(color)
                    x += steps
                    total_pixels += min(x + steps, int(h_x+1)) - x
                    continue
                elif b < -sqrt2:
                    steps = int(-b - (sqrt2 - 1))
                    x += steps
                    continue
                s = 0
                for x_, y_ in corners:
                    if self.contains(corner + Vector(x_, y_)):
                        s += 1
                if s == 4:
                    total_pixels += 1
                    image.pixels[y][x].draw(color)
                elif s > 0:
                    total_pixels += 1
                    coverage = 0
                    for j in jitter:
                        if self.contains(corner + j):
                            coverage += 1.0
                    image.pixels[y][x].draw(color.fainter(coverage / lj))
                x += 1
