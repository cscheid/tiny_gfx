import math
import array
import copy
from itertools import product
import random
import sys

class Vector:
    def __init__(self, x, y): self.x, self.y = x, y
    def __add__(self, o): return Vector(self.x + o.x, self.y + o.y)
    def __sub__(self, o): return Vector(self.x - o.x, self.y - o.y)
    def __neg__(self): return Vector(-self.x, -self.y)
    def __mul__(self, k): return Vector(self.x * k, self.y * k)
    def cross(self, o): return self.x * o.y - self.y * o.x
    def min(self, o): return Vector(min(self.x, o.x), min(self.y, o.y))
    def max(self, o): return Vector(max(self.x, o.x), max(self.y, o.y))
    def __str__(self): return "v[%s, %s]" % (self.x, self.y)
    def __repr__(self): return "v[%s, %s]" % (self.x, self.y)
    def union(*args):
        return AABox(reduce(Vector.min, args), reduce(Vector.max, args))
    # true if v3 is to the right side of the vector going from v1 to v2
    def is_cw(v1, v2, v3): return (v2 - v1).cross(v3 - v1) < 0

# axis-aligned box
class AABox:
    def __init__(self, p1, p2): self.low, self.high = p1.min(p2), p1.max(p2)
    def midpoint(self): return (self.low + self.high) * 0.5
    def bounds(self): return self
    def size(self): return self.high - self.low
    def contains(s, p):
        return (s.low.x <= p.x <= s.high.x and s.low.y <= p.y <= s.high.y)
    def overlaps(self, r):
        return not (r.low.x >= self.high.x or r.high.x <= self.low.x or
                    r.low.y >= self.high.y or r.high.y <= self.low.y)
    def area(s): return (s.high.y - s.low.y) * (s.high.x - s.low.x)

# a *convex* poly, in ccw order of vertices, with no repeating vertices
# non-convex polys will break this
class Poly:
    def __init__(self, *ps):
        mn = min(enumerate(ps), key=lambda (i,v): (v.y, v.x))[0]
        self.vs = list(ps[mn:]) + list(ps[:mn])
        self.bound = None
    def bounds(self):
        if self.bound is None: self.bound = Vector.union(*self.vs)
        return self.bound
    def area(self):
        return sum(a.cross(b) for (a,b) in
                   zip(self.vs, self.vs[1:] + [self.vs[0]])) / 2.0
    def slop(self): return self.bounds().area() / self.area()
    def contains(s, p):
        l = len(s.vs)
        return not any(s.vs[i].is_cw(s.vs[(i+1) % l], p) for i in xrange(l))
    def flip(self): return Poly(*list(Vector(v.y, v.x) for v in self.vs[::-1]))
    def split(self):
        def split_chain(a, v, key):
            # based on http://hg.python.org/cpython/file/2.7/Lib/bisect.py
            lo, hi, kv = 0, len(a), key(v)
            while lo < hi:
                mid = (lo+hi)//2
                if key(a[mid]) < kv: lo = mid+1
                else: hi = mid
            if key(a[lo]) <> kv:
                u = (v.y - a[lo-1].y) / (a[lo].y - a[lo-1].y)
                v.x = u * a[lo].x + (1-u) * a[lo-1].x
                a.insert(lo, v)
            return lo
        sz = self.bounds().size()
        if sz.y >= sz.x:
            max_index_y = max(enumerate(self.vs),
                              key=lambda (i,v): (v.y, v.x))[0]
            upward = self.vs[0:max_index_y+1]
            downward = self.vs[max_index_y:] + [self.vs[0]]
            mid_y = self.bounds().midpoint().y
            i_u = split_chain(upward, Vector(None, mid_y), lambda k: k.y)
            i_d = split_chain(downward, Vector(None, mid_y), lambda k: -k.y)
            return [Quad(*(upward[:i_u+1] + downward[i_d:-1])),
                    Quad(*(upward[i_u:-1] + downward[:i_d+1]))]
        else: return list(x.flip() for x in self.flip().split())
Triangle, Quad = Poly, Poly

def Rectangle(v1, v2):
    return Quad(Vector(min(v1.x, v2.x), min(v1.y, v2.y)),
                Vector(max(v1.x, v2.x), min(v1.y, v2.y)),
                Vector(max(v1.x, v2.x), max(v1.y, v2.y)),
                Vector(min(v1.x, v2.x), max(v1.y, v2.y)))

def Circle(center=Vector(0,0), radius=1, k=64):
    d = math.pi * 2 / k
    lst = [center + Vector(math.cos(i*d), math.sin(i*d)) * radius
           for i in xrange(k)]
    return Poly(*lst)

class Color:
    def __init__(self, r, g, b, a=1): self.r, self.g, self.b, self.a = r,g,b,a
    def over(s, o):
        if s.a == o.a == 0.0: return s
        a = 1.0 - (1.0 - s.a) * (1.0 - o.a)
        u, v = s.a / a, 1 - s.a / a
        return Color(u * s.r + v * o.r, u * s.g + v * o.g, u * s.b + v * o.b, a)
    def __mul__(s, k): return Color(s.r * k, s.g * k, s.b * k, s.a * k)
    def as_ppm(s):
        def byte(v): return int(v ** (1.0 / 2.2) * 255)
        return "%c%c%c" % (byte(s.r * s.a), byte(s.g * s.a), byte(s.b * s.a))
        
class Image:
    def __init__(self, resolution, bg=Color(0,0,0,0)):
        self.resolution = 2 ** resolution
        self.pixels = [[bg for i in xrange(self.resolution)]
                       for j in xrange(self.resolution)]
    def bounds(self): return AABox(Vector(0,0), Vector(1,1))
    def __getitem__(self, a): return self.pixels[a.y][a.x]
    def __setitem__(self, a, color): self.pixels[a.y][a.x] = color
    def write_ppm(self, out):
        n = self.resolution
        out.write("P6\n%s\n%s\n255\n" % (n,n))
        for y, x in product(xrange(n-1,-1,-1), xrange(n)):
            out.write(self.pixels[y][x].as_ppm())

# affine 2D transforms, encoded by a matrix
# ( m11 m12 tx )
# ( m21 m22 ty )
# (  0   0   1 )
# vectors are to be interpreted as (vx vy 1)
class Transform:
    def __init__(self, m11, m12, tx, m21, m22, ty):
        self.m = [[m11, m12, tx], [m21, m22, ty], [0, 0, 1]]
    def __mul__(self, other): # ugly
        if isinstance(other, Transform):
            t = [[0] * 3 for i in xrange(3)]
            for i, j, k in product(xrange(3), repeat=3):
                t[i][j] += self.m[i][k] * other.m[k][j]
            return Transform(t[0][0], t[0][1], t[0][2],
                             t[1][0], t[1][1], t[1][2])
        else:
            nx = self.m[0][0] * other.x + self.m[0][1] * other.y + self.m[0][2]
            ny = self.m[1][0] * other.x + self.m[1][1] * other.y + self.m[1][2]
            return Vector(nx, ny)
    def det(s): return s.m[0][0] * s.m[1][1] - s.m[0][1] * s.m[1][0]
    def inverse(self):
        d = 1.0 / self.det()
        return Transform(d * self.m[1][1], -d * self.m[0][1], -self.m[0][2],
                         -d * self.m[1][0], d * self.m[0][0], -self.m[1][2])
    def __str__(self):
        return str(self.m)

def rotate(theta):
    s, c = math.sin(theta), math.cos(theta)
    return Transform(c, -s, 0, s, c, 0)
def translate(tx, ty): return Transform(1, 0, tx, 0, 1, ty)
def scale(x, y): return Transform(x, 0, 0, 0, y, 0)
def around(v, t): return translate(v.x, v.y) * t * translate(-v.x, -v.y)    

# "Grob" for graphics object, only supports solid colors for now
class Grob:
    def __init__(self, color):
        self.color = color
    def color_at(self, point):
        if self.contains(point): return self.color
        else: return Color(0,0,0,0)
with_ms = 0
without_ms = 0

class ShapeGrob(Grob):
    def __init__(self, shape, color):
        Grob.__init__(self, color)
        self.shape = shape
    def contains(self, p): return self.shape.contains(p)
    def draw(self, image):
        shapes, r = [self.shape], image.resolution
        jitter = [Vector(x + random.random(),
                         y + random.random()) * 0.25 * (1.0 / r)
                  for (x, y) in product(xrange(6), repeat=2)]
        # jitter = [Vector(0.5, 0.5)]
        ib = image.bounds()
        min_area = 4.0 / (image.resolution ** 2)
        global with_ms, without_ms
        while len(shapes):
            shape = shapes.pop()
            tb = shape.bounds()
            if not tb.overlaps(ib): continue
            if shape.slop() < 3 or tb.area() < min_area:
                l_x = max(0, int(tb.low.x * r))
                l_y = max(0, int(tb.low.y * r))
                h_x = min(r-1, int(tb.high.x * r))
                h_y = min(r-1, int(tb.high.y * r))
                for y, x in product(xrange(l_y, h_y+1), xrange(l_x, h_x+1)):
                    corner = Vector(x,y) * (1.0/r)
                    s = sum(1 for x,y in product([0,1.0/r], repeat=2)
                            if shape.contains(corner + Vector(x,y)))
                    if s == 4:
                        without_ms += 1
                        image.pixels[y][x] = self.color.over(image.pixels[y][x])
                    elif s > 0:
                        with_ms += 1
                        coverage = sum(1.0 for j in jitter
                                       if shape.contains(corner+j)) / len(jitter)
                        image.pixels[y][x] = (self.color * coverage).over(image.pixels[y][x])
            else:
                print >> sys.stderr, "SPLIT!"
                shapes.extend(shape.split())
        print >>sys.stderr, with_ms, without_ms
def TransformedShapeGrob(xform, shape, color):
    c = copy.copy(shape)
    c.vs = [xform * v for v in shape.vs]
    return ShapeGrob(c, color)

