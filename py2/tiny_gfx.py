import math
import array
import copy
from itertools import product
import random
import sys
import time

##############################################################################
# Geometry

class Vector:
    def __init__(self, *args):
        self.x, self.y = args
    def __add__(self, o):
        return Vector(self.x + o.x, self.y + o.y)
    def __sub__(self, o):
        return Vector(self.x - o.x, self.y - o.y)
    def __neg__(self):
        return Vector(-self.x, -self.y)
    def __mul__(self, k):
        return Vector(self.x * k, self.y * k)
    def cross(self, o):
        return self.x * o.y - self.y * o.x
    def min(self, o):
        return Vector(min(self.x, o.x), min(self.y, o.y))
    def max(self, o):
        return Vector(max(self.x, o.x), max(self.y, o.y))
    def union(*args):
        return AABox(reduce(Vector.min, args), reduce(Vector.max, args))
    def length(self):
        return (self.x * self.x + self.y * self.y) ** 0.5
    def __str__(self):
        return "v[%s, %s]" % (self.x, self.y)
    def __repr__(self):
        return "v[%s, %s]" % (self.x, self.y)

class AABox:
    def __init__(self, p1, p2):
        self.low = p1.min(p2)
        self.high = p1.max(p2)
    def midpoint(self):
        return (self.low + self.high) * 0.5
    def bounds(self):
        return self
    def size(self):
        return self.high - self.low
    def contains(self, p):
        return self.low.x <= p.x <= self.high.x and \
               self.low.y <= p.y <= self.high.y
    def overlaps(self, r):
        return not (r.low.x >= self.high.x or r.high.x <= self.low.x or
                    r.low.y >= self.high.y or r.high.y <= self.low.y)

class HalfPlane:
    def __init__(self, p1, p2):
        self.a = -p2.y + p1.y
        self.b = p2.x - p1.x
        l = (self.a * self.a + self.b * self.b) ** 0.5
        self.c = -self.b * p1.y - self.a * p1.x
        self.a /= l
        self.b /= l
        self.c /= l

##############################################################################
# Shapes

# a *convex* poly, in ccw order of vertices, with no repeating vertices
# non-convex polys will break this
class Poly:
    def __init__(self, *ps):
        mn = min(enumerate(ps), key=lambda (i,v): (v.y, v.x))[0]
        self.vs = list(ps[mn:]) + list(ps[:mn])
        self.bound = Vector.union(*self.vs)
        self.half_planes = []
        for i in xrange(len(self.vs)):
            h = HalfPlane(self.vs[i], self.vs[(i+1) % len(self.vs)])
            self.half_planes.append(h)
    def bounds(self):
        return self.bound
    def signed_distance_bound(self, p):
        plane = self.half_planes[0]
        min_inside = 1e30
        max_outside = -1e30
        for plane in self.half_planes:
            d = plane.a * p.x + plane.b * p.y + plane.c
            if d <= 0 and d > max_outside:
                max_outside = d
            if d >= 0 and d < min_inside:
                min_inside = d
        return max_outside if max_outside <> -1e30 else min_inside
    def contains(self, p):
        for plane in self.half_planes:
            if plane.a * p.x + plane.b * p.y + plane.c < 0:
                return False
        return True
    def flip(self):
        return Poly(*list(Vector(v.y, v.x) for v in self.vs[::-1]))
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
            return [Poly(*(upward[:i_u+1] + downward[i_d:-1])),
                    Poly(*(upward[i_u:-1] + downward[:i_d+1]))]
        else:
            return list(x.flip() for x in self.flip().split())
    def __str__(self):
        return "[ Poly %s ]" % len(self.vs)
    def transform(self, xform):
        return Poly(*(xform * v for v in self.vs))
    def draw(self, image, color):
        shape = self
        t = time.time()
        r = float(image.resolution)
        super_sampling = 6
        jitter = [Vector((x + random.random()) / super_sampling / r,
                         (y + random.random()) / super_sampling / r)
                  for (x, y) in product(xrange(super_sampling), repeat=2)]
        lj = len(jitter)
        total_pixels = 0
        tb = shape.bounds()
        if not tb.overlaps(image.bounds()):
            return
        l_x = max(0, int(tb.low.x * r))
        l_y = max(0, int(tb.low.y * r))
        h_x = min(r-1, int(tb.high.x * r))
        h_y = min(r-1, int(tb.high.y * r))
        corners = list(product([0,1.0/r], repeat=2))
        for y in xrange(l_y, int(h_y+1)):
            x = l_x
            while x <= h_x:
                corner = Vector(x / r, y / r)
                b = shape.signed_distance_bound(corner) * r
                if b > 1.414:
                    steps = int(b - 0.414)
                    for x_ in xrange(x, min(x + steps, int(h_x+1))):
                        image.pixels[y][x_].draw(color)
                    x += steps
                    total_pixels += min(x + steps, int(h_x+1)) - x
                    continue
                elif b < -1.414:
                    steps = int(-b - 0.414)
                    x += steps
                    continue
                s = 0
                for x_, y_ in corners:
                    if shape.contains(corner + Vector(x_, y_)):
                        s += 1
                if s == 4:
                    total_pixels += 1
                    image.pixels[y][x].draw(color)
                elif s > 0:
                    total_pixels += 1
                    coverage = 0
                    for j in jitter:
                        if shape.contains(corner + j):
                            coverage += 1.0
                    image.pixels[y][x].draw(color.fainter(coverage / lj))
                x += 1
        elapsed = time.time() - t
        print >>sys.stderr, "%s\t%s\t%.3f %.8f" % (shape, total_pixels, elapsed, elapsed/total_pixels)

Triangle, Quad = Poly, Poly

class HierarchicalPoly(Poly):
    def __init__(self, *ps):
        Poly.__init__(self, *ps)
        self.children = None
        self.split_until_simple()
    def contains(self, p):
        if not self.bound.contains(p):
            return False
        if self.children is None:
            return Poly.contains(self, p)
        return self.children[0].contains(p) or \
               self.children[1].contains(p)
    def signed_distance_bound(self, p):
        return 0
    def split_until_simple(self, max_points=5):
        if len(self.vs) <= max_points:
            return
        poly_children = self.split()
        self.children = [HierarchicalPoly(*poly_children[0].vs),
                         HierarchicalPoly(*poly_children[1].vs)]
    def __str__(self):
        return "HierarchicalPoly"
    def transform(self, xform):
        return HierarchicalPoly(*(xform * v for v in self.vs))

def Rectangle(v1, v2):
    return Quad(Vector(min(v1.x, v2.x), min(v1.y, v2.y)),
                Vector(max(v1.x, v2.x), min(v1.y, v2.y)),
                Vector(max(v1.x, v2.x), max(v1.y, v2.y)),
                Vector(min(v1.x, v2.x), max(v1.y, v2.y)))

def Circle(center=Vector(0,0), radius=1, k=256):
    d = math.pi * 2 / k
    lst = [center + Vector(math.cos(i*d), math.sin(i*d)) * radius
           for i in xrange(k)]
    return HierarchicalPoly(*lst)

def LineSegment(v1, v2, thickness):
    d = v2 - v1
    d.x, d.y = -d.y, d.x
    d *= thickness / d.length() / 2
    return Quad(v1 + d, v1 - d, v2 - d, v2 + d)
    
##############################################################################

class Color:
    def __init__(self, r, g, b, a=1):
        self.r = r
        self.g = g
        self.b = b
        self.a = a
    def draw(self, o):
        if self.a == o.a == 0.0:
            return
        if o.a == 1.0:
            self.r = o.r
            self.g = o.g
            self.b = o.b
            self.a = 1
        else:
            u = 1.0 - o.a
            self.r = u * self.r + o.a * o.r
            self.g = u * self.g + o.a * o.g
            self.b = u * self.b + o.a * o.b
            self.a = 1.0 - (1.0 - self.a) * (1.0 - o.a)
    def over(self, o):
        if self.a == o.a == 0.0:
            return self
        a = 1.0 - (1.0 - self.a) * (1.0 - o.a)
        u = self.a / a
        v = 1 - u
        return Color(u * self.r + v * o.r,
                     u * self.g + v * o.g,
                     u * self.b + v * o.b, a)
    def fainter(self, k):
        return Color(self.r, self.g, self.b, self.a * k)
    def as_ppm(self):
        def byte(v):
            return int(v ** (1.0 / 2.2) * 255)
        return "%c%c%c" % (byte(self.r * self.a),
                           byte(self.g * self.a),
                           byte(self.b * self.a))
    def __str__(self):
        return "c[%.3f %.3f %.3f %.3f]" % (self.r, self.g, self.b, self.a)
        
class Image:
    def __init__(self, resolution, bg=Color(0,0,0,0)):
        self.resolution = resolution
        self.pixels = []
        for i in xrange(self.resolution):
            lst = []
            for j in xrange(self.resolution):
                lst.append(copy.copy(bg))
            self.pixels.append(lst)
    def bounds(self):
        return AABox(Vector(0,0), Vector(1,1))
    def __getitem__(self, a):
        return self.pixels[a.y][a.x]
    def __setitem__(self, a, color):
        self.pixels[a.y][a.x] = color
    def write_ppm(self, out):
        n = self.resolution
        out.write("P6\n%s\n%s\n255\n" % (n,n))
        for y, x in product(xrange(n-1,-1,-1), xrange(n)):
            out.write(self.pixels[y][x].as_ppm())

################################################################################
# affine 2D transforms, encoded by a matrix
# ( m11 m12 tx )
# ( m21 m22 ty )
# (  0   0   1 )
# vectors are to be interpreted as (vx vy 1)

class Transform:
    def __init__(self, m11, m12, tx, m21, m22, ty):
        self.m = [[m11, m12, tx],
                  [m21, m22, ty],
                  [0, 0, 1]]
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
    def det(s):
        return s.m[0][0] * s.m[1][1] - s.m[0][1] * s.m[1][0]
    def inverse(self):
        d = 1.0 / self.det()
        return Transform(d * self.m[1][1], -d * self.m[0][1], -self.m[0][2],
                         -d * self.m[1][0], d * self.m[0][0], -self.m[1][2])
    def __str__(self):
        return str(self.m)

def rotate(theta):
    s, c = math.sin(theta), math.cos(theta)
    return Transform(c, -s, 0, s, c, 0)

def translate(tx, ty):
    return Transform(1, 0, tx, 0, 1, ty)

def scale(x, y):
    return Transform(x, 0, 0, 0, y, 0)

def around(v, t):
    return translate(v.x, v.y) * t * translate(-v.x, -v.y)

################################################################################
        
class ShapeGrob:
    def __init__(self, shape, color):
        self.shape = shape
        self.color = color
    def contains(self, p):
        return self.shape.contains(p)
    def draw(self, image):
        self.shape.draw(image, self.color)

def TransformedShapeGrob(xform, shape, color):
    xform_shape = shape.transform(xform)
    return ShapeGrob(xform_shape, color)
