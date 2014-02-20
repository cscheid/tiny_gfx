import math
import copy
from itertools import product
import random

def quadratic(a, b, c):
    d = (b * b - 4 * a * c) ** 0.5
    if b >= 0:
        return (-b - d) / (2 * a), (2 * c) / (-b - d)
    else:
        return (2 * c) / (-b + d), (-b + d) / (2 * a)

class Vector:
    def __init__(self, *args):
        self.x, self.y = args
    def __add__(self, o):
        return Vector(self.x + o.x, self.y + o.y)
    def __sub__(self, o):
        return Vector(self.x - o.x, self.y - o.y)
    def __mul__(self, k):
        return Vector(self.x * k, self.y * k)
    def dot(self, o):
        return self.x * o.x + self.y * o.y
    def min(self, o):
        return Vector(min(self.x, o.x), min(self.y, o.y))
    def max(self, o):
        return Vector(max(self.x, o.x), max(self.y, o.y))
    def union(*args):
        return AABox(reduce(Vector.min, args), reduce(Vector.max, args))
    def length(self):
        return (self.x * self.x + self.y * self.y) ** 0.5

class AABox:
    def __init__(self, p1, p2):
        self.low = p1.min(p2)
        self.high = p1.max(p2)
    def midpoint(self):
        return (self.low + self.high) * 0.5
    def size(self):
        return self.high - self.low
    def contains(self, p):
        return self.low.x <= p.x <= self.high.x and \
               self.low.y <= p.y <= self.high.y
    def overlaps(self, r):
        return not (r.low.x >= self.high.x or r.high.x <= self.low.x or
                    r.low.y >= self.high.y or r.high.y <= self.low.y)
    def intersection(self, other):
        return AABox(self.low.max(other.low), self.high.min(other.high))

class HalfPlane:
    def __init__(self, p1, p2):
        self.v = Vector(-p2.y + p1.y, p2.x - p1.x)
        l = self.v.length()
        self.c = -self.v.dot(p1) / l
        self.v = self.v * (1.0 / l)
    def signed_distance(self, p):
        return self.v.dot(p) + self.c

class Shape:
    def draw(self, image, color, super_sampling = 6):
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

class Poly(Shape): # a *convex* poly, in ccw order, with no repeating vertices
    def __init__(self, *ps):
        mn = min(enumerate(ps), key=lambda (i,v): (v.y, v.x))[0]
        self.vs = list(ps[mn:]) + list(ps[:mn])
        self.bound = Vector.union(*self.vs)
        self.half_planes = []
        for i in xrange(len(self.vs)):
            h = HalfPlane(self.vs[i], self.vs[(i+1) % len(self.vs)])
            self.half_planes.append(h)
    def signed_distance_bound(self, p):
        plane = self.half_planes[0]
        min_inside = 1e30
        max_outside = -1e30
        for plane in self.half_planes:
            d = plane.signed_distance(p)
            if d <= 0 and d > max_outside:
                max_outside = d
            if d >= 0 and d < min_inside:
                min_inside = d
        return max_outside if max_outside <> -1e30 else min_inside
    def contains(self, p):
        for plane in self.half_planes:
            if plane.signed_distance(p) < 0:
                return False
        return True
    def transform(self, xform):
        return Poly(*(xform * v for v in self.vs))

Triangle, Quad = Poly, Poly

def Rectangle(v1, v2):
    return Quad(Vector(min(v1.x, v2.x), min(v1.y, v2.y)),
                Vector(max(v1.x, v2.x), min(v1.y, v2.y)),
                Vector(max(v1.x, v2.x), max(v1.y, v2.y)),
                Vector(min(v1.x, v2.x), max(v1.y, v2.y)))

class Ellipse(Shape):
    def __init__(self, a=1.0, b=1.0, c=0.0, d=0.0, e=0.0, f=-1.0):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        t = Transform(2 * a, c, 0, c, 2 * b, 0)
        self.center = t.inverse() * Vector(-d, -e)
        l1, l0 = quadratic(1, 2 * (-a - b), 4 * a * b - c * c)
        v = t.eigv()
        axes = [v[0] * ((l0 / 2) ** -0.5), v[1] * ((l1 / 2) ** -0.5)]
        self.bound = Vector.union(self.center - axes[0] - axes[1],
                                  self.center - axes[0] + axes[1],
                                  self.center + axes[0] - axes[1],
                                  self.center + axes[0] + axes[1])
    def value(self, p):
        return self.a*p.x*p.x + self.b*p.y*p.y + self.c*p.x*p.y \
               + self.d*p.x + self.e*p.y + self.f
    def contains(self, p):
        return self.value(p) < 0
    def transform(self, transform):
        i = transform.inverse()
        ((m00, m01, m02), (m10, m11, m12),_) = i.m
        aa = self.a*m00*m00 + self.b*m10*m10 + self.c*m00*m10
        bb = self.a*m01*m01 + self.b*m11*m11 + self.c*m01*m11
        cc = 2*self.a*m00*m01 + 2*self.b*m10*m11 \
             + self.c*(m00*m11 + m01*m10)
        dd = 2*self.a*m00 * m02 + 2*self.b*m10*m12 \
             + self.c*(m00*m12 + m02*m10) + self.d*m00 + self.e*m10
        ee = 2*self.a*m10*m02 + 2*self.b*m11*m12 \
             + self.c*(m01*m12 + m02*m11) + self.d*m01 + self.e*m11
        ff = self.a*m02*m02 + self.b*m12*m12 + self.c*m02*m12 \
             + self.d*m02 + self.e*m12 + self.f
        return Ellipse(aa, bb, cc, dd, ee, ff)
    def signed_distance_bound(self, p):
        def sgn(x):
            return 0 if x == 0 else x / abs(x)
        v = -sgn(self.value(p))
        c = self.center
        pc = p - c
        u2 = self.a*pc.x**2 + self.b*pc.y**2 + self.c*pc.x*pc.y
        u1 = 2*self.a*c.x*pc.x + 2*self.b*c.y*pc.y \
             + self.c*c.y*pc.x + self.c*c.x*pc.y + self.d*pc.x \
             + self.e*pc.y
        u0 = self.a*c.x**2 + self.b*c.y**2 + self.c*c.x*c.y \
             + self.d*c.x + self.e*c.y + self.f
        sols = quadratic(u2, u1, u0)
        crossings = c+pc*sols[0], c+pc*sols[1]
        if (p - crossings[0]).length() < (p - crossings[1]).length():
            surface_pt = crossings[0]
        else:
            surface_pt = crossings[1]
        d = Vector(2*self.a*surface_pt.x + self.c*surface_pt.y + self.d,
                   2*self.b*surface_pt.y + self.c*surface_pt.x + self.e)
        return v * abs(d.dot(p - surface_pt) / d.length())

def Circle(center, radius):
    return Ellipse().transform(
        scale(radius, radius)).transform(
        translate(center.x, center.y))

def LineSegment(v1, v2, thickness):
    d = v2 - v1
    d.x, d.y = -d.y, d.x
    d *= thickness / d.length() / 2
    return Quad(v1 + d, v1 - d, v2 - d, v2 + d)

class CSG(Shape):
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2
    def transform(self, t):
        return self.__class__(self.v1.transform(t), self.v2.transform(t))

class Union(CSG):
    def __init__(self, v1, v2):
        CSG.__init__(self, v1, v2)
        self.bound = Vector.union(v1.bound.low, v1.bound.high,
                                  v2.bound.low, v2.bound.high)
    def contains(self, p):
        return self.v1.contains(p) or self.v2.contains(p)
    def signed_distance_bound(self, p):
        b1 = self.v1.signed_distance_bound(p)
        b2 = self.v2.signed_distance_bound(p)
        return b1 if b1 > b2 else b2

class Intersection(CSG):
    def __init__(self, v1, v2):
        CSG.__init__(self, v1, v2)
        self.bound = v1.bound.intersection(v2.bound)
    def contains(self, p):
        return self.v1.contains(p) and self.v2.contains(p)
    def signed_distance_bound(self, p):
        b1 = self.v1.signed_distance_bound(p)
        b2 = self.v2.signed_distance_bound(p)
        return b1 if b1 < b2 else b2

class Subtraction(CSG):
    def __init__(self, v1, v2):
        CSG.__init__(self, v1, v2)
        self.bound = self.v1.bound
    def contains(self, p):
        return self.v1.contains(p) and not self.v2.contains(p)
    def signed_distance_bound(self, p):
        b1 = self.v1.signed_distance_bound(p)
        b2 = -self.v2.signed_distance_bound(p)
        return b1 if b1 < b2 else b2

class Color:
    def __init__(self, r=0, g=0, b=0, a=1, rgb=None):
        self.rgb = rgb or (r, g, b)
        self.a = a
    def draw(self, o):
        if self.a == o.a == 0.0:
            return
        if o.a == 1.0:
            self.rgb = o.rgb
            self.a = 1
        else:
            u = 1.0 - o.a
            self.rgb = (u * self.rgb[0] + o.a * o.rgb[0],
                        u * self.rgb[1] + o.a * o.rgb[1],
                        u * self.rgb[2] + o.a * o.rgb[2])
            self.a = 1.0 - (1.0 - self.a) * (1.0 - o.a)
    def fainter(self, k):
        return Color(rgb=self.rgb, a=self.a*k)
    def as_ppm(self):
        def byte(v):
            return int(v ** (1.0 / 2.2) * 255)
        return "%c%c%c" % (byte(self.rgb[0] * self.a),
                           byte(self.rgb[1] * self.a),
                           byte(self.rgb[2] * self.a))

class PPMImage:
    def __init__(self, resolution, bg=Color()):
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

class Scene:
    def __init__(self, nodes=None, transform=None):
        if transform is None:
            transform = identity()
        if nodes is None:
            nodes = []
        self.transform = transform
        self.nodes = nodes
    def add(self, node):
        self.nodes.append(node)
    def draw(self, image):
        for grob in self.traverse(identity()):
            grob.draw(image)
    def traverse(self, xform):
        this_xform = xform * self.transform
        for node in self.nodes:
            if isinstance(node, Scene):
                for n in node.traverse(this_xform):
                    yield n
            elif isinstance(node, Grob):
                yield node.transform(this_xform)
        
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
    def eigv(self): # power iteration, ignores translation, assumes SPD
        a = Vector(random.random(), random.random())
        last = Vector(0,0)
        t = Transform(self.m[0][0], self.m[0][1], 0,
                      self.m[1][0], self.m[1][1], 0)
        while (a - last).length() > 1e-6:
            last = a
            a = t * a
            a = a * (1.0 / a.length())
        return a, Vector(-a.y, a.x)
        
def identity():
    return Transform(1, 0, 0, 0, 1, 0)

def rotate(theta):
    s = math.sin(theta)
    c = math.cos(theta)
    return Transform(c, -s, 0, s, c, 0)

def translate(tx, ty):
    return Transform(1, 0, tx, 0, 1, ty)

def scale(x, y):
    return Transform(x, 0, 0, 0, y, 0)

def around(v, t):
    return translate(v.x, v.y) * t * translate(-v.x, -v.y)
        
class Grob:
    def __init__(self, shape, color):
        self.shape = shape
        self.color = color
    def contains(self, p):
        return self.shape.contains(p)
    def draw(self, image):
        self.shape.draw(image, self.color)
    def transform(self, transform):
        return Grob(self.shape.transform(transform), self.color)
