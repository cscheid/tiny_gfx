import math
import array
import copy
from itertools import product
import random
import sys
import time

def sgn(x):
    return 0 if x == 0 else x / abs(x)

def quadratic(a, b, c):
    d = (b * b - 4 * a * c) ** 0.5
    if b >= 0:
        return (-b - d) / (2 * a), (2 * c) / (-b - d)
    else:
        return (2 * c) / (-b + d), (-b + d) / (2 * a)

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
    def dot(self, o):
        return self.x * o.x + self.y * o.y
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
    def signed_distance_bound(self, p):
        if self.contains(p):
            return 0
        if p.x < self.low.x:
            x = self.low.x - p.x
        elif p.x > self.high.x:
            x = p.x - self.high.x
        else:
            x = 0
        if p.y < self.low.y:
            y = self.low.y - p.y
        elif p.y > self.high.y:
            y = p.y - self.high.y
        else:
            y = 0
        return -(x ** 2 + y ** 2) ** 0.5
    def __repr__(self):
        return "b[%s %s]" % (self.low, self.high)

class HalfPlane:
    def __init__(self, p1, p2):
        self.a = -p2.y + p1.y
        self.b = p2.x - p1.x
        l = (self.a * self.a + self.b * self.b) ** 0.5
        self.c = (-self.b * p1.y - self.a * p1.x) / l
        self.a /= l
        self.b /= l

##############################################################################
# Shapes

class Shape:
    # def signed_distance_bound(self, p): pass
    # self.bound
    # def contains(self, p): pass
    # def transform(self, transform): pass
    def draw(self, image, color, super_sampling = 6):
        t = time.time()
        r = float(image.resolution)
        jitter = [Vector((x + random.random()) / super_sampling / r,
                         (y + random.random()) / super_sampling / r)
                  for (x, y) in product(xrange(super_sampling), repeat=2)]
        # jitter = [Vector(0,0)]
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
                # else:
                #     image.pixels[y][x].draw(Color(1,0,0,1))
                x += 1
        elapsed = time.time() - t
        print >>sys.stderr, "%s\t%s\t%.3f %.8f" % (self, total_pixels, elapsed, elapsed/(total_pixels+1))

# a *convex* poly, in ccw order of vertices, with no repeating vertices
# non-convex polys will break this
class Poly(Shape):
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
        sz = self.bound.size()
        if sz.y >= sz.x:
            max_index_y = max(enumerate(self.vs),
                              key=lambda (i,v): (v.y, v.x))[0]
            upward = self.vs[0:max_index_y+1]
            downward = self.vs[max_index_y:] + [self.vs[0]]
            mid_y = self.bound.midpoint().y
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

class Ellipse(Shape):
    def __init__(self, a=1.0, b=1.0, c=0.0, d=0.0, e=0.0, f=-1.0):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        # TODO
        self.bound = AABox(Vector(0,0),Vector(1,1))
        # kind of a terrible hack, but I'm running out of LOCs
        t = Transform(2 * a, c, 0, c, 2 * b, 0)
        self.center = t.inverse() * Vector(-d, -e)
        s = t.svd()
        a1 = (s[1].m[0][0] / 2) ** -0.5
        a2 = (s[1].m[1][1] / 2) ** -0.5
        self.axes = [Vector(s[0].m[0][0], s[0].m[1][0]) * a1,
                     Vector(s[0].m[0][1], s[0].m[1][1]) * a2]
    def value(self, p):
        return self.a * p.x * p.x + self.b * p.y * p.y + self.c * p.x * p.y \
            + self.d * p.x + self.e * p.y + self.f
    def contains(self, p):
        return self.value(p) < 0
    def transform(self, transform):
        i = transform.inverse()
        ((m00, m01, m02), (m10, m11, m12),_) = i.m
        aa = self.a * m00 * m00 + self.b * m10 * m10 + self.c * m00 * m10
        bb = self.a * m01 * m01 + self.b * m11 * m11 + self.c * m01 * m11
        cc = 2 * self.a * m00 * m01 + 2 * self.b * m10 * m11 \
             + self.c * (m00 * m11 + m01 * m10)
        dd = 2 * self.a * m00 * m02 + 2 * self.b * m10 * m12 \
             + self.c * (m00 * m12 + m02 * m10) + self.d * m00 + self.e * m10
        ee = 2 * self.a * m10 * m02 + 2 * self.b * m11 * m12 \
             + self.c * (m01 * m12 + m02 * m11) + self.d * m01 + self.e * m11
        ff = self.a * m02 * m02 + self.b * m12 * m12 + self.c * m02 * m12 \
             + self.d * m02 + self.e * m12 + self.f
        return Ellipse(aa, bb, cc, dd, ee, ff)
    def signed_distance_bound(self, p):
        v = -sgn(self.value(p))
        c = self.center
        pc = p - c
        u2 = self.a * pc.x ** 2 + self.b * pc.y ** 2 + self.c * pc.x * pc.y
        u1 = self.a * 2 * c.x * pc.x + self.b * 2 * c.y * pc.y \
             + self.c * c.y * pc.x + self.c * c.x * pc.y + self.d * pc.x \
             + self.e * pc.y
        u0 = self.a * c.x ** 2 + self.b * c.y ** 2 + self.c * c.x * c.y \
             + self.d * c.x + self.e * c.y + self.f
        sols = quadratic(u2, u1, u0)
        crossings = c + pc * sols[0], c + pc * sols[1]
        if (p - crossings[0]).length() < (p - crossings[1]).length():
            surface_pt = crossings[0]
        else:
            surface_pt = crossings[1]
        d = Vector(2 * self.a * surface_pt.x + self.c * surface_pt.y + self.d,
                   2 * self.b * surface_pt.y + self.c * surface_pt.x + self.e)
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
        
class PPMImage:
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

class Scene:
    def __init__(self, nodes=[], transform=None):
        if transform is None:
            transform = identity()
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
            elif isinstance(node, ShapeGrob):
                yield node.transform(this_xform)
        
################################################################################
# affine 2D transforms, encoded by a 3x3 matrix in homogeneous coordinates
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
    def transpose(self):
        t = self.m
        return Transform(t[0][0], t[1][0], 0,
                         t[0][1], t[1][1], 0)
    def svd(self):
        selft = self.transpose()
        Su = self * selft
        phi = 0.5 * math.atan2(Su.m[0][1] + Su.m[1][0], Su.m[0][0] - Su.m[1][1])
        U = rotate(phi)
        Sw = selft * self
        theta = 0.5 * math.atan2(Sw.m[0][1] + Sw.m[1][0], Sw.m[0][0] - Sw.m[1][1])
        W = rotate(theta)
        SUsum = Su.m[0][0] + Su.m[1][1]
        SUdif = ((Su.m[0][0] - Su.m[1][1]) ** 2 + 4 * Su.m[0][1] * Su.m[1][0]) ** 0.5
        SIG = scale(((SUsum + SUdif) / 2) ** 0.5, ((SUsum - SUdif) / 2) ** 0.5)
        S = U.transpose() * self * W
        C = scale(sgn(S.m[0][0]), sgn(S.m[1][1]))
        V = W * C
        return U * C, SIG, W
    def __str__(self):
        return "(%.3f %.3f %.3f)\n(%.3f %.3f %.3f)" % tuple(self.m[0] + self.m[1])
    def __repr__(self):
        return "(%.3f %.3f %.3f)\n(%.3f %.3f %.3f)" % tuple(self.m[0] + self.m[1])
        
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

################################################################################
        
class ShapeGrob:
    def __init__(self, shape, color):
        self.shape = shape
        self.color = color
    def contains(self, p):
        return self.shape.contains(p)
    def draw(self, image):
        self.shape.draw(image, self.color)
    def transform(self, transform):
        return ShapeGrob(self.shape.transform(transform), self.color)

def TransformedShapeGrob(xform, shape, color):
    xform_shape = shape.transform(xform)
    return ShapeGrob(xform_shape, color)
