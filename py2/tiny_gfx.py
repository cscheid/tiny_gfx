import math
import array
from itertools import product

class Vector:
    def __init__(self, x, y): self.x, self.y = x, y
    def __add__(self, o): return Vector(self.x + o.x, self.y + o.y)
    def __sub__(self, o): return Vector(self.x - o.x, self.y - o.y)
    def __neg__(self): return Vector(-self.x, -self.y)
    def __mul__(self, k): return Vector(self.x * k, self.y * k)
    def cross(self, o): return self.x * o.y - self.y * o.x
    def bounds(self): return Rectangle(self, self)
    def min(self, o): return Vector(min(self.x, o.x), min(self.y, o.y))
    def max(self, o): return Vector(max(self.x, o.x), max(self.y, o.y))
    def length(self): return (self.x ** 2 + self.y ** 2) ** 0.5
    def __str__(self): return "[%f, %f]" % (self.x, self.y)

class Rectangle:
    def __init__(self, p1, p2): self.low, self.high = p1.min(p2), p1.max(p2)
    def midpoint(self): return (self.low + self.high) * 0.5
    def bounds(self): return self
    def contains(self, p):
        return (self.low.x <= p.x <= self.high.x and
                self.low.y <= p.y <= self.high.y)
    def overlaps(self, r):
        return not (r.low.x >= self.high.x or r.high.x <= self.low.x or
                    r.low.y >= self.high.y or r.high.y <= self.low.y)
    def area(self): return (self.high.y - self.low.y) * (self.high.x - self.low.x)

class Triangle:
    def __init__(self, a, b, c):
        self.p1, self.p2, self.p3 = (a, b, c) if is_ccw(a, b, c) else (a, c, b)
        self.b = union(self.p1, self.p2, self.p3)
    def bounds(self): return self.b
    def contains(self, p):
        return all([is_ccw(self.p1, self.p2, p),
                    is_ccw(self.p2, self.p3, p), is_ccw(self.p3, self.p1, p)])

class Circle:
    def __init__(self, c, r):
        self.center, self.radius = c, r
        d = Vector(self.radius, self.radius)
        self.b = Rectangle(self.center - d, self.center + d)
    def bounds(self): return self.b
    def contains(self, p):
        return (p - self.center).length() <= self.radius

class Color:
    def __init__(self, r, g, b, a=1): self.r, self.g, self.b, self.a = r,g,b,a
    def over(o, s):
        if s.a == o.a == 0.0: return s
        a = 1.0 - (1.0 - s.a) * (1.0 - o.a)
        u, v = s.a / a, 1 - s.a / a
        return Color(u * s.r + v * o.r, u * s.g + v * o.g, u * s.b + v * o.b, a)
    def as_ppm(s):
        def byte(v): return int(v ** (1.0 / 2.2) * 255)
        return "%c%c%c" % (byte(s.r * s.a), byte(s.g * s.a), byte(s.b * s.a))

class Address:
    def __init__(self, x, y, level):
        self.x, self.y, self.level = x, y, level
    def bounds(self):
        return Rectangle(Vector(self.x, self.y) * 0.5 ** self.level,
                         Vector(self.x + 1, self.y + 1) * 0.5 ** self.level)
    def split(self):
        for dx, dy in ((0,0), (1, 0), (0,1), (1, 1)):
            yield Address(2 * self.x + dx, 2 * self.y + dy, self.level + 1)
    def __str__(self): return "[%s,%s,%s]" % (self.x, self.y, self.level)

class Image:
    def __init__(self, resolution, bg=Color(0,0,0,0)):
        self.resolution = resolution
        self.pixels = [[bg for i in xrange(2 ** resolution)]
                       for j in xrange(2 ** resolution)]
    def __getitem__(self, a):
        return self.pixels[a.y][a.x]
    def __setitem__(self, a, color):
        self.pixels[a.y][a.x] = color
    # def set(self, a, color):
    #     w = 2 ** max(0, self.resolution - a.level)
    #     n = self.npixels
    #     for y, x in product(xrange(w), xrange(w)):
    #         self.pixels[a.y * self.npixels + a.x] = color
    def write_ppm(self, out):
        n = 2 ** self.resolution
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
def union(b, *rest):
    b = b.bounds()
    for r in rest:
        r = r.bounds()
        b = Rectangle(b.low.min(r.low), b.high.max(r.high))
    return b

# true if p3 is to the left side of the vector going from p1 to p2
def is_ccw(p1, p2, p3): return (p2 - p1).cross(p3 - p1) > 0

def iterate_pixels_inside(resolution, address, shape):
    if not shape.bounds().overlaps(address.bounds()): return
    if resolution > address.level:
        for child in address.split():
            for i in iterate_pixels_inside(resolution, child, shape):
                yield i
    else: yield address

# "Grob" for graphics object
class ShapeGrob:
    def __init__(self, shape, color):
        self.shape, self.color = shape, color
        self.leaf = True
    def contains(self, p): return self.shape.contains(p)
    def color_at(self, point):
        if self.contains(point): return self.color
        else: return Color(0,0,0,0)
    
def draw(image, grob):
    resolution = image.resolution
    b = grob.shape.bounds()
    base = Address(0,0,0)
    for pixel in iterate_pixels_inside(resolution, base, grob.shape):
        image[pixel] = image[pixel].over(grob.color_at(pixel.bounds().midpoint()))
