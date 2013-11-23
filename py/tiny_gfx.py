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

class Rectangle:
    def __init__(self, p1, p2): self.low, self.high = p1.min(p2), p1.max(p2)
    def midpoint(self): return (self.low + self.high) * 0.5
    def bounds(self): return self
    def contains(self, p):
        return (self.low.x <= self.x <= self.high.x and
                self.low.y <= self.y <= self.high.y)
    def overlaps(self, r):
        return not (r.low.x >= self.high.x or r.high.x <= self.low.x or
                    r.low.y >= self.high.y or r.high.y <= self.low.y)

class Triangle:
    def __init__(self, a, b, c):
        self.p1, self.p2, self.p3 = (a, b, c) if is_ccw(a, b, c) else (a, c, b)
    def bounds(self): return union(self.p1, self.p2, self.p3)
    def contains(self, p):
        return all([is_ccw(self.p1, self.p2, p),
                    is_ccw(self.p2, self.p3, p), is_ccw(self.p3, self.p1, p)])

class Color:
    def __init__(self, r, g, b, a=1): self.r, self.g, self.b, self.a = r,g,b,a
    def over(o, s):
        if s.a == o.a == 0.0: return s
        a = 1.0 - (1.0 - s.a) * (1.0 - o.a)
        u, v = s.a / a, 1 - s.a / a
        return Color(u * s.r + v * o.r, u * s.g + v * o.g, u * s.b + v * o.b, a)
    def as_ppm(s):
        def byte(v): return int(v ** (1.0 / 2.2) * 255)
        return bytes([byte(s.r * s.a), byte(s.g * s.a), byte(s.b * s.a)])

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
        self.resolution = 2 ** resolution
        self.pixels = [[bg for x in range(2 ** resolution)]
                       for y in range(2 ** resolution)]
    def __getitem__(self, address): return self.pixels[address.x][address.y]
    def __setitem__(self, address, color):
        self.pixels[address.x][address.y] = color
    def write_ppm(self, out):
        out.write("P6\n%s\n%s\n255\n" % (self.resolution, self.resolution))
        out.flush()
        for y in range(self.resolution-1, -1, -1):
            for x in range(self.resolution):
                out.buffer.write(self.pixels[x][y].as_ppm())

def union(b, *rest):
    b = b.bounds()
    for r in rest:
        r = r.bounds()
        b = Rectangle(b.low.min(r.low), b.high.max(r.high))
    return b

# true if p3 is to the left side of the vector going from p1 to p2
def is_ccw(p1, p2, p3): return (p2 - p1).cross(p3 - p1) > 0

# "Grob" for graphics object
class ShapeGrob:
    def __init__(self, shape, color):
        self.shape, self.color = shape, color
    def contains(self, p): return self.shape.contains(p)
    def pixel_color(self, address):
        if self.contains(address.bounds().midpoint()): return self.color
        else: return Color(0,0,0,0)

def iterate_pixels_inside(resolution, address, shape):
    if not shape.bounds().overlaps(address.bounds()): return
    if resolution > address.level:
        for child in address.split():
            for i in iterate_pixels_inside(resolution, child, shape):
                yield i
    else: yield address
    
def draw_on_image(resolution, grob, image):
    b = grob.shape.bounds()
    base = Address(0,0,0)
    for pixel in iterate_pixels_inside(resolution, base, grob.shape):
        image[pixel] = image[pixel].over(grob.pixel_color(pixel))
