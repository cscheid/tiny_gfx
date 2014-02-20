from tiny_gfx import *
import sys
import cProfile
import random

def do_it():
    f = open(sys.argv[1], 'w')
    i = PPMImage(512, Color(1,1,1,1))
    s = Scene()
    s2 = Scene([
        ShapeGrob(LineSegment(Vector(0.0,0), Vector(0.0,1), 0.01), Color(1,0,0,1)),
        ShapeGrob(LineSegment(Vector(0.1,0), Vector(0.1,1), 0.01), Color(1,0,0,1)),
        ShapeGrob(LineSegment(Vector(0.2,0), Vector(0.2,1), 0.01), Color(1,0,0,1)),
        ShapeGrob(LineSegment(Vector(0.3,0), Vector(0.3,1), 0.01), Color(1,0,0,1)),
        ShapeGrob(LineSegment(Vector(0.4,0), Vector(0.4,1), 0.01), Color(1,0,0,1)),
        ShapeGrob(LineSegment(Vector(0.5,0), Vector(0.5,1), 0.01), Color(1,0,0,1)),
        ShapeGrob(LineSegment(Vector(0.6,0), Vector(0.6,1), 0.01), Color(1,0,0,1)),
        ShapeGrob(LineSegment(Vector(0.2,0.1), Vector(0.7, 0.4), 0.01), Color(0,0.5,0,1))])

    for grob in [
        ShapeGrob(Poly(Vector(0.2,0.1), Vector(0.9,0.3), Vector(0.1,0.4)),
                  Color(1,0,0,0.5)),
        ShapeGrob(Poly(Vector(0.5,0.2), Vector(1,0.4), Vector(0.2,0.7)),
                  Color(0,1,0,0.5)),
        ShapeGrob(Rectangle(Vector(0.1,0.7), Vector(0.6,0.8)),
                  Color(0,0.5,0.8,0.5)),
        ShapeGrob(Poly(Vector(-1, 0.6), Vector(0.2, 0.8), Vector(-2, 0.7)),
                  Color(1,0,1,0.9)),
        ShapeGrob(Circle(Vector(0.5,0.9), 0.2), Color(0,0,0,1)),
        TransformedShapeGrob(around(Vector(0.9, 0.5), scale(1.0, 0.5)),
                             Circle(Vector(0.9, 0.5), 0.2),
                             Color(0,1,1,0.5)),
        s2,
        Scene([s2], around(Vector(0.5, 0.5), rotate(math.radians(90)))),
        ]:
        s.add(grob)

    # s = Scene([ShapeGrob(Circle(Vector(0.5,0.9), 0.2), Color(0,0,0,1))])
    # s = Scene([ShapeGrob(Ellipse().transform(scale(0.5, 1)).transform(translate(0.5,0)), Color(0,0,0,1))])
    # s = Scene([ShapeGrob(Ellipse().transform(translate(0.5,0)).transform(scale(0.5, 1)), Color(0,0,0,1))])
    # s = Scene([ShapeGrob(Ellipse()
    #                      .transform(scale(0.1, 0.1))
    #                      .transform(translate(0.2, 0.2))
    #                      .transform(around(Vector(0.2, 0.2), scale(0.5, 1)))
    #                      ,Color(0,0,0,1))])

    s.draw(i)
    i.write_ppm(f)
    f.close()

def test_svd():
    id = identity()
    for i in xrange(10000):
        t = Transform(random.normalvariate(0,1), random.normalvariate(0,1), 0,
                      random.normalvariate(0,1), random.normalvariate(0,1), 0)
        s = t.svd()
        tt = s[0] * s[1] * s[2].transpose()
        v = t * tt.inverse()
        for l1, l2 in zip(id.m, v.m):
            for v1, v2 in zip(l1, l2):
                if abs(v1 - v2) > 1e-5:
                    print "Matrix failed!"
                    print s
                    print t
                    print tt
                    print v1, v2
                    print t.det()
                    raise Exception("boo")
    print "10000 svd tests passed with matrices from IID unit gaussians"

if __name__ == '__main__':
    do_it()
    # test_svd()
    # cProfile.run("do_it()", sort="time")
