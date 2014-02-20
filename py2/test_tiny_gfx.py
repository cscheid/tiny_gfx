from tiny_gfx import *
import sys
import cProfile
import random

def do_it():
    f = open(sys.argv[1], 'w')
    i = PPMImage(512, Color(1,1,1,1))
    s = Scene()
    s3 = Scene()
    s3.add(Grob(Union(Circle(Vector(0.3, 0.1), 0.1),
                      Circle(Vector(0.35, 0.1), 0.1)),
                Color(0,0,0,0.5)))
    s3.add(Grob(Intersection(Circle(Vector(0.3, 0.3), 0.1),
                             Circle(Vector(0.35, 0.3), 0.1)),
                Color(0,0.5,0,1)))
    s3.add(Grob(Subtraction(Circle(Vector(0.3, 0.5), 0.1),
                            Circle(Vector(0.35, 0.5), 0.1)),
                Color(0,0,0.5,1)))
    s3.add(Grob(Subtraction(Circle(Vector(0.35, 0.7), 0.1),
                            Circle(Vector(0.3, 0.7), 0.1)),
                Color(0,0.5,0.5,1)))
    s2 = Scene([
        Grob(LineSegment(Vector(0.0,0), Vector(0.0,1), 0.01), Color(1,0,0,1)),
        Grob(LineSegment(Vector(0.1,0), Vector(0.1,1), 0.01), Color(1,0,0,1)),
        Grob(LineSegment(Vector(0.2,0), Vector(0.2,1), 0.01), Color(1,0,0,1)),
        Grob(LineSegment(Vector(0.3,0), Vector(0.3,1), 0.01), Color(1,0,0,1)),
        Grob(LineSegment(Vector(0.4,0), Vector(0.4,1), 0.01), Color(1,0,0,1)),
        Grob(LineSegment(Vector(0.5,0), Vector(0.5,1), 0.01), Color(1,0,0,1)),
        Grob(LineSegment(Vector(0.6,0), Vector(0.6,1), 0.01), Color(1,0,0,1)),
        Grob(LineSegment(Vector(0.2,0.1), Vector(0.7, 0.4), 0.01), Color(0,0.5,0,1))])
    for grob in [
        Grob(Poly(Vector(0.2,0.1), Vector(0.9,0.3), Vector(0.1,0.4)),
             Color(1,0,0,0.5)),
        Grob(Poly(Vector(0.5,0.2), Vector(1,0.4), Vector(0.2,0.7)),
             Color(0,1,0,0.5)),
        Grob(Rectangle(Vector(0.1,0.7), Vector(0.6,0.8)),
             Color(0,0.5,0.8,0.5)),
        Grob(Poly(Vector(-1, 0.6), Vector(0.2, 0.8), Vector(-2, 0.7)),
             Color(1,0,1,0.9)),
        Grob(Circle(Vector(0.5,0.9), 0.2), Color(0,0,0,1)),
        Grob(Circle(Vector(0.9, 0.5), 0.2),
             Color(0,1,1,0.5)).transform(around(Vector(0.9, 0.5), scale(1.0, 0.5))),
        s2,
        Scene([s2], around(Vector(0.5, 0.5), rotate(math.radians(90)))),
        Scene([s3], translate(0.5, 0)),
        ]:
        s.add(grob)

    # s = Scene([Grob(Circle(Vector(0.5,0.9), 0.2), Color(0,0,0,1))])
    # s = Scene([Grob(Ellipse().transform(scale(0.5, 1)).transform(translate(0.5,0)), Color(0,0,0,1))])
    # s = Scene([Grob(Ellipse().transform(translate(0.5,0)).transform(scale(0.5, 1)), Color(0,0,0,1))])
    # s = Scene([Grob(Ellipse()
    #                      .transform(scale(0.1, 0.1))
    #                      .transform(translate(0.2, 0.2))
    #                      .transform(around(Vector(0.2, 0.2), scale(0.5, 1)))
    #                      ,Color(0,0,0,1))])

    s.draw(i)
    i.write_ppm(f)
    f.close()

def test_ellipse():
    Ellipse().transform(
        scale(1.5, 1)).transform(
        translate(0, 2)).transform(
        rotate(math.radians(45))).transform(
        scale(-1, 2))

def test_eigv():
    v = Transform(2, 0, 0, 0, 1, 0)
    print >>sys.stderr, v.eigv()

# def test_svd():
#     id = identity()
#     for i in xrange(10000):
#         t = Transform(random.normalvariate(0,1), random.normalvariate(0,1), 0,
#                       random.normalvariate(0,1), random.normalvariate(0,1), 0)
#         s = t.svd()
#         tt = s[0] * s[1] * s[2].transpose()
#         v = t * tt.inverse()
#         for l1, l2 in zip(id.m, v.m):
#             for v1, v2 in zip(l1, l2):
#                 if abs(v1 - v2) > 1e-5:
#                     print "Matrix failed!"
#                     print sx
#                     print t
#                     print tt
#                     print v1, v2
#                     print t.det()
#                     raise Exception("boo")
#     print "10000 svd tests passed with matrices from IID unit gaussians"

if __name__ == '__main__':
    test_ellipse()
    test_eigv()
    # test_svd()
    do_it()
    # cProfile.run("do_it()", sort="time")
