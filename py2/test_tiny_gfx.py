from tiny_gfx import *
import sys
import cProfile

def do_it():
    f = open(sys.argv[1], 'w')
    i = Image(8, Color(0,0,0,1))
    
    grobs = [
        ShapeGrob(Poly(Vector(0.2,0.1), Vector(0.9,0.3), Vector(0.1,0.4)),
                  Color(1,0,0,0.5)),
        ShapeGrob(Poly(Vector(0.5,0.2), Vector(1,0.4), Vector(0.2,0.7)),
                  Color(0,1,0,0.5)),
        ShapeGrob(Rectangle(Vector(0.1,0.7), Vector(0.6,0.8)),
                  Color(0,0.5,0.8,0.5)),
        ShapeGrob(Poly(Vector(-1, 0.6), Vector(0.2, 0.8), Vector(-2, 0.7)),
                  Color(1,0,1,0.9)),
        ShapeGrob(Circle(Vector(0.5,0.9), 0.2), Color(1,1,0,1)),
        TransformedShapeGrob(around(Vector(0.9, 0.5), scale(1.0, 0.5)),
                             Circle(Vector(0.9, 0.5), 0.2),
                             Color(0,1,1,0.5))
        ]
    for grob in grobs:
        grob.draw(i)

    i.write_ppm(f)
    f.close()
    
if __name__ == '__main__':
    do_it()
    # cProfile.run("do_it()", sort="time")
