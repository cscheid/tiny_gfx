from tiny_gfx import *
import sys
import cProfile

def do_it():
    i = Image(8, Color(0,0,0,1))
    ShapeGrob(Triangle(Vector(0.2,0.1), Vector(0.9,0.3), Vector(0.1,0.4)),
              Color(1,0,0,1)).draw_on_image(i)
    ShapeGrob(Triangle(Vector(0.5,0.2), Vector(1,0.4), Vector(0.2,0.7)),
              Color(0,1,0,0.5)).draw_on_image(i)
    ShapeGrob(Rectangle(Vector(0.1,0.7), Vector(0.6,0.8)),
              Color(0,0.5,0.8,0.5)).draw_on_image(i)
    ShapeGrob(Circle(Vector(0.5,0.9), 0.2),
              Color(1,1,0,1)).draw_on_image(i)
    f = open(sys.argv[1], 'w')
    i.write_ppm(f)
    f.close()
    
if __name__ == '__main__':
    do_it()
    # cProfile.run("do_it()", sort="time")
