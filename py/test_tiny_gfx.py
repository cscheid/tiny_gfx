from tiny_gfx import *
import sys

if __name__ == '__main__':
    i = Image(8, Color(0,0,0,1))
    g = ShapeGrob(Triangle(Vector(0.2,0.1), Vector(0.9, 0.3), Vector(0.1, 0.4)),
                  Color(1,0,0,1))
    draw_on_image(8, g, i)
    g = ShapeGrob(Triangle(Vector(0.5,0.2), Vector(1, 0.4), Vector(0.2, 0.7)),
                  Color(0,1,0,0.5))
    draw_on_image(8, g, i)
    with open(sys.argv[1], 'w') as f:
        i.write_ppm(f)
    
