from geometry import identity
from shape import Shape

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
        for shape in self.traverse(identity()):
            shape.draw(image)
    def traverse(self, xform):
        this_xform = xform * self.transform
        for node in self.nodes:
            if isinstance(node, Scene):
                for n in node.traverse(this_xform):
                    yield n
            elif isinstance(node, Shape):
                yield node.transform(this_xform)
