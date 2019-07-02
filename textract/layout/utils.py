from enum import Enum

class LayoutType(Enum):
    Page = 0
    Paragraph = 1
    Line = 2
    Image = 3

class LayoutUtils():
    def __init__(self, id_, src, type_):
        self._id   = id_
        self._src  = src
        self._type = type_

    def get_id(self):
        return self._id

    def get_src(self):
        return self._src

    def get_type(self):
        return self._type

class Page(LayoutUtils):
    def __init__(self, id_, src):
        super().__init__(id_, src, LayoutType.Page)

class Paragraph(LayoutUtils):
    def __init__(self, id_, src):
        super().__init__(id_, src, LayoutType.Paragraph)

class Line(LayoutUtils):
    def __init__(self, id_, src):
        super().__init__(id_, src, LayoutType.Line)

class Image(LayoutUtils):
    def __init__(self, id_, src):
        super().__init__(id_, src, LayoutType.Image)

class BBox(object):
    def __init__(self, id_, x, y, w, h):
        self._id = id_
        self._coords = (x, y, x+w, y+h)

    def get_id(self):
        return self._id

    def get_coords(self):
        return self._coords
