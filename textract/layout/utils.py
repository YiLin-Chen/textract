'''
util.py is the module that defines several basic unit for analyzing layout.
1. Page: a page of a document
2. Paragraph: text paragram in a page
3. Line: text line in a paragraph
4. Image: image block in a page
5. BBox: bounding box
'''

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

    # axis = 0 (x-axis), 1 (y-axis)
    def is_overlap(self, bbox, axis=0):
        coords = bbox.get_coords()

        return True if max(0, min(self._coords[2+axis], coords[2+axis]) - max(self._coords[0+axis], coords[0+axis])) else False

