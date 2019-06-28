import cv2, argparse, os
import numpy as np
from pythonRLSA import rlsa
from collections import defaultdict
from textract.layout.utils import LayoutType, Page, Paragraph, Image, Line, BBox
from textract.model.classifier import BlockClassifier, ClassifierType

class Graph(object):
	def __init__(self, vertex_num):
		# adjacency List
		self._graph = defaultdict(list)
		self._nodes = [i for i in range(vertex_num)]

	def add_edge(self, u, v):
		self._graph[u].append(v) 

	def check_edge(self, u, v):
		return v in self._graph[u]

	def _topological_sort(self, v, visited, stack): 
		visited[v] = True

		for i in self._graph[v]: 
			if not visited[i]: 
				self._topological_sort(i, visited, stack) 

		stack.insert(0,v)

	def check_cyclic(self, u, v):
		visited = set([u])

		# dfs check cyclic
		stack = [v]
		while stack:
			node = stack.pop()
			if node in visited:
				return True
			
			if node not in self._graph:
				return False

			visited.add(node)

			for nei in self._graph[node]:
				stack.append(nei)

		return False
 
	def topological_sort(self): 
		visited = {}
		for i in self._nodes:
			visited[i] = False

		stack =[] 

		for i in self._nodes: 
			if not visited[i]: 
				self._topological_sort(i, visited, stack) 

		return stack 


class LayoutAnalyzer(object):
	def __init__(self):
		self._classifier = BlockClassifier(ClassifierType.CNN_VGG16)

	def extract(self, layout):
		if layout.get_type() == LayoutType.Page:
			return self._extract_boxes(layout)
		
		elif layout.get_type() == LayoutType.Paragraph:
			return self._extract_lines(layout)

	def _extract_boxes(self, layout):
		img_src = layout.get_src()

		# preprocess
		img_pre = self._preprocess(img_src)

		# rlsa
		img_rlsa = rlsa.rlsa(img_pre, True, True, 10)

		# dilation
		img_rlsa = 255 - img_rlsa # invert color
		kernel = np.ones((5,5), np.uint8)
		dilate = cv2.dilate(img_rlsa, kernel, iterations=2)

		# find bbox
		bboxes = self._calculate_bbox(dilate)

		# calculate order
		graph = self._build_graph(bboxes)
		order = graph.topological_sort()

		# extract subgraph box
		boxes = []
		for id_ in order:
			bbox = bboxes[id_]	
			x1, y1, x2, y2 = bbox.get_coords()
			subgraph = img_src[y1:y2, x1:x2]

			# cnn check type
			type_, _ = self._classifier.classify(subgraph)

			if type_ == 'Text':
				para_box = Paragraph(id_, subgraph)
				boxes.append(para_box)

			elif type_ == 'Image':
				img_box = Image(id_, subgraph)
				boxes.append(img_box)

		return boxes

	def _extract_lines(self, layout):
		img_src = layout.get_src()

		# preprocess
		img_pre = self._preprocess(img_src, inverse = True)

		# horizatidilation 
		kernel = np.ones((1,20), np.uint8)
		dilate = cv2.dilate(img_pre, kernel, iterations=3)

		# find find contours
		(contours, _) = cv2.findContours(dilate, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

		bboxes = self._calculate_bbox(dilate)
		bboxes.sort(key=lambda box:box.get_coords()[1]) # top to bottom

		lines = []
		for i in range(len(bboxes)):
			bbox = bboxes[i]	
			x1, y1, x2, y2 = bbox.get_coords()
			subgraph = img_src[y1:y2, x1:x2]

			lines.append(Line(i, subgraph))

		return lines

	def _preprocess(self, src, inverse = False):
		# gray scale
		src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

		# clean the image using otsu method with the inversed binarized image
		if inverse:
			_, src = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

		else:
			_, src = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

		return src

	def _calculate_bbox(self, src):
		img = src.copy()
		bboxes = []

		# get bounding box and keep merging until no intersection
		while True:
			(img_contours, _) = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
			
			# make mask 
			mask = np.zeros(img.shape[:2], dtype=np.uint8)

			for cnt in img_contours:
				# adding bouding box 
				x,y,w,h = cv2.boundingRect(cnt)           
				cv2.rectangle(mask, (x+1,y+1), (x+w-1,y+h-1), (255), -1)

			(mask_contours, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

			if len(img_contours) == len(mask_contours):
				# no intersection
				for i in range(len(mask_contours)):
					x, y, w, h = cv2.boundingRect(mask_contours[i])
					bbox = BBox(i, x, y, w, h)
					bboxes.append(bbox)
				break
			else:
				# still have intersection, keep iterating
				img = mask.copy()

		return bboxes

	def _build_graph(self, bboxes):
		'''
		1. Line segment a comes before line segment b 
			- if their ranges of x-coordinates overlap 
		and - if line segment a is above line segment b on the page
		
		2. Line segment a comes before line segment b 
			- if a is entirely to the left of b
		and - there does not exist a line segment c whose y-coordinates are between a and b and whose range of xcoordinates overlaps both a and b.
		'''

		graph = Graph(len(bboxes))

		# sort by (y, x) coordinate (top down -> left right)
		bboxes_sorty = sorted(bboxes, key = lambda box : (box.get_coords()[1], box.get_coords()[0]))
		lookup_table = {}

		# rule 1
		for i in range(len(bboxes_sorty)):
			bbox1 = bboxes_sorty[i]
			lookup_table[bbox1.get_id()] = i

			for j in range(i+1, len(bboxes_sorty)):
				bbox2 = bboxes_sorty[j]

				if self._find_overlap(bbox1, bbox2) and not graph.check_cyclic(bbox1.get_id(), bbox2.get_id()):
					graph.add_edge(bbox1.get_id(), bbox2.get_id())

		# sort by (x, y) coordinate (left right -> top down)
		bboxes_sortx = sorted(bboxes, key = lambda box : (box.get_coords()[0], box.get_coords()[1]))
		
		# rule 2
		for i in range(len(bboxes_sortx)):
			bbox1 = bboxes_sortx[i]

			for j in range(i+1, len(bboxes_sortx)):
				bbox2 = bboxes_sortx[j]

				if not self._find_overlap(bbox1, bbox2):
					is_neighbor = True
					for k in range(min(lookup_table[bbox1.get_id()], lookup_table[bbox2.get_id()]), max(lookup_table[bbox1.get_id()], lookup_table[bbox2.get_id()])):
						if self._find_overlap(bbox1, bboxes_sorty[k]) and self._find_overlap(bbox2, bboxes_sorty[k]):
							is_neighbor = False
							break

					if is_neighbor and not graph.check_cyclic(bbox1.get_id(), bbox2.get_id()):
						graph.add_edge(bbox1.get_id(), bbox2.get_id())

		return graph

	# axis = 0 (x-axis), 1 (y-axis)
	def _find_overlap(self, bbox1, bbox2, axis=0):
		coords_1 = bbox1.get_coords()
		coords_2 = bbox2.get_coords()

		return True if max(0, min(coords_1[2+axis], coords_2[2+axis]) - max(coords_1[0+axis], coords_2[0+axis])) else False
