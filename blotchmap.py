# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 10:49:53 2017

@author: Calvin Schmidt
blotchmap
draws a map composed of a bunch of shapless blotches
"""

import numpy as np

#Hex Graph section

class node:
	def __init__(self, coords=(0,0)):
		self.coords = coords
		self.edges = []
	def add_edge(self, e):
		if e in self.edges: return -1
		self.edges.append(e)
	def delete_edge(self, e):
		if not (e in self.edges): return -1
		self.edges.remove(e)
	def __str__(self):
		return str(self.coords)
	def __repr__(self):
		return str(self.coords)


class graph:
	def __init__(self):
		self.nodes = []
	def add_node(self, n):
		self.nodes.append(n)


class tri_node(node):
	NE, NW, W, SW, SE, E = (0,1,2,3,4,5)
	debug_id = 0
	def __init__(self, pos=(0,0), coords= (0,0)):
		node.__init__(self, coords)
		self.pos = pos
		self.edges = []
		self.debug_id = tri_node.debug_id
		tri_node.debug_id += 1
	def __str__(self):
		return "<" + str(self.debug_id) + ">"
	def __repr__(self):
		return "<" + str(self.debug_id) + ">"


class hex_node(node):
	N, NW, SW, S, SE, NE = (0,1,2,3,4,5)
	def __init__(self, coords=(0,0), pos=(0,0)):
		node.__init__(self, coords)
		self.pos = pos
		self.tri_nodes = []
		self.tri_node_directions = []
		self.edges = []
		self.edge_directions = []
		self.merged_nodes = []
		self.merged_node_directions = []

	def add_edge(self, hn, direction):
		if hn in self.edges: return -1
		self.edges.append(hn)
		self.edge_directions.append(direction)

	def get_edge(self, direction):
		for d in range(len(self.edge_directions)):
			if direction == self.edge_directions[d]:
				return self.edges[d]
		return None

	def add_merged(self, hn, direction):
		if hn in self.merged_nodes: return -1
		self.merged_nodes.append(hn)
		self.merged_node_directions.append(direction)

	def add_trinode(self, tn, direction):
		if tn in self.tri_nodes: return -1
		self.tri_nodes.append(tn)
		self.tri_node_directions.append(direction)

	def get_trinode(self, direction):
		for d in range(len(self.tri_node_directions)):
			if direction == self.tri_node_directions[d]:
				return self.tri_nodes[d]
		return None

	def delete_trinode(self, tn):
		if not (tn in self.tri_nodes): return -1
		for t in range(len(self.tri_nodes)):
			if tn == self.tri_nodes[t]:
				self.tri_nodes.pop(t)
				self.tri_node_directions.pop(t)
				break

	def mirror_tri_node(self, h, t):
		index = (t + 2 + 2*(t==h)) % 6
		if not (h == t or t == (h + 1) % 6):
			return -1
		if self.get_edge(h):
			return self.get_edge(h).get_trinode(index)
		return None

	def triangulate_hn(self, hn2_dir, hn3_dir):
		d12 = hn2_dir
		d13 = hn3_dir % 6
		d21 = d12 + 3 % 6
		d23 = d21
		if d13 == (d12 - 1) % 6:
			d23 += 1
		elif d13 == (d12 + 1) % 6:
			d23 -= 1
		else:
			return None#hn2 and hn3 must be adjacent
		return d23 % 6#the direction of hn3 from hn2's perspective


#grid coordinates are [i,j] where i designates row, and
#j designates column. (changing column is the one that zigzags)
class hex_grid:
	def __init__(self, shape=(1, 1), nnd=1):
		self.shape = shape
		self.node_array = np.empty(shape, dtype=object)
		self.trinode_pairs = None
		self.nnd = nnd#nearest neighbour distance
		self.pairs = None
		for j in range(self.shape[1]):
			for i in range(self.shape[0]):
				x = j*self.nnd*np.cos(np.pi/6)
				y = i*self.nnd
				#odd colums get offset vertically
				if j%2 == 1: y += self.nnd*np.sin(np.pi/6)
				position = (x,y)
				new_node = hex_node((i,j), position)
				self.node_array[i, j] = new_node
		for j in range(self.shape[1]):
			for i in range(self.shape[0]):
				edgs = np.zeros((6,2), dtype=int)
				edgs[0] = (i + 1, j) #N
				edgs[1] = (i + j%2, j - 1) #NW
				edgs[2] = (i - 1 + j%2, j - 1) #SW
				edgs[3] = (i - 1, j) #S
				edgs[4] = (i - 1 + j%2, j + 1) #SE
				edgs[5] = (i + j%2, j + 1) #NE
				for e in range(len(edgs)):
					if self.valid_coord(edgs[e]):
						self.node_array[i, j].add_edge(self.node_array[edgs[e][0], edgs[e][1]], e)
		self.create_tri_node_graph()
		self.pairs = np.array(get_node_pairs(self.node_array[0, 0]))
		self.trinode_pairs = np.array(get_node_pairs(self.node_array[0, 0].tri_nodes[0]))
		self.trinode_visible = np.full((self.trinode_pairs.shape[0]), True, dtype=bool)

	def valid_coord(self, coords):
		i, j = coords
		return (j >= 0 and
			j < self.shape[1] and
			i >= 0 and
			i < self.shape[0])

	def create_tri_node_graph(self):
		for j in range(self.shape[1]):
			for i in range(self.shape[0]):
				r = (2/(3**(1/2)))*(self.nnd/2)
				tn = np.array([1, 2, 3, 4, 5, 0])
				x = r*np.cos(tn*np.pi/3)
				y = r*np.sin(tn*np.pi/3)

				node = self.node_array[i, j]
				for t in range(len(tn)):
					if node.mirror_tri_node(t - 1, t):
						node.add_trinode(node.mirror_tri_node(t - 1, t), t)
					elif node.mirror_tri_node(t, t):
						node.add_trinode(node.mirror_tri_node(t, t), t)
					else:
						#create new tri_node
						xn, yn = node.pos
						p = (xn + x[t], yn + y[t])
						node.add_trinode(tri_node(pos=p), t)
					
				for t in range(len(node.tri_nodes)):
					node.tri_nodes[t].add_edge(node.get_trinode((t - 1) % 6))
					node.tri_nodes[t].add_edge(node.get_trinode((t + 1) % 6))

	def get_trinode_pairs_with(self, tn1, tn2=None, pairs=None, vmask=False):
		if not pairs:
			pairs=self.trinode_pairs
		else:
			vmask = False
		if tn2:
			bool1 = np.any(pairs == tn1, axis=1)
			bool2 = np.any(pairs == tn2, axis=1)
			bool_arr = np.all((bool1, bool2),axis=0)
			if vmask: bool_arr = np.all([bool_arr, self.trinode_visible], axis=0)
			return np.where(bool_arr)
		else:
			bool_arr = np.any(pairs == tn1, axis=1)
			if vmask: bool_arr = np.all([bool_arr, self.trinode_visible], axis=0)
			return np.where(bool_arr)

	def merge_nodes(self, hn1, hn2):
		hn1m = np.concatenate((np.array(hn1.merged_nodes), np.array([hn1,])))
		hn2m = np.concatenate((np.array(hn2.merged_nodes), np.array([hn2,])))
		#cartesian product of p1m on p2m contatenated with
		#cartesian product of p2m on p1m
		merged_pairs_1 = np.transpose([np.tile(hn1m, len(hn2m)), np.repeat(hn2m, len(hn1m))])
		merged_pairs_2 = np.transpose([np.tile(hn2m, len(hn1m)), np.repeat(hn1m, len(hn2m))])
		possible_merged_pairs = np.concatenate((merged_pairs_1, merged_pairs_2), axis=0)
		for pair in possible_merged_pairs:
			hn1, hn2 = pair
			if hn1 in hn2.edges and hn2 in hn1.edges:
				dir_12 = hn1.edge_directions[hn1.edges.index(hn2)]
				dir_21 = (dir_12 + 3) % 6
				#disconnect trinodes
				tn1 = hn1.get_trinode(dir_12 % 6)
				tn2 = hn1.get_trinode((dir_12 + 1) % 6)
				tnp12 = np.array([tn1, tn2])
				tnp21 = np.array([tn2, tn1])
				self.trinode_visible[np.where(np.all(tnp12 == self.trinode_pairs, axis=1))] = False
				self.trinode_visible[np.where(np.all(tnp21 == self.trinode_pairs, axis=1))] = False

				#add nodes to merged list
				hn1.add_merged(hn2, dir_12)
				hn2.add_merged(hn1, dir_21)
			else:
				#add nodes to merged list
				hn1.add_merged(hn2, None)
				hn2.add_merged(hn1, None)			


	def merg_nodes_list(self, hnlist):
		for hn1 in hnlist:
			for hn2 in hnlist:
				if not (hn1 is hn2):
					if hn1 in hn2.edges and hn2 in hn1.edges:
						dir_12 = hn1.edge_directions[hn1.edges.index(hn2)]
						#disconnect trinodes
						tn1 = hn1.get_trinode(dir_12 % 6)
						tn2 = hn1.get_trinode((dir_12 + 1) % 6)
						tnp12 = np.array([tn1, tn2])
						tnp21 = np.array([tn2, tn1])
						self.trinode_visible[np.where(np.all(tnp12 == self.trinode_pairs, axis=1))] = False
						self.trinode_visible[np.where(np.all(tnp21 == self.trinode_pairs, axis=1))] = False
						#add hn2 to hn1's merged list
						hn1.add_merged(hn2, dir_12)
					else:
						#add hn2 to hn1's merged list
						hn1.add_merged(hn2, None)


#Random section

import scipy.interpolate, scipy.stats
#traverses graphs and returns all connected pairs of nodes
def get_node_pairs(node):
	todo_list = [node,]
	done_list = []
	pairs = []
	while len(todo_list) > 0:
		node = todo_list.pop(0)
		done_list.append(node)
		for e in node.edges:
			if not (e in done_list):
				pairs.append((node, e))
		for e in node.edges:
			if not (e in done_list) and not (e in todo_list):
				todo_list.append(e)
	return pairs

#I might not need this
def get_merged_pairs(node):
	todo_list = [node,]
	done_list = []
	pairs = []
	while len(todo_list) > 0:
		node = todo_list.pop(0)
		done_list.append(node)
		for mn in node.merged_nodes:
			if not (mn in done_list):
				pairs.append((node, mn))
		for mn in node.merged_nodes:
			if not (mn in done_list) and not (mn in todo_list):
				todo_list.append(mn)
	return pairs


def jostle_trinode(tn, r, theta):
	x, y = tn.pos
	x += r*np.cos(theta)
	y += r*np.sin(theta)
	tn.pos = (x, y)

def jostle_trinodes_random(tn_list, sigma=5):
	r = np.abs(sigma*np.random.randn(len(tn_list)))
	theta = np.random.rand(len(tn_list))*2*np.pi
	for t in range(len(tn_list)):
		jostle_trinode(tn_list[t], r[t], theta[t])



def to_pointlist(tn_chains, div=10):
	pointlists = []
	for chain in tn_chains:
		p_list = []
		for t in range(len(chain) - 1):
			x1, y1 = chain[t].pos
			x2, y2 = chain[t + 1].pos
			x_div = np.linspace(x1, x2, div)
			y_div = np.linspace(y1, y2, div)
			if t != len(chain) - 2:
				x_div = x_div[:-1]
				y_div = y_div[:-1]
			p_list.append(np.array([x_div, y_div]).T)
		pointlists.append(np.concatenate(p_list, axis=0))
	return pointlists


#chain together trinodes
#returns a list of trinode chains
def get_connected_trinode_pairs(hgrid):
	trinode_chains = []
	used_pairs = []
	for p in range(len(hgrid.trinode_pairs)):
		pair = hgrid.trinode_pairs[p]

		if tuple(pair) in used_pairs or not hgrid.trinode_visible[p]: continue
		used_pairs.append(tuple(pair))
		chain = [pair[0], pair[1]]
		p1_pairs = hgrid.get_trinode_pairs_with(pair[0], vmask=True)[0]
		p2_pairs = hgrid.get_trinode_pairs_with(pair[1], vmask=True)[0]
		ptn = pair[0]
		tn = pair[1]
		tn_pairs = p2_pairs
		tne_len = len(p2_pairs)

		#forward append
		while tne_len == 2:
			prev_pair = hgrid.get_trinode_pairs_with(ptn, tn, vmask=True)[0][0]
			tn_pairs = tn_pairs[np.where(tn_pairs != prev_pair)]
			ntnp = hgrid.trinode_pairs[tn_pairs[0]]

			if tuple(ntnp) in used_pairs:
				break
			used_pairs.append(tuple(ntnp))
			ntn = ntnp[np.where(ntnp != tn)][0]


			chain.append(ntn)
			ptn = tn
			tn = ntn
			tn_pairs = hgrid.get_trinode_pairs_with(tn, vmask=True)[0]
			tne_len = len(tn_pairs)

		ptn = pair[1]
		tn = pair[0]
		tn_pairs = p1_pairs
		tne_len = len(p1_pairs)
		#backward append
		while tne_len == 2:
			prev_pair = hgrid.get_trinode_pairs_with(ptn, tn, vmask=True)[0][0]
			tn_pairs = tn_pairs[np.where(tn_pairs != prev_pair)]
			ntnp = hgrid.trinode_pairs[tn_pairs[0]]

			if tuple(ntnp) in used_pairs:
				break
			used_pairs.append(tuple(ntnp))
			ntn = ntnp[np.where(ntnp != tn)][0]
			temp = [ntn,]
			temp.extend(chain)
			chain = temp
			ptn = tn
			tn = ntn
			tn_pairs = hgrid.get_trinode_pairs_with(tn, vmask=True)[0]
			tne_len = len(tn_pairs)

		trinode_chains.append(chain)

	return trinode_chains

def jostle_pointlist(p_list, r, theta):
	p_list[0] = p_list[0] + r*np.cos(theta)
	p_list[1] = p_list[1] + r*np.sin(theta)

def jostle_pointlist_random(p_list, sigma_t):
	n = p_list.shape[0]
	xa, ya = p_list[0]
	xb, yb = p_list[1]
	alpha = np.arctan((yb - ya)/(xb - xa))
	rng_t = np.random.randn(n - 2)

	p_list[1:-1, 0] += sigma_t*rng_t*np.cos(alpha + np.pi/2)
	p_list[1:-1, 1] += sigma_t*rng_t*np.sin(alpha + np.pi/2)


def splinterpolate(p_list, n_new, s=0):
	x, y = p_list.T
	u_new = np.linspace(0, 1, n_new)
	tck, u = scipy.interpolate.splprep([x, y], s=s, k=3)
	new_points = scipy.interpolate.splev(u_new, tck)
	p_list = np.array([new_points[0], new_points[1]], dtype=int).T
	return p_list

def poly_exagerate(p_list, n_new, deg=1):
	x, y = p_list.T
	t = np.linspace(0, 1, len(x))
	t_new = np.linspace(0, 1, n_new)
	x_new = np.poly1d(np.polyfit(x, t, 1))
	y_new = np.poly1d(np.polyfit(y, t, 1))
	p_list = np.array([x_new(t_new), y_new(t_new)],dtype=int).T
	return p_list


def wherein(a, b):#basically: a in b element-wise (only works for 2d arrays right now)
	boolean = np.any(np.array([np.all(b==a[i],axis=1) for i in range(a.shape[0])]), axis=0)
	return np.where(boolean)

def merge_random_nodes(hgrid, ngroups):
	ys = np.arange(hgrid.shape[0])
	xs = np.arange(hgrid.shape[1])
	crds = np.transpose([np.tile(ys, len(xs)), np.repeat(xs, len(ys))])
	np.random.shuffle(crds)
	ys = crds[:ngroups, 0]
	xs = crds[:ngroups, 1]

	todo_groups = []
	done_groups = []
	for g in range(ngroups):
		todo_groups.append([hgrid.node_array[ys[g],xs[g]],])
		done_groups.append([])	

	turns_each = np.ones((ngroups,), dtype=int)
	#in the future, turns_each can be set to differnt numbers for random sizes

	def isclaimed(node):
		for g in range(ngroups):
			if node in done_groups[g]: return True
		return False

	def isdone():
		alldone = np.all([len(todo_groups[g]) == 0 for g in range(ngroups)])
		return alldone
	
	r = 0
	while not isdone():
		r += 1
		for g in range(ngroups):
			for t in range(turns_each[g]):
				if len(todo_groups[g]) == 0: break
				node = todo_groups[g].pop(0)
				if isclaimed(node): break
				done_groups[g].append(node)
				for e in range(len(node.edges)):
					if not isclaimed(node.edges[e]):
						todo_groups[g].append(node.edges[e])

	for g in range(ngroups):
		hgrid.merg_nodes_list(done_groups[g])


#Drawing section
import pygame, sys
BLACK = (0,0,0)
WHITE = (255, 255,255)
BLUE =  (0,0,255)
GREEN = (0,255,0)
RED =   (255,0,0)
OFFSET = 100

def draw_hex_grid(surf, hgrid):
	dot_rad = 5#radius of the dot
	for j in range(hgrid.shape[1]):
		for i in range(hgrid.shape[0]):
			x, y = hgrid.node_array[i, j].pos
			x += OFFSET
			y += OFFSET
			x, y = (int(x), surf.get_height() - int(y))
			pygame.draw.circle(surf, BLUE, (x, y), dot_rad)

def draw_tri_graph(surf, hgrid):
	dot_rad = 3#radius of the dot
	for j in range(hgrid.shape[1]):
		for i in range(hgrid.shape[0]):
			hn = hgrid.node_array[i, j]			
			for t in range(len(hn.tri_nodes)):
				tn = hn.tri_nodes[t]
				x, y = tn.pos
				x += OFFSET
				y += OFFSET
				x, y = (int(x), surf.get_height() - int(y))
				pygame.draw.circle(surf, RED, (x, y), dot_rad)
	node = hgrid.node_array[0, 0].tri_nodes[0]
	pairs = get_node_pairs(node)
	for p in pairs:
		xy1 = (p[0].pos[0] + OFFSET, surf.get_height() - (p[0].pos[1] + OFFSET))
		xy2 = (p[1].pos[0] + OFFSET, surf.get_height() - (p[1].pos[1] + OFFSET))
		pygame.draw.line(surf, RED, xy1, xy2)

def draw_pointlist(surf, p_list):
	for p in range(p_list.shape[0] - 1):
		xy1 = (p_list[p, 0] + OFFSET, surf.get_height() - (p_list[p, 1] + OFFSET))
		xy2 = (p_list[p + 1, 0] + OFFSET, surf.get_height() - (p_list[p + 1, 1] + OFFSET))
		pygame.draw.line(surf, RED, xy1, xy2)



def save_image(hg, pts, filename):
	surf = pygame.Surface((hg.nnd*hg.shape[1] + 200, hg.nnd*hg.shape[0] + 200))
	for p in range(len(pts)):
		draw_pointlist(surf, pts[p])
	pygame.image.save(surf, filename)


def on_screen(hg, pts):

	screen = pygame.display.set_mode((1024, 768))
	screen.fill(BLACK)
	dots = pygame.Surface((hg.nnd*hg.shape[1] + 200, hg.nnd*hg.shape[0] + 200))
	for p in range(len(pts)):
		draw_pointlist(dots, pts[p])
	pygame.draw.rect(dots, WHITE, pygame.Rect(0, 0, dots.get_width(), dots.get_height()), 3)
	screen.blit(dots, (35, 35))
	
	
	
	pygame.display.flip()
	while 1:
		event = pygame.event.wait()
		if event.type == pygame.QUIT: sys.exit(0)



import argparse
def __main__():
	parser = argparse.ArgumentParser()
	parser.add_argument("-S", "--Seed", help="Seed Number for rng.", default=None, type=int, metavar='S')
	D_help = """Dimensions of hexgrid.
	x and y are horizontal and vertical number of hexes respectively.
	nnd is nearest neighbour distance, which determines size of each hex.
	"""
	parser.add_argument("-D", "--Dim", help=D_help, default=(6, 6, 75), type=int, nargs=3, metavar=('x', 'y', 'nnd'))
	M_help = """Merge hexes to have M cells remain.
	"""
	parser.add_argument("-M", "--Merge", help=M_help, default=None, type=int, metavar='M')
	J_help = """Jostle the points that make up the hex grid.
	vdev and ddev are standard deviation of the
	vertices in each hex and the subdivisions of a side, respectively.
	"""
	parser.add_argument("-J", "--Jostle", help=J_help, default=None, type=float, nargs=2, metavar=('vdev','ddev'))
	I_help = """Interpolate over subdivision using cubic spline.
	res is resolution, number of points per hex side.
	"""
	parser.add_argument("-I", "--Interp", help=I_help, default=None, type=int, metavar='res')
	O_help = """Save blotchmap as image to given filename.
	Use file extensions .bmp, .tga, .png, or .jpeg.
	"""
	parser.add_argument("-O", "--Output", help=O_help, type=str, metavar='filename')
	parser.add_argument("-V", "--Visual", help="Show blotchmap as pygame drawing.", action="store_true")

	args = parser.parse_args()
	if args.Seed:
		np.random.seed(args.Seed)

	#build hex grid
	shp = (args.Dim[1], args.Dim[0])
	hg = hex_grid(shape=shp, nnd=args.Dim[2])

	#merge nodes
	if args.Merge:
		merge_random_nodes(hg, args.Merge)
	
	#jostle points, subdivide, jostle subdivisions
	tripairs = hg.trinode_pairs
	if args.Jostle:
		for p in tripairs: jostle_trinodes_random(p, sigma=args.Jostle[0])
	trinode_chains = get_connected_trinode_pairs(hg)
	pts = to_pointlist(trinode_chains, div=6)
	pts = np.array(pts)
	if args.Jostle:
		for p in pts: jostle_pointlist_random(p, args.Jostle[1])

	#splinterpolate
	if args.Interp:
		spldivs = args.Interp
		splpts = np.empty((pts.shape[0], spldivs, 2))
		for p in range(len(pts)): splpts[p] = splinterpolate(pts[p], spldivs)
		pts = splpts

	if args.Output:
		save_image(hg, pts, args.Output)
	
	if args.Visual:
		on_screen(hg, pts)




if __name__ == "__main__":
	__main__()