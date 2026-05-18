import math

tm = [
    '############################################################',
    '#..........................................................#',
    '#.......#######################............................#',
    '#.............................#............................#',
    '#.............................#............................#',
    '#.......S.....................#............................#',
    '#.......############..........#............................#',
    '#.............................#............................#',
    '#.............................#............E...............#',
    '#.............................#............................#',
    '#..........................................................#',
    '#..........................................................#',
    '#..........................................................#',
    '############################################################',
]

# mutable copy of the map
test_map = []


class Node_Elem:
    def __init__(self, parent, x, y, dist):
        self.parent = parent
        self.x = x
        self.y = y
        self.dist = dist


class A_Star:
    def __init__(self, s_x, s_y, e_x, e_y, w=60, h=14):
        self.s_x = s_x
        self.s_y = s_y
        self.e_x = e_x
        self.e_y = e_y
        self.width = w
        self.height = h
        self.open = []
        self.close = []
        self.path = []

    def find_path(self):
        p = Node_Elem(None, self.s_x, self.s_y, 0.0)
        while True:
            self.extend_round(p)
            if not self.open:
                return
            idx, p = self.get_best()
            if self.is_target(p):
                self.make_path(p)
                return
            self.close.append(p)
            del self.open[idx]

    def make_path(self, p):
        while p:
            self.path.append((p.x, p.y))
            p = p.parent

    def is_target(self, i):
        return i.x == self.e_x and i.y == self.e_y

    def get_best(self):
        best = None
        bv = 1e18
        bi = -1
        for idx, i in enumerate(self.open):
            value = self.get_dist(i)
            if value < bv:
                best = i
                bv = value
                bi = idx
        return bi, best

    def get_dist(self, i):
        return i.dist + math.sqrt(
            (self.e_x - i.x) ** 2 + (self.e_y - i.y) ** 2) * 1.2

    def extend_round(self, p):
        xs = (0, -1, 1, 0, -1, 1, -1, 1)
        ys = (-1, 0, 0, 1, -1, -1, 1, 1)
        for x, y in zip(xs, ys):
            new_x, new_y = x + p.x, y + p.y
            if not self.is_valid_coord(new_x, new_y):
                continue
            node = Node_Elem(p, new_x, new_y,
                             p.dist + self.get_cost(p.x, p.y, new_x, new_y))
            if self.node_in_close(node):
                continue
            i = self.node_in_open(node)
            if i != -1:
                if self.open[i].dist > node.dist:
                    self.open[i].parent = p
                    self.open[i].dist = node.dist
                continue
            self.open.append(node)

    def get_cost(self, x1, y1, x2, y2):
        if x1 == x2 or y1 == y2:
            return 1.0
        return 1.4

    def node_in_close(self, node):
        for i in self.close:
            if node.x == i.x and node.y == i.y:
                return True
        return False

    def node_in_open(self, node):
        for i, n in enumerate(self.open):
            if node.x == n.x and node.y == n.y:
                return i
        return -1

    def is_valid_coord(self, x, y):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return test_map[y][x] != '#'

    def get_searched(self):
        l = []
        for i in self.open:
            l.append((i.x, i.y))
        for i in self.close:
            l.append((i.x, i.y))
        return l


def print_test_map():
    for line in test_map:
        print(''.join(line))


def get_start_XY():
    return get_symbol_XY('S')


def get_end_XY():
    return get_symbol_XY('E')


def get_symbol_XY(s):
    for y, line in enumerate(test_map):
        try:
            x = line.index(s)
        except ValueError:
            continue
        else:
            return x, y
    return -1, -1


def mark_path(l):
    mark_symbol(l, '*')


def mark_searched(l):
    mark_symbol(l, ' ')


def mark_symbol(l, s):
    for x, y in l:
        test_map[y][x] = s


def mark_start_end(s_x, s_y, e_x, e_y):
    test_map[s_y][s_x] = 'S'
    test_map[e_y][e_x] = 'E'


def tm_to_test_map():
    for line in tm:
        test_map.append(list(line))


def find_path():
    s_x, s_y = get_start_XY()
    e_x, e_y = get_end_XY()
    a_star = A_Star(s_x, s_y, e_x, e_y, w=len(tm[0]), h=len(tm))
    a_star.find_path()
    searched = a_star.get_searched()
    path = a_star.path
    mark_searched(searched)
    mark_path(path)
    print("path length is %d" % (len(path)))
    print("searched squares count is %d" % (len(searched)))
    mark_start_end(s_x, s_y, e_x, e_y)


if __name__ == "__main__":
    tm_to_test_map()
    print("Original layout:")
    print_test_map()
    print()
    find_path()
    print()
    print("After A* search (path marked with *):")
    print_test_map()
