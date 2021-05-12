import argparse

def parse_args():
  parser = argparse.ArgumentParser(description="AORM: Arbitrary-order reachability matrix framework")
  parser.add_argument('-m', nargs='?', default='i', help='Methods (i: i-aorm, s: s-aorm, p: p-aorm, v: v-bfs, p: p-sm)')
  parser.add_argument('-i', nargs='?', default='./datasets/synthetic/ba_graph_n500.gpickle', help='Input graph file path')
  parser.add_argument('-k', nargs='?', default=100, type=int, help='Reachability constraints (k-order)')
  parser.add_argument('-r', action='store_true', help='Real-world network experiment')
  parser.add_argument('-d', action='store_true', help='Directed (True) or undirected (False)')
  parser.add_argument('-p', action='store_true', help='Computations per step unit')
  parser.add_argument('-o', action='store_true', help='Print out the distance matrix')
  parser.add_argument('-mx', action='store_true', help='Set as compressed sparse row matrix')
  
  return parser.parse_args()