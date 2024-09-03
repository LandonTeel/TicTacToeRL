# import torch  

# torch.device("cuda" if torch.cuda.is_available() else "cpu")

P1, P2 = 0, 1

class Node:
    def __init__(self, data: list[list[int]], parent: "Node", children=[]) -> None:
        self.children = children
        self.data = data
        self.parent = parent
    
# class Tree:
parents = []
# we are 0 and they are 1
def minimax_helper(current_state: list[list[int]], parent: Node, turn=0) -> tuple[list[list[int]], Node, int]:
    visited = []
    nodes = []
    for i in range(len(current_state)):
        for j in range(len(current_state[0])):
            if current_state[i][j] == -1 and (i, j) not in visited:
                currentNode = Node(current_state, parent)
                new_state = current_state
                new_state[i][j] = turn
                minimax_helper(new_state, currentNode, turn^1)
                visited.append((i, j))
                nodes.append(currentNode)

    parent.children = nodes

    parents.append(parent)

    return list(), Node(list, None), 0


def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)

    print(start)

    for next in graph[start] - visited:
        dfs(graph, next, visited)
    return visited


def minimax(state: list[list[int]]) -> list[int]:
    head = Node(None, None)
    res = minimax_helper(state, head)


def main():
    state = [[-1 for j in range(3)] for i in range(3)]
    minimax(state)
    for p in parents:
        for c in p.children:
            print(c.data)


if __name__ == "__main__":
    main()
