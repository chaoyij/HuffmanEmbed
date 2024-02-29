import heapq 
import torch
import pickle
from typing import Dict, Optional

class Node: 
    def __init__(self, freq: int, symbol: int, left=None, right=None, children=None): 
        self.freq = freq 
        self.symbol = symbol 
        self.left = left 
        self.right = right 
        self.children = children if children is not None else []
        self.huff = '' 

    def __lt__(self, nxt): 
        return self.freq < nxt.freq 


# utility function to print huffman 
# codes for all symbols in the newly 
# created Huffman tree 
def printNodes(node: Node, val: str, huffman_coding_dict: Dict[str, int]) -> int:
    newVal = val + str(node.huff)
    max_len = 0

    # if node is not an edge node
    # then traverse inside it
    if(node.left):
        left_len = printNodes(node.left, newVal, huffman_coding_dict)
        if left_len > max_len:
            max_len = left_len
    if(node.right): 
        right_len = printNodes(node.right, newVal, huffman_coding_dict)
        if right_len > max_len:
            max_len = right_len

        # if node is edge node then 
        # display its huffman code 
    if(not node.left and not node.right):
        huffman_coding_dict[node.symbol] = newVal
        if len(newVal) > max_len:
            max_len = len(newVal)

    return max_len

def print_n_ary_nodes(node: Node, val: int, huffman_coding_dict: Dict[str, int]) -> int:
    newVal = val + "," + str(node.huff) if val != '' else str(node.huff)

    max_len = 0

    for child in node.children:
        if child:
            child_len = print_n_ary_nodes(child, newVal, huffman_coding_dict)
            if child_len > max_len:
                max_len = child_len
    
    if len(node.children) == 0 and node.symbol != -1:
        huffman_coding_dict[node.symbol] = newVal
        coding_length = newVal.count(",") + 1
        if coding_length > max_len:
            max_len = coding_length

    return max_len


def huffman_coding(freq_dict: Dict[str, int]):
    # list containing unused nodes 
    nodes = []
    # converting characters and frequencies 
    # into huffman tree nodes 
    for key, value in freq_dict.items(): 
        heapq.heappush(nodes, Node(value, key))

    while len(nodes) > 1:
        # sort all the nodes in ascending order 
        # based on their frequency 
        left = heapq.heappop(nodes) 
        right = heapq.heappop(nodes) 

        # assign directional value to these nodes 
        left.huff = 0
        right.huff = 1

        # combine the 2 smallest nodes to create 
        # new node as their parent 
        newNode = Node(left.freq+right.freq, left.symbol+right.symbol, left, right) 

        heapq.heappush(nodes, newNode)
    
    huffman_coding_dict = dict()

    max_len = printNodes(nodes[0], '', huffman_coding_dict)

    return huffman_coding_dict, max_len


def n_ary_huffman_coding(freq_dict: Dict[str, int], n: int):
    nodes = []
    remain = (len(freq_dict) - 1) % (n - 1)
    if remain != 0:
        for _ in range(n - 1 - remain):
            heapq.heappush(nodes, Node(0, -1))

    for key, value in freq_dict.items(): 
        heapq.heappush(nodes, Node(value, key))

    while len(nodes) > 1:

        candidate_nodes = []
        for i in range(n):
            candidate_node = heapq.heappop(nodes)
            candidate_nodes.append(candidate_node)
            candidate_nodes[-1].huff = i

        newNode = Node(sum(candidate_node.freq for candidate_node in candidate_nodes), -1, None, None, candidate_nodes) 

        heapq.heappush(nodes, newNode)
    
    huffman_coding_dict = dict()

    max_len = print_n_ary_nodes(nodes[0], '', huffman_coding_dict)

    return huffman_coding_dict, max_len


def test_huffman_coding():
    print("test_huffman_coding")
    keys = range(6)
    freq = [5, 9, 12, 13, 16, 45]

    freq_dict = dict()

    for i in range(len(keys)):
        freq_dict[keys[i]] = freq[i]

    huffman_coding_dict, max_len = huffman_coding(freq_dict)

    for key, value in huffman_coding_dict.items():
        print(f"{key}:{value}")

    print("max_len=", max_len)


def test_n_ary_huffman_coding(n: Optional[int] = 3):
    print(f"test_n_ary_huffman_coding with n={n}")
    keys = range(6)
    freq = [5, 9, 12, 13, 16, 45]

    freq_dict = dict()

    for i in range(len(keys)):
        freq_dict[keys[i]] = freq[i]

    huffman_coding_dict, max_len = n_ary_huffman_coding(freq_dict, 3)

    for key, value in huffman_coding_dict.items():
        print(f"{key}:{value}")

    print("max_len=", max_len)


def generate_huffman_coding_tensor(huffman_coding_dict, table_offset, num_features, max_len):
    huffman_coding_tensor = torch.zeros(num_features, max_len, dtype=torch.int64)
    print("num_features:", num_features, " max_len:", max_len, " table_offset:", table_offset)
    for key, value in huffman_coding_dict.items():
        for i in range(len(value)):
            if value[i] == '0':
                huffman_coding_tensor[key, i] = i + 1
            else:
                huffman_coding_tensor[key, i] = i + 1 + table_offset
    return huffman_coding_tensor


def generate_n_ary_huffman_coding_tensor(huffman_coding_dict, table_offset, num_features, max_len, n):
    huffman_coding_tensor = torch.zeros(num_features, max_len, dtype=torch.int64)
    print("num_features:", num_features, " max_len:", max_len, " table_offset:", table_offset, " n:", n)
    for key, value in huffman_coding_dict.items():
        if key == -1:
            continue
        value = value.split(",")
        for i in range(len(value)):
            num_table = int(value[i])
            huffman_coding_tensor[key, i] = i + 1 + num_table * table_offset
    return huffman_coding_tensor


def generate_and_dump_huffman_coding_tensors():
    huffman_coding_tensors = []

    with open('huffman_coding.pkl', 'rb') as file:
        huffman_coding_dicts = pickle.load(file)
        max_lens = pickle.load(file)
        print("huffman_coding_dicts.size()=", len(huffman_coding_dicts))
        for i in range(26):
            huffman_coding_tensors.append(generate_huffman_coding_tensor(huffman_coding_dicts[i], max_lens[i], len(huffman_coding_dicts[i]), max_lens[i]))
        
    with open('huffman_coding_tensors.pkl', 'wb') as file:
        pickle.dump(huffman_coding_tensors, file)
        pickle.dump(max_lens, file)


def test_generate_huffman_coding_tensor():
    print("test_generate_huffman_coding_tensor")
    huffman_coding_dict = {0:"01100", 1:"1001", 2:"110010"}
    table_offset = 6
    num_features = 3
    max_len = 8
    print("generate_huffman_coding_tensor")
    print("huffman_coding_dict:", huffman_coding_dict)
    print(generate_huffman_coding_tensor(huffman_coding_dict, table_offset, num_features, max_len))


def test_generate_n_ary_huffman_coding_tensor_n_2(n = 2):
    print("test_generate_huffman_coding_tensor")
    huffman_coding_dict = {0:"01100", 1:"1001", 2:"110010"}
    table_offset = 6
    num_features = 3
    max_len = 8
    print("generate_huffman_coding_tensor")
    print("huffman_coding_dict:", huffman_coding_dict)
    print(generate_n_ary_huffman_coding_tensor(huffman_coding_dict, table_offset, num_features, max_len, n))


def test_generate_n_ary_huffman_coding_tensor_n_3(n = 3):
    print("test_generate_n_ary_huffman_coding_tensor")
    huffman_coding_dict = {0:"01102", 1:"1201", 2:"110210"}
    table_offset = 6
    num_features = 3
    max_len = 8
    print("generate_n_ary_huffman_coding_tensor")
    print("huffman_coding_dict:", huffman_coding_dict)
    print(generate_n_ary_huffman_coding_tensor(huffman_coding_dict, table_offset, num_features, max_len, n))


def huffman_coding_lookup(sparse_index_group_batch, huffman_coding_matrix, is_reversed, reverse_and_shift_huffman_coding, table_size):
    selected_rows = huffman_coding_matrix[sparse_index_group_batch]
    if is_reversed:
        # reverse each row
        selected_rows = torch.flip(selected_rows, dims=[1])

    if reverse_and_shift_huffman_coding:
        selected_rows = torch.flip(selected_rows, dims=[1])
        nonzero_mask = selected_rows != 0
        nonzero_indices = torch.argmax(nonzero_mask.to(torch.int), dim=1)
        row_indices = torch.arange(selected_rows.size(0)).reshape(-1, 1)
        first_nonzero_indices = torch.cat((row_indices, nonzero_indices.unsqueeze(1)), dim=1)
        selected_elements = selected_rows[first_nonzero_indices[:, 0], first_nonzero_indices[:, 1]]
        broadcasted_tensor = selected_elements.unsqueeze(0).expand(selected_rows.size(1),-1)
        broadcasted_tensor=broadcasted_tensor.T
        selected_rows[nonzero_mask]-=broadcasted_tensor[nonzero_mask]
        mask = torch.zeros_like(selected_rows, dtype=torch.bool)
        mask[first_nonzero_indices[:, 0], first_nonzero_indices[:, 1]] = True
        selected_rows[mask]+=broadcasted_tensor[mask]
        selected_rows %= table_size


    update_sparse_index_group_batch = selected_rows.view(-1)
    non_zero_sparse_index_group_batch = update_sparse_index_group_batch[update_sparse_index_group_batch != 0]
    
    non_zero_counts = torch.count_nonzero(selected_rows, dim=1)
    cumulative_sum_counts = torch.cumsum(non_zero_counts, dim=0)
    cumulative_sum_counts = cumulative_sum_counts - non_zero_counts
    return non_zero_sparse_index_group_batch, cumulative_sum_counts


def test_huffman_coding_lookup():
    print("test_huffman_coding_lookup")
    # sparse_index_group_batch = torch.tensor([1, 3])
    # huffman_coding_matrix = torch.tensor([[1, 2, 3, 3], [4, 5, 0, 0], [6, 0, 0, 0], [7, 8, 0, 0]])
    # print("sparse_index_group_batch:", sparse_index_group_batch)
    # print("huffman_coding_matrix:", huffman_coding_matrix)
    # print("huffman_coding_lookup:")
    # print(huffman_coding_lookup(sparse_index_group_batch, huffman_coding_matrix))

    sparse_index_group_batch_2 = torch.tensor([1, 3, 4])
    huffman_coding_matrix_2 = torch.tensor([[1, 2, 3, 3], [4, 5, 0, 0], [6, 0, 0, 0], [7, 8, 0, 0], [2, 4, 5, 0]])
    print("sparse_index_group_batch_2:", sparse_index_group_batch_2)
    print("huffman_coding_matrix_2:", huffman_coding_matrix_2)
    huffman_sparse_index_group_batch_2, huffman_offset_index_group_batch_2 = huffman_coding_lookup(sparse_index_group_batch_2, huffman_coding_matrix_2)
    print("huffman_sparse_index_group_batch_2:", huffman_sparse_index_group_batch_2)
    print("huffman_offset_index_group_batch_2:", huffman_offset_index_group_batch_2)
    assert torch.equal(huffman_sparse_index_group_batch_2, torch.tensor([4,5,7,8,2,4,5])), "wrong huffman_sparse_index_group_batch_2"
    assert torch.equal(huffman_offset_index_group_batch_2, torch.tensor([0,2,4])), "wrong huffman_offset_index_group_batch_2"

    print()

    sparse_index_group_batch_3 = torch.tensor([1, 3, 4, 5])
    huffman_coding_matrix_3 = torch.tensor([[1, 2, 3, 3], [4, 5, 0, 0], [6, 0, 0, 0], [7, 8, 0, 0], [2, 4, 5, 0], [23,0,0,0]])
    print("sparse_index_group_batch_3:", sparse_index_group_batch_3)
    print("huffman_coding_matrix:_3", huffman_coding_matrix_3)
    huffman_sparse_index_group_batch_3, huffman_offset_index_group_batch_3 = huffman_coding_lookup(sparse_index_group_batch_3, huffman_coding_matrix_3)
    print("huffman_sparse_index_group_batch_3:", huffman_sparse_index_group_batch_3)
    print("huffman_offset_index_group_batch_3:", huffman_offset_index_group_batch_3)
    assert torch.equal(huffman_sparse_index_group_batch_3, torch.tensor([4,5,7,8,2,4,5,23])), "wrong huffman_sparse_index_group_batch_3"
    assert torch.equal(huffman_offset_index_group_batch_3, torch.tensor([0,2,4,7])), "wrong huffman_offset_index_group_batch_3"

    print()

    sparse_index_group_batch_4 = torch.tensor([5])
    huffman_coding_matrix_4 = torch.tensor([[1, 2, 3, 3], [4, 5, 0, 0], [6, 0, 0, 0], [7, 8, 0, 0], [2, 4, 5, 0], [23,0,0,0]])
    print("sparse_index_group_batch_4:", sparse_index_group_batch_4)
    print("huffman_coding_matrix_4:", huffman_coding_matrix_4)
    huffman_sparse_index_group_batch_4, huffman_offset_index_group_batch_4 = huffman_coding_lookup(sparse_index_group_batch_4, huffman_coding_matrix_4)
    print("huffman_sparse_index_group_batch_4:", huffman_sparse_index_group_batch_4)
    print("huffman_offset_index_group_batch_4:", huffman_offset_index_group_batch_4)
    assert torch.equal(huffman_sparse_index_group_batch_4, torch.tensor([23])), "wrong huffman_sparse_index_group_batch_4"
    assert torch.equal(huffman_offset_index_group_batch_4, torch.tensor([0])), "wrong huffman_offset_index_group_batch_4"

    print()

    sparse_index_group_batch_5 = torch.tensor([1, 3, 4])
    huffman_coding_matrix_5 = torch.tensor([[1, 2, 3, 3], [4, 5, 0, 0], [6, 0, 0, 0], [7, 8, 0, 0], [2, 4, 5, 0]])
    print("sparse_index_group_batch_5:", sparse_index_group_batch_5)
    print("huffman_coding_matrix_5:", huffman_coding_matrix_5)
    huffman_sparse_index_group_batch_5, huffman_offset_index_group_batch_5 = huffman_coding_lookup(sparse_index_group_batch_5, huffman_coding_matrix_5, is_reversed=True)
    print("huffman_sparse_index_group_batch_5:", huffman_sparse_index_group_batch_5)
    print("huffman_offset_index_group_batch_5:", huffman_offset_index_group_batch_5)
    assert torch.equal(huffman_sparse_index_group_batch_5, torch.tensor([5,4,8,7,5,4,2])), "wrong huffman_sparse_index_group_batch_5"
    assert torch.equal(huffman_offset_index_group_batch_5, torch.tensor([0,2,4])), "wrong huffman_offset_index_group_batch_5"

    print()

    sparse_index_group_batch_6 = torch.tensor([1, 3, 4, 5])
    huffman_coding_matrix_6 = torch.tensor([[1, 2, 3, 3], [4, 5, 0, 0], [6, 0, 0, 0], [7, 8, 0, 0], [2, 4, 5, 0], [23,0,0,0]])
    print("sparse_index_group_batch_6:", sparse_index_group_batch_6)
    print("huffman_coding_matrix:_6", huffman_coding_matrix_6)
    huffman_sparse_index_group_batch_6, huffman_offset_index_group_batch_6 = huffman_coding_lookup(sparse_index_group_batch_6, huffman_coding_matrix_6, is_reversed=True)
    print("huffman_sparse_index_group_batch_6:", huffman_sparse_index_group_batch_6)
    print("huffman_offset_index_group_batch_6:", huffman_offset_index_group_batch_6)
    assert torch.equal(huffman_sparse_index_group_batch_6, torch.tensor([5,4,8,7,5,4,2,23])), "wrong huffman_sparse_index_group_batch_6"
    assert torch.equal(huffman_offset_index_group_batch_6, torch.tensor([0,2,4,7])), "wrong huffman_offset_index_group_batch_6"


if __name__ == '__main__':
    # test_huffman_coding()
    # test_n_ary_huffman_coding()
    test_huffman_coding_lookup()
    # test_generate_huffman_coding_tensor()
    # test_generate_n_ary_huffman_coding_tensor_n_2()
    # test_generate_n_ary_huffman_coding_tensor_n_3()
    # generate_and_dump_huffman_coding_tensors()
