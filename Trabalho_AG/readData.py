import numpy as np

tsp_name = ["dantzig42",
            "fri26",
            "gr21"]

atsp_name = [
            "ftv33",
            "ft53",
            "ftv38",
            "ftv170",
            "kro124p",
            "rbg323",
            "rbg358",
            # "rbg453",
            # "rbg443"
            ]
lengths = [
          34, 
          53, 
          38, 
          171, 
          100, 
          323, 
          358, 
          # 403, 443
          ]

def readAsymmetric(filename, length):
  with open(filename + '.atsp', 'r') as file:
        data = file.read()

  matrix = data.split("EDGE_WEIGHT_SECTION")[1].split("EOF")[0].strip().split('\n')
  diag = [item.strip().split() for item in matrix]
  diag = [list(map(int, i)) for i in diag]

  full_matrix = np.array(diag)
  full_matrix = np.concatenate(full_matrix)
  full_matrix = full_matrix.reshape(length, length)
  full_matrix = full_matrix.tolist()

  return full_matrix