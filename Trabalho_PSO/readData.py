import numpy as np

tsp_name = ["dantzig42",
            "fri26",
            "gr21"]

atsp_name = ["ft53",
            "ftv33",
            "ftv38",
            "ftv170",
            "kro124p",
            "rbg323",
            "rbg358",
            "rbg453",
            "rbg443"]
lengths = [53, 34, 38, 171, 100, 323, 358, 403, 443]

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
        

  # coords = data.split("DISPLAY_DATA_SECTION")[1].split("DISPLAY_DATA_SECTION_END")[0].strip().split('\n')
  # cities = [item.strip().split() for item in coords]
  # cities = [list(map(float, i)) for i in cities]

if __name__ == '__main__':
  # for i in tsp_name:
  #   print(i)
  # #   run_file('data/symmetric/'+i)

  # for i in atsp_name:
  #   print(i)
  #   readAsymmetric('data/assymetric/'+i)

  print(readAsymmetric('../data/assymetric/ftv170', 171))