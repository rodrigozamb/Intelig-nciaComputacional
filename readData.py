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

def run_file(filename):
  with open(filename + '.tsp', 'r') as file:
        data = file.read()

  matrix = data.split("EDGE_WEIGHT_SECTION")[1].split("EDGE_WEIGHT_SECTION_END")[0].strip().split('\n')
  matrix_list = [item.strip().split() for item in matrix]
  matrix_list = [list(map(float, i)) for i in matrix_list]

  coords = data.split("DISPLAY_DATA_SECTION")[1].split("DISPLAY_DATA_SECTION_END")[0].strip().split('\n')
  cities = [item.strip().split() for item in coords]
  cities = [list(map(float, i)) for i in cities]
  
  print('\nmatrix_list')
  for i in matrix_list:
    print(i)
  
  print('\ncities coordinates')
  for i in cities:
    print(i)

if __name__ == '__main__':
  for i in tsp_name:
    print(i)
    run_file('data/symmetric/'+i)