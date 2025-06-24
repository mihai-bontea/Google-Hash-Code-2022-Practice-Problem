solutions = ["sol1", "sol2", "sol3"]
input_files = ["a_an_example", "b_basic", "c_coarse", "d_difficult", "e_elaborate"]

def read_input_file(filename):
    with open(filename) as fin:
        lines = [line.strip().split() for line in fin if line.strip()]

    like_map, dislike_map = {}, {}
    nr_clients = int(lines[0][0])
    for i in range(nr_clients):
        likes = lines[1 + 2*i][1:]
        dislikes = lines[2 + 2*i][1:]
        for ingr in likes:
            like_map[ingr] = like_map.get(ingr, set()) | {i}
        for ingr in dislikes:
            dislike_map[ingr] = dislike_map.get(ingr, set()) | {i}

    return nr_clients, like_map, dislike_map

def read_solution_file(filename):
    with open(filename) as fin:
        return fin.read().strip().split()[1:]
    
def filter_clients(nr_clients, like_map, dislike_map, ingredients):
    valid_clients = set(range(nr_clients))

    for ingr in ingredients:
        # Remove clients who dislike any included ingredient
        if ingr in dislike_map:
            valid_clients -= dislike_map[ingr]

    for ingr, clients in like_map.items():
        # Remove clients who like an ingredient that is missing
        if ingr not in ingredients:
            valid_clients -= clients

    return valid_clients

for solution in solutions:
    total_score = 0
    print(f"For {solution} we got:")
    for input_file in input_files:
        full_input_path = f"input_files/{input_file}.in"
        nr_clients, like_map, dislike_map = read_input_file(full_input_path)

        try:
            full_solution_path = f"output_files/{solution}/{input_file}.out"
            ingredients_chosen = read_solution_file(full_solution_path)

            # No duplicated ingredients in output
            assert(len(ingredients_chosen) == len(set(ingredients_chosen)))

            valid_clients = filter_clients(nr_clients, like_map, dislike_map, ingredients_chosen)
            score = len(valid_clients)
            total_score += score
            print(f"--->For {input_file} we obtained score = {score:,}")

        except FileNotFoundError:
            print(f"Error: File '{full_solution_path}' not found.")
    
    print(f"Total score = {total_score:,}\n")