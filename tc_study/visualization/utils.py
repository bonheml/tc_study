from itertools import combinations_with_replacement


def get_variables_combinations():
    comb = ["{}_{}".format(v1, v2) for v1, v2 in combinations_with_replacement(["active", "mixed", "passive"], 2)]
    comb += ["full"]
    return comb
