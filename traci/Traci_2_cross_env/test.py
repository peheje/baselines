m = {'hr_east': {16: 3, 17: 12.812095995016087},
     'hr_north': {16: 1, 17: 12.596320443819277},
     'hr_south': {16: 0, 17: -1.0},
     'hr_west': {16: 6, 17: 13.236164319454334},
     'mc_east': {16: 0, 17: -1.0},
     'mc_north': {16: 0, 17: -1.0},
     'mc_south': {16: 0, 17: -1.0},
     'mc_west': {16: 0, 17: -1.0},
     'nd_east': {16: 7, 17: 12.987324268326962},
     'nd_north': {16: 0, 17: -1.0},
     'nd_south': {16: 13, 17: 10.832798358669255},
     'nd_west': {16: 7, 17: 13.08636695211354},
     'st_east': {16: 0, 17: -1.0},
     'st_north': {16: 0, 17: -1.0},
     'st_south': {16: 0, 17: -1.0},
     'st_west': {16: 0, 17: -1.0}}


def extract_list(m, subscription_id):
    values = [m[id][subscription_id] for id in sorted(m)]
    return values


extracted = extract_list(m, 16)

print(extracted)