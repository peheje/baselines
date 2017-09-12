def discrete_to_multidiscrete(self,action,num_actions):
    mod_rest=action%num_actions
    div_floor=action//num_actions
    return [mod_rest,div_floor]

for i in range(9):
    print(discrete_to_multidiscrete(i))