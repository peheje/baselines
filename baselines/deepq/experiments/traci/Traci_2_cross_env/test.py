import numpy as np

def random():
    return np.random.uniform()


def random_int(low, high):
    return np.random.randint(low, high)


def spawn_cars():
    froms = ["As", "Bs", "Cs", "Ds", "Es", "Fs", "Gs", "Hs", "Is", "Js"]
    tos = ["Ae", "Be", "Ce", "De", "Ee", "Ge", "He", "Ie", "Je"]
    paths = []

    for f in froms:
        for t in tos:
            paths.append((f, t))

    print("<routes>")
    print(
        '<vType id="carType" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>')
    for i, p in enumerate(paths):
        print('<route id="route{}" edges="{} {}" />'.format(i, p[0], p[1]))

    prop = 0.1
    vehid = 0
    for i in range(1000):
        print('<vehicle id="{}" type="carType" route="route{}" depart="{}"'.format(vehid, random_int(0, len(paths)), i))
        vehid += 1

    print("</routes>")

    debug = 0

spawn_cars()