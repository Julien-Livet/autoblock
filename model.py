import autoblock as ab
import math
import numpy as np
import os

model = ab.Map()
filename = "model.bin"

def search(m, inputType, outputType, input, output, name = ""):
    f = m.search(inputType, outputType, input, output)
    func = ab.function(f)
    t = type(f)

    if ((t is tuple or t is list) and len(f) > 1):
        m.add(func, inputType, outputType, name)
    
    return func

def searchList(m, inputType, outputType, inputOutputList, name = ""):
    if (len(name)):
        for key, value in m.functions.items():
            for x in value:
                if (x[1] == name):
                    return x[0]

    f = m.searchList(inputType, outputType, inputOutputList)
    func, inputType = ab.functionList(f, inputType, inputOutputList)
    t = type(f)

    if ((t is tuple or t is list) and len(f) > 1):
        m.add(func, inputType, outputType, name)

    return func

if (os.path.isfile(filename)):
    model.load(filename)
else:
    ab.add_bool(model)
    ab.add_float(model)
    ab.add_int(model)
    ab.add_list(model)
    ab.add_ndarray(model)
    ab.add_str(model)
    ab.add_tuple(model)

search(model, list, list, [256, 16, 4, 1], [1, 2, 4, 16])
isodd = searchList(model, (int, int), bool, (((0, 2), False), ((1, 2), True), ((2, 2), False), ), "isodd")
iseven = searchList(model, int, bool, ((0, True), (1, False), (2, True), ), "iseven")

model.save(filename)
