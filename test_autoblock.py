import autoblock as ab

def test_1():
    m = ab.Map()
    ab.add_int(m)
    ab.add_float(m)
    f = m.search(int, str, 16, "4")
    func = ab.function(f)
    t = type(f)

    print([x[1] for x in f])

    assert((t is tuple or t is list) and len(f) > 1)
    assert("float_sqrt" in [x[1] for x in f])

def test_2():
    m = ab.Map()
    ab.add_list(m)
    ab.add_ndarray(m)
    f = m.search(list, list, [4.0], [2.0])
    func = ab.function(f)
    t = type(f)
    
    print([x[1] for x in f])

    assert((t is tuple or t is list) and len(f) > 1)

def test_3():
    m = ab.Map()
    f = m.search(int, str, 4, "2")

    assert(f is None)

def test_4():
    m = ab.Map()
    ab.add_list(m)
    ab.add_ndarray(m)
    f = m.search(list, int, [256.0, 16.0], 4)
    func = ab.function(f)
    t = type(f)
    
    print([x[1] for x in f])

    assert((t is tuple or t is list) and len(f) > 1)

def test_5():
    m = ab.Map()
    ab.add_list(m)
    ab.add_ndarray(m)
    f = m.search(list, list, [256, 16, 4, 1], [1, 2, 4, 16])
    func = ab.function(f)
    t = type(f)

    print([x[1] for x in f])

    assert((t is tuple or t is list) and len(f) > 1)

def test_6():
    m = ab.Map()
    ab.add_int(m)
    ab.add_float(m)
    inputOutputList = ((4, "2"), (1, "1"))
    f = m.searchList(int, str, inputOutputList)
    func, inputType = ab.functionList(f, int, inputOutputList)
    t = type(f)

    print([x[1] for x in f])

    assert((t is tuple or t is list) and len(f) > 1)
    assert("float_sqrt" in [x[1] for x in f])

def test_7():
    m = ab.Map()
    ab.add_bool(m)
    ab.add_int(m)
    inputOutputList = (((0, 2), False), ((1, 2), True), ((2, 2), False), )
    f = m.searchList((int, int), bool, inputOutputList)
    func, inputType = ab.functionList(f, (int, int), inputOutputList)
    t = type(f)

    print([x[1] for x in f])

    assert((t is tuple or t is list) and len(f) > 1)
    assert("int_mod" in [x[1] for x in f])
    
    print(func(4))
    print(func(5))

    m.add(func, inputType, bool, "isodd")

    inputOutputList = ((0, True), (1, False), (2, True), )
    f = m.searchList(int, bool, inputOutputList)
    func, inputType = ab.functionList(f, int, inputOutputList)
    t = type(f)

    print([x[1] for x in f])

    assert((t is tuple or t is list) and len(f) > 1)
    assert("isodd" in [x[1] for x in f])

    print(func(4))
    print(func(5))
