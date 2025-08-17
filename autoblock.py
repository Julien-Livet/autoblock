from collections import deque
import dill
import functools
import math
import numpy as np

def is_equal(a, b):
    if (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
        return np.array_equal(a, b)
    elif (isinstance(a, list) and isinstance(b, list)):
        if (len(a) != len(b)):
            return False

        return all(is_equal(x, y) for x, y in zip(a, b))
    elif (isinstance(a, float) and isinstance(b, float)):
        return abs(a - b) < 1e-9

    return a == b

class Map:
    def __init__(self):
        self.functions = {}

    def add(self, function, inputType, outputType, name = ""):
        l = self.functions.get((inputType, outputType), [])
        l.append((function, name))
        self.functions[(inputType, outputType)] = l

    def searchList(self, inputType, outputType, inputOutputValues):
        if (len(inputOutputValues) == 0):
            return None

        visited = set()
        queue = deque()
        queue.append((inputType, [x[0] for x in inputOutputValues], []))  # (type, current_value, path)

        while (queue):
            current_type, listValues, path = queue.popleft()

            if (len(listValues) == 0):
                continue

            values = []
            
            for current_value in listValues:
                if (type(current_value) != current_type):
                    if (current_type is np.ndarray):
                        current_value = np.array([current_value])
                        
                        assert(type(current_value) == current_type)
                    elif (current_type is float or current_type is int):
                        current_value = current_type(current_value)
                        
                        assert(type(current_value) == current_type)
                        
                values.append(current_value)

            key = (current_type, self._hashable(values))
            
            if (key in visited):
                continue
                
            visited.add(key)

            for (src_type, dst_type), funcs in self.functions.items():
                if (src_type != current_type):
                    continue

                for f in funcs:
                    new_result = []
                    count = 0
                    
                    for i in range(0, len(listValues)):
                        current_value = listValues[i]
                    
                        try:
                            result = f[0](*current_value)
                        except:
                            try:
                                result = f[0](current_value)
                            except:
                                break
                                
                        new_result.append(result)
                        
                        if (dst_type == outputType and is_equal(result, inputOutputValues[i][1])):
                            count += 1

                    new_path = path + [f]
                    
                    if (count == len(listValues)):
                        return new_path
                    
                    queue.append((dst_type, new_result, new_path))

        return None
        
    def search(self, inputType, outputType, input_value, expected_output):
        visited = set()
        queue = deque()
        queue.append((inputType, input_value, []))  # (type, current_value, path)

        while (queue):
            current_type, current_value, path = queue.popleft()
            
            if (type(current_value) != current_type):
                if (current_type is np.ndarray):
                    current_value = np.array([current_value])
                    
                    assert(type(current_value) == current_type)
                elif (current_type is float or current_type is int):
                    current_value = current_type(current_value)
                    
                    assert(type(current_value) == current_type)

            key = (current_type, self._hashable(current_value))
            
            if (key in visited):
                continue
                
            visited.add(key)

            for (src_type, dst_type), funcs in self.functions.items():
                if (src_type != current_type):
                    continue

                for f in funcs:
                    try:
                        result = f[0](*current_value)
                    except:
                        try:
                            result = f[0](current_value)
                        except:
                            continue
                        
                    new_path = path + [f]
                    
                    if (dst_type == outputType and is_equal(result, expected_output)):
                        return new_path
                    
                    queue.append((dst_type, result, new_path))

        return None

    def _hashable(self, x):
        if (isinstance(x, np.ndarray)):
            return (x.shape, tuple(x.flatten()))
        elif (isinstance(x, (list, tuple))):
            return tuple(self._hashable(e) for e in x)
        elif (isinstance(x, dict)):
            return tuple(sorted((k, self._hashable(v)) for k, v in x.items()))

        return x

    def save(self, filename):
        with open(filename, "wb") as f:
            dill.dump(self.functions, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            self.functions = dill.load(f)

def function(functions):
    return lambda x: functools.reduce(
        lambda acc, f: f(*acc) if isinstance(acc, (tuple, list)) else f(acc),
        [x[0] for x in functions],
        x
    )

def functionList(functions, inputType, inputOutputList):
    if (type(inputType) is tuple):
        inputType = list(inputType)
    
    assert(len(inputOutputList))
    
    t = type(inputOutputList[0][0])
    
    indices = []
    constants = []
    
    if (t is list or t is tuple):
        counters = [0] * len(inputOutputList[0])
        
        for i in range(1, len(inputOutputList)):
            for j in range(0, len(inputOutputList[i][0])):
                if (inputOutputList[i][0][j] == inputOutputList[0][0][j]):
                    counters[j] += 1
                    
                    if (counters[j] == len(inputOutputList) - 1):
                        indices.append(j)
                        constants.append(inputOutputList[i][0][j])

    def wrap_input(x):
        if (not indices):
            return x
            
        if (not isinstance(x, (tuple, list))):
            x = (x,)
            
        result = list(x)
        
        for idx, const in sorted(zip(indices, constants)):
            result.insert(idx, const)
            
        return tuple(result)

    def final_func(x):
        full_input = wrap_input(x)
        
        return functools.reduce(
            lambda acc, f: f(*acc) if isinstance(acc, (tuple, list)) else f(acc),
            [x[0] for x in functions],
            full_input
        )

    for i in reversed(indices):
        del inputType[i]

    if (type(inputType) is list):
        inputType = tuple(inputType)
        
        if (len(inputType) == 1):
            inputType = inputType[0]

    return final_func, inputType

def add_int(m):
    m.add(lambda x: str(x), int, str, "int_str")
    m.add(lambda x, y: x + y, (int, int), int, "int_add")
    m.add(lambda x, y: x - y, (int, int), int, "int_sub")
    m.add(lambda x: -x, int, int, "int_neg")
    m.add(lambda x, y: x * y, (int, int), int, "int_mul")
    m.add(lambda x, y: x % y, (int, int), int, "int_mod")
    m.add(lambda x, y: x ** y, (int, int), int, "int_pow")
    m.add(lambda x, y: x / y, (int, int), int, "int_div")
    m.add(lambda x, y: min(x, y), (int, int), int, "int_min")
    m.add(lambda x, y: max(x, y), (int, int), int, "int_max")
    m.add(lambda x: abs(x), int, int, "int_abs")
    m.add(lambda x: float(x), int, float, "int_float")
    m.add(lambda x: bool(x), int, bool, "int_bool")
    m.add(lambda x: np.ndarray(x), int, np.ndarray, "int_np")
    m.add(lambda x, y: (y, x), (int, int), (int, int), "int_swap")
    m.add(lambda x, y: (y, x), (int, int), (int, int), "int_swap")
    m.add(lambda x, y: x > y, (int, int), bool, "int_greater")
    m.add(lambda x, y: x >= y, (int, int), bool, "int_greater_equal")
    m.add(lambda x, y: x < y, (int, int), bool, "int_less")
    m.add(lambda x, y: x <= y, (int, int), bool, "int_less_equal")
    m.add(lambda x, y: x == y, (int, int), bool, "int_equal")
    m.add(lambda x, y: x != y, (int, int), bool, "int_not_equal")
    m.add(lambda stop: list(range(stop)), (int), list, "range")
    m.add(lambda start, stop: list(range(start, stop)), (int, int), list, "range")
    m.add(lambda start, stop, step: list(range(start, stop, step)), (int, int, int), list, "range")

def add_float(m):
    m.add(lambda x: str(x), float, str, "float_str")
    m.add(lambda x, y: x + y, (float, float), float, "float_add")
    m.add(lambda x, y: x - y, (float, float), float, "float_sub")
    m.add(lambda x: -x, float, float, "float_neg")
    m.add(lambda x, y: x * y, (float, float), float, "float_mul")
    m.add(lambda x, y: x ** y, (float, float), float, "float_pow")
    m.add(lambda x, y: x / y, (float, float), float, "float_div")
    m.add(lambda x, y: min(x, y), (float, float), float, "float_min")
    m.add(lambda x, y: max(x, y), (float, float), float, "float_max")
    m.add(lambda x: int(x), float, int, "float_int")
    m.add(lambda x: math.exp(x), float, float, "float_exp")
    m.add(lambda x: math.log(x), float, float, "float_log")
    m.add(lambda x: math.cos(x), float, float, "float_cos")
    m.add(lambda x: math.sin(x), float, float, "float_sin")
    m.add(lambda x: math.tan(x), float, float, "float_tan")
    m.add(lambda x: math.acos(x), float, float, "float_acos")
    m.add(lambda x: math.asin(x), float, float, "float_asin")
    m.add(lambda x: math.atan(x), float, float, "float_atan")
    m.add(lambda x: math.cosh(x), float, float, "float_cosh")
    m.add(lambda x: math.sinh(x), float, float, "float_sinh")
    m.add(lambda x: math.tanh(x), float, float, "float_tanh")
    m.add(lambda x: math.acosh(x), float, float, "float_acosh")
    m.add(lambda x: math.asinh(x), float, float, "float_asinh")
    m.add(lambda x: math.atanh(x), float, float, "float_atanh")
    m.add(lambda x: 1 / x, float, float, "float_inverse")
    m.add(lambda x: math.sqrt(x), float, float, "float_sqrt")
    m.add(lambda x: math.floor(x), float, float, "float_floor")
    m.add(lambda x: math.ceil(x), float, float, "float_ceil")
    m.add(lambda x: math.abs(x), float, float, "float_abs")
    m.add(lambda x: np.ndarray(x), float, np.ndarray, "float_np")
    m.add(lambda x, y: (y, x), (float, float), (float, float), "float_swap")
    m.add(lambda x, y: x > y, (float, float), bool, "float_greater")
    m.add(lambda x, y: x >= y, (float, float), bool, "float_greater_equal")
    m.add(lambda x, y: x < y, (float, float), bool, "float_less")
    m.add(lambda x, y: x <= y, (float, float), bool, "float_less_equal")
    m.add(lambda x, y: x == y, (float, float), bool, "float_equal")
    m.add(lambda x, y: x != y, (float, float), bool, "float_not_equal")
    m.add(lambda start, stop: np.linspace(start, stop), (float, float), np.ndarray, "linspace")
    m.add(lambda start, stop, num: np.linspace(start, stop, num), (float, float, float), np.ndarray, "linspace")

def add_list(m):
    m.add(lambda x: np.array(x), list, np.ndarray, "list_np")
    m.add(lambda x: x[::-1], list, list, "list_reverse")
    m.add(lambda x, y: x * y, (list, int), list, "list_repeat")
    m.add(lambda x, y: x + y, (list, list), list, "list_concat")
    m.add(lambda x: [int(xx) for xx in x], list, list, "list_int")
    m.add(lambda x: [float(xx) for xx in x], list, list, "list_float")
    m.add(lambda x, y: (y, x), (list, list), (list, list), "list_swap")
    m.add(lambda x: all(x), list, bool, "list_all")
    m.add(lambda x: any(x), list, bool, "list_any")
    m.add(lambda x: sorted(x), list, list, "list_sorted")
    m.add(lambda x: reversed(x), list, list, "list_reversed")
    m.add(lambda x: len(x), list, int, "list_len")
    m.add(lambda x: min(x), list, int, "list_min_int")
    m.add(lambda x: min(x), list, float, "list_min_float")
    m.add(lambda x: max(x), list, int, "list_max_int")
    m.add(lambda x: max(x), list, float, "list_max_float")
    m.add(lambda x: tuple(x), list, tuple, "list_tuple")

def add_ndarray(m):
    m.add(lambda x, y: x + y, (np.ndarray, np.ndarray), np.ndarray, "np_add")
    m.add(lambda x, y: x - y, (np.ndarray, np.ndarray), np.ndarray, "np_sub")
    m.add(lambda x, y: x * y, (np.ndarray, np.ndarray), np.ndarray, "np_mul")
    m.add(lambda x, y: x / y, (np.ndarray, np.ndarray), np.ndarray, "np_div")
    m.add(lambda x, y: x ** y, (np.ndarray, np.ndarray), np.ndarray, "np_pow")
    m.add(lambda x, y: np.logical_and(x, y), (np.ndarray, np.ndarray), np.ndarray, "np_and")
    m.add(lambda x, y: np.logical_or(x, y), (np.ndarray, np.ndarray), np.ndarray, "np_or")
    m.add(lambda x, y: np.logical_xor(x, y), (np.ndarray, np.ndarray), np.ndarray, "np_xor")
    m.add(lambda x: np.logical_not(x), np.ndarray, np.ndarray, "np_not")
    m.add(lambda x, y: np.greater(x, y), (np.ndarray, np.ndarray), np.ndarray, "np_greater")
    m.add(lambda x, y: np.greater_equal(x, y), (np.ndarray, np.ndarray), np.ndarray, "np_greater_equal")
    m.add(lambda x, y: np.less(x, y), (np.ndarray, np.ndarray), np.ndarray, "np_less")
    m.add(lambda x, y: np.less_equal(x, y), (np.ndarray, np.ndarray), np.ndarray, "np_less_equal")
    m.add(lambda x, y: np.equal(x, y), (np.ndarray, np.ndarray), np.ndarray, "np_equal")
    m.add(lambda x, y: np.not_equal(x, y), (np.ndarray, np.ndarray), np.ndarray, "np_not_equal")
    m.add(lambda x: -x, np.ndarray, np.ndarray, "np_neg")
    m.add(lambda x: np.min(x), np.ndarray, np.ndarray, "np_min")
    m.add(lambda x: np.max(x), np.ndarray, np.ndarray, "np_max")
    m.add(lambda x: np.min(x), np.ndarray, float, "np_min_float")
    m.add(lambda x: np.max(x), np.ndarray, float, "np_max_float")
    m.add(lambda x: np.log(x), np.ndarray, np.ndarray, "np_log")
    m.add(lambda x: np.exp(x), np.ndarray, np.ndarray, "np_exp")
    m.add(lambda x: np.cos(x), np.ndarray, np.ndarray, "np_cos")
    m.add(lambda x: np.sin(x), np.ndarray, np.ndarray, "np_sin")
    m.add(lambda x: np.tan(x), np.ndarray, np.ndarray, "np_tan")
    m.add(lambda x: np.acos(x), np.ndarray, np.ndarray, "np_acos")
    m.add(lambda x: np.asin(x), np.ndarray, np.ndarray, "np_asin")
    m.add(lambda x: np.atan(x), np.ndarray, np.ndarray, "np_atan")
    m.add(lambda x: np.cosh(x), np.ndarray, np.ndarray, "np_cosh")
    m.add(lambda x: np.sinh(x), np.ndarray, np.ndarray, "np_sinh")
    m.add(lambda x: np.tanh(x), np.ndarray, np.ndarray, "np_tanh")
    m.add(lambda x: np.acosh(x), np.ndarray, np.ndarray, "np_acosh")
    m.add(lambda x: np.asinh(x), np.ndarray, np.ndarray, "np_asinh")
    m.add(lambda x: np.atanh(x), np.ndarray, np.ndarray, "np_atanh")
    m.add(lambda x: np.floor(x), np.ndarray, np.ndarray, "np_floor")
    m.add(lambda x: np.ceil(x), np.ndarray, np.ndarray, "np_ceil")
    m.add(lambda x: np.abs(x), np.ndarray, np.ndarray, "np_abs")
    m.add(lambda x: np.sqrt(x), np.ndarray, np.ndarray, "np_sqrt")
    m.add(lambda x: 1 / x, np.ndarray, np.ndarray, "np_inverse")
    m.add(lambda x: np.sum(x), np.ndarray, np.ndarray, "np_sum")
    m.add(lambda x: np.sum(x), np.ndarray, float, "np_sum_float")
    m.add(lambda x: list(x), np.ndarray, list, "np_list")
    m.add(lambda x: int(x), np.ndarray, int, "np_int")
    m.add(lambda x: float(x), np.ndarray, float, "np_float")
    m.add(lambda x, y: (y, x), (np.ndarray, np.ndarray), (np.ndarray, np.ndarray), "np_swap")
    m.add(lambda x: np.all(x), np.ndarray, np.ndarray, "np_all")
    m.add(lambda x: np.any(x), np.ndarray, np.ndarray, "np_any")
    m.add(lambda x: np.square(x), np.ndarray, np.ndarray, "np_square")
    m.add(lambda x: np.min(x), nd.array, int, "np_min_int")
    m.add(lambda x: np.min(x), nd.array, float, "np_min_float")
    m.add(lambda x: np.max(x), nd.array, int, "np_max_int")
    m.add(lambda x: np.max(x), nd.array, float, "np_max_float")

def add_str(m):
    m.add(lambda x: int(x), str, int, "str_int")
    m.add(lambda x: float(x), str, float, "str_float")
    m.add(lambda x: bool(x), str, bool, "str_bool")
    m.add(lambda x, y: (y, x), (str, str), (str, str), "str_swap")
    m.add(lambda x: len(x), str, int, "str_len")
    m.add(lambda x: x.lower(), str, str, "str_lower")
    m.add(lambda x: x.upper(), str, str, "str_upper")

def add_tuple(m):
    m.add(lambda x: tuple(reversed(x)), tuple, tuple, "tuple_swap")
    m.add(lambda x: list(x), tuple, list, "tuple_list")

def add_bool(m):
    m.add(lambda x: not x, bool, bool, "bool_not")
    m.add(lambda x, y: x and y, (bool, bool), bool, "bool_and")
    m.add(lambda x, y: x or y, (bool, bool), bool, "bool_or")
    m.add(lambda x, y: x ^ y, (bool, bool), bool, "bool_xor")
    m.add(lambda x, y: (y, x), (bool, bool), (bool, bool), "bool_swap")
    m.add(lambda x: str(x), bool, str, "bool_str")
