import numpy as np
import numpy.testing as npt

def test_nextpow2():
    from osl_dynamics.utils.misc import nextpow2

    import numpy.testing as npt

    # Test case 1: Positive integer
    npt.assert_equal(nextpow2(10), 4)

    # Test case 2: Negative integer
    npt.assert_equal(nextpow2(-5), 3)

    # Test case 3: Zero
    npt.assert_equal(nextpow2(0), 0)

    # Test case 4: Large positive integer
    npt.assert_equal(nextpow2(1000), 10)

    # Test case 5: Large negative integer
    npt.assert_equal(nextpow2(-500), 9)


def test_leading_zeros():
    from osl_dynamics.utils.misc import leading_zeros

    import numpy.testing as npt

    # Test case 1: Single-digit number with largest number less than 10
    npt.assert_equal(leading_zeros(3, 9), '3')

    # Test case 2: Single-digit number with largest number equal to it
    npt.assert_equal(leading_zeros(7, 7), '7')

    # Test case 3: Single-digit number with largest number greater than 10
    npt.assert_equal(leading_zeros(5, 15), '05')

    # Test case 4: Double-digit number with largest number less than 100
    npt.assert_equal(leading_zeros(50, 99), '50')

    # Test case 5: Double-digit number with largest number equal to it
    npt.assert_equal(leading_zeros(100, 100), '100')

def test_override_dict_defaults():
    from osl_dynamics.utils.misc import override_dict_defaults

    # Test case 1: Override some default values
    default_dict = {'a': 1, 'b': 2, 'c': 3}
    override_dict = {'b': 20, 'c': 30}
    expected_result = {'a': 1, 'b': 20, 'c': 30}
    npt.assert_equal(override_dict_defaults(default_dict, override_dict), expected_result)

    # Test case 2: No override dictionary provided
    default_dict = {'a': 1, 'b': 2, 'c': 3}
    expected_result = {'a': 1, 'b': 2, 'c': 3}
    npt.assert_equal(override_dict_defaults(default_dict), expected_result)

    # Test case 3: Empty default dictionary and override some values
    default_dict = {}
    override_dict = {'a': 10, 'b': 20, 'c': 30}
    expected_result = {'a': 10, 'b': 20, 'c': 30}
    npt.assert_equal(override_dict_defaults(default_dict, override_dict), expected_result)

def test_listify():
    from osl_dynamics.utils.misc import listify

    # Test case 1: None input
    result = listify(None)
    expected_result = []
    npt.assert_equal(result, expected_result)

    # Test case 2: List input
    input_list = [1, 2, 3]
    result = listify(input_list)
    expected_result = [1, 2, 3]
    npt.assert_equal(result, expected_result)

    # Test case 3: Tuple input
    input_tuple = (4, 5, 6)
    result = listify(input_tuple)
    expected_result = [4, 5, 6]
    npt.assert_equal(result, expected_result)

    # Test case 4: Other object input
    input_obj = "hello"
    result = listify(input_obj)
    expected_result = ["hello"]
    npt.assert_equal(result, expected_result)



def test_replace_argument():
    from osl_dynamics.utils.misc import replace_argument
    # Define a dummy function for testing purposes
    def dummy_function(a, b, c=None):
        return (a, b, c)

    # Test case 1: Replace an existing argument in args list
    args = [1, 2, 3]
    kwargs = {'c': 4}
    name = 'b'
    item = 5
    append = False
    expected_args = [1, 5, 3]
    expected_kwargs = {'c': 4}
    modified_args, modified_kwargs = replace_argument(dummy_function, name, item, args, kwargs, append)
    npt.assert_equal(modified_args, expected_args)
    npt.assert_equal(modified_kwargs, expected_kwargs)

    # Test case 2: Replace an existing argument in kwargs
    args = [1, 2]
    kwargs = {'c': 3}
    name = 'c'
    item = 4
    append = False
    expected_args = [1, 2]
    expected_kwargs = {'c': 4}
    modified_args, modified_kwargs = replace_argument(dummy_function, name, item, args, kwargs, append)
    npt.assert_equal(modified_args, expected_args)
    npt.assert_equal(modified_kwargs, expected_kwargs)

    # Test case 3: Append to an existing argument list
    args = [1, [2, 3]]
    kwargs = {'c': [4, 5]}
    name = 'b'
    item = 6
    append = True
    expected_args = [1, [2, 3, 6]]
    expected_kwargs = {'c': [4, 5]}
    modified_args, modified_kwargs = replace_argument(dummy_function, name, item, args, kwargs, append)
    npt.assert_equal(modified_args, expected_args)
    npt.assert_equal(modified_kwargs, expected_kwargs)




