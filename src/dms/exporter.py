# Export a live Python function's source code to a .py file.
#
# Useful for taking functions defined interactively (e.g. in a notebook
# after symbolic derivation with sympy) and saving them to a reusable
# module. If the target file already contains a function with the same
# name, it is replaced in place. Otherwise the function is appended.

import inspect
import ast


def ExportFunPy(function: callable, rename=None, filepath='exportedfuns.py'):
    """Export a function's source code to a Python file.

    Args:
        function: The live function object to export.
        rename:   Optional new name for the function in the output file.
        filepath: Path to the target .py file (created if it doesn't exist).
    """
    func_code = inspect.getsource(function)

    # Optionally rename the function in the AST
    if rename is not None:
        tree = ast.parse(func_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                node.name = rename
        func_code = ast.unparse(tree)
    else:
        rename = function.__name__

    # Read existing file content (or start empty)
    try:
        with open(filepath, 'r') as file:
            code = file.read()
    except FileNotFoundError:
        code = ''

    # Parse existing code and look for a function with the same name
    tree = ast.parse(code)
    found = False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == rename:
            # Replace the existing function's args and body
            new_func = ast.parse(func_code).body[0]
            node.args = new_func.args
            node.body = new_func.body
            found = True

    # Write back: either the updated tree or append the new function
    new_code = ast.unparse(tree)
    with open(filepath, 'w') as file:
        file.write(new_code)

    if not found:
        with open(filepath, 'a') as file:
            file.write('\n')
            file.write(func_code)


if __name__ == '__main__':
    import math

    # Define a function we want to export
    def pendulum_torque(theta, theta_dot):
        g, L, m, c = 9.81, 1.0, 1.0, 0.1
        return -m * g * L * math.sin(theta) - c * theta_dot

    # Export it to a file
    ExportFunPy(pendulum_torque, filepath='exported_example.py')
    print("Exported 'pendulum_torque' to exported_example.py")

    # Export again with a different name -- appends a second function
    ExportFunPy(pendulum_torque, rename='my_torque', filepath='exported_example.py')
    print("Exported 'my_torque' to exported_example.py")

    # Show the result
    with open('examples/exported_example.py', 'r') as f:
        print("\n--- exported_example.py ---")
        print(f.read())