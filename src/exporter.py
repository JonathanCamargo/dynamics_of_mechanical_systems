import inspect
import ast

def ExportFunPy(function: callable, rename=None, filepath='exportedfuns.py'):    
    func_code = inspect.getsource(function)
    if rename!=None:        
        tree = ast.parse(func_code)
        # Find function definitions and replace the function name with "myfun"
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                node.name = rename
        func_code = ast.unparse(tree)
    else:
        rename=function.__name__
    
    # Load the content of the file
    file_exists=True
    try:
        with open(filepath, 'r') as file:
            code = file.read()
    except FileNotFoundError:
        file_exists=False
    if file_exists==False:
        with open(filepath, 'w') as file:
            code = ''
        

    # Parse the code into an abstract syntax tree (AST)
    tree = ast.parse(code)
        
    # Find function definitions with name 
    found=False
    for node in tree.body:       
        if isinstance(node, ast.FunctionDef) and node.name == rename:
            # Replace the function's body with new content
            node.body = ast.parse(func_code).body[0].body            
            found=True
                
        # Generate Python code from the modified AST
    new_code = ast.unparse(tree)    
    # Write the modified code back to the file
    with open(filepath, 'w') as file:
        file.write(new_code)
            
    if found==False:            
        with open(filepath, 'a') as file:
            file.write('\n')
            file.write(func_code)      