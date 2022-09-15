
# ==============================================================================
#                        Checking a directory
# ==============================================================================
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import path_utility
path_utility.get_path('prj/main.py')
print('\n\n')

# ==============================================================================
#             Importing mnodule in case they have same paranet
# ==============================================================================
# [Case 1] Direct usage
def divide(a, b):
    return a/b
print('\nCase1 : ', divide(4, 2))

# [case 2] Direct importing a function  Using an Absolute Path
from inmodule2 import divide
print('\nCase2 : ',divide(4, 2))

# [case 3] Direct importing a module and use a function in the file
import inmodule3
print('\nCase3 : ',inmodule3.divide(4, 2))

# [case 4] Direct importing a module from a package
from pack1 import module1
print('\nCase4 : ',module1.addInPack1(4, 2))

# ==============================================================================
#      Importing module in different package in case they have same paranet
# ==============================================================================
# [case 5] Direct importing a function from a module
from pack1.module1 import addInPack1
print('\nCase5 : ',addInPack1(4, 2))


# [Case 6] Direct importing a module and use a function in the module
from pack1 import module3
print('\nCase6 :',module3.divideInPack1(4, 2))

# ==============================================================================
#             Importing mnodule in case they have different paranet
# ==============================================================================
# [Case 7] Importing module from parent package - Make parent dir same with current dir
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from pack2 import module4
print('\nCase7 :',module4.multiplyInPack2(4, 2))

# [Case 8] Importing module from parent package - Make parent dir same with current dir
from pack3 import module5
print('\nCase8 :',module5.multiplyInPack3(4, 2))

# [Case 9] Importing module from parent package - Make parent dir same with current dir
sys.path.append('E:/TorchProject/venv/GITHUB/PathExample/pack3')
import module5
print('\nCase9 :',module5.multiplyInPack3(4, 2))
