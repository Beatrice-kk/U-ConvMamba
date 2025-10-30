# from ..VMamba.vmamba import VSSBlock

import os
import sys


moudle_dir=os.path.abspath(os.path.join(os.path.dirname(__file__),'../VMamba'))


sys.path.append(moudle_dir)


from vmamba import VSSBlock

# import vmamba

print("imported import_test.py successfully")
