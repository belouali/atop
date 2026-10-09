"""Deprecated: this script rebuilt tensors_test.pt from ``combined_full`` and
dropped the in-sequence [PAD] placeholders that encode out-of-vocabulary
codes, shifting token positions for 7.3% of test patients. Use
``scripts/regen_tensors_test_fixed.py``, which rebuilds from ``combined_map``
and reproduces the original model inputs exactly.
"""
import sys

sys.exit("regen_tensors_test.py is deprecated (it misaligns positions); "
         "run scripts/regen_tensors_test_fixed.py instead.")
