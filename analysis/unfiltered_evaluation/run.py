"""Score all test observations with the shared benchmark scorer."""
import runpy
import sys
from pathlib import Path

if __name__ == '__main__':
    sys.argv += ['--population', 'full']
    runpy.run_path(str(Path(__file__).resolve().parents[1] / 'score_benchmark.py'), run_name='__main__')
