import codecs
import re

def modify_test_pipeline():
    with codecs.open('test_pipeline.py', 'r', 'utf-8') as f:
        content = f.read()

    # Fix imports
    content = content.replace("from environment import FastGaiaSimEnv, NativeGaiaEnv, make_gaia_env", "from environment import NativeGaiaEnv, make_gaia_env")

    # Remove the FastGaiaSimEnv specific test around line 100
    # Let's just find "def test_fast_gaia_sim_env" or similar, or just parse AST?
    pass

if __name__ == '__main__':
    modify_test_pipeline()
