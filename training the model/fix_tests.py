import codecs
import re

def fix_tests():
    with codecs.open('test_pipeline.py', 'r', 'utf-8') as f:
        content = f.read()

    # Fix imports
    content = content.replace("from environment import FastGaiaSimEnv, NativeGaiaEnv, make_gaia_env", "from environment import NativeGaiaEnv, make_gaia_env")

    # Remove test_environment() block
    # It starts at "def test_environment():" and ends at "def test_trainer_single_step():"
    
    start_str = "def test_environment():"
    end_str = "def test_trainer_single_step():"
    start_idx = content.find(start_str)
    end_idx = content.find(end_str)
    
    if start_idx != -1 and end_idx != -1:
        content = content[:start_idx] + content[end_idx:]

    # Replace remaining FastGaiaSimEnv with make_gaia_env
    content = content.replace("FastGaiaSimEnv(", "make_gaia_env(")
    
    with codecs.open('test_pipeline.py', 'w', 'utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    fix_tests()
