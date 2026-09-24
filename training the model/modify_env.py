import codecs

def modify_env():
    with codecs.open('environment.py', 'r', 'utf-8') as f:
        lines = f.readlines()
        
    new_lines = []
    skip = False
    for line in lines:
        if line.startswith('class FastGaiaSimEnv:'):
            skip = True
        if line.startswith('class RestGaiaEnv:'):
            skip = False
        if not skip:
            new_lines.append(line)
            
    content = "".join(new_lines)
    
    content = content.replace('"""Factory creating NativeGaiaEnv (DLL) > RestGaiaEnv > FastGaiaSimEnv."""', '"""Factory creating NativeGaiaEnv (DLL) > RestGaiaEnv."""')
    
    old_ret = '    return FastGaiaSimEnv(players=players, seed=seed if seed is not None else 42, **kwargs)'
    new_ret = '    raise RuntimeError("Native Gaia DLL not found. FastGaiaSimEnv has been removed. Please build the Rust engine.")'
    content = content.replace(old_ret, new_ret)
    
    with codecs.open('environment.py', 'w', 'utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    modify_env()
