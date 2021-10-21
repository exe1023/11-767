import sys
import yaml


with open(sys.argv[1], "r") as f:    
    config = yaml.safe_load(f)

with open('token_list.txt', 'w') as f:
    for token in config['token_list']:
        f.write(f'{token}\n')
