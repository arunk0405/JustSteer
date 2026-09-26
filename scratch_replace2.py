import os

path = 'tests/test_astar.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('"S"', '0')
with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
