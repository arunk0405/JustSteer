import os

def replace_nodes(content):
    mapping = {
        '"A"': '1', '"B"': '2', '"C"': '3', '"D"': '4', '"E"': '5', '"F"': '6',
        "'A'": '1', "'B'": '2', "'C'": '3', "'D'": '4', "'E'": '5', "'F'": '6'
    }
    for old, new in mapping.items():
        content = content.replace(old, new)
    return content

for root, dirs, files in os.walk('tests'):
    for file in files:
        if file.endswith('.py'):
            path = os.path.join(root, file)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            new_content = replace_nodes(content)
            if new_content != content:
                print(f"Updated {path}")
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
