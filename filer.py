import os
import re
import json
from pathlib import Path

def detect_naming_convention(name):
    if re.match(r'^[a-z0-9_]+$', name):
        return 'snake_case'
    elif re.match(r'^[a-z]+([A-Z][a-z0-9]+)+$', name):
        return 'camelCase'
    elif re.match(r'^[A-Z][a-zA-Z0-9]+$', name):
        return 'PascalCase'
    else:
        return 'unknown'

def scan_folder_structure(root_path):
    structure_info = []

    for root, dirs, files in os.walk(root_path):
        for d in dirs:
            dir_path = os.path.join(root, d)
            relative_dir = os.path.relpath(dir_path, root_path)
            structure_info.append({
                'type': 'folder',
                'path': relative_dir,
                'naming_convention': detect_naming_convention(Path(relative_dir).name)
            })
        
        for f in files:
            file_path = os.path.join(root, f)
            relative_file = os.path.relpath(file_path, root_path)
            structure_info.append({
                'type': 'file',
                'path': relative_file,
                'naming_convention': detect_naming_convention(Path(relative_file).stem) # 拡張子を除く
            })

    return structure_info

def save_structure_info(structure_info, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(structure_info, f, indent=2, ensure_ascii=False)

def main():
    target_root = input("解析するルートフォルダパスを入力してください: ").strip()
    output_file = input("出力するJSONファイル名を入力してください (例: structure.json): ").strip()

    structure_info = scan_folder_structure(target_root)
    save_structure_info(structure_info, output_file)
    print(f"✅ スキャン完了！結果を '{output_file}' に保存しました。")

if __name__ == "__main__":
    main()