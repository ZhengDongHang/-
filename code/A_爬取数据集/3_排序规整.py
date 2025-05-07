import os
import time

def sort_and_rename_files_by_mtime(root_folder):
    # 遍历主文件夹中的所有子文件夹
    for subdir, dirs, files in os.walk(root_folder):
        # 跳过根目录，只处理子文件夹
        if subdir == root_folder:
            continue

        # 构建文件的完整路径
        file_paths = [os.path.join(subdir, f) for f in files if os.path.isfile(os.path.join(subdir, f))]

        # 如果文件夹为空，跳过
        if not file_paths:
            continue

        # 按照修改时间排序
        file_paths.sort(key=lambda x: os.path.getmtime(x))

        # 给每个文件添加编号前缀
        for i, old_path in enumerate(file_paths, start=1):
            folder, old_name = os.path.split(old_path)

            # 如果已经有类似"1_"前缀的名字，先移除旧前缀
            parts = old_name.split("_", 1)
            if parts[0].isdigit() and len(parts) > 1:
                base_name = parts[1]
            else:
                base_name = old_name

            new_name = f"{i}_{base_name}"
            new_path = os.path.join(folder, new_name)

            # 避免重名覆盖
            if old_path != new_path:
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")


root_directory = "novel_data"
sort_and_rename_files_by_mtime(root_directory)
