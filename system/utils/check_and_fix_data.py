"""
Data File Integrity Checker and Fixer
检查和修复损坏的数据文件

Usage:
    python utils/check_and_fix_data.py --check       # 检查所有数据文件
    python utils/check_and_fix_data.py --fix         # 修复损坏的文件
"""

import os
import sys
import numpy as np
import glob
import zipfile
from pathlib import Path

# Windows UTF-8 encoding fix
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def check_npz_file(file_path):
    """
    检查单个 .npz 文件是否损坏

    Returns:
        (bool, str): (是否正常, 错误信息)
    """
    try:
        with open(file_path, 'rb') as f:
            data = np.load(f, allow_pickle=True)

            # 检查是否包含 'data' 键
            if 'data' not in data:
                return False, "Missing 'data' key"

            # 尝试读取数据
            content = data['data'].tolist()

            # 检查数据结构
            if not isinstance(content, dict):
                return False, f"Data is not dict, got {type(content)}"

            if 'x' not in content or 'y' not in content:
                return False, "Missing 'x' or 'y' in data"

            # 检查数据大小
            x_shape = np.array(content['x']).shape
            y_shape = np.array(content['y']).shape

            if x_shape[0] != y_shape[0]:
                return False, f"Shape mismatch: x={x_shape}, y={y_shape}"

            return True, f"OK (samples={x_shape[0]})"

    except zipfile.BadZipFile as e:
        return False, f"BadZipFile: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def scan_all_data_files(base_dir='../dataset'):
    """
    扫描所有数据文件并检查完整性

    Returns:
        (list, list): (正常文件列表, 损坏文件列表)
    """
    print(f"[Scan] Scanning data files in {base_dir}...")

    # 查找所有 .npz 文件
    pattern = os.path.join(base_dir, '**', '*.npz')
    all_files = glob.glob(pattern, recursive=True)

    print(f"[Scan] Found {len(all_files)} .npz files")
    print()

    good_files = []
    bad_files = []

    for idx, file_path in enumerate(all_files, 1):
        is_ok, message = check_npz_file(file_path)

        # 简化路径显示
        rel_path = os.path.relpath(file_path, base_dir)

        if is_ok:
            status = "✓"
            good_files.append(file_path)
        else:
            status = "✗"
            bad_files.append((file_path, message))

        print(f"[{idx}/{len(all_files)}] {status} {rel_path}")
        if not is_ok:
            print(f"         Error: {message}")

    return good_files, bad_files


def delete_corrupted_files(bad_files):
    """删除损坏的文件"""
    print(f"\n[Delete] Deleting {len(bad_files)} corrupted files...")

    for file_path, error_msg in bad_files:
        try:
            os.remove(file_path)
            print(f"✓ Deleted: {file_path}")
        except Exception as e:
            print(f"✗ Failed to delete {file_path}: {e}")


def regenerate_data_suggestion(bad_files, base_dir='../dataset'):
    """提供重新生成数据的建议"""
    print("\n[Suggestion] To regenerate corrupted data files:")
    print("=" * 80)

    # 按数据集和alpha分组
    datasets = {}
    for file_path, _ in bad_files:
        rel_path = os.path.relpath(file_path, base_dir)
        parts = Path(rel_path).parts

        if len(parts) >= 3:
            alpha = parts[0]
            dataset = parts[1]
            key = f"{dataset}_alpha{alpha}"

            if key not in datasets:
                datasets[key] = []
            datasets[key].append(file_path)

    print("\nAffected datasets:")
    for key, files in datasets.items():
        print(f"  - {key}: {len(files)} corrupted files")

    print("\nRegeneration commands:")
    print("-" * 80)

    # 根据数据集提供重新生成命令
    for key in datasets.keys():
        parts = key.split('_alpha')
        if len(parts) == 2:
            dataset = parts[0]
            alpha = parts[1]

            print(f"\n# Regenerate {key}:")
            print(f"cd ../generate")

            if 'cifar' in dataset.lower():
                print(f"python generate_CIFAR10.py noniid - dir {alpha}")
            elif 'mnist' in dataset.lower():
                print(f"python generate_MNIST.py noniid - dir {alpha}")
            else:
                print(f"# Unknown dataset type: {dataset}")

    print("\n" + "=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Data File Integrity Checker')
    parser.add_argument('--check', action='store_true',
                        help='Check all data files for corruption')
    parser.add_argument('--fix', action='store_true',
                        help='Delete corrupted files (use with caution!)')
    parser.add_argument('--base_dir', type=str, default='../dataset',
                        help='Base directory for data files')

    args = parser.parse_args()

    if not args.check and not args.fix:
        parser.print_help()
        return

    print("=" * 80)
    print("Data File Integrity Checker")
    print("=" * 80)
    print()

    # 扫描所有文件
    good_files, bad_files = scan_all_data_files(args.base_dir)

    # 显示统计
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total files: {len(good_files) + len(bad_files)}")
    print(f"Good files: {len(good_files)} ✓")
    print(f"Bad files: {len(bad_files)} ✗")
    print()

    if bad_files:
        print("Corrupted files:")
        for file_path, error_msg in bad_files:
            rel_path = os.path.relpath(file_path, args.base_dir)
            print(f"  - {rel_path}")
            print(f"    Error: {error_msg}")

        # 提供修复建议
        regenerate_data_suggestion(bad_files, args.base_dir)

        if args.fix:
            print("\n" + "=" * 80)
            response = input("Are you sure you want to DELETE corrupted files? (yes/no): ")
            if response.lower() == 'yes':
                delete_corrupted_files(bad_files)
                print("\n[Info] Please regenerate the data using the commands above.")
            else:
                print("[Cancelled] No files deleted.")
    else:
        print("✓ All data files are intact!")
        print("=" * 80)


if __name__ == '__main__':
    main()
