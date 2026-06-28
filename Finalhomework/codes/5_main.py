"""
主程序：按顺序执行所有步骤
"""
import subprocess
import os
def run_script(script_name):
    print(f"\n{'=' * 60}")
    print(f"执行: {script_name}")
    print('=' * 60)
    result = subprocess.run(['python', script_name], capture_output=False)
    return result.returncode
def main():
    scripts = [
        '1_data_preprocessing.py',
        '2_seir_model.py',
        '3_lstm_model.py',
        '4_visualization.py'
    ]
    for script in scripts:
        if not os.path.exists(script):
            print(f"警告: {script} 不存在，跳过")
            continue
        ret = run_script(script)
        if ret != 0:
            print(f"错误: {script} 执行失败，退出")
            break
    print("\n" + "=" * 60)
    print("所有步骤执行完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()