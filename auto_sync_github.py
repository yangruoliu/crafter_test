#!/usr/bin/env python3
"""
自动Git同步脚本
确保所有代码修改自动同步到GitHub分支
"""

import subprocess
import os
import time
import sys
from datetime import datetime

class AutoGitSync:
    def __init__(self, branch_name="cursor/check-ppo-loss-normalization-for-direction-loss-c7d3"):
        self.branch_name = branch_name
        self.repo_url = "https://github.com/yangruoliu/crafter_test"
        
    def run_command(self, command, check=True):
        """运行shell命令并返回结果"""
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True, 
                check=check
            )
            return result.stdout.strip(), result.stderr.strip(), result.returncode
        except subprocess.CalledProcessError as e:
            return "", e.stderr, e.returncode
    
    def check_git_status(self):
        """检查git状态"""
        stdout, stderr, code = self.run_command("git status --porcelain")
        return stdout.strip() != ""  # 如果有输出，说明有变化
    
    def get_git_status_summary(self):
        """获取git状态摘要"""
        stdout, stderr, code = self.run_command("git status --short")
        return stdout
    
    def add_all_changes(self):
        """添加所有更改到暂存区"""
        stdout, stderr, code = self.run_command("git add .")
        return code == 0
    
    def commit_changes(self, message=None):
        """提交更改"""
        if message is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = f"Auto-sync: Update code changes - {timestamp}"
        
        stdout, stderr, code = self.run_command(f'git commit -m "{message}"')
        return code == 0, stdout, stderr
    
    def push_to_remote(self):
        """推送到远程仓库"""
        stdout, stderr, code = self.run_command(f"git push origin {self.branch_name}")
        return code == 0, stdout, stderr
    
    def sync_now(self, commit_message=None):
        """立即同步所有更改"""
        print("🔄 开始自动同步...")
        
        # 检查是否有更改
        if not self.check_git_status():
            print("✅ 没有需要同步的更改")
            return True
        
        # 显示待同步的文件
        status_summary = self.get_git_status_summary()
        print(f"📁 待同步的文件:")
        for line in status_summary.split('\n'):
            if line.strip():
                print(f"   {line}")
        
        # 添加所有更改
        print("📦 添加文件到暂存区...")
        if not self.add_all_changes():
            print("❌ 添加文件失败")
            return False
        
        # 提交更改
        print("💾 提交更改...")
        success, stdout, stderr = self.commit_changes(commit_message)
        if not success:
            print(f"❌ 提交失败: {stderr}")
            return False
        
        print(f"✅ 提交成功: {stdout}")
        
        # 推送到远程
        print("🚀 推送到GitHub...")
        success, stdout, stderr = self.push_to_remote()
        if not success:
            print(f"❌ 推送失败: {stderr}")
            return False
        
        print("✅ 同步完成!")
        print(f"🔗 查看更改: {self.repo_url}/tree/{self.branch_name}")
        return True
    
    def setup_auto_sync_hook(self):
        """设置自动同步钩子 (每次保存文件时自动同步)"""
        hook_script = '''#!/bin/bash
# Auto-sync git hook
python3 auto_sync_github.py --auto
'''
        
        # 创建git hooks目录（如果不存在）
        hooks_dir = ".git/hooks"
        if not os.path.exists(hooks_dir):
            os.makedirs(hooks_dir)
        
        # 创建post-commit hook
        hook_path = os.path.join(hooks_dir, "post-commit")
        with open(hook_path, 'w') as f:
            f.write(hook_script)
        
        # 设置执行权限
        os.chmod(hook_path, 0o755)
        
        print(f"✅ 自动同步钩子已设置: {hook_path}")
        print("💡 现在每次提交后会自动推送到GitHub")

def main():
    sync = AutoGitSync()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--auto":
            # 自动模式，静默运行
            sync.sync_now("Auto-sync: Automated update")
        elif sys.argv[1] == "--setup":
            # 设置自动同步钩子
            sync.setup_auto_sync_hook()
        elif sys.argv[1] == "--message" and len(sys.argv) > 2:
            # 带自定义消息的同步
            message = " ".join(sys.argv[2:])
            sync.sync_now(message)
        else:
            print("用法:")
            print("  python auto_sync_github.py              # 手动同步")
            print("  python auto_sync_github.py --auto       # 自动同步 (钩子使用)")
            print("  python auto_sync_github.py --setup      # 设置自动同步钩子")
            print("  python auto_sync_github.py --message 'commit message'  # 带消息同步")
    else:
        # 手动同步模式
        sync.sync_now()

if __name__ == "__main__":
    main()