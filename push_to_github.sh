#!/bin/bash

echo "🚀 潟湖水域分割项目 - GitHub推送脚本"
echo "=================================="

# 检查是否在正确的目录
if [ ! -f "README.md" ]; then
    echo "❌ 错误：请在项目根目录运行此脚本"
    exit 1
fi

echo "📁 当前目录：$(pwd)"
echo "📊 Git状态："
git status --short

echo ""
echo "🔧 尝试推送到GitHub..."
echo "仓库地址：https://github.com/retailer-yang/lagoon-water-segmentation"

# 尝试推送
if git push -u origin main; then
    echo "✅ 推送成功！"
    echo "🌐 查看仓库：https://github.com/retailer-yang/lagoon-water-segmentation"
else
    echo "❌ 推送失败"
    echo ""
    echo "🔧 可能的解决方案："
    echo "1. 确保SSH密钥已添加到GitHub"
    echo "2. 检查网络连接"
    echo "3. 确认仓库已创建且为公开"
    echo ""
    echo "📋 手动推送命令："
    echo "git push -u origin main"
fi
