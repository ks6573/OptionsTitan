#!/bin/bash
# OptionsTitan Installation Script with UV
# Modern, fast installation using UV package manager

echo "╔══════════════════════════════════════════════════════════╗"
echo "║         OptionsTitan - Installation with UV              ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 UV not found. Installing UV..."
    echo ""
    
    # Install UV
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    if [ $? -eq 0 ]; then
        echo "✅ UV installed successfully"
        
        # Source the shell configuration to make uv available
        if [ -f "$HOME/.bashrc" ]; then
            source "$HOME/.bashrc"
        elif [ -f "$HOME/.zshrc" ]; then
            source "$HOME/.zshrc"
        fi
    else
        echo "❌ Failed to install UV"
        echo "Falling back to pip installation..."
        pip install -r requirements.txt
        exit $?
    fi
else
    echo "✅ UV is already installed"
fi

echo ""
echo "📦 Installing OptionsTitan dependencies..."
echo ""

# Sync dependencies using UV
uv sync

if [ $? -eq 0 ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║              Installation Complete! ✨                    ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""
    echo "🚀 You can now run:"
    echo "   • Modern GUI:  uv run python options_gui_qt.py"
    echo "   • Classic GUI: uv run python options_gui.py"
    echo "   • Train AI:    uv run python main.py"
    echo ""
    echo "📖 Next steps:"
    echo "   1. Read GETTING_STARTED.md for setup guide"
    echo "   2. Optional: Enable LLAMA AI (./scripts/setup_llama.sh)"
    echo "   3. Run: uv run python verify_installation.py"
    echo ""
else
    echo ""
    echo "❌ Installation failed with UV"
    echo ""
    echo "🔧 Try manual installation:"
    echo "   pip install -r requirements.txt"
    echo ""
    exit 1
fi
