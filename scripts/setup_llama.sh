#!/bin/bash
# OptionsTitan LLAMA AI Setup Script

echo "╔══════════════════════════════════════════════════════════╗"
echo "║      OptionsTitan - LLAMA AI Enhancement Setup          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed."
    exit 1
fi

# Step 1: Install LLAMA API client
echo "📦 Step 1: Installing LLAMA API client..."

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "Using uv for installation..."
    uv pip install llama-api-client
else
    echo "Using pip for installation..."
    pip3 install llama-api-client
fi

if [ $? -eq 0 ]; then
    echo "✅ LLAMA API client installed successfully"
else
    echo "❌ Failed to install LLAMA API client"
    exit 1
fi

echo ""

# Step 2: Create .env file
echo "📝 Step 2: Setting up environment file..."

if [ -f ".env" ]; then
    echo "⚠️  .env file already exists"
    read -p "Do you want to update it? (y/n): " response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Keeping existing .env file"
        exit 0
    fi
fi

# Prompt for API key
echo ""
echo "Please enter your LLAMA API key:"
echo "(Get it from: https://api.llama.com/)"
echo ""
read -p "LLAMA API Key: " api_key

if [ -z "$api_key" ]; then
    echo "❌ No API key provided"
    exit 1
fi

# Create .env file
cat > .env << EOF
# OptionsTitan - Meta LLAMA API Configuration
# Generated: $(date)

LLAMA_API_KEY=$api_key

# This file is automatically loaded by the GUI
# Keep this file private - never commit to version control
EOF

if [ $? -eq 0 ]; then
    echo "✅ .env file created successfully"
    chmod 600 .env  # Restrict permissions
    echo "🔒 Permissions set to 600 (owner read/write only)"
else
    echo "❌ Failed to create .env file"
    exit 1
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete! ✨                     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "🚀 Next Steps:"
echo "   1. Launch the GUI: ./launch_gui.sh"
echo "   2. Look for '(LLAMA AI Enhanced)' in window title"
echo "   3. Run an analysis to see AI-powered insights!"
echo ""
echo "📖 For more details, see: LLAMA_AI_SETUP.md"
echo ""
