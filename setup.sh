#!/bin/bash

# CS336 Assignment 1 Setup Script
# This script sets up the development environment and builds the Rust BPE tokenizer

set -e  # Exit on any error

echo "🚀 Setting up CS336 Assignment 1 environment..."

# Check if we're in the right directory
if [[ ! -f "pyproject.toml" ]]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo "📦 Checking dependencies..."

# Check for Rust installation
if ! command_exists rustc; then
    echo "🦀 Installing Rust..."
    if command_exists curl; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    else
        echo "❌ Error: curl is required to install Rust. Please install curl first."
        exit 1
    fi
else
    echo "✅ Rust is already installed ($(rustc --version))"
fi

# Check for Python 3.11+
if command_exists python3; then
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "✅ Python $PYTHON_VERSION found"
    if [[ $(echo "$PYTHON_VERSION >= 3.11" | bc -l 2>/dev/null || echo "0") == "0" ]]; then
        echo "⚠️  Warning: Python 3.11+ is recommended. Current version: $PYTHON_VERSION"
    fi
else
    echo "❌ Error: Python 3 is required. Please install Python 3.11+ first."
    exit 1
fi

# Check for uv (Python package manager)
if ! command_exists uv; then
    echo "📦 Installing uv (Python package manager)..."
    if command_exists curl; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source "$HOME/.local/bin/env" 2>/dev/null || true
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo "❌ Error: curl is required to install uv. Please install curl first."
        exit 1
    fi
else
    echo "✅ uv is already installed ($(uv --version))"
fi

echo "🐍 Setting up Python environment..."

echo "🚀 Starting installation of npm and Claude Code..."

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install Node.js and npm
echo "📦 Installing Node.js and npm..."

if command_exists node && command_exists npm; then
    echo "✅ Node.js and npm are already installed"
    echo "Node.js version: $(node --version)"
    echo "npm version: $(npm --version)"
else
    # Detect OS and install accordingly
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command_exists apt-get; then
            # Ubuntu/Debian
            echo "🐧 Detected Ubuntu/Debian system"
            apt-get update
            apt-get install -y curl
            curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -
            apt-get install -y nodejs
        elif command_exists yum; then
            # CentOS/RHEL/Fedora
            echo "🐧 Detected CentOS/RHEL/Fedora system"
            curl -fsSL https://rpm.nodesource.com/setup_lts.x | bash -
            yum install -y nodejs npm
        elif command_exists pacman; then
            # Arch Linux
            echo "🐧 Detected Arch Linux system"
            pacman -S nodejs npm
        else
            echo "❌ Unsupported Linux distribution"
            exit 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        echo "🍎 Detected macOS system"
        if command_exists brew; then
            brew install node
        else
            echo "❌ Homebrew not found. Please install Homebrew first or install Node.js manually"
            echo "Visit: https://nodejs.org/en/download/"
            exit 1
        fi
    else
        echo "❌ Unsupported operating system: $OSTYPE"
        echo "Please install Node.js manually from: https://nodejs.org/en/download/"
        exit 1
    fi
fi

# Verify npm installation
if command_exists npm; then
    echo "✅ npm successfully installed"
    echo "npm version: $(npm --version)"
else
    echo "❌ npm installation failed"
    exit 1
fi

# Install Claude Code
echo "🤖 Installing Claude Code..."

if command_exists claude-code; then
    echo "✅ Claude Code is already installed"
    echo "Claude Code version: $(claude-code --version)"
else
    echo "📥 Installing Claude Code via npm..."
    npm install -g @anthropic-ai/claude-code
    
    # Verify installation
    if command_exists claude-code; then
        echo "✅ Claude Code successfully installed"
        echo "Claude Code version: $(claude-code --version)"
    else
        echo "❌ Claude Code installation failed"
        echo "💡 Try running: npm install -g @anthropic-ai/claude-code"
        exit 1
    fi
fi

echo ""
echo "🎉 Installation complete!"
echo ""
echo "📋 Next steps:"
echo "1. Set up your Anthropic API key:"
echo "   export ANTHROPIC_API_KEY='your-api-key-here'"
echo "   (Add this to your ~/.bashrc or ~/.zshrc for persistence)"
echo ""
echo "2. Start using Claude Code:"
echo "   claude-code --help"
echo ""
echo "📚 For more information about Claude Code, visit Anthropic's blog"
