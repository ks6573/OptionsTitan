# 🛠️ Scripts Directory

Helper scripts for launching and configuring OptionsTitan.

## 📜 Available Scripts

### Launch Scripts

#### [launch_gui.sh](launch_gui.sh)
**Mac/Linux GUI Launcher**
```bash
./scripts/launch_gui.sh
```

**What it does:**
- Checks if Python 3 is installed
- Verifies dependencies
- Installs missing packages if needed
- Launches the GUI

**Use when:**
- On Mac or Linux
- Want automated dependency checking
- First time running the GUI

---

#### [launch_gui.bat](launch_gui.bat)
**Windows GUI Launcher**
```cmd
scripts\launch_gui.bat
```

**What it does:**
- Checks if Python is installed
- Verifies dependencies
- Installs missing packages if needed
- Launches the GUI

**Use when:**
- On Windows
- Want automated dependency checking
- First time running the GUI

---

### Setup Scripts

#### [setup_llama.sh](setup_llama.sh)
**LLAMA AI Setup Automation**
```bash
./scripts/setup_llama.sh
```

**What it does:**
- Installs llama-api-client package
- Creates .env file
- Prompts for your API key
- Sets secure file permissions
- Validates setup

**Use when:**
- Want to enable AI features
- Have a LLAMA API key
- Want automated setup

---

## 🚀 Quick Start

### First Time Setup:

1. **Launch GUI (checks dependencies automatically):**
   ```bash
   # Mac/Linux
   ./scripts/launch_gui.sh
   
   # Windows
   scripts\launch_gui.bat
   ```

2. **Optional - Enable AI:**
   ```bash
   ./scripts/setup_llama.sh
   ```

### After Initial Setup:

You can run the GUI directly:
```bash
python options_gui.py
```

Or continue using the launch scripts if you prefer.

---

## 🔧 Making Scripts Executable (Mac/Linux)

If you get "permission denied":

```bash
chmod +x scripts/launch_gui.sh
chmod +x scripts/setup_llama.sh
```

---

## 📝 Notes

### Launch Scripts vs Direct Python

**Use launch scripts when:**
- First time running
- Not sure if dependencies installed
- Want automated checks
- On a new machine

**Run Python directly when:**
- Dependencies confirmed installed
- Running frequently
- Want faster startup
- Integrating into other tools

### Windows Notes

- Use `\` instead of `/` in paths
- Double-click `.bat` files to run
- Or run from Command Prompt/PowerShell

### Mac/Linux Notes

- Scripts need executable permission
- Use `./` prefix when running
- Or add `scripts/` to PATH

---

## 🆘 Troubleshooting

**"Command not found"**
- Make scripts executable (see above)
- Use `./scripts/` prefix
- Check you're in project root

**"Python not found"**
- Install Python 3.7+
- Add Python to PATH
- Use `python3` instead of `python`

**"Permission denied"**
- Run `chmod +x scripts/*.sh`
- On Windows, run as Administrator if needed

**Scripts seem slow**
- They check dependencies each time
- After first run, use Python directly
- Or skip dependency checks in scripts

---

## 🔗 Related Documentation

- **GUI Guide:** [../docs/gui/GUI_GUIDE.md](../docs/gui/GUI_GUIDE.md)
- **LLAMA Setup:** [../docs/llama/LLAMA_AI_SETUP.md](../docs/llama/LLAMA_AI_SETUP.md)
- **Main README:** [../readme.md](../readme.md)

---

*OptionsTitan Helper Scripts - Simplified Setup & Launch*
