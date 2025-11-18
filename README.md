# Neural Network Implementation in JAX

A simple neural network implementation using JAX

## Installation
1. Install python accordingly and run virtual envirnoment

<details>
<summary>MacOS Installation Guide</summary>

### Installing Python on MacOS
1. Using Homebrew:
```bash
brew install python
```
2. Or download from [Python's official website](https://www.python.org/downloads/macos/)

### Creating Virtual Environment on MacOS
```bash
# Navigate to your project directory
cd your_project

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# To deactivate
deactivate
```
</details>

<details>
<summary>Windows Installation Guide</summary>

### Installing Python on Windows
1. Download Python installer from [Python's official website](https://www.python.org/downloads/windows/)
2. Run the installer (Make sure to check "Add Python to PATH")

### Creating Virtual Environment on Windows
```bash
# Navigate to your project directory
cd your_project

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# To deactivate
deactivate
```
</details>

<details>
<summary>Linux Installation Guide</summary>

### Installing Python on Linux
#### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install python3
```

#### Fedora:
```bash
sudo dnf install python3
```

#### Arch Linux:
```bash
sudo pacman -S python
```

### Creating Virtual Environment on Linux
```bash
# Navigate to your project directory
cd your_project

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# To deactivate
deactivate
```
</details>

2. git clone and install requirements
```bash
git clone https://github.com/paxamans/paxjaxlib.git
```
```bash
pip install -r requirements.txt
```
3. Install the package in development mode:
```bash
pip install -e .
```

4. Run tests:
```bash
python -m pytest tests/
```

If you want to proceed with CNN for MNIST, install more pip packages so that you are able to download MNIST dataset.

5. Install dependencies for MNIST dataset
```bash
pip install tensorflow_datasets tensorflow
```

6. Run the example:
```bash
python examples/usage.py
```


