#!/bin/bash

# make file executable: chmod +x setup.sh

# Exit immediately if a command exits with a non-zero status.
set -e

PYTHON_VERSION="3.12"
VENV_DIR=".venv"
PYTHON_EXEC="python${PYTHON_VERSION}"
REQUIREMENTS_FILE="requirements.txt"

# update packages first
sudo apt update && sudo apt upgrade -y
echo " "
echo "==== Packages updated."

# check for correct Python version
echo "Checking for ${PYTHON_EXEC}..."
if ! command -v $PYTHON_EXEC &> /dev/null; then
    echo "Error: ${PYTHON_EXEC} could not be found."
    echo "Please install it first. On Ubuntu/Debian, use:"
    echo "sudo apt update && sudo apt install python${PYTHON_VERSION} python${PYTHON_VERSION}-venv"
    exit 1
fi
echo "${PYTHON_EXEC} found."

# 2. Check for requirements.txt
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "Error: ${REQUIREMENTS_FILE} not found in this directory."
    exit 1
fi

# 3. Create the virtual environment if it doesn't exist
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment '${VENV_DIR}' already exists. Skipping creation."
else
    echo "Creating virtual environment with ${PYTHON_EXEC}..."
    $PYTHON_EXEC -m venv $VENV_DIR
fi

# 4. Upgrade pip and install packages
echo "Upgrading pip and installing packages from ${REQUIREMENTS_FILE}..."
./$VENV_DIR/bin/python -m pip install --upgrade pip > /dev/null
./$VENV_DIR/bin/pip install -r $REQUIREMENTS_FILE

# 5. Final success message
echo ""
echo "==== Setup complete!"
echo "To activate your new virtual environment, run:"
echo "source ${VENV_DIR}/bin/activate"