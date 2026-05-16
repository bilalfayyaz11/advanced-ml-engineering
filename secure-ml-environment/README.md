# Secure Advanced Machine Learning Environment Setup

## Objective
Build a secure Linux-based machine learning environment optimized for modern AI and deep learning workflows using TensorFlow, PyTorch, JupyterLab, and Python virtual environments.

## Technologies Used
- Ubuntu 24.04 LTS
- Python 3.12
- JupyterLab
- TensorFlow
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- OpenCV
- Pillow
- fail2ban
- OpenSSL
- Linux Virtual Environments

## Key Skills Demonstrated
- Linux environment hardening
- Secure ML workstation setup
- Python virtual environment management
- JupyterLab secure configuration
- HTTPS certificate generation
- Machine learning dependency management
- TensorFlow and PyTorch environment validation
- Brute-force protection using fail2ban

## Security Enhancements
- Password-protected JupyterLab
- HTTPS encryption using self-signed SSL certificates
- fail2ban intrusion prevention
- Isolated Python environment for dependency control

## Troubleshooting & Fixes

### Updated Python Stack
Modernized the environment from Python 3.9 to Python 3.12 for improved compatibility with modern ML libraries.

### Fixed PyTorch Installation
Replaced:
pip install pytorch

With:
pip install torch torchvision torchaudio

### Updated Jupyter Configuration
Migrated deprecated:
c.NotebookApp.*

To:
c.ServerApp.*

For compatibility with modern JupyterLab architecture.

## Verification
Validated:
- TensorFlow imports
- PyTorch imports
- OpenCV imports
- GPU detection checks
- JupyterLab functionality
- Secure HTTPS access
