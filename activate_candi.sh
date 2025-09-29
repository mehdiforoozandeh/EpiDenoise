#!/bin/bash
# CANDI Environment Activation Script

echo "🚀 Activating CANDI environment..."

# Load required modules
module load StdEnv/2023 gcc/12.3 cuda/12.2 cudnn/8.9.5.29 python/3.10 scipy-stack/2025a

# Activate the virtual environment
source /project/6014832/mforooz/EpiDenoise/candi_venv/bin/activate

echo "✅ CANDI environment activated!"
echo "   Python: $(python --version)"
echo "   Location: $(which python)"
echo ""
echo "🔧 Available packages:"
echo "   ✓ PyTorch, NumPy, SciPy, Pandas"
echo "   ✓ Scikit-learn, Matplotlib, Seaborn"
echo "   ✓ TorchInfo, ImageIO"
echo "   ✓ pyBigWig, pybedtools, intervaltree"
echo ""
echo "📖 To deactivate: deactivate"
