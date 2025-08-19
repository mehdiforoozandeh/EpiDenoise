#!/bin/bash
#SBATCH --job-name=candi_eic_complete
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --output=eic_complete_%j.out
#SBATCH --error=eic_complete_%j.err
#SBATCH --account=def-maxwl

# Load required bioinformatics modules (same as successful tests)
module load samtools/1.22.1
module load bedtools/2.31.0

# Change to working directory
cd /home/mforooz/projects/def-maxwl/mforooz/EpiDenoise

# Set environment variables
export PYTHONPATH="/home/mforooz/projects/def-maxwl/mforooz/EpiDenoise:$PYTHONPATH"

# Print job information
echo "================================================================================"
echo "🎯 CANDI EIC COMPLETE DATASET PROCESSING JOB"
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 80G (5GB per experiment × 16 parallel)"
echo "Max parallel experiments: 16"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Dataset: EIC (363 experiments across 89 biosamples)"
echo "Output Directory: /home/mforooz/projects/def-maxwl/mforooz/DATA_CANDI_EIC"
echo "================================================================================"

# Check if tools are available
echo "🔧 Checking tool availability:"
echo "samtools: $(which samtools)"
echo "bedtools: $(which bedtools)"
echo "Python version: $(python --version)"
echo ""

# Check available disk space
echo "💾 Checking available disk space:"
df -h /home/mforooz/projects/def-maxwl/mforooz/
echo ""

# Check memory and CPU info
echo "⚡ System resources:"
echo "Available memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo "Available CPUs: $(nproc)"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated Memory: 80G"
echo ""

# Run the EIC complete processing
echo "🚀 Starting EIC dataset complete processing..."
echo "📋 Expected: 363 experiments across 89 biosamples"
echo "⏱️  Estimated time: ~2 hours based on optimization analysis (16 parallel workers)"
echo ""

# Execute the processing script
python process_eic_complete.py

# Check exit status
EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ EIC DATASET PROCESSING COMPLETED SUCCESSFULLY"
else
    echo "❌ EIC DATASET PROCESSING FAILED (Exit code: $EXIT_CODE)"
fi
echo "================================================================================"
echo "End Time: $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Final disk usage check
echo ""
echo "💾 Final disk usage:"
du -sh /home/mforooz/projects/def-maxwl/mforooz/DATA_CANDI_EIC/ 2>/dev/null || echo "Directory not found"

# Show log file location
echo ""
echo "📁 Log files:"
echo "   - SLURM output: eic_complete_${SLURM_JOB_ID}.out"
echo "   - SLURM error: eic_complete_${SLURM_JOB_ID}.err"
echo "   - Detailed log: eic_complete_processing.log"

exit $EXIT_CODE
