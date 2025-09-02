#!/bin/bash
#SBATCH --job-name=candi_update_metadata
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=update_metadata_%j.out
#SBATCH --error=update_metadata_%j.err
#SBATCH --account=def-maxwl

# Load required modules
module load python/3.9

# Change to working directory
cd /home/mforooz/projects/def-maxwl/mforooz/EpiDenoise

# Set environment variables
export PYTHONPATH="/home/mforooz/projects/def-maxwl/mforooz/EpiDenoise:$PYTHONPATH"

# Print job information
echo "================================================================================"
echo "🎯 CANDI METADATA UPDATE JOB - EIC & MERGED DATASETS"
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 32G"
echo "Max parallel workers: 8"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Output Directory: /home/mforooz/projects/def-maxwl/mforooz/log"
echo "================================================================================"

# Check if tools are available
echo "🔧 Checking tool availability:"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
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
echo "Allocated Memory: 32G"
echo ""

# Create log directory if it doesn't exist
mkdir -p /home/mforooz/projects/def-maxwl/mforooz/log

# Update EIC dataset metadata
echo "🚀 Starting EIC dataset metadata update..."
echo "📋 Dataset: EIC"
echo "📁 Directory: /home/mforooz/projects/def-maxwl/mforooz/DATA_CANDI_EIC"
echo "⏱️  Estimated time: ~30-45 minutes"
echo ""

# Execute EIC metadata update
python get_candi_data.py update-metadata eic /home/mforooz/projects/def-maxwl/mforooz/DATA_CANDI_EIC --backup --force --max-workers 8

# Check EIC exit status
EIC_EXIT_CODE=$?

echo ""
if [ $EIC_EXIT_CODE -eq 0 ]; then
    echo "✅ EIC DATASET METADATA UPDATE COMPLETED SUCCESSFULLY"
else
    echo "❌ EIC DATASET METADATA UPDATE FAILED (Exit code: $EIC_EXIT_CODE)"
fi
echo ""

# Update MERGED dataset metadata
echo "🚀 Starting MERGED dataset metadata update..."
echo "📋 Dataset: MERGED"
echo "📁 Directory: /home/mforooz/projects/def-maxwl/mforooz/DATA_CANDI_MERGED"
echo "⏱️  Estimated time: ~30-45 minutes"
echo ""

# Execute MERGED metadata update
python get_candi_data.py update-metadata merged /home/mforooz/projects/def-maxwl/mforooz/DATA_CANDI_MERGED --backup --force --max-workers 8

# Check MERGED exit status
MERGED_EXIT_CODE=$?

echo ""
if [ $MERGED_EXIT_CODE -eq 0 ]; then
    echo "✅ MERGED DATASET METADATA UPDATE COMPLETED SUCCESSFULLY"
else
    echo "❌ MERGED DATASET METADATA UPDATE FAILED (Exit code: $MERGED_EXIT_CODE)"
fi
echo ""

# Overall job status
echo "================================================================================"
if [ $EIC_EXIT_CODE -eq 0 ] && [ $MERGED_EXIT_CODE -eq 0 ]; then
    echo "🎉 ALL METADATA UPDATES COMPLETED SUCCESSFULLY"
    OVERALL_EXIT_CODE=0
else
    echo "⚠️  SOME METADATA UPDATES FAILED"
    echo "   EIC: $([ $EIC_EXIT_CODE -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
    echo "   MERGED: $([ $MERGED_EXIT_CODE -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
    OVERALL_EXIT_CODE=1
fi
echo "================================================================================"
echo "End Time: $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Show log file locations
echo ""
echo "📁 Log files created:"
echo "   - SLURM output: update_metadata_${SLURM_JOB_ID}.out"
echo "   - SLURM error: update_metadata_${SLURM_JOB_ID}.err"
echo "   - Metadata update logs: /home/mforooz/projects/def-maxwl/mforooz/log/"

# List the metadata update log files
echo ""
echo "📋 Metadata update log files:"
ls -la /home/mforooz/projects/def-maxwl/mforooz/log/metadata_update_*.log 2>/dev/null || echo "No metadata update logs found"

# Show backup file counts
echo ""
echo "💾 Backup files created:"
echo "   EIC backups: $(find /home/mforooz/projects/def-maxwl/mforooz/DATA_CANDI_EIC -name "file_metadata.json.backup" 2>/dev/null | wc -l)"
echo "   MERGED backups: $(find /home/mforooz/projects/def-maxwl/mforooz/DATA_CANDI_MERGED -name "file_metadata.json.backup" 2>/dev/null | wc -l)"

exit $OVERALL_EXIT_CODE
