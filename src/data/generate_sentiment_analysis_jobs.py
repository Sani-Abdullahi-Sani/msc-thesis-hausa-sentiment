import os

# Sentiment analysis configuration
DATASETS = ['hausa']  # Start with Hausa sentiment, expand to ['hausa', 'swahili', 'english'] later
TOKENIZERS = ['bpe', 'sentencepiece', 'morphological', 'phoneme']
SEEDS = [42, 123, 456, 789, 999]

# Complete initialization methods for sentiment analysis research
INITIALIZATION_METHODS = [
    ('xavier', {}),
    ('he', {}),
    ('generalized', {'beta': 1.0}),
    ('generalized_imbalanced', {'beta_input': 1.0, 'beta_output': 4.0})  # Champion 4x ratio
]

def create_job_script(dataset, tokenizer, init_method, init_params, seed, job_dir="jobs_sentiment_analysis"):
    """Create a SLURM job script for sentiment analysis configuration"""
    os.makedirs(job_dir, exist_ok=True)

    # Create job name based on initialization method
    if init_method == 'generalized':
        job_name = f"sentiment_{dataset}_{tokenizer}_{init_method}_beta{init_params['beta']}_seed{seed}"
    elif init_method == 'generalized_imbalanced':
        ratio = init_params['beta_output'] / init_params['beta_input']
        job_name = f"sentiment_{dataset}_{tokenizer}_{init_method}_{ratio:.0f}x_seed{seed}"
    else:
        job_name = f"sentiment_{dataset}_{tokenizer}_{init_method}_seed{seed}"

    script_path = os.path.join(job_dir, f"{job_name}.slurm")

    # Build command arguments for sentiment analysis
    cmd_args = f"{tokenizer} {seed} {dataset} --init_method {init_method} --track_every 5 --output_dir results_sentiment_analysis"

    if init_method == 'generalized':
        cmd_args += f" --beta {init_params['beta']}"
    elif init_method == 'generalized_imbalanced':
        cmd_args += f" --beta_input {init_params['beta_input']} --beta_output {init_params['beta_output']}"

    script_content = f"""#!/bin/bash
#SBATCH -p stampede
#SBATCH -N 1
#SBATCH -t 06:00:00
#SBATCH -J {job_name}
#SBATCH -o /home-mscluster/ssani/multilingual_emotion_classification/logs/slurm_{job_name}.%N.%j.out
#SBATCH -e /home-mscluster/ssani/multilingual_emotion_classification/logs/slurm_{job_name}.%N.%j.err

echo "🎭 Starting Sentiment Analysis Experiment: {dataset} {tokenizer} {init_method} seed{seed}"
echo "📊 Task: Sentiment Classification (positive/negative/neutral)"
echo "🔧 Configuration: Fixed hyperparameters for research consistency"
echo "🎯 Initialization: {init_method}"
"""

    if init_method == 'generalized':
        beta_val = init_params['beta']
        script_content += f'echo "📐 Beta value: {beta_val}"\n'
    elif init_method == 'generalized_imbalanced':
        beta_in = init_params['beta_input']
        beta_out = init_params['beta_output']
        ratio = beta_out / beta_in
        script_content += f'echo "📐 Beta Input: {beta_in}"\n'
        script_content += f'echo "📐 Beta Output: {beta_out}"\n'
        script_content += f'echo "⚖️  Imbalance Ratio: {ratio}x (OUTPUT {ratio}x LARGER than INPUT)"\n'

    script_content += f"""

cd $SLURM_SUBMIT_DIR

source ~/miniconda3/etc/profile.d/conda.sh
conda activate emotion_env

echo "🚀 Running sentiment analysis with imbalanced initialization research..."
python train_experiment.py {cmd_args}

echo "✅ Sentiment analysis experiment completed!"
"""

    with open(script_path, 'w') as f:
        f.write(script_content)

    os.chmod(script_path, 0o755)
    return script_path

def main():
    print("🎭 GENERATING SENTIMENT ANALYSIS EXPERIMENTS")
    print("=" * 60)
    print("🔬 Research Focus: Imbalanced Initialization for Sentiment Classification")
    print("🎯 Task: Multi-language sentiment analysis (positive/negative/neutral)")
    print("")

    # Create logs and results directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results_sentiment_analysis", exist_ok=True)

    total_jobs = 0
    jobs_by_init = {}

    # Calculate total experiments
    total_experiments = len(DATASETS) * len(TOKENIZERS) * len(INITIALIZATION_METHODS) * len(SEEDS)
    print(f"📈 Experimental Design:")
    print(f"   🗂️  Datasets: {len(DATASETS)} ({', '.join(DATASETS)})")
    print(f"   🔤 Tokenizers: {len(TOKENIZERS)} (BPE, SentencePiece, Morphological, Phoneme)")
    print(f"   ⚡ Initialization Methods: {len(INITIALIZATION_METHODS)}")
    print(f"   🎲 Seeds: {len(SEEDS)} (for statistical reliability)")
    print(f"   🎯 Total Experiments: {total_experiments}")
    print("")
    
    # Show initialization methods
    print("🔬 Initialization Methods for Sentiment Analysis:")
    for i, (method, params) in enumerate(INITIALIZATION_METHODS, 1):
        if method == 'generalized_imbalanced':
            ratio = params['beta_output'] / params['beta_input']
            print(f"   {i}. {method}: {params['beta_input']}→{params['beta_output']} ({ratio:.0f}x imbalance)")
            print(f"      💡 Tests if larger output initialization improves sentiment classification")
        elif method == 'generalized':
            print(f"   {i}. {method}: β={params['beta']}")
        else:
            print(f"   {i}. {method}: standard baseline")
    print("")

    # Generate all job scripts
    for dataset in DATASETS:
        for tokenizer in TOKENIZERS:
            for init_method, init_params in INITIALIZATION_METHODS:
                for seed in SEEDS:
                    script_path = create_job_script(dataset, tokenizer, init_method, init_params, seed)
                    total_jobs += 1
                    
                    if init_method not in jobs_by_init:
                        jobs_by_init[init_method] = 0
                    jobs_by_init[init_method] += 1

    print(f"✅ Generated {total_jobs} sentiment analysis job scripts!")

    # Create submission script
    submit_script_content = f"""#!/bin/bash
# Submit Sentiment Analysis Experiments
# Research: Imbalanced initialization for sentiment classification

echo "🎭 Submitting {total_jobs} Sentiment Analysis Experiments"
echo "🔬 Research Question: Does imbalanced initialization improve sentiment classification?"
echo "📊 Task: Sentiment analysis (positive/negative/neutral classification)"
echo ""

echo "⚡ Initialization Methods:"
"""

    for method, count in jobs_by_init.items():
        submit_script_content += f'echo "   {method}: {count} experiments"\n'

    submit_script_content += f"""
echo ""
echo "🚀 Starting job submission..."

"""

    job_count = 0
    for dataset in DATASETS:
        for tokenizer in TOKENIZERS:
            for init_method, init_params in INITIALIZATION_METHODS:
                for seed in SEEDS:
                    if init_method == 'generalized':
                        job_name = f"sentiment_{dataset}_{tokenizer}_{init_method}_beta{init_params['beta']}_seed{seed}"
                    elif init_method == 'generalized_imbalanced':
                        ratio = init_params['beta_output'] / init_params['beta_input']
                        job_name = f"sentiment_{dataset}_{tokenizer}_{init_method}_{ratio:.0f}x_seed{seed}"
                    else:
                        job_name = f"sentiment_{dataset}_{tokenizer}_{init_method}_seed{seed}"
                    
                    submit_script_content += f"""echo "📤 Submitting job {job_count + 1}: {job_name}"
sbatch jobs_sentiment_analysis/{job_name}.slurm
sleep 1
"""
                    job_count += 1

    submit_script_content += f"""
echo ""
echo "✅ All {total_jobs} sentiment analysis jobs submitted!"
echo ""
echo "📊 Job breakdown by initialization method:"
"""

    for method, count in jobs_by_init.items():
        submit_script_content += f'echo "   {method}: {count} jobs"\n'

    submit_script_content += f"""
echo ""
echo "🎯 Sentiment Analysis Research Goals:"
echo "   • Test imbalanced initialization on sentiment classification"
echo "   • Compare 4x output scaling vs standard methods"
echo "   • Multi-tokenizer validation for robustness"
echo "   • Statistical validation with 5 seeds per configuration"
echo ""
echo "📈 Expected outcomes:"
echo "   • Validate initialization benefits for sentiment tasks"
echo "   • Identify optimal initialization for sentiment analysis"
echo "   • Compare with emotion classification results"
echo ""
echo "💾 Results will be saved to: results_sentiment_analysis/"
echo "📊 Monitor with: squeue -u ssani"
echo "🔍 Check progress: ./monitor_sentiment_jobs.sh"
"""

    with open("submit_sentiment_analysis_jobs.sh", 'w') as f:
        f.write(submit_script_content)
    
    os.chmod("submit_sentiment_analysis_jobs.sh", 0o755)

    # Create monitoring script
    monitor_script_content = f"""#!/bin/bash
# Monitor Sentiment Analysis Experiments

echo "🎭 SENTIMENT ANALYSIS EXPERIMENT MONITORING"
echo "=" * 50

echo ""
echo "📊 Current job status:"
squeue -u ssani -o "%.10i %.9P %.30j %.8u %.8T %.10M %.6D %R"

echo ""
echo "📈 Job summary:"
echo "Running: $(squeue -u ssani -h -t R | wc -l)"
echo "Pending: $(squeue -u ssani -h -t PD | wc -l)"
echo "Total active: $(squeue -u ssani -h | wc -l)"

echo ""
echo "✅ Completed experiments:"
completed=$(ls -1 results_sentiment_analysis/ 2>/dev/null | wc -l)
echo "Total: $completed/{total_experiments}"

echo ""
echo "🔬 Results by initialization method:"
for init in xavier he generalized generalized_imbalanced; do
    count=$(ls -1d results_sentiment_analysis/*${{init}}* 2>/dev/null | wc -l)
    echo "  $init: $count experiments completed"
done

echo ""
echo "📝 Recent log activity:"
ls -lt logs/slurm*sentiment*.out 2>/dev/null | head -5

echo ""
echo "🎯 Research Progress:"
if [ $completed -gt 0 ]; then
    progress=$((completed * 100 / {total_experiments}))
    echo "Progress: $progress% ($completed/{total_experiments} experiments)"
    
    if [ $completed -eq {total_experiments} ]; then
        echo ""
        echo "🎉 ALL EXPERIMENTS COMPLETED!"
        echo "Ready for analysis: python plot_sentiment_results.py"
    fi
else
    echo "No experiments completed yet"
fi
"""

    with open("monitor_sentiment_jobs.sh", 'w') as f:
        f.write(monitor_script_content)
    
    os.chmod("monitor_sentiment_jobs.sh", 0o755)

    print("")
    print("=" * 70)
    print("🎉 SENTIMENT ANALYSIS EXPERIMENT GENERATION COMPLETE!")
    print("=" * 70)
    print("")
    print("📊 Research Design Summary:")
    print(f"   🎯 Total Experiments: {total_experiments}")
    print(f"   🧪 Initialization Methods: {len(INITIALIZATION_METHODS)}")
    print(f"   📈 Focus: Imbalanced initialization for sentiment classification")
    print(f"   🎭 Task: Sentiment analysis (positive/negative/neutral)")
    print("")
    print("🏗️  Generated Files:")
    print(f"   📁 jobs_sentiment_analysis/              {total_jobs} SLURM scripts")
    print(f"   🚀 submit_sentiment_analysis_jobs.sh     Submit all experiments")
    print(f"   👀 monitor_sentiment_jobs.sh             Monitor progress")
    print("")
    print("🔬 Research Hypothesis:")
    print("   ❓ Question: Does imbalanced initialization improve sentiment classification?")
    print("   📐 Test: 4x output layer initialization vs standard methods")
    print("   🎯 Goal: Validate initialization benefits across sentiment tasks")
    print("")
    print("🚀 Next Steps:")
    print("   1. Prepare data:    python prepare_sentiment_data.py")
    print("   2. Update config:   config/experiment_config.yaml")
    print("   3. Submit jobs:     ./submit_sentiment_analysis_jobs.sh")
    print("   4. Monitor:         ./monitor_sentiment_jobs.sh")
    print("")
    print("⏱️  Expected Runtime:")
    print(f"   🕐 ~6 hours per job")
    print(f"   📊 {total_experiments} total experiments")
    print(f"   ⚡ Results: Sentiment analysis initialization research!")

if __name__ == "__main__":
    main()
