#!/usr/bin/env python3
"""
Generate comprehensive English-only experiments with all initialization methods
Including proper imbalanced initialization for research purposes
"""

import os

# English-only configuration with proper research setup
DATASETS = ['english']  # Focus on English for research
TOKENIZERS = ['bpe', 'sentencepiece', 'morphological', 'phoneme']
SEEDS = [42, 123, 456, 789, 999]  # 5 seeds for statistical reliability

# Complete initialization methods for research
INITIALIZATION_METHODS = [
    # Standard methods
    ('xavier', {}),
    ('he', {}),
    ('generalized', {'beta': 1.0}),
    
    # Research: Imbalanced initialization with different ratios
    ('generalized_imbalanced', {'beta_input': 1.0, 'beta_output': 2.0}),  # 2x imbalance
    ('generalized_imbalanced', {'beta_input': 1.0, 'beta_output': 3.0}),  # 3x imbalance  
    ('generalized_imbalanced', {'beta_input': 1.0, 'beta_output': 4.0}),  # 4x imbalance (champion)
]

def create_job_script(dataset, tokenizer, init_method, init_params, seed, job_dir="jobs_english_research"):
    """Create a SLURM job script for specific configuration"""
    os.makedirs(job_dir, exist_ok=True)

    # Create job name based on initialization method
    if init_method == 'generalized':
        job_name = f"{dataset}_{tokenizer}_{init_method}_beta{init_params['beta']}_seed{seed}"
    elif init_method == 'generalized_imbalanced':
        ratio = init_params['beta_output'] / init_params['beta_input']
        job_name = f"{dataset}_{tokenizer}_{init_method}_{ratio:.0f}x_seed{seed}"
    else:
        job_name = f"{dataset}_{tokenizer}_{init_method}_seed{seed}"

    script_path = os.path.join(job_dir, f"{job_name}.slurm")

    # Build command arguments
    cmd_args = f"{tokenizer} {seed} {dataset} --config config/experiment_config_fixed.yaml --init_method {init_method} --track_every 5 --output_dir results_english_research"
    
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

echo "🧪 Starting English Research Experiment: {dataset} {tokenizer} {init_method} seed{seed}"
echo "📊 Configuration: Fixed hyperparameters (dropout=0.1, weight_decay=0.01)"
echo "🎯 Initialization: {init_method}"
"""

    if init_method == 'generalized':
        script_content += f'echo "📐 Beta value: {init_params[\'beta\']}"\n'
    elif init_method == 'generalized_imbalanced':
        ratio = init_params['beta_output'] / init_params['beta_input']
        script_content += f'echo "📐 Beta Input: {init_params[\'beta_input\']}"\n'
        script_content += f'echo "📐 Beta Output: {init_params[\'beta_output\']}"\n'
        script_content += f'echo "⚖️  Imbalance Ratio: {ratio:.1f}x (OUTPUT {ratio:.1f}x LARGER than INPUT)"\n'

    script_content += f"""

cd $SLURM_SUBMIT_DIR

source ~/miniconda3/etc/profile.d/conda.sh
conda activate emotion_env

echo "🚀 Running research experiment..."
python train_experiment.py {cmd_args}

echo "✅ English research experiment completed!"
"""

    with open(script_path, 'w') as f:
        f.write(script_content)

    os.chmod(script_path, 0o755)
    return script_path

def main():
    print("🧪 Generating English Research Experiments")
    print("=" * 60)
    print("📊 Research Focus: Imbalanced Initialization Impact")
    print("🔧 Fixed Configuration: Restored working hyperparameters")
    print("")

    # Create logs and results directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results_english_research", exist_ok=True)

    total_jobs = 0
    jobs_by_init = {}
    
    # Calculate total experiments
    total_experiments = len(DATASETS) * len(TOKENIZERS) * len(INITIALIZATION_METHODS) * len(SEEDS)
    print(f"📈 Experimental Design:")
    print(f"   🗂️  Datasets: {len(DATASETS)} (English only)")
    print(f"   🔤 Tokenizers: {len(TOKENIZERS)} (BPE, SentencePiece, Morphological, Phoneme)")
    print(f"   ⚡ Initialization Methods: {len(INITIALIZATION_METHODS)}")
    print(f"   🎲 Seeds: {len(SEEDS)} (for statistical reliability)")
    print(f"   🎯 Total Experiments: {total_experiments}")
    print("")
    
    # Show initialization methods
    print("🔬 Initialization Methods Being Tested:")
    for i, (method, params) in enumerate(INITIALIZATION_METHODS, 1):
        if method == 'generalized_imbalanced':
            ratio = params['beta_output'] / params['beta_input']
            print(f"   {i}. {method}: {params['beta_input']}→{params['beta_output']} ({ratio:.0f}x imbalance)")
        elif method == 'generalized':
            print(f"   {i}. {method}: β={params['beta']}")
        else:
            print(f"   {i}. {method}: standard")
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
                    
                    # Print progress every 20 jobs
                    if total_jobs % 20 == 0:
                        print(f"   Generated {total_jobs}/{total_experiments} job scripts...")

    print(f"✅ Generated {total_jobs} job scripts!")

    # Create submission script
    submit_script_content = f"""#!/bin/bash
# Submit English Research Experiments
# Tests imbalanced initialization hypothesis

echo "🧪 Submitting {total_jobs} English Research Experiments"
echo "📊 Research Question: Does imbalanced initialization improve performance?"
echo "🔧 Using FIXED hyperparameters (dropout=0.1, weight_decay=0.01)"
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
                        job_name = f"{dataset}_{tokenizer}_{init_method}_beta{init_params['beta']}_seed{seed}"
                    elif init_method == 'generalized_imbalanced':
                        ratio = init_params['beta_output'] / init_params['beta_input']
                        job_name = f"{dataset}_{tokenizer}_{init_method}_{ratio:.0f}x_seed{seed}"
                    else:
                        job_name = f"{dataset}_{tokenizer}_{init_method}_seed{seed}"
                    
                    submit_script_content += f"""echo "📤 Submitting job {job_count + 1}: {job_name}"
sbatch jobs_english_research/{job_name}.slurm
sleep 1
"""
                    job_count += 1

    submit_script_content += f"""
echo ""
echo "✅ All {total_jobs} English research jobs submitted!"
echo ""
echo "📊 Job breakdown by initialization method:"
"""

    for method, count in jobs_by_init.items():
        submit_script_content += f'echo "   {method}: {count} jobs"\n'

    submit_script_content += f"""
echo ""
echo "🎯 Research Hypothesis Testing:"
echo "   • Standard methods: Xavier, He, Generalized(β=1)"
echo "   • Imbalanced methods: 2x, 3x, 4x output scaling"
echo "   • Statistical validation: 5 seeds per configuration"
echo ""
echo "📈 Expected outcomes:"
echo "   • Validate imbalanced initialization benefits"
echo "   • Compare different imbalance ratios"
echo "   • Identify optimal β_input → β_output ratios"
echo ""
echo "💾 Results will be saved to: results_english_research/"
echo "📊 Monitor with: squeue -u ssani"
echo "🔍 Check progress: ./monitor_english_research_jobs.sh"
"""

    with open("submit_english_research_jobs.sh", 'w') as f:
        f.write(submit_script_content)
    
    os.chmod("submit_english_research_jobs.sh", 0o755)

    # Create monitoring script
    monitor_script_content = """#!/bin/bash
# Monitor English Research Experiments

echo "🧪 ENGLISH RESEARCH EXPERIMENT MONITORING"
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
completed=$(ls -1 results_english_research/ 2>/dev/null | wc -l)
echo "Total: $completed/""" + str(total_experiments) + """

echo ""
echo "🔬 Results by initialization method:"
for init in xavier he generalized generalized_imbalanced; do
    count=$(ls -1d results_english_research/*${init}* 2>/dev/null | wc -l)
    echo "  $init: $count experiments completed"
done

echo ""
echo "📝 Recent log activity:"
ls -lt logs/slurm*english*.out 2>/dev/null | head -5

echo ""
echo "🎯 Research Progress:"
if [ $completed -gt 0 ]; then
    progress=$((completed * 100 / """ + str(total_experiments) + """))
    echo "Progress: $progress% ($completed/""" + str(total_experiments) + """ experiments)"
    
    if [ $completed -eq """ + str(total_experiments) + """ ]; then
        echo ""
        echo "🎉 ALL EXPERIMENTS COMPLETED!"
        echo "Ready for analysis: python plot_english_research_results.py"
    fi
else
    echo "No experiments completed yet"
fi
"""

    with open("monitor_english_research_jobs.sh", 'w') as f:
        f.write(monitor_script_content)
    
    os.chmod("monitor_english_research_jobs.sh", 0o755)

    # Create analysis script placeholder
    analysis_script_content = f"""#!/bin/bash
# Analyze English Research Results

echo "🧪 ENGLISH RESEARCH RESULTS ANALYSIS"
echo "=" * 50

completed=$(find results_english_research -name "final_results.yaml" | wc -l)
expected={total_experiments}

echo "📊 Experiment Status: $completed/$expected completed"

if [ $completed -eq $expected ]; then
    echo ""
    echo "🎉 All experiments completed! Running analysis..."
    
    # Run the research analysis
    python plot_english_research_results.py
    
    echo ""
    echo "📈 Analysis complete! Check these outputs:"
    echo "   📊 plots_english_research/ - Visualization plots"
    echo "   📋 research_summary.txt - Statistical summary"
    echo "   🏆 best_configurations.yaml - Top performing setups"
    
else
    echo ""
    echo "⏳ Waiting for experiments to complete..."
    echo "📊 Progress: $(echo "scale=1; $completed * 100 / $expected" | bc)%"
    echo ""
    echo "🔍 Monitor progress: ./monitor_english_research_jobs.sh"
fi
"""

    with open("analyze_english_research_results.sh", 'w') as f:
        f.write(analysis_script_content)
    
    os.chmod("analyze_english_research_results.sh", 0o755)

    print("")
    print("=" * 70)
    print("🎉 ENGLISH RESEARCH EXPERIMENT GENERATION COMPLETE!")
    print("=" * 70)
    print("")
    print("📊 Research Design Summary:")
    print(f"   🎯 Total Experiments: {total_experiments}")
    print(f"   🧪 Initialization Methods: {len(INITIALIZATION_METHODS)}")
    print(f"   📈 Focus: Imbalanced initialization impact")
    print(f"   🔧 Configuration: FIXED working hyperparameters")
    print("")
    print("🏗️  Generated Files:")
    print(f"   📁 jobs_english_research/              {total_jobs} SLURM scripts")
    print(f"   🚀 submit_english_research_jobs.sh     Submit all experiments")
    print(f"   👀 monitor_english_research_jobs.sh    Monitor progress")
    print(f"   📊 analyze_english_research_results.sh Analyze results")
    print("")
    print("🔬 Research Hypothesis:")
    print("   ❓ Question: Does imbalanced initialization improve performance?")
    print("   📐 Test: Different β_input → β_output ratios (1→2, 1→3, 1→4)")
    print("   🎯 Goal: Identify optimal imbalance ratio for emotion classification")
    print("")
    print("🚀 Next Steps:")
    print("   1. Review setup:    ls jobs_english_research/ | head -10")
    print("   2. Submit jobs:     ./submit_english_research_jobs.sh")
    print("   3. Monitor:         ./monitor_english_research_jobs.sh")
    print("   4. Analyze:         ./analyze_english_research_results.sh")
    print("")
    print("⏱️  Expected Runtime:")
    print(f"   🕐 ~6 hours per job")
    print(f"   📊 {total_experiments} total experiments")
    print(f"   ⚡ Results: publication-ready research data!")

if __name__ == "__main__":
    main()
