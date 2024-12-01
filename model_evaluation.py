import subprocess
import re
import numpy as np
import argparse
import os
from datetime import datetime

def run_evaluation(checkpoint_path, seed):
    command = [
        "python", "ppo.py", 
        "--env_id=Cinnamon-Reach-v2", 
        "--num_envs=1024", 
        "--num-steps=200", 
        "--num-eval-steps=200", 
        f"--checkpoint={checkpoint_path}", 
        "--evaluate", 
        f"--seed={seed}"
    ]
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    # extract Success Rate with regex
    success_rate_match = re.search(r"Success Rate: (\d+\.\d+)", result.stdout)
    
    if success_rate_match:
        return float(success_rate_match.group(1))
    else:
        print(f"Could not find Success Rate in the output for {checkpoint_path}")
        return None

def evaluate_checkpoint(checkpoint_path, num_runs):
    success_rates = []
    
    for i in range(num_runs):
        print(f"Running evaluation {i+1}/{num_runs} for {checkpoint_path}")
        success_rate = run_evaluation(checkpoint_path, seed=i)
        
        if success_rate is not None:
            success_rates.append(success_rate)
    
    if success_rates:
        return {
            'checkpoint': checkpoint_path,
            'total_runs': len(success_rates),
            'average_success_rate': np.mean(success_rates),
            'std_dev_success_rate': np.std(success_rates),
            'min_success_rate': np.min(success_rates),
            'max_success_rate': np.max(success_rates)
        }
    else:
        return {
            'checkpoint': checkpoint_path,
            'total_runs': 0,
            'error': 'No successful evaluations'
        }

def main():
    parser = argparse.ArgumentParser(description="Run multiple evaluations of PPO models")
    parser.add_argument("checkpoints", nargs='+', type=str, help="Paths to model checkpoint files")
    parser.add_argument("--num_runs", type=int, default=20, help="Number of evaluation runs per checkpoint (default: 20)")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save output files")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_file_path = os.path.join(args.output_dir, f"evaluation_summary_{timestamp}.txt")
    
    all_results = []
    
    with open(output_file_path, 'w') as output_file:
        output_file.write("Checkpoint Evaluation Summary\n")
        output_file.write("============================\n\n")
        
        for checkpoint in args.checkpoints:
            print(f"\nEvaluating checkpoint: {checkpoint}")
            result = evaluate_checkpoint(checkpoint, args.num_runs)
            all_results.append(result)
            
            output_file.write(f"Checkpoint: {result['checkpoint']}\n")
            if 'error' in result:
                output_file.write(f"Error: {result['error']}\n\n")
            else:
                output_file.write(f"Total Runs: {result['total_runs']}\n")
                output_file.write(f"Average Success Rate: {result['average_success_rate']:.4f}\n")
                output_file.write(f"Success Rate Standard Deviation: {result['std_dev_success_rate']:.4f}\n")
                output_file.write(f"Min Success Rate: {result['min_success_rate']:.4f}\n")
                output_file.write(f"Max Success Rate: {result['max_success_rate']:.4f}\n\n")
            
            print(f"Results for {checkpoint} written to {output_file_path}")
    
    print(f"\nFinal summary saved to {output_file_path}")

if __name__ == "__main__":
    main()