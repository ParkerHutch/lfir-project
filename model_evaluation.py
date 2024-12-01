import subprocess
import re
import numpy as np
import argparse
import os
from datetime import datetime

def run_evaluation(env_id, checkpoint_path, seed):
    # Command to run the evaluation
    command = [
        "python", "ppo.py", 
        f"--env_id={env_id}", 
        "--num_envs=1024", 
        "--num-steps=200", 
        "--num-eval-steps=200", 
        f"--checkpoint={checkpoint_path}", 
        "--evaluate", 
        f"--seed={seed}"
    ]
    
    result = subprocess.run(command, capture_output=True, text=True)
    output = result.stdout
    
    # Extract metrics using regex
    metrics = {}
    
    success_rate_match = re.search(r"Success Rate: (\d+\.\d+)", output)
    if success_rate_match:
        metrics['success_rate'] = float(success_rate_match.group(1))
    
    success_once_match = re.search(r"eval_success_once_mean=(\d+\.\d+)", output)
    if success_once_match:
        metrics['success_once_mean'] = float(success_once_match.group(1))
    
    if not metrics:
        print(f"Could not find metrics in the output for {checkpoint_path}")
        return None
    
    return metrics

def evaluate_checkpoint(env_id, checkpoint_path, num_runs):
    success_rates = []
    success_once_means = []
    
    for i in range(num_runs):
        print(f"Running evaluation {i+1}/{num_runs} for {os.path.basename(checkpoint_path)}")
        metrics = run_evaluation(env_id, checkpoint_path, seed=i)
        
        if metrics is not None:
            if 'success_rate' in metrics:
                success_rates.append(metrics['success_rate'])
            if 'success_once_mean' in metrics:
                success_once_means.append(metrics['success_once_mean'])
    
    # Compute summary statistics
    result = {
        'checkpoint': checkpoint_path,
        'total_runs': len(success_rates)
    }
    
    if success_rates:
        result.update({
            'average_success_rate': np.mean(success_rates),
            'std_dev_success_rate': np.std(success_rates),
            'min_success_rate': np.min(success_rates),
            'max_success_rate': np.max(success_rates)
        })
    
    if success_once_means:
        result.update({
            'average_success_once_mean': np.mean(success_once_means),
            'std_dev_success_once_mean': np.std(success_once_means),
            'min_success_once_mean': np.min(success_once_means),
            'max_success_once_mean': np.max(success_once_means)
        })
    
    if not success_rates and not success_once_means:
        result['error'] = 'No successful evaluations'
    
    return result

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run multiple evaluations of PPO models")
    parser.add_argument("checkpoints", nargs='+', type=str, help="Paths to model checkpoint files")
    parser.add_argument("--env_id", type=str, required=True, help="Environment ID to evaluate")
    parser.add_argument("--num_runs", type=int, default=20, help="Number of evaluation runs per checkpoint (default: 20)")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save output files")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_file_path = os.path.join(args.output_dir, f"evaluation_summary_{timestamp}.txt")
    
    all_results = []
    
    # Evaluate each checkpoint
    with open(output_file_path, 'w') as output_file:
        output_file.write(f"Checkpoint Evaluation Summary for Environment: {args.env_id}\n")
        output_file.write("============================\n\n")
        
        for checkpoint in args.checkpoints:
            print(f"\nEvaluating checkpoint: {checkpoint}")
            result = evaluate_checkpoint(args.env_id, checkpoint, args.num_runs)
            all_results.append(result)
            
            # Write results to file
            output_file.write(f"Checkpoint: {result['checkpoint']}\n")
            if 'error' in result:
                output_file.write(f"Error: {result['error']}\n\n")
            else:
                output_file.write(f"Total Runs: {result['total_runs']}\n")
                
                # Success Rate statistics
                if 'average_success_rate' in result:
                    output_file.write(f"Average Success Rate: {result['average_success_rate']:.4f}\n")
                    output_file.write(f"Success Rate Standard Deviation: {result['std_dev_success_rate']:.4f}\n")
                    output_file.write(f"Min Success Rate: {result['min_success_rate']:.4f}\n")
                    output_file.write(f"Max Success Rate: {result['max_success_rate']:.4f}\n")
                
                # Success Once Mean statistics
                if 'average_success_once_mean' in result:
                    output_file.write(f"Average Success Once Mean: {result['average_success_once_mean']:.4f}\n")
                    output_file.write(f"Success Once Mean Standard Deviation: {result['std_dev_success_once_mean']:.4f}\n")
                    output_file.write(f"Min Success Once Mean: {result['min_success_once_mean']:.4f}\n")
                    output_file.write(f"Max Success Once Mean: {result['max_success_once_mean']:.4f}\n")
                
                output_file.write("\n")
            
            print(f"Results for {checkpoint} written to {output_file_path}")
    
    print(f"\nFinal summary saved to {output_file_path}")

if __name__ == "__main__":
    main()