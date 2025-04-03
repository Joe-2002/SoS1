import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import glob
import numpy as np


def create_performance_chart(df: pd.DataFrame, model_name: str, output_dir: str):
    """Create performance analysis chart for a single model"""
    # Prepare data
    model_data = df[df[f'{model_name}_output_1'].notna()]

    # If no valid data, create an empty chart
    if len(model_data) == 0:
        plt.figure(figsize=(12, 8))
        plt.title(
            f'Polynomial Type Prediction Accuracy\n{model_name}\n(No Valid Prediction Data)')
        plt.savefig(os.path.join(output_dir, f'{model_name}_performance.png'))
        plt.close()
        return {}

    label_stats = model_data.groupby('label').apply(
        lambda x: pd.Series({
            'total': len(x),
            'correct': (x[f'{model_name}_output_1'] == x['ans']).sum()
        })
    ).to_dict('index')

    # Create chart
    plt.figure(figsize=(12, 8))
    labels = list(label_stats.keys())
    correct_counts = [label_stats[label]['correct'] for label in labels]
    incorrect_counts = [label_stats[label]['total'] -
                        label_stats[label]['correct'] for label in labels]

    # Draw stacked bar chart
    y_pos = range(len(labels))
    plt.barh(y_pos, correct_counts, label='Correct',
             color='#2ecc71', alpha=0.8)  # Green represents correct
    plt.barh(y_pos, incorrect_counts, left=correct_counts,
             label='Incorrect', color='#e74c3c', alpha=0.8)  # Red represents incorrect

    # Set labels
    plt.yticks(y_pos, labels)
    plt.xlabel('Number of Samples')
    plt.title(f'Polynomial Type Prediction Accuracy\n{model_name}')

    # Add value labels and percentages
    for i, (correct, incorrect) in enumerate(zip(correct_counts, incorrect_counts)):
        total = correct + incorrect
        if total > 0:  # Add label only when there is data
            if correct > 0:
                plt.text(correct/2, i, f'{correct} ({correct/total:.1%})',
                         ha='center', va='center', color='white', fontweight='bold')
            if incorrect > 0:
                plt.text(correct + incorrect/2, i, f'{incorrect} ({incorrect/total:.1%})',
                         ha='center', va='center', color='white', fontweight='bold')

    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    # Save chart
    plt.savefig(os.path.join(output_dir, f'{model_name}_performance.png'))
    plt.close()

    return label_stats


def calculate_model_statistics(df: pd.DataFrame, model_name: str) -> dict:
    """Calculate statistics for a single model"""
    # Get output and time columns
    output_col = f'{model_name}_output_1'
    time_col = f'{model_name}_time_1'
    raw_response_col = f'{model_name}_raw_response_1'

    total_samples = len(df)
    valid_mask = df[output_col].notna()
    valid_predictions = df[valid_mask]
    valid_samples = len(valid_predictions)
    correct_predictions = (
        valid_predictions[output_col] == valid_predictions['ans']).sum()

    # Calculate generation success rate
    error_patterns = [
        'error', 'Error', 'ERROR',
        'timeout', 'Timeout', 'TIMEOUT',
        'failed', 'Failed', 'FAILED',
        'unsupported_value', 'invalid_request',
        'BadRequestError', 'TimeoutException',
        'Input length exceeds model limit'
    ]
    error_pattern = '|'.join(error_patterns)

    # Check if response exists and does not contain error information, and output is not None
    generation_success = len(df[
        df[raw_response_col].notna() &
        ~df[raw_response_col].str.contains(error_pattern, na=False, regex=True) &
        df[output_col].notna()  # Ensure output is not None
    ])
    generation_success_rate = generation_success / \
        total_samples if total_samples > 0 else 0

    # Calculate accuracy
    accuracy_total = correct_predictions / total_samples if total_samples > 0 else 0
    accuracy_valid = correct_predictions / valid_samples if valid_samples > 0 else 0

    # Calculate average response time
    valid_times = df[df[time_col].notna()][time_col]
    avg_time = valid_times.mean() if not valid_times.empty else None

    return {
        'model': model_name,
        'total_samples': total_samples,
        'valid_samples': valid_samples,
        'correct_predictions': correct_predictions,
        'accuracy_total': accuracy_total,
        'accuracy_valid': accuracy_valid,
        'generation_success_rate': generation_success_rate,
        'generation_success': generation_success,
        'avg_response_time': avg_time
    }


def create_models_comparison_chart(model_stats: list, output_dir: str):
    """Create comparison chart for all models"""
    # Prepare data
    models = [stat['model'] for stat in model_stats]

    # Simplify model name display
    def simplify_model_name(name):
        # Remove common prefix
        name = name.replace('qwen2.5-', 'qwen-')
        name = name.replace('-instruct', '')
        name = name.replace('-preview', '')
        # Keep main features
        if '_' in name:
            base, prompt = name.split('_')
            return f"{base}\n({prompt})"
        return name

    display_models = [simplify_model_name(model) for model in models]
    accuracies_total = [stat['accuracy_total'] for stat in model_stats]
    accuracies_valid = [stat['accuracy_valid'] for stat in model_stats]
    generation_rates = [stat['generation_success_rate']
                        for stat in model_stats]
    avg_times = [stat['avg_response_time'] for stat in model_stats]

    # 1. Performance metrics comparison chart
    plt.figure(figsize=(20, 10))  # Increase chart size
    x = range(len(display_models))
    width = 0.25

    # Adjust x-axis position
    x1 = [i - width for i in x]
    x2 = x
    x3 = [i + width for i in x]

    # Use more vibrant colors
    bars1 = plt.bar(x1, accuracies_total, width, label='Overall Accuracy',
                    color='#2ecc71', alpha=0.9)
    bars2 = plt.bar(x2, accuracies_valid, width, label='Valid Accuracy',
                    color='#3498db', alpha=0.9)
    bars3 = plt.bar(x3, generation_rates, width, label='Generation Success Rate',
                    color='#e74c3c', alpha=0.9)

    # Optimize value label display
    def add_value_labels(bars, offset=0):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 + offset,
                     f'{height:.1%}',
                     ha='center', va='bottom',
                     rotation=0,
                     fontsize=9,
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # Interleave value labels to avoid overlap
    add_value_labels(bars1, offset=0)
    add_value_labels(bars2, offset=0.02)
    add_value_labels(bars3, offset=0.04)

    plt.xlabel('Models', fontsize=12, labelpad=15)
    plt.ylabel('Rate', fontsize=12, labelpad=15)
    plt.title('Model Performance Metrics Comparison', fontsize=16, pad=20)

    # Optimize x-axis label display
    plt.xticks(x, display_models, rotation=45, ha='right', fontsize=10)

    # Add legend and optimize position
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center',
               ncol=3, fontsize=11, frameon=True,
               facecolor='white', edgecolor='gray')

    # Add grid lines for readability
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Adjust y-axis range to ensure all labels are visible
    plt.ylim(0, max(max(accuracies_total), max(accuracies_valid),
                    max(generation_rates)) * 1.2)

    # Add model group background color
    def add_model_group_background():
        current_group = None
        group_start = 0

        for i, model in enumerate(models):
            group = model.split('-')[0]  # Group based on model name prefix

            if current_group is None:
                current_group = group
                group_start = i
            elif group != current_group:
                # Add background color for previous group
                plt.axvspan(group_start - 0.5, i - 0.5,
                            alpha=0.1, color='gray')
                current_group = group
                group_start = i

        # Add background color for last group
        plt.axvspan(group_start - 0.5, len(models) - 0.5,
                    alpha=0.1, color='gray')

    add_model_group_background()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_performance_comparison.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

    # 2. Response time comparison chart
    plt.figure(figsize=(20, 8))
    valid_times = [(model, time) for model, time in zip(display_models, avg_times)
                   if time is not None]

    if valid_times:
        models_with_time, times = zip(*valid_times)
        bars = plt.bar(models_with_time, times, color='#9b59b6', alpha=0.9)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2f}s',
                     ha='center', va='bottom',
                     fontsize=10,
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

        plt.xlabel('Models', fontsize=12, labelpad=15)
        plt.ylabel('Average Response Time (seconds)', fontsize=12, labelpad=15)
        plt.title('Response Time Comparison', fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=10)

        # Add grid lines
        plt.grid(True, alpha=0.3, axis='y', linestyle='--')

        # Adjust y-axis range
        plt.ylim(0, max(times) * 1.15)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'response_time_comparison.png'),
                bbox_inches='tight', dpi=300)
    plt.close()


def create_prompt_comparison_for_model(df: pd.DataFrame, model_base_name: str, output_dir: str):
    """Create comparison chart for different prompts for a single model"""
    prompts = ['base', 'ours', 'cot']

    # Create a larger chart
    plt.figure(figsize=(24, 8))

    for idx, prompt in enumerate(prompts, 1):
        model_name = f"{model_base_name}_{prompt}"
        output_col = f'{model_name}_output_1'

        if output_col not in df.columns:
            continue

        plt.subplot(1, 3, idx)

        # Prepare data
        model_data = df[df[output_col].notna()]
        if len(model_data) == 0:
            plt.title(f'{prompt.capitalize()}\n(No Valid Data)', fontsize=14)
            continue

        label_stats = model_data.groupby('label').apply(
            lambda x: pd.Series({
                'total': len(x),
                'correct': (x[output_col] == x['ans']).sum()
            })
        ).to_dict('index')

        # Draw stacked bar chart
        labels = list(label_stats.keys())
        correct_counts = [label_stats[label]['correct'] for label in labels]
        incorrect_counts = [label_stats[label]['total'] -
                            label_stats[label]['correct'] for label in labels]

        y_pos = range(len(labels))
        plt.barh(y_pos, correct_counts, label='Correct',
                 color='#2ecc71', alpha=0.8)
        plt.barh(y_pos, incorrect_counts, left=correct_counts,
                 label='Incorrect', color='#e74c3c', alpha=0.8)

        # Optimize label display
        for i, (correct, incorrect) in enumerate(zip(correct_counts, incorrect_counts)):
            total = correct + incorrect
            if total > 0:
                # Correct prediction label
                if correct > 0:
                    plt.text(correct/2, i,
                             f'{correct}\n({correct/total:.1%})',
                             ha='center', va='center',
                             color='white', fontweight='bold',
                             fontsize=10)
                # Incorrect prediction label
                if incorrect > 0:
                    plt.text(correct + incorrect/2, i,
                             f'{incorrect}\n({incorrect/total:.1%})',
                             ha='center', va='center',
                             color='white', fontweight='bold',
                             fontsize=10)

        plt.yticks(y_pos, labels, fontsize=10)
        plt.xlabel('Number of Samples', fontsize=12, labelpad=10)
        plt.title(f'{prompt.capitalize()} Prompt', fontsize=14, pad=20)

        # Add legend only in the first subplot
        if idx == 1:
            plt.legend(loc='upper right', fontsize=10)

        plt.grid(True, alpha=0.3, axis='x')

    # Add total title
    plt.suptitle(f'Prompt Comparison for {model_base_name}',
                 fontsize=16, y=1.02)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save chart
    plt.savefig(os.path.join(output_dir, f'{model_base_name}_prompt_comparison.png'),
                bbox_inches='tight', dpi=300)
    plt.close()


def analyze_results(df: pd.DataFrame, output_dir: str):
    """Analyze results and generate visualizations"""
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get all model columns
        model_columns = [
            col for col in df.columns if col.endswith('_output_1')]
        model_names = [col.replace('_output_1', '') for col in model_columns]

        print(f"\nFound results for the following models: {model_names}")

        # Store analysis results
        model_stats = []
        label_stats_dict = {}

        # Analyze each model
        for model_name in model_names:
            print(f"\nAnalyzing model: {model_name}")
            try:
                # Calculate statistics
                stats = calculate_model_statistics(df, model_name)
                model_stats.append(stats)

                # Print detailed statistics information
                print(f"- Total samples: {stats['total_samples']}")
                print(f"- Valid predictions: {stats['valid_samples']}")
                print(f"- Correct predictions: {stats['correct_predictions']}")
                print(f"- Overall accuracy: {stats['accuracy_total']:.4%}")
                print(f"- Valid accuracy: {stats['accuracy_valid']:.4%}")
                print(
                    f"- Generation success rate: {stats['generation_success_rate']:.4%}")
                if stats['avg_response_time']:
                    print(
                        f"- Average response time: {stats['avg_response_time']:.2f} seconds")

                # Create performance chart
                try:
                    label_stats = create_performance_chart(
                        df, model_name, output_dir)
                    label_stats_dict[model_name] = label_stats
                    print(
                        f"- Generated performance chart: {model_name}_performance.png")
                except Exception as e:
                    print(
                        f"Warning: Error generating performance chart for {model_name}: {str(e)}")
            except Exception as e:
                print(f"Warning: Error analyzing {model_name}: {str(e)}")
                continue

        # Create model comparison chart
        try:
            create_models_comparison_chart(model_stats, output_dir)
            print("- Generated models comparison chart: model_performance_comparison.png and response_time_comparison.png")
        except Exception as e:
            print(
                f"Warning: Error generating models comparison chart: {str(e)}")

        # Create prompt comparison chart for each base model
        try:
            # Get all base model names
            base_models = set()
            for model_name in model_names:
                if '_' in model_name:  # Ensure model name contains prompt information
                    base_name = model_name.rsplit('_', 1)[0]
                    base_models.add(base_name)

            # Create comparison chart for each base model
            for base_model in base_models:
                create_prompt_comparison_for_model(df, base_model, output_dir)
                print(f"- Generated prompt comparison chart for {base_model}")
        except Exception as e:
            print(
                f"Warning: Error generating prompt comparison charts: {str(e)}")

        # Save original data
        try:
            df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
            print("- Saved prediction data: predictions.csv")
        except Exception as e:
            print(f"Warning: Error saving prediction data: {str(e)}")

        # Save summary data
        try:
            summary_df = pd.DataFrame(model_stats)
            summary_df.to_csv(os.path.join(
                output_dir, 'summary.csv'), index=False)
            print("- Saved summary data: summary.csv")
        except Exception as e:
            print(f"Warning: Error saving summary data: {str(e)}")

        # Generate analysis report
        try:
            with open(os.path.join(output_dir, 'analysis_report.txt'), 'w', encoding='utf-8') as f:
                f.write("Model Performance Analysis Report\n")
                f.write("=" * 50 + "\n\n")

                # Overall performance analysis
                f.write("1. Overall Performance Analysis\n")
                f.write("-" * 30 + "\n")

                # Group by model base name
                model_groups = {}
                for stats in model_stats:
                    model_name = stats['model']
                    if '_prompt' in model_name:
                        base_name = model_name.split('_prompt')[0]
                        if base_name not in model_groups:
                            model_groups[base_name] = []
                        model_groups[base_name].append(stats)
                    else:
                        model_groups[model_name] = [stats]

                # Output performance for each model
                for base_name, stats_list in model_groups.items():
                    f.write(f"\n{base_name}:\n")
                    for stats in stats_list:
                        prompt_suffix = f" (Prompt {stats['model'].split('_prompt')[1]})" if '_prompt' in stats['model'] else ""
                        f.write(f"{prompt_suffix}\n")
                        f.write(f"- Total samples: {stats['total_samples']}\n")
                        f.write(
                            f"- Valid predictions: {stats['valid_samples']}\n")
                        f.write(
                            f"- Correct predictions: {stats['correct_predictions']}\n")
                        f.write(
                            f"- Overall accuracy: {stats['accuracy_total']:.4%}\n")
                        f.write(
                            f"- Valid accuracy: {stats['accuracy_valid']:.4%}\n")
                        f.write(
                            f"- Generation success rate: {stats['generation_success_rate']:.4%}\n")
                        if stats['avg_response_time']:
                            f.write(
                                f"- Average response time: {stats['avg_response_time']:.2f} seconds\n")

                # Prompt template comparison analysis
                if any('_prompt' in stat['model'] for stat in model_stats):
                    f.write("\n2. Prompt Template Comparison Analysis\n")
                    f.write("-" * 30 + "\n")
                    for base_name, stats_list in model_groups.items():
                        if len(stats_list) > 1:
                            f.write(f"\n{base_name}:\n")
                            # Sort by accuracy
                            sorted_stats = sorted(
                                stats_list, key=lambda x: x['accuracy_total'], reverse=True)
                            for stats in sorted_stats:
                                prompt_num = stats['model'].split(
                                    '_prompt')[1] if '_prompt' in stats['model'] else "N/A"
                                f.write(f"Prompt {prompt_num}:\n")
                                f.write(
                                    f"- Overall accuracy: {stats['accuracy_total']:.4%}\n")
                                f.write(
                                    f"- Valid accuracy: {stats['accuracy_valid']:.4%}\n")
                                f.write(
                                    f"- Generation success rate: {stats['generation_success_rate']:.4%}\n")

                # Performance ranking analysis
                f.write("\n3. Performance Ranking Analysis\n")
                f.write("-" * 30 + "\n")

                # Sort by overall accuracy
                f.write("\nOverall Accuracy Ranking:\n")
                sorted_by_total = sorted(
                    model_stats, key=lambda x: x['accuracy_total'], reverse=True)
                for i, stats in enumerate(sorted_by_total, 1):
                    f.write(
                        f"{i}. {stats['model']}: {stats['accuracy_total']:.4%}\n")

                # Sort by valid accuracy
                f.write("\nValid Accuracy Ranking:\n")
                sorted_by_valid = sorted(
                    model_stats, key=lambda x: x['accuracy_valid'], reverse=True)
                for i, stats in enumerate(sorted_by_valid, 1):
                    f.write(
                        f"{i}. {stats['model']}: {stats['accuracy_valid']:.4%}\n")

                # Sort by generation success rate
                f.write("\nGeneration Success Rate Ranking:\n")
                sorted_by_success = sorted(
                    model_stats, key=lambda x: x['generation_success_rate'], reverse=True)
                for i, stats in enumerate(sorted_by_success, 1):
                    f.write(
                        f"{i}. {stats['model']}: {stats['generation_success_rate']:.4%}\n")

                # Response time ranking
                valid_times = [(stat['model'], stat['avg_response_time'])
                               for stat in model_stats if stat['avg_response_time'] is not None]
                if valid_times:
                    f.write("\nResponse Time Ranking (fastest to slowest):\n")
                    sorted_times = sorted(valid_times, key=lambda x: x[1])
                    for i, (model, time) in enumerate(sorted_times, 1):
                        f.write(f"{i}. {model}: {time:.2f} seconds\n")

                # Polynomial type analysis
                if 'label' in df.columns and label_stats_dict:
                    f.write("\n4. Polynomial Type Analysis\n")
                    f.write("-" * 30 + "\n")
                    for model_name, label_stats in label_stats_dict.items():
                        f.write(f"\n{model_name}:\n")
                        # Sort by accuracy
                        sorted_labels = sorted(
                            [(label, stats['correct']/stats['total'] if stats['total'] > 0 else 0)
                             for label, stats in label_stats.items()],
                            key=lambda x: x[1],
                            reverse=True
                        )
                        for label, accuracy in sorted_labels:
                            stats = label_stats[label]
                            f.write(f"\n  {label}:\n")
                            f.write(f"    - Total samples: {stats['total']}\n")
                            f.write(
                                f"    - Correct predictions: {stats['correct']}\n")
                            f.write(
                                f"    - Incorrect predictions: {stats['total'] - stats['correct']}\n")
                            f.write(f"    - Accuracy: {accuracy:.4%}\n")

            print("- Generated analysis report: analysis_report.txt")
        except Exception as e:
            print(f"Warning: Error generating analysis report: {str(e)}")

        print(f"\nAnalysis results saved to: {output_dir}")

    except Exception as e:
        print(f"Error: Exception occurred during analysis: {str(e)}")
        raise


def main():
    """Main function"""
    results_files = glob.glob('../Data/results_*.jsonl')
    if not results_files:
        print("No result files found!")
        return

    for file in results_files:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f'../Analysis/Analysis_{timestamp}'
        print(f"Analyzing file: {file}")
        analyze_results(file, output_dir)
        print(f"Analysis results saved to: {output_dir}")


if __name__ == "__main__":
    main()
