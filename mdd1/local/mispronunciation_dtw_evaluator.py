import os
import glob
from sklearn.metrics import f1_score, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pprint

def parse_arguments():
    """ Set up and return the argument parser. """
    parser = argparse.ArgumentParser(description='Evaluate pronunciation using DTW cost.')
    # Training set arguments
    parser.add_argument('--valid_transcript', type=str, required=True, help='Path to the training set transcript file.')
    parser.add_argument('--valid_targets', type=str, required=True, help='Path to the training set detection targets file.')
    parser.add_argument('--valid_dtw_folder', type=str, required=True, help='Path to the training set folder containing DTW log files.')
    # Test set arguments
    parser.add_argument('--test_transcript', type=str, required=True, help='Path to the test set transcript file.')
    parser.add_argument('--test_targets', type=str, required=True, help='Path to the test set detection targets file.')
    parser.add_argument('--test_dtw_folder', type=str, required=True, help='Path to the test set folder containing DTW log files.')
    parser.add_argument('--save_path', type=str, required=True, help='Path to the test set folder containing DTW log files.')
    return parser.parse_args()

def read_transcript(file_path):
    """ Read the transcript file and return a dictionary of utterances and their phones. """
    transcript = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            utterance_id = parts[0]
            phones = [phone.lower() for phone in parts[1:] if phone != 'sil']
            transcript[utterance_id] = phones
    return transcript

def read_detection_targets(file_path):
    """ Read the detection targets file and return a dictionary of utterance IDs and their corresponding targets. """
    targets = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            utterance_id = parts[0]
            target_values = [int(value) for value in parts[1:]]
            targets[utterance_id] = target_values
    return targets

def read_dtw_costs(dtw_folder_path, utt_ids):
    """ Read DTW log files for the provided Utterance IDs and return a dictionary of DTW costs for each utterance. """
    dtw_costs = {}
    for utt_id in utt_ids:
        log_file_path = os.path.join(dtw_folder_path, utt_id + '.log')
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r') as file:
                costs = []
                for line in file:
                    phone, cost = line.strip().split(',')
                    costs.append((phone.lower(), float(cost)))
                dtw_costs[utt_id] = costs
    return dtw_costs


def combine_data(transcript, detection_targets, dtw_costs):
    """
    Combine transcript, detection_targets, and dtw_costs into a single data structure.
    """
    combined_data = {}
    for utt_id in transcript:
        if utt_id in detection_targets and utt_id in dtw_costs:
            combined_data[utt_id] = {
                'phones': transcript[utt_id],
                'targets': detection_targets[utt_id],
                'dtw_costs': [cost for _, cost in dtw_costs[utt_id]]
            }
    return combined_data


def calculate_optimal_thresholds(combined_data):
    """
    Calculate the optimal threshold for each phone to maximize the F1 score.
    """
    phone_thresholds = {}
    phone_data = {}

    # Aggregate DTW costs and targets for each phone
    for utt_data in combined_data.values():
        assert len(utt_data['phones']) == len(utt_data['targets']) and len(utt_data['dtw_costs']) == len(utt_data['targets'])

        for phone, target, cost in zip(utt_data['phones'], utt_data['targets'], utt_data['dtw_costs']):
            if phone not in phone_data:
                phone_data[phone] = {'costs': [], 'targets': []}
            phone_data[phone]['costs'].append(cost)
            phone_data[phone]['targets'].append(target)

    # Calculate optimal threshold for each phone
    for phone, data in phone_data.items():
        costs = np.array(data['costs'])
        targets = np.array(data['targets'])
        thresholds = np.unique(sorted(costs))
        
        best_f1 = 0
        best_threshold = 0
        for threshold in thresholds:
            predictions = costs > threshold
            f1 = f1_score(targets, predictions)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        phone_thresholds[phone] = best_threshold

    return phone_thresholds

def calculate_eer_thresholds(combined_data):
    """
    Calculate the Equal Error Rate (EER) threshold for each phone.
    """
    phone_thresholds = {}
    phone_data = {}

    # Aggregate DTW costs and targets for each phone
    for utt_data in combined_data.values():
        for phone, target, cost in zip(utt_data['phones'], utt_data['targets'], utt_data['dtw_costs']):
            if phone not in phone_data:
                phone_data[phone] = {'costs': [], 'targets': []}
            phone_data[phone]['costs'].append(cost)
            phone_data[phone]['targets'].append(target)

    # Calculate EER threshold for each phone
    for phone, data in phone_data.items():
        costs = np.array(data['costs'])
        targets = np.array(data['targets'])

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(targets, costs)
        fnr = 1 - tpr

        # Find the EER threshold
        eer_index = np.nanargmin(np.absolute(fnr - fpr))
        eer_threshold = thresholds[eer_index]

        phone_thresholds[phone] = eer_threshold

    return phone_thresholds

def evaluate_pronunciation(combined_data, optimal_thresholds):
    """
    Evaluate pronunciation based on the optimal thresholds using sklearn metrics.
    Calculate metrics for correct pronunciations (targets = 1),
    mispronunciations (targets = 0), and overall accuracy.
    Also calculates true accept rate, false reject rate, false accept rate, and true reject rate.
    """
    all_predictions = []
    all_targets = []

    # Collect predictions and targets
    for utt_data in combined_data.values():
        for phone, target, cost in zip(utt_data['phones'], utt_data['targets'], utt_data['dtw_costs']):
            threshold = optimal_thresholds.get(phone, float('inf'))
            prediction = int(cost > threshold)
            all_predictions.append(prediction)
            all_targets.append(target)

    # Convert to numpy arrays for sklearn functions
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Calculate metrics for correct pronunciations (targets = 1)
    correct_precision = precision_score(all_targets, all_predictions, pos_label=1)
    correct_recall = recall_score(all_targets, all_predictions, pos_label=1)
    correct_f1 = f1_score(all_targets, all_predictions, pos_label=1)

    # Calculate metrics for mispronunciations (targets = 0)
    mispron_precision = precision_score(all_targets, all_predictions, pos_label=0)
    mispron_recall = recall_score(all_targets, all_predictions, pos_label=0)
    mispron_f1 = f1_score(all_targets, all_predictions, pos_label=0)

    # Calculate overall accuracy
    overall_accuracy = accuracy_score(all_targets, all_predictions)

    # Confusion matrix to calculate true/false accept/reject rates
    tn, fp, fn, tp = confusion_matrix(all_targets, all_predictions).ravel()
    true_accept_rate = tp / (tp + fn) if tp + fn > 0 else 0
    false_reject_rate = fn / (tp + fn) if tp + fn > 0 else 0
    false_accept_rate = fp / (fp + tn) if fp + tn > 0 else 0
    true_reject_rate = tn / (fp + tn) if fp + tn > 0 else 0

    metrics = {
        'correct_precision': correct_precision,
        'correct_recall': correct_recall,
        'correct_f1': correct_f1,
        'mispronunciation_precision': mispron_precision,
        'mispronunciation_recall': mispron_recall,
        'mispronunciation_f1': mispron_f1,
        'overall_accuracy': overall_accuracy,
        'true_accept_rate': true_accept_rate,
        'false_reject_rate': false_reject_rate,
        'false_accept_rate': false_accept_rate,
        'true_reject_rate': true_reject_rate
    }

    return metrics

def plot_dtw_costs(combined_data, save_path):
    """
    Plot and save DTW cost distributions for each phone with thresholds.
    Red dots represent mispronunciations (target=0), blue dots represent correct pronunciations (target=1).
    """
    phone_costs = {}

    # Collect DTW costs for each phone
    for utt_data in combined_data.values():
        for phone, target, cost in zip(utt_data['phones'], utt_data['targets'], utt_data['dtw_costs']):
            if phone not in phone_costs:
                phone_costs[phone] = {'CP': [], 'MP': []}
            if target == 0:
                phone_costs[phone]['MP'].append(cost)
            else:
                phone_costs[phone]['CP'].append(cost)

    # Plotting
    fig, ax = plt.subplots(figsize=(15, 8))
    phone_ids = range(len(phone_costs))

    # Plot each phone's DTW costs and thresholds
    for i, phone in enumerate(phone_costs.keys()):
        correct_costs = phone_costs[phone]['CP']
        incorrect_costs = phone_costs[phone]['MP']
        threshold = optimal_thresholds.get(phone, max(correct_costs + incorrect_costs))  # Use max cost if no threshold
        ax.scatter([i] * len(correct_costs), correct_costs, color='blue', label='CP (target=1)' if i == 0 else "")
        ax.scatter([i] * len(incorrect_costs), incorrect_costs, color='red', label='MP (target=0)' if i == 0 else "")
        ax.hlines(threshold, i-0.4, i+0.4, color='green', label='Threshold' if i == 0 else "")  # Threshold line

    # Labeling
    ax.set_xlabel('Phone ID')
    ax.set_ylabel('DTW Cost')
    ax.set_title('DTW Cost Distribution by Phone with Thresholds')
    ax.set_xticks(phone_ids)
    ax.set_xticklabels(phone_costs.keys(), rotation=90)
    ax.legend()

    # Save the figure
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    # valid
    valid_transcript = args.valid_transcript
    valid_targets = args.valid_targets
    valid_dtw_folder = args.valid_dtw_folder
    # test
    test_transcript = args.test_transcript
    test_targets = args.test_targets
    test_dtw_folder = args.test_dtw_folder

    save_path = args.save_path

    # Read training set data
    valid_transcript = read_transcript(valid_transcript)
    valid_detection_targets = read_detection_targets(args.valid_targets)
    valid_utt_ids = list(valid_transcript.keys())
    valid_dtw_costs = read_dtw_costs(valid_dtw_folder, valid_utt_ids)

    # Read test set data
    test_transcript = read_transcript(test_transcript)
    test_detection_targets = read_detection_targets(args.test_targets)
    test_utt_ids = list(test_transcript.keys())
    test_dtw_costs = read_dtw_costs(test_dtw_folder, test_utt_ids)

    # Process training set data to calculate optimal thresholds
    valid_combined_data = combine_data(valid_transcript, valid_detection_targets, valid_dtw_costs)
    optimal_thresholds = calculate_optimal_thresholds(valid_combined_data)

    # Process test set data using the calculated optimal thresholds
    test_combined_data = combine_data(test_transcript, test_detection_targets, test_dtw_costs)
    metrics = evaluate_pronunciation(test_combined_data, optimal_thresholds)

    # Output results
    print("Optimal Thresholds:", optimal_thresholds)
    print()
    print("Evaluation Metrics:", metrics)
    plot_dtw_costs(test_combined_data, save_path)
