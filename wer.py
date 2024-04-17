import csv
from typing import List, Any, Tuple
import pandas as pd
import torch
from evaluate import load


def rerank_sentences_for_wer(model: Any, wer_data: List[Any], savepath: str):
    """
    Function to rerank candidate sentences in the HUB dataset. For each set of sentences,
    you must assign each sentence a score in the form of the sentence's acoustic score plus
    the sentence's log probability. You should then save the top scoring sentences in a .csv
    file similar to those found in the results directory.

    Inputs:
        model (Any): An n-gram or Transformer model.
        wer_data (List[Any]): Processed data from the HUB dataset. 
        savepath (str): The path to save the csv file pairing sentence set ids and the top ranked sentences.
    """
    # model.eval()
    if model.__class__.__name__ == "NGramLM":
        with open(savepath, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            # Write the header
            csvwriter.writerow(['id', 'sentences'])

            for doc_id in wer_data:
                sentences = wer_data[doc_id]['sentences']
                top_score = float('-inf')
                top_sentence = None

                for sentence in sentences:
                    acoustic_score = wer_data[doc_id]['acoustic_scores'][sentence]

                    # Preprocess the sentence for the model
                    # Implement this function based on your model's input requirements
                    model_input = wer_data[doc_id]['tokens'][sentence]

                    # Compute log probability using the model
                    # Ensure this matches your model's method for computing log probability
                    log_prob = model.log_probability(model_input)

                    # Total score for the sentence is the sum of its acoustic score and log probability
                    total_score = log_prob + acoustic_score

                    if total_score > top_score:
                        top_score = total_score
                        top_sentence = sentence

                # Write the top-scoring sentence for this set to the CSV
                csvwriter.writerow([doc_id, top_sentence])
    else:
        with open(savepath, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['id', 'sentences'])

            for doc_id in wer_data:
                sentences = wer_data[doc_id]['sentences']
                top_score = float('-inf')
                top_sentence = None

                for sentence in sentences:
                    acoustic_score = wer_data[doc_id]['acoustic_scores'][sentence]
                    model_input = torch.tensor(wer_data[doc_id]['tokens'][sentence], dtype=torch.long).unsqueeze(0)

                    # Assuming the model expects input and target to be the same for log_probability calculation
                    target_tokens = model_input.clone()

                    # Compute log probability
                    with torch.no_grad():
                        log_prob = model.log_probability(model_input, target_tokens).item()  # Assuming it returns a scalar

                    # Calculate total score
                    total_score = acoustic_score + log_prob

                    # Update top score and sentence
                    if total_score > top_score:
                        top_score = total_score
                        top_sentence = sentence

                # Write the top sentence for this doc_id to CSV
                csvwriter.writerow([doc_id, top_sentence])


def compute_wer(gt_path, model_path):
    # Load the sentences
    ground_truths = pd.read_csv(gt_path)['sentences'].tolist()
    guesses = pd.read_csv(model_path)['sentences'].tolist()

    # Compute wer
    wer = load("wer")
    wer_value = wer.compute(predictions=guesses, references=ground_truths)
    return wer_value
