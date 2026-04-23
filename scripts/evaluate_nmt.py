import socket
import logging
import sacrebleu
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def translate_via_tcp(text: str, host='127.0.0.1', port=18080) -> str:
    """Send a single english string to the C++ server via TCP and return the Italian result."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            # Send english text (newline framing matches TCP server; enables long lines)
            s.sendall(text.encode('utf-8') + b'\n')
            
            # Receive Italian result
            data = s.recv(1024)
            return data.decode('utf-8')
    except Exception as e:
        logging.error(f"Error communicating with TCP server: {e}")
        return ""

def main():
    logging.info("Starting NMT Accuracy Evaluation...")
    
    # We will use the opus_books English -> Italian dataset.
    logging.info("Downloading OPUS Books Eng-Ita dataset...")
    try:
        # load train split (contains ~32k sentences)
        dataset = load_dataset("opus_books", "en-it", split="train")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return

    # Extract up to 1000 sentence pairs for the evaluation test
    max_sentences = 1000
    english_sentences = []
    reference_italian = []
    
    for item in dataset["translation"][:max_sentences]:
        if "en" in item and "it" in item:
            english_sentences.append(item["en"])
            reference_italian.append(item["it"])
    
    total_sentences = len(english_sentences)
    logging.info(f"Loaded {total_sentences} sentences to evaluate.")

    predictions = []
    
    # Loop over English, send to TCP Server, fetch Italian
    logging.info("Connecting to C++ TCP Engine and processing...")
    for eng in tqdm(english_sentences, desc="Translating"):
        # The C++ pipeline takes the raw text. 
        # (It adds the source language token and </s> automatically inside NMTWrapper.cpp)
        ita = translate_via_tcp(eng)
        predictions.append(ita)
    
    if len(predictions) != total_sentences:
         logging.error("Mismatch between predictions and references. Aborting scoring.")
         return

    logging.info("Calculating BLEU and chrF++ scores via SacreBLEU...")
    
    # Exact Match + BLEU
    # SacreBLEU takes a list of candidate strings, and a list of lists of reference strings.
    refs = [reference_italian]
    
    bleu = sacrebleu.corpus_bleu(predictions, refs)
    chrf = sacrebleu.corpus_chrf(predictions, refs)
    
    logging.info(f"Final BLEU Score: {bleu.score:.2f}")
    logging.info(f"Final chrF++ Score: {chrf.score:.2f}")
    
    # Write to Report
    report_path = "evaluation_report.txt"
    with open(report_path, "w", encoding='utf-8') as f:
        f.write("=== NMT-MenKan English-to-Italian Accuracy Evaluation ===\n")
        f.write("Dataset: OPUS Books (en-it)\n")
        f.write(f"Total Sentences Tested: {total_sentences}\n\n")
        f.write("--- Metrics ---\n")
        f.write(f"BLEU Score: {bleu.score:.2f}\n")
        f.write(f"{bleu.format()}\n\n")
        f.write(f"chrF++ Score: {chrf.score:.2f}\n")
        f.write(f"{chrf.format()}\n\n")
        
        f.write("--- Sample Outputs (First 5) ---\n")
        for i in range(min(5, len(predictions))):
            f.write(f"ENG (Source) : {english_sentences[i]}\n")
            f.write(f"ITA (Predict): {predictions[i]}\n")
            f.write(f"ITA (Target) : {reference_italian[i]}\n")
            f.write("-" * 40 + "\n")
            
    logging.info(f"Evaluation report successfully saved to {report_path}")

if __name__ == "__main__":
    main()
