from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate large synthetic CSV for confusion evaluation.")
    parser.add_argument("--rows", type=int, default=5000, help="Number of rows.")
    parser.add_argument("--output", default="reports/confusion_matrices/mock_big_eval.csv", help="Output CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    src_words = ["run", "beautiful", "quickly", "teacher", "book", "write", "slowly", "idea", "bright", "eat"]
    exp_words = ["correre", "bello", "rapidamente", "insegnante", "libro", "scrivere", "lentamente", "idea", "luminoso", "mangiare"]
    act_words = ["corre", "bella", "rapido", "professore", "libri", "scrive", "lento", "idea", "luminosi", "mangia"]

    formal_sentences = [
        "Dear Sir, could you please review the attached document?",
        "Thank you for your consideration regarding this request.",
        "However, we would kindly ask for your confirmation.",
        "Please accept our sincere appreciation for your support.",
    ]
    informal_sentences = [
        "hey can you check this real quick?",
        "yo this is kinda cool lol",
        "gonna send it now, pls reply",
        "thx bro, wanna meet later?",
    ]

    rows = []
    for _ in range(args.rows):
        idx = random.randrange(len(src_words))
        expected_is_formal = random.random() > 0.4
        actual_flip = random.random() < 0.2
        actual_is_formal = not expected_is_formal if actual_flip else expected_is_formal

        expected_sentence = random.choice(formal_sentences if expected_is_formal else informal_sentences)
        actual_sentence = random.choice(formal_sentences if actual_is_formal else informal_sentences)

        rows.append(
            {
                "source_word": src_words[idx],
                "expected_target_word": exp_words[idx],
                "actual_target_word": random.choice([act_words[idx], exp_words[idx]]),
                "expected_sentence": expected_sentence,
                "actual_sentence": actual_sentence,
            }
        )

    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {args.rows} rows to {out}")


if __name__ == "__main__":
    main()
