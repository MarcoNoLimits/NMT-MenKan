import json
import os
from pathlib import Path

# Custom slang / culinary phrases
custom_pairs = [
    # Culinary
    ("meatballs", "polpette"),
    ("spaghetti and meatballs", "spaghetti con le polpette"),
    ("I am cooking spaghetti and meatballs for dinner.", "Sto cucinando spaghetti con le polpette per cena."),
    ("This sauce is amazing.", "Questo sugo è incredibile."),
    ("Pass the parmesan cheese, please.", "Passami il parmigiano, per favore."),
    ("I'm starving, let's grab some grub.", "Sto morendo di fame, andiamo a mettere qualcosa sotto i denti."),
    ("Let's order some takeout.", "Ordiniamo qualcosa da asporto."),
    
    # Casual/Slang
    ("What's up, bro?", "Come butta, fra?"),
    ("That's so cool!", "Che figata!"),
    ("I don't care, whatever.", "Non m'importa, fa lo stesso."),
    ("Chill out, man.", "Datti una calmata, amico."),
    ("You're driving me crazy.", "Mi fai impazzire."),
    ("Awesome!", "Mito!"),
    ("Stop messing around.", "Smettila di fare cazzate."),
    ("I screwed up.", "Ho fatto un casino."),
    ("No way!", "Non ci credo!"),
    ("Are you kidding me?", "Mi stai prendendo in giro?"),
    ("Let's bounce.", "Tagliamo la corda."),
    ("I'm broke.", "Sono al verde."),
    ("It's a piece of cake.", "È una passeggiata."),
    ("Don't pull my leg.", "Non prendermi in giro."),
    ("I'm totally out of it today.", "Oggi sono proprio fuso."),
    ("Catch you later.", "Ci becchiamo dopo."),
    ("That sucks.", "Fa schifo."),
    ("Give me a break.", "Fammi il piacere."),
    ("I'm pissed off.", "Sono incazzato nero."),
    ("It costs an arm and a leg.", "Costa un occhio della testa."),
    ("I'm dead tired.", "Sono stanco morto."),
    ("He completely lost it.", "Ha perso completamente la brocca.")
]

def main():
    # 29 pairs * 2 directions = 58 rows per iteration.
    # We want these to firmly embed themselves against 800k rows. 
    # An oversample of 200 gives us exactly 11,600 injected records.
    OVERSAMPLE = 200
    
    data_dir = Path("data/en_it_v3")
    train_file = data_dir / "train.jsonl"
    meta_file = data_dir / "metadata.json"
    
    if not train_file.exists():
        print(f"File {train_file} not found!")
        return
        
    rows = []
    for en, it in custom_pairs:
        # Eng -> Ita
        rows.append({
            "source_text": en, "target_text": it, 
            "source_lang": "eng_Latn", "target_lang": "ita_Latn", "dataset": "custom_slang"
        })
        # Ita -> Eng
        rows.append({
            "source_text": it, "target_text": en, 
            "source_lang": "ita_Latn", "target_lang": "eng_Latn", "dataset": "custom_slang"
        })
        
    print(f"Generated {len(rows)} custom pairs.")
    
    with open(train_file, "a", encoding="utf-8") as f:
        for _ in range(OVERSAMPLE):
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                
    added_count = len(rows) * OVERSAMPLE
    print(f"Successfully appended {added_count} injected records to {train_file}")
    
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["counts"]["train"] += added_count
        meta["counts"]["total"] += added_count
        if "custom_slang" not in meta["datasets"]:
            meta["datasets"].append("custom_slang")
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print("Updated metadata.json")

if __name__ == "__main__":
    main()
