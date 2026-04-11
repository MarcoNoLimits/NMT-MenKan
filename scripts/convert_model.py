import ctranslate2
import os

# Define model name and output directory
model_name = "facebook/nllb-200-distilled-600M"
output_dir = "nllb_int8"

print(f"Converting model {model_name} to {output_dir} with int8 quantization...")

# Configurate the converter
converter = ctranslate2.converters.TransformersConverter(
    model_name,
    copy_files=["tokenizer.json", "sentencepiece.bpe.model"]
)

# Convert to Int8
converter.convert(output_dir, quantization="int8", force=True)

print("Conversion complete!")
print(f"Model saved to {os.path.abspath(output_dir)}")
