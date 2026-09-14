#!/usr/bin/env python3
"""
Translate English sentiment data to Hausa using dammyogt/english-to-hausa-translation
Optimized for cluster usage with batch processing and progress tracking
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from tqdm import tqdm
import os
import sys
import argparse
from pathlib import Path

def setup_model(model_name="dammyogt/english-to-hausa-translation"):
    """Load the English→Hausa translation model"""
    print(f"🔄 Loading model: {model_name}")
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_id = 0 if device == "cuda" else -1
    print(f"   Device: {device}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Create pipeline
    translator = pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        device=device_id,
        batch_size=16 if device == "cuda" else 4
    )
    
    print(f"✅ Model loaded successfully!")
    return translator

def translate_texts(texts, translator, batch_size=16, max_length=128):
    """Translate a list of texts with progress bar"""
    translations = []
    
    print(f"📝 Translating {len(texts)} texts...")
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Translation progress"):
        batch = texts[i:i+batch_size]
        
        try:
            # Translate batch
            results = translator(
                batch,
                max_length=max_length,
                num_beams=4,  # Beam search for better quality
                early_stopping=True
            )
            
            # Extract translated text
            batch_translations = [r['translation_text'] for r in results]
            translations.extend(batch_translations)
            
        except Exception as e:
            print(f"\n⚠️  Error in batch {i//batch_size}: {e}")
            # Fallback: keep original text
            translations.extend(batch)
    
    return translations

def translate_dataset(input_file, output_file, translator):
    """Translate a complete TSV dataset"""
    print(f"\n{'='*70}")
    print(f"📖 Processing: {input_file}")
    print(f"{'='*70}")
    
    # Load data
    try:
        df = pd.read_csv(input_file, sep='\t', encoding='utf-8')
    except:
        # Try comma separator if tab doesn't work
        df = pd.read_csv(input_file, encoding='utf-8')
    
    print(f"   Loaded {len(df)} samples")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Check for text column
    text_col = 'text' if 'text' in df.columns else df.columns[0]
    label_col = 'label' if 'label' in df.columns else df.columns[1]
    
    print(f"   Text column: '{text_col}'")
    print(f"   Label column: '{label_col}'")
    
    # Show sample
    print(f"\n   Sample original text:")
    print(f"   '{df[text_col].iloc[0]}'")
    
    # Translate
    translations = translate_texts(
        df[text_col].tolist(),
        translator,
        batch_size=16
    )
    
    # Show sample translation
    print(f"\n   Sample translation:")
    print(f"   '{translations[0]}'")
    
    # Create output dataframe
    df_output = pd.DataFrame({
        'text': translations,
        'label': df[label_col],
        'original_english': df[text_col]
    })
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_output.to_csv(output_file, sep='\t', index=False, encoding='utf-8')
    
    print(f"\n✅ Saved to: {output_file}")
    print(f"   Translated samples: {len(df_output)}")
    
    return len(df_output)

def main():
    parser = argparse.ArgumentParser(
        description='Translate English sentiment data to Hausa',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  %(prog)s --input_dir data/raw/english --output_dir data/raw/english_hausa_translated
  
This will translate train.tsv, dev.tsv, and test.tsv
        """
    )
    
    parser.add_argument('--input_dir', 
                       default='data/raw/english',
                       help='Input directory with English TSV files')
    parser.add_argument('--output_dir',
                       default='data/raw/english_hausa_translated',
                       help='Output directory for translated files')
    parser.add_argument('--model',
                       default='dammyogt/english-to-hausa-translation',
                       help='Translation model to use')
    parser.add_argument('--splits',
                       nargs='+',
                       default=['train', 'dev', 'test'],
                       help='Which splits to translate')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🌍 ENGLISH → HAUSA TRANSLATION")
    print("=" * 70)
    print(f"📁 Input:  {args.input_dir}")
    print(f"📁 Output: {args.output_dir}")
    print(f"🤖 Model:  {args.model}")
    print(f"📊 Splits: {args.splits}")
    print("=" * 70)
    
    # Load model
    translator = setup_model(args.model)
    
    # Translate each split
    total_samples = 0
    for split in args.splits:
        input_file = os.path.join(args.input_dir, f'{split}.tsv')
        output_file = os.path.join(args.output_dir, f'{split}.tsv')
        
        if not os.path.exists(input_file):
            print(f"\n⚠️  File not found: {input_file}")
            continue
        
        try:
            n_samples = translate_dataset(input_file, output_file, translator)
            total_samples += n_samples
        except Exception as e:
            print(f"\n❌ Error processing {split}: {e}")
            continue
    
    print("\n" + "=" * 70)
    print("🎉 TRANSLATION COMPLETE!")
    print("=" * 70)
    print(f"📊 Total samples translated: {total_samples}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"\n✅ Ready for training!")
    print(f"   Next: Update config and run multilayer analysis")
    print("=" * 70)

if __name__ == "__main__":
    main()
