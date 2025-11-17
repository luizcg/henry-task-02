#!/usr/bin/env python3
"""
Script to convert CSV Q&A dataset to plain text document.
Transforms Financial-QA CSV into structured Q&A format for RAG system.
"""

import csv
import os


def convert_csv_to_text(csv_path: str, output_path: str, max_items: int = None):
    """
    Convert CSV Q&A dataset to plain text document.
    
    Args:
        csv_path: Path to input CSV file
        output_path: Path to output text file
        max_items: Maximum number of Q&A pairs to include (None = all)
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(csv_path, 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        
        with open(output_path, 'w', encoding='utf-8') as text_file:
            # Write header
            text_file.write("=" * 80 + "\n")
            text_file.write("NVIDIA FINANCIAL INFORMATION - FAQ DOCUMENT\n")
            text_file.write("Source: 10-K Filing Q&A Dataset\n")
            text_file.write("=" * 80 + "\n\n")
            
            count = 0
            for row in reader:
                if max_items and count >= max_items:
                    break
                
                question = row['question'].strip()
                answer = row['answer'].strip()
                context = row.get('context', '').strip()
                ticker = row.get('ticker', '').strip()
                filing = row.get('filing', '').strip()
                
                # Write Q&A pair
                text_file.write(f"Q{count + 1}: {question}\n\n")
                text_file.write(f"A{count + 1}: {answer}\n\n")
                
                # Add context if available and different from answer
                if context and context != answer:
                    text_file.write(f"Context: {context}\n\n")
                
                # Add metadata
                if ticker or filing:
                    text_file.write(f"Source: {ticker} - {filing}\n")
                
                text_file.write("-" * 80 + "\n\n")
                count += 1
            
            # Write footer
            text_file.write("=" * 80 + "\n")
            text_file.write(f"Total Q&A Pairs: {count}\n")
            text_file.write("=" * 80 + "\n")
    
    print(f"✅ Successfully converted {count} Q&A pairs to {output_path}")
    
    # Calculate file size
    file_size = os.path.getsize(output_path)
    print(f"📄 Output file size: {file_size:,} bytes ({file_size / 1024:.2f} KB)")
    
    # Estimate word count (rough)
    with open(output_path, 'r', encoding='utf-8') as f:
        content = f.read()
        word_count = len(content.split())
    print(f"📝 Estimated word count: {word_count:,} words")


if __name__ == "__main__":
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "dataset", "Financial-QA-10k.csv")
    output_path = os.path.join(base_dir, "data", "faq_document.txt")
    
    # Convert all Q&A pairs (or set max_items to limit)
    # For testing, you can set max_items=50 or 100
    convert_csv_to_text(csv_path, output_path, max_items=None)
    
    print("\n🎉 Conversion complete!")
