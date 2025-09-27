import numpy as np
import pathlib
from typing import List, Dict, Any
from tqdm import tqdm
import re
from google import genai
from google.genai import types
import fitz  # PyMuPDF

class PDFClusterSemanticChunker:
    def __init__(self, gemini_model="gemini-embedding-001", max_chunk_size=400, min_chunk_size=50):
        self.client = genai.Client()
        self.model = gemini_model
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_cluster = max_chunk_size // min_chunk_size
        
    def _extract_text_with_pages(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF with page information."""
        doc = fitz.open(pdf_path)
        text_with_pages = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            # Split page text into paragraphs/sentences
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            for para in paragraphs:
                # Further split long paragraphs into sentences
                sentences = self._split_into_sentences(para)
                for sentence in sentences:
                    if len(sentence.strip()) > 10:  # Filter very short sentences
                        text_with_pages.append({
                            'text': sentence.strip(),
                            'page': page_num + 1  # 1-indexed pages
                        })
        
        doc.close()
        return text_with_pages
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using basic punctuation."""
        # Simple sentence splitting - can be improved with more sophisticated methods
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _token_count(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        return len(text.split()) * 1.3  # Rough token estimation
    
    def _split_by_size(self, text_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split text items to ensure they don't exceed min_chunk_size."""
        result = []
        
        for item in text_items:
            text = item['text']
            page = item['page']
            
            if self._token_count(text) <= self.min_chunk_size:
                result.append(item)
            else:
                # Split large text into smaller chunks
                words = text.split()
                current_chunk = []
                
                for word in words:
                    current_chunk.append(word)
                    if self._token_count(' '.join(current_chunk)) >= self.min_chunk_size:
                        result.append({
                            'text': ' '.join(current_chunk),
                            'page': page
                        })
                        current_chunk = []
                
                # Add remaining words
                if current_chunk:
                    result.append({
                        'text': ' '.join(current_chunk),
                        'page': page
                    })
        
        return result
    
    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts using Gemini."""
        result = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(
                output_dimensionality=768,
                task_type="QUESTION_ANSWERING"
            )
        )
        
        embeddings = []
        for embedding_obj in result.embeddings:
            values = np.array(embedding_obj.values)
            normed = values / np.linalg.norm(values)
            embeddings.append(normed.tolist())
        
        return embeddings
    
    def _get_similarity_matrix(self, text_items: List[Dict[str, Any]]) -> np.ndarray:
        """Get similarity matrix for text items."""
        texts = [item['text'] for item in text_items]
        BATCH_SIZE = 100  # Gemini API allows max 100 requests per batch
        N = len(texts)
        embedding_matrix = None

        print(f"Computing embeddings for {N} text segments...")
        for i in tqdm(range(0, N, BATCH_SIZE), desc="Getting embeddings"):
            batch_texts = texts[i:i+BATCH_SIZE]
            embeddings = self._get_embeddings_batch(batch_texts)

            # Convert embeddings to numpy array
            batch_embedding_matrix = np.array(embeddings)

            # Append to main embedding matrix
            if embedding_matrix is None:
                embedding_matrix = batch_embedding_matrix
            else:
                embedding_matrix = np.concatenate((embedding_matrix, batch_embedding_matrix), axis=0)

        similarity_matrix = np.dot(embedding_matrix, embedding_matrix.T)
        return similarity_matrix

    def _calculate_reward(self, matrix: np.ndarray, start: int, end: int) -> float:
        """Calculate reward for a cluster segment."""
        sub_matrix = matrix[start:end+1, start:end+1]
        return np.sum(sub_matrix)

    def _optimal_segmentation(self, matrix: np.ndarray, max_cluster_size: int) -> List[tuple]:
        """Find optimal segmentation using dynamic programming."""
        mean_value = np.mean(matrix[np.triu_indices(matrix.shape[0], k=1)])
        matrix = matrix - mean_value  # Normalize the matrix
        np.fill_diagonal(matrix, 0)  # Set diagonal to 0

        n = matrix.shape[0]
        dp = np.zeros(n)
        segmentation = np.zeros(n, dtype=int)

        for i in range(n):
            for size in range(1, max_cluster_size + 1):
                if i - size + 1 >= 0:
                    reward = self._calculate_reward(matrix, i - size + 1, i)
                    adjusted_reward = reward
                    if i - size >= 0:
                        adjusted_reward += dp[i - size]
                    if adjusted_reward > dp[i]:
                        dp[i] = adjusted_reward
                        segmentation[i] = i - size + 1

        clusters = []
        i = n - 1
        while i >= 0:
            start = segmentation[i]
            clusters.append((start, i))
            i = start - 1

        clusters.reverse()
        return clusters
    
    def chunk_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Chunk a PDF using cluster semantic chunking with page information.
        
        Returns:
            List of chunks with metadata including page numbers.
        """
        print(f"Processing PDF: {pdf_path}")
        
        # Extract text with page information
        text_items = self._extract_text_with_pages(pdf_path)
        
        if not text_items:
            return []
        
        print(f"Extracted {len(text_items)} text segments from PDF")
        
        # Split by size to ensure no segment exceeds min_chunk_size
        text_items = self._split_by_size(text_items)
        
        print(f"After size splitting: {len(text_items)} segments")
        
        # Get similarity matrix
        similarity_matrix = self._get_similarity_matrix(text_items)
        
        # Find optimal clusters
        print("Finding optimal clusters...")
        clusters = self._optimal_segmentation(similarity_matrix, max_cluster_size=self.max_cluster)
        
        # Create final chunks with metadata
        chunks = []
        for chunk_id, (start, end) in enumerate(clusters):
            # Combine texts in cluster
            cluster_items = text_items[start:end+1]
            chunk_text = ' '.join([item['text'] for item in cluster_items])
            
            # Get page range
            pages = [item['page'] for item in cluster_items]
            page_start = min(pages)
            page_end = max(pages)
            
            chunk = {
                'chunk_id': f'chunk_{chunk_id}',
                'original_index': chunk_id,
                'content': chunk_text,
                'page_start': page_start,
                'page_end': page_end,
                'token_count': self._token_count(chunk_text)
            }
            
            chunks.append(chunk)
        
        print(f"Created {len(chunks)} semantic clusters")
        return chunks
    
    def create_dataset_from_pdf(self, pdf_path: str, doc_id: str = None) -> Dict[str, Any]:
        """
        Create a dataset entry from a PDF file.
        
        Returns:
            Dictionary with document and chunks information.
        """
        if doc_id is None:
            doc_id = pathlib.Path(pdf_path).stem
        
        chunks = self.chunk_pdf(pdf_path)
        
        dataset_entry = {
            'doc_id': doc_id,
            'original_uuid': doc_id,
            'pdf_path': pdf_path,
            'chunks': chunks
        }
        
        return dataset_entry

# Example usage
def main():
    # Initialize the chunker
    chunker = PDFClusterSemanticChunker(
        max_chunk_size=400,
        min_chunk_size=50
    )
    
    # Process a PDF
    pdf_path = 'data/your_document.pdf'  # Replace with your PDF path
    
    try:
        # Create dataset entry
        dataset_entry = chunker.create_dataset_from_pdf(pdf_path)
        
        print(f"\nProcessed PDF: {pdf_path}")
        print(f"Document ID: {dataset_entry['doc_id']}")
        print(f"Number of chunks: {len(dataset_entry['chunks'])}")
        
        # Show sample chunks
        for i, chunk in enumerate(dataset_entry['chunks'][:3]):
            print(f"\nChunk {i+1}:")
            print(f"Pages: {chunk['page_start']}-{chunk['page_end']}")
            print(f"Token count: {chunk['token_count']}")
            print(f"Content preview: {chunk['content'][:200]}...")
        
        # Save to JSON (optional)
        import json
        with open(f'data/{dataset_entry["doc_id"]}_chunks.json', 'w') as f:
            json.dump([dataset_entry], f, indent=2)
        
        print(f"\nSaved chunks to data/{dataset_entry['doc_id']}_chunks.json")
        
    except Exception as e:
        print(f"Error processing PDF: {e}")

if __name__ == "__main__":
    main()