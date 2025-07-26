import fs from 'fs';
import pdfParse from 'pdf-parse';
import axios from 'axios';
import dotenv from 'dotenv';
import { Pinecone } from '@pinecone-database/pinecone';

dotenv.config();

// Initialize Pinecone client
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY
});
const index = pinecone.Index('chatwithpdffrag');

// Main processing function
export async function processPdfAndStoreInPinecone(filePath) {
  try {
    // Step 1: Read PDF
    const buffer = fs.readFileSync(filePath);
    const data = await pdfParse(buffer);
    const text = data.text;

    // Step 2: Chunk text
    const chunks = chunkText(text);

    // Step 3: Embed and Upsert
    const vectors = [];

    for (let i = 0; i < chunks.length; i++) {
      const embedding = await getEmbedding(chunks[i]);
      vectors.push({
        id: `chunk-${Date.now()}-${i}`,
        values: embedding,
        metadata: { text: chunks[i] }
      });
    }

    await index.upsert({ vectors });
    console.log('✅ Successfully uploaded to Pinecone.');
  } catch (err) {
    console.error('❌ Error processing PDF:', err);
    throw err;
  }
}

// Helper: Text chunking with overlap
function chunkText(text, chunkSize = 500, overlap = 50) {
  const chunks = [];
  for (let i = 0; i < text.length; i += chunkSize - overlap) {
    chunks.push(text.slice(i, i + chunkSize));
  }
  return chunks;
}

// Helper: Get embedding from Hugging Face
async function getEmbedding(text) {
  const HF_API_URL = 'h';

  const response = await axios.post(
    HF_API_URL,
    { inputs: text },
    {
      headers: {
        Authorization: `Bearer ${process.env.HF_API_KEY}`,
        'Content-Type': 'application/json'
      }
    }
  );

  return response.data.embedding || response.data[0];
}
