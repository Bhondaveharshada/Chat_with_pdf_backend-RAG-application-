import express from 'express';
import cors from "cors";
import multer from 'multer';
import session from 'express-session';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/pinecone";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import fs from 'fs';
import { Groq } from 'groq-sdk';
import { v4 as uuidv4 } from 'uuid';
import 'dotenv/config';

const app = express();
app.use(cors());

app.use(session({
  secret: 'SECRET', // Change this to a strong secret in production
  resave: false,
  saveUninitialized: true,
  cookie: { secure: false } // Set to true only if using HTTPS
}));

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });



const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, 'upload/');
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
    cb(null, `${uniqueSuffix}-${file.originalname}`);
  },
});

const upload = multer({ storage: storage });

// Initialize Pinecone
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY ,
});

// Initialize HuggingFace embeddings
const embeddings = new HuggingFaceInferenceEmbeddings({
  apiKey: process.env.HF_API_KEY ,
   model: "sentence-transformers/all-MiniLM-L6-v2", 
});

app.get('/', (req, res) => {
  return res.json({ status: 'All Good!' });
});



app.post('/upload/pdf', upload.single('pdf'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    // Step 1: Load the PDF
    const filePath = req.file.path;
    const loader = new PDFLoader(filePath, {
      splitPages: true,
    });
    
    const rawDocs = await loader.load();

    // Step 2: Split documents into optimal chunks
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 500, 
      chunkOverlap: 200, 
     // separators: ["\n\n", "\n", " ", ""] // Default separators
    });

    const docs = await textSplitter.splitDocuments(rawDocs);

    // Step 3: Prepare Pinecone connection
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME || "pdf-chat");
    const namespace = uuidv4();
    req.session.namespace = namespace;
    
    // Step 4: Create vector store and upsert with batching
    const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
      pineconeIndex,
      namespace,
    });

    // Process in batches to avoid timeouts/rate limits
    const batchSize = 100;
    for (let i = 0; i < docs.length; i += batchSize) {
      const batch = docs.slice(i, i + batchSize);
      await vectorStore.addDocuments(batch);
      console.log(`Processed batch ${i / batchSize + 1}/${Math.ceil(docs.length / batchSize)}`);
    }

    // Alternative direct upsert with more control:
    /*
    const embeddingsList = await embeddings.embedDocuments(docs.map(d => d.pageContent));
    const vectors = docs.map((doc, idx) => ({
      id: `${namespace}-${idx}`,
      metadata: {
        ...doc.metadata,
        text: doc.pageContent,
        source: req.file.originalname
      },
      values: embeddingsList[idx]
    }));

    // Upsert in batches
    for (let i = 0; i < vectors.length; i += batchSize) {
      const batch = vectors.slice(i, i + batchSize);
      await pineconeIndex.namespace(namespace).upsert(batch);
    }
    */

  
    fs.unlinkSync(filePath);

    return res.json({ 
      message: 'PDF processed and stored successfully',
      namespace,
      chunkCount: docs.length, 
      originalPageCount: rawDocs.length
    });
    
  } catch (error) {
    console.error('Error processing PDF:', error);
    if (req.file?.path) fs.unlinkSync(req.file.path);
    return res.status(500).json({ 
      error: 'Failed to process PDF',
      details: error.message,
      stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
    });
  }
});




app.post('/query', express.json(), async (req, res) => {
  try {
    const namespace = "6763088c-17a1-433d-9f7f-066974723312"
    //const namespace = req.session.namespace;
    const { question } = req.body;
    
    if (!question) {
      return res.status(400).json({ error: 'Question is required' });
    }
    
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME || "pdf-chat");
    const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
      pineconeIndex,
      namespace,
    });

    const results = await vectorStore.similaritySearch(question, 5); // Get more context
     
    const context = results.map((doc, i) => 
      `[Context ${i+1}]: ${doc.pageContent}\nSource: ${doc.metadata.source || 'PDF'}`)
      .join('\n\n');

  
    const SYSTEM_PROMPT = `
    You are an AI assistant that answers questions based on the provided context from PDF documents.
    - Always stay truthful to the context
    - If you don't know, say you don't know
    - Keep answers concise and accurate
    - give all the details (can give in paragraph) 
    
    Context:
     ${JSON.stringify(context)}
    `;

    const chatCompletion = await groq.chat.completions.create({
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: question }
      ],
      model: "llama-3.3-70b-versatile", // Or "llama2-70b-4096"
      temperature: 0.3,
      max_tokens: 1024
    });

    // 5. Return both answer and sources
    return res.json({
      answer: chatCompletion.choices[0]?.message?.content,
      sources: results.map(doc => ({
        content: doc.pageContent,
        metadata: doc.metadata
      }))
    });
    
  } catch (error) {
    console.error('Error querying:', error);
    return res.status(500).json({ 
      error: 'Failed to process query',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
});

const PORT = process.env.PORT || 8000;
app.listen(PORT, () => {
  console.log(`Server started at ${PORT}`);
});