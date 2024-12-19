import pandas as pd
import re
import spacy
from mpi4py import MPI
import numpy as np
import time

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Get the rank of the process
size = comm.Get_size()  # Get the total number of processes

# Load spaCy model in each worker process
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

# Preprocessing function
def preprocess_text_fast(text):
    # Clean text (removing URLs, punctuation, etc.)
    text = re.sub(r'http\S+|www\S+|@\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()

    # Tokenization and lemmatization in one pass
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc 
                         if not token.is_stop and token.pos_ not in ['ADP', 'DET'] and not token.is_digit]

    return ' '.join(lemmatized_tokens)

# Function to divide data among MPI processes
def distribute_data(df):
    if rank == 0:  # Root process (rank 0) splits the data and sends to other processes
        # Split the data into chunks (preserving all columns)
        chunks = np.array_split(df, size)
    else:
        chunks = None
    chunk = comm.scatter(chunks, root=0)  # Scatter chunks to all processes
    return chunk

# Function to gather results from all processes
def gather_results(results):
    all_results = comm.gather(results, root=0)  # Gather results from all processes to the root process
    if rank == 0:
        # Combine all the chunks back into a single DataFrame
        combined_df = pd.concat(all_results, ignore_index=True)
        return combined_df
    else:
        return None

# Main function
if __name__ == "__main__":
    start = time.time()
    if rank == 0:
        # Load the data (only on root process)
        data = pd.read_csv('train.csv', encoding='latin1')
        
        # Sample a smaller subset for testing
        data = data.sample(1000, random_state=42)  # Use a sample size of 1000 rows
        print("Number of rows sampled for testing:", len(data))
    else:
        data = None

    # Distribute data chunks to all processes
    chunk = distribute_data(data)

    # Each process preprocesses its chunk of data
    chunk['processed_text'] = chunk['text'].apply(preprocess_text_fast)

    # Gather all the processed results
    processed_df = gather_results(chunk)

    # Save the results (only in the root process)
    if rank == 0:
        # Save all columns along with the new 'processed_text' column
        processed_df.to_csv('combined_output_small.csv', index=False)
        print("Output saved to 'combined_output_small.csv'")
        print("Time taken:", time.time() - start)
