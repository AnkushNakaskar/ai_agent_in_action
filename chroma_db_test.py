import chromadb

# Initialize the Chroma client
client = chromadb.Client()
collection = client.create_collection(name="my_collection")
# Sample data
documents = ["Document 1 text", "Document Ankush 2 text", "Document 3 text"]
ids = ["doc1", "doc2", "doc3"]

# Add documents to the collection
collection.add(documents=documents, ids=ids)
# Query the collection
results = collection.query(query_texts=["Ankush"], n_results=1)

# Display results
print(results)

# https://docs.trychroma.com/docs/overview/getting-started
# https://www.datacamp.com/tutorial/chromadb-tutorial-step-by-step-guide
#https://www.databasemart.com/blog/how-to-install-and-use-chromadb?srsltid=AfmBOooFRVjT89EpWbrs7wsS9wdntS-1c-8QGvc7RHur8oj7Qlp7xorY