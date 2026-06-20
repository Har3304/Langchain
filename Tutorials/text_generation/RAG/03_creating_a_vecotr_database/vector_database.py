import chromadb

client = chromadb.Client()

collection_name = client.get_or_create_collection(name='mycollection')


documents = [
    {'id':'doc1','text': 'Hello world!'},
    {'id':'doc2','text': 'How are you today?'},
    {'id':'doc3','text': 'Good, bye see you later!'}
]

for doc in documents:
  collection_name.upsert(ids=doc['id'], documents=doc['text'])

query = 'Hello world!'

results = collection_name.query(query_texts=[query], n_results=3)

print(results)
